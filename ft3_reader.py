"""
ParaView Python plugin to read GLS3D .ft3 binary dump files.

Usage:
  1. In ParaView: Tools -> Manage Plugins -> Load New -> select this file
  2. File -> Open -> select any F*.ft3 file
  3. All F*.ft3 files in the same directory are loaded as a time series

Output: vtkMultiBlockDataSet
  - Block 0: "Eulerian Grid" (vtkImageData with cell-centered fields)
  - Block 1..N: "Bubble 0", "Bubble 1", ... (vtkPolyData triangle meshes)
"""

import os
import re
import glob
import struct
import numpy as np

from vtkmodules.vtkCommonDataModel import (
    vtkImageData,
    vtkMultiBlockDataSet,
    vtkPolyData,
    vtkCellArray,
)
from vtkmodules.vtkCommonCore import vtkPoints, vtkDoubleArray, vtkIdTypeArray
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain, smhint

# --------------------------------------------------------------------------
# Binary format constants
# --------------------------------------------------------------------------
HEADER_SIZE = 400  # bytes


def _read_header(f):
    """Read the 400-byte header and return a dict of parameters."""
    buf = f.read(HEADER_SIZE)
    if len(buf) < HEADER_SIZE:
        raise IOError("File too short to contain a valid FT3 header")

    hdr = {}
    # Offset 0: cycle (int) + padding (int)
    hdr['cycle'] = struct.unpack_from('<i', buf, 0)[0]
    # Offset 8: tim (double)
    hdr['tim'] = struct.unpack_from('<d', buf, 8)[0]
    # Offset 16: OriginShift[0..2]
    hdr['origin'] = struct.unpack_from('<3d', buf, 16)
    # Offset 40: nx (int), pad, ny (int), pad, nz (int), pad
    hdr['nx'] = struct.unpack_from('<i', buf, 40)[0]
    hdr['ny'] = struct.unpack_from('<i', buf, 48)[0]
    hdr['nz'] = struct.unpack_from('<i', buf, 56)[0]
    # Offset 64: dx, dy, dz (doubles)
    hdr['dx'], hdr['dy'], hdr['dz'] = struct.unpack_from('<3d', buf, 64)
    # Offset 88: nph (int), pad
    hdr['nph'] = struct.unpack_from('<i', buf, 88)[0]
    # Offset 96: neli (int), pad
    hdr['neli'] = struct.unpack_from('<i', buf, 96)[0]
    # Offset 160: PeriodicBoundaryX, Y, Z (3 ints)
    hdr['periodicX'] = struct.unpack_from('<i', buf, 160)[0]
    hdr['periodicY'] = struct.unpack_from('<i', buf, 164)[0]
    hdr['periodicZ'] = struct.unpack_from('<i', buf, 168)[0]
    # Offset 184: dt (double)
    hdr['dt'] = struct.unpack_from('<d', buf, 184)[0]
    # Offset 392: version (int)
    hdr['version'] = struct.unpack_from('<i', buf, 392)[0]
    # Offset 396: UseMassTransfer (int)
    hdr['useMassTransfer'] = struct.unpack_from('<i', buf, 396)[0]

    return hdr


def _read_eulerian_fields(f, hdr):
    """Read phase fractions, pressure, and velocities. Returns dict of arrays."""
    nx, ny, nz = hdr['nx'], hdr['ny'], hdr['nz']
    nph = hdr['nph']
    ncells_full = (nx + 2) * (ny + 2) * (nz + 2)

    fields = {}

    # --- Phase fractions: nph blocks of (nz+2)*(ny+2)*(nx+2) doubles ---
    # File loop order: p(outer) -> k -> j -> i(fastest)
    # numpy reshape with C order: last index varies fastest -> shape (nph, nz+2, ny+2, nx+2)
    fff_raw = np.frombuffer(f.read(nph * ncells_full * 8), dtype='<f8')
    fff_raw = fff_raw.reshape(nph, nz + 2, ny + 2, nx + 2)
    for p in range(nph):
        # Trim ghost cells: keep interior [1:nz+1, 1:ny+1, 1:nx+1]
        arr = fff_raw[p, 1:nz + 1, 1:ny + 1, 1:nx + 1]
        fields[f'fff_p{p + 1}'] = np.ascontiguousarray(arr)

    # --- Pressure: (nz+2)*(ny+2)*(nx+2) doubles ---
    ppp_raw = np.frombuffer(f.read(ncells_full * 8), dtype='<f8')
    ppp_raw = ppp_raw.reshape(nz + 2, ny + 2, nx + 2)
    fields['pressure'] = np.ascontiguousarray(ppp_raw[1:nz + 1, 1:ny + 1, 1:nx + 1])

    # --- u_x: staggered in x ---
    # Written as (nz+2) * (ny+2) rows, each row = (nx+1) values + 1 dummy = (nx+2) doubles
    ux_raw = np.frombuffer(f.read(ncells_full * 8), dtype='<f8')
    ux_raw = ux_raw.reshape(nz + 2, ny + 2, nx + 2)
    # u_x is defined at faces i=0..nx. Cell I (1-based) has faces I-1 and I.
    # Interior cells are I=1..nx -> faces I-1=0..nx-1 and I=1..nx
    # In 0-based array indexing: face indices 0..nx-1 and 1..nx
    # Trim k to [1:nz+1], j to [1:ny+1]
    ux_cc = 0.5 * (ux_raw[1:nz + 1, 1:ny + 1, 0:nx] +
                   ux_raw[1:nz + 1, 1:ny + 1, 1:nx + 1])
    fields['u_x'] = np.ascontiguousarray(ux_cc)

    # --- u_y: staggered in y ---
    # Written as (nz+2) planes, each plane = (ny+1) rows of (nx+2) + 1 dummy row = (ny+2)*(nx+2)
    uy_raw = np.frombuffer(f.read(ncells_full * 8), dtype='<f8')
    uy_raw = uy_raw.reshape(nz + 2, ny + 2, nx + 2)
    # u_y at faces j=0..ny. Cell J (1-based) has faces J-1 and J.
    uy_cc = 0.5 * (uy_raw[1:nz + 1, 0:ny, 1:nx + 1] +
                   uy_raw[1:nz + 1, 1:ny + 1, 1:nx + 1])
    fields['u_y'] = np.ascontiguousarray(uy_cc)

    # --- u_z: staggered in z ---
    # Written as (nz+1) planes of (ny+2)*(nx+2) + 1 dummy plane = (nz+2)*(ny+2)*(nx+2)
    uz_raw = np.frombuffer(f.read(ncells_full * 8), dtype='<f8')
    uz_raw = uz_raw.reshape(nz + 2, ny + 2, nx + 2)
    # u_z at faces k=0..nz. Cell K (1-based) has faces K-1 and K.
    uz_cc = 0.5 * (uz_raw[0:nz, 1:ny + 1, 1:nx + 1] +
                   uz_raw[1:nz + 1, 1:ny + 1, 1:nx + 1])
    fields['u_z'] = np.ascontiguousarray(uz_cc)

    return fields


def _read_bubbles(f, hdr):
    """Read bubble surface meshes. Returns list of (points, triangles) tuples."""
    neli = hdr['neli']
    bubbles = []

    for _ in range(neli):
        # nmar (int), npos (int)
        raw = f.read(8)
        if len(raw) < 8:
            break
        nmar, npos = struct.unpack_from('<2i', raw, 0)

        # positon: npos * 3 doubles (x, y, z per node)
        pts = np.frombuffer(f.read(npos * 3 * 8), dtype='<f8').reshape(npos, 3)

        # Connectivity: nmar * 3 * 2 ints (interleaved connect/markpos)
        conn_raw = np.frombuffer(f.read(nmar * 3 * 2 * 4), dtype='<i4').reshape(nmar, 3, 2)
        # markpos is the second element in the interleaved pair -> [:, :, 1]
        triangles = conn_raw[:, :, 1].copy()

        bubbles.append((pts, triangles))

    return bubbles


def _build_image_data(hdr, fields):
    """Build vtkImageData for the Eulerian grid."""
    nx, ny, nz = hdr['nx'], hdr['ny'], hdr['nz']
    img = vtkImageData()
    # Point dimensions = cell dims + 1
    img.SetDimensions(nx + 1, ny + 1, nz + 1)
    img.SetOrigin(hdr['origin'][0], hdr['origin'][1], hdr['origin'][2])
    img.SetSpacing(hdr['dx'], hdr['dy'], hdr['dz'])

    ncells = nx * ny * nz

    # Add scalar fields as cell data
    for name, arr in fields.items():
        # VTK ImageData stores cells with x varying fastest (i + j*nx + k*nx*ny).
        # Our arrays are shaped (nz, ny, nx), so C-order ravel gives x-fastest.
        flat = arr.ravel(order='C')
        vtk_arr = vtkDoubleArray()
        vtk_arr.SetName(name)
        vtk_arr.SetNumberOfTuples(ncells)
        for i in range(ncells):
            vtk_arr.SetValue(i, flat[i])
        img.GetCellData().AddArray(vtk_arr)

    # Add a 3-component velocity vector
    ux = fields['u_x'].ravel(order='C')
    uy = fields['u_y'].ravel(order='C')
    uz = fields['u_z'].ravel(order='C')
    vel = vtkDoubleArray()
    vel.SetName('velocity')
    vel.SetNumberOfComponents(3)
    vel.SetNumberOfTuples(ncells)
    for i in range(ncells):
        vel.SetTuple3(i, ux[i], uy[i], uz[i])
    img.GetCellData().AddArray(vel)

    return img


def _build_polydata(pts_np, tri_np):
    """Build vtkPolyData for a single bubble surface."""
    poly = vtkPolyData()

    # Points
    points = vtkPoints()
    npts = pts_np.shape[0]
    points.SetNumberOfPoints(npts)
    for i in range(npts):
        points.SetPoint(i, pts_np[i, 0], pts_np[i, 1], pts_np[i, 2])
    poly.SetPoints(points)

    # Triangles
    nmar = tri_np.shape[0]
    cells = vtkCellArray()
    for i in range(nmar):
        cells.InsertNextCell(3)
        cells.InsertCellPoint(int(tri_np[i, 0]))
        cells.InsertCellPoint(int(tri_np[i, 1]))
        cells.InsertCellPoint(int(tri_np[i, 2]))
    poly.SetPolys(cells)

    return poly


def _apply_periodic_bc(bubbles, hdr):
    """Wrap bubbles into the domain and create periodic images where needed.

    For each bubble:
      1. Shift all nodes so the center of mass is inside the domain.
      2. If any node still extends past a periodic boundary, create an image
         (copy shifted by ±domain_length) on that side.
      3. For edges/corners (multiple directions), create all combinations.
    """
    from itertools import product as iterproduct

    domain_lengths = np.array([hdr['nx'] * hdr['dx'],
                               hdr['ny'] * hdr['dy'],
                               hdr['nz'] * hdr['dz']])
    domain_lo = np.array(hdr['origin'])
    domain_hi = domain_lo + domain_lengths
    periodic = [bool(hdr['periodicX']), bool(hdr['periodicY']), bool(hdr['periodicZ'])]

    result = []
    for pts, tris in bubbles:
        # Compute center of mass
        com = pts.mean(axis=0)

        # Wrap center of mass into domain for periodic directions
        shift = np.zeros(3)
        for d in range(3):
            if periodic[d]:
                shift[d] = domain_lo[d] + (com[d] - domain_lo[d]) % domain_lengths[d] - com[d]

        pts_shifted = pts + shift

        # For each periodic direction, determine if images are needed
        shifts_per_dir = []
        for d in range(3):
            dir_shifts = [0.0]
            if periodic[d]:
                if pts_shifted[:, d].min() < domain_lo[d]:
                    dir_shifts.append(domain_lengths[d])
                if pts_shifted[:, d].max() > domain_hi[d]:
                    dir_shifts.append(-domain_lengths[d])
            shifts_per_dir.append(dir_shifts)

        # Cartesian product gives all image combinations (including original at [0,0,0])
        for sx, sy, sz in iterproduct(*shifts_per_dir):
            offset = np.array([sx, sy, sz])
            result.append((pts_shifted + offset, tris))

    return result


def _read_ft3(filepath):
    """Read a complete .ft3 file and return (hdr, fields, bubbles)."""
    with open(filepath, 'rb') as f:
        hdr = _read_header(f)
        fields = _read_eulerian_fields(f, hdr)
        bubbles = _read_bubbles(f, hdr)
    bubbles = _apply_periodic_bc(bubbles, hdr)
    return hdr, fields, bubbles


def _discover_time_series(filepath):
    """Find all F*.ft3 files in the same directory and read their times.
    Returns sorted list of (time, filepath) tuples."""
    dirname = os.path.dirname(os.path.abspath(filepath))
    pattern = os.path.join(dirname, 'F*.ft3')
    files = glob.glob(pattern)

    series = []
    ft3_re = re.compile(r'F(\d+)\.ft3$')

    for fpath in files:
        basename = os.path.basename(fpath)
        m = ft3_re.match(basename)
        if not m:
            continue
        # Read time from header (offset 8, double)
        try:
            with open(fpath, 'rb') as fh:
                fh.seek(8)
                tim = struct.unpack('<d', fh.read(8))[0]
            series.append((tim, fpath))
        except (IOError, struct.error):
            continue

    # Sort by time (or by cycle number as fallback for identical times)
    series.sort(key=lambda x: x[0])
    return series


# --------------------------------------------------------------------------
# ParaView Plugin Class
# --------------------------------------------------------------------------
@smproxy.reader(
    name="GLS3DFT3Reader",
    label="GLS3D FT3 Reader",
    extensions="ft3",
    file_description="GLS3D FT3 simulation dump files",
)
class FT3Reader(VTKPythonAlgorithmBase):
    """ParaView reader for GLS3D .ft3 binary dump files."""

    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self,
            nInputPorts=0,
            nOutputPorts=1,
            outputType='vtkMultiBlockDataSet',
        )
        self._filename = None
        self._file_series = None  # list of (time, path)
        self._time_values = None  # numpy array of times

    @smproperty.stringvector(name="FileName", number_of_elements="1")
    @smdomain.filelist()
    @smhint.filechooser(extensions="ft3", file_description="GLS3D FT3 files")
    def SetFileName(self, filename):
        if self._filename != filename:
            self._filename = filename
            self._file_series = None
            self._time_values = None
            self.Modified()

    def _setup_series(self):
        """Discover the time series if not already done."""
        if self._file_series is not None:
            return
        if not self._filename:
            self._file_series = []
            self._time_values = np.array([], dtype=np.float64)
            return

        self._file_series = _discover_time_series(self._filename)
        if self._file_series:
            self._time_values = np.array([t for t, _ in self._file_series],
                                         dtype=np.float64)
        else:
            self._time_values = np.array([], dtype=np.float64)

    def _get_file_for_time(self, time):
        """Return the filepath closest to the requested time."""
        if not self._file_series:
            return self._filename
        idx = np.argmin(np.abs(self._time_values - time))
        return self._file_series[idx][1]

    def RequestInformation(self, request, inInfo, outInfo):
        self._setup_series()
        executive = self.GetExecutive()
        info = outInfo.GetInformationObject(0)

        if self._time_values is not None and len(self._time_values) > 0:
            info.Remove(executive.TIME_STEPS())
            for t in self._time_values:
                info.Append(executive.TIME_STEPS(), t)
            info.Remove(executive.TIME_RANGE())
            info.Append(executive.TIME_RANGE(), self._time_values[0])
            info.Append(executive.TIME_RANGE(), self._time_values[-1])

        return 1

    def RequestData(self, request, inInfo, outInfo):
        info = outInfo.GetInformationObject(0)
        output = self.GetOutputData(outInfo, 0)

        # Determine which time step is requested
        executive = self.GetExecutive()
        if info.Has(executive.UPDATE_TIME_STEP()):
            req_time = info.Get(executive.UPDATE_TIME_STEP())
        else:
            req_time = 0.0

        self._setup_series()
        filepath = self._get_file_for_time(req_time)

        if not filepath or not os.path.isfile(filepath):
            return 1

        # Read the file
        hdr, fields, bubbles = _read_ft3(filepath)

        # Build multi-block dataset
        mb = vtkMultiBlockDataSet()

        # Block 0: Eulerian grid
        img = _build_image_data(hdr, fields)
        mb.SetNumberOfBlocks(1 + len(bubbles))
        mb.SetBlock(0, img)
        mb.GetMetaData(0).Set(vtkMultiBlockDataSet.NAME(), "Eulerian Grid")

        # Blocks 1..N: Bubble surfaces
        for i, (pts, tris) in enumerate(bubbles):
            poly = _build_polydata(pts, tris)
            mb.SetBlock(i + 1, poly)
            mb.GetMetaData(i + 1).Set(vtkMultiBlockDataSet.NAME(), f"Bubble {i}")

        output.ShallowCopy(mb)

        # Set the actual time value on the output
        output.GetInformation().Set(output.DATA_TIME_STEP(), hdr['tim'])

        return 1
