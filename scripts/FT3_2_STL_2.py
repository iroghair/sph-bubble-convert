# Project to demonstrate extraction of a bubble from Front Tracking dump files
import numpy as np
from collections import namedtuple
import os
import struct
from multiprocessing import Pool
from functools import partial

Bubble = namedtuple('Bubble', ['positions', 'connectivity', 'points'])

def ft3int(fid):
  return int.from_bytes(fid.read(4), "little", signed=True)
  # return struct.unpack('i',fid.read(4))[0]
def ft3double(fid):
  return struct.unpack('d',fid.read(8))[0]
  # return float.from_bytes(fid.read(8), "little", signed=True)

def ft3skipbytes(fid,n):
  return fid.read(n)

class Ft3file:
  def __init__(self,fname,sname):
    # Set filename
    self.fname = fname
    self.sname = sname
    self.Bubbles = []

  def loadFT3(self):
    # Read through the binary data file format, and
    # store relevant parameters to extract bubble meshes
    fid = open(self.fname,"rb")
    cycle = ft3int(fid)
    print("Reading FT3 file: Cycle number", cycle)

    # print(struct.pack('f',0.4))
    # exit()
    # Skip bytes: 4 (dummy int) + 4*8 (time and originshift x,y,z)
    ft3skipbytes(fid,4) # Dummy
    self.time = ft3double(fid)
    self.time = round(self.time * 1e5)
    ft3skipbytes(fid,24) # originshift

    # Read grid dimensions: 
    self.nx = ft3int(fid)
    ft3skipbytes(fid,4) # Dummy
    self.ny = ft3int(fid)
    ft3skipbytes(fid,4) # Dummy
    self.nz = ft3int(fid)
    ft3skipbytes(fid,4) # Dummy

    # Skip bytes: 3*8: dx, dy, dz (doubles)
    # fid.read(24)
    dx = ft3double(fid)
    dy = ft3double(fid)
    dz = ft3double(fid)

    # Calculate the grid size and volume
    L = (self.nx)*dx, (self.ny)*dy, (self.nz)*dz
    self.volume_grid = L[0] * L[1] * L[2]

    self.nph = ft3int(fid)
    ft3skipbytes(fid,4) # Dummy
    self.neli = ft3int(fid)
    
    # Skip bytes in header: 7*4 + 4*8 + 4*4 + 28*8 = 300
    ft3skipbytes(fid,300)

    # Skip phase fractions: nph * (nz+2) * (ny+2) * (nx+2)
    self.ncells = (self.nz+2) * (self.ny+2) * (self.nx+2)
    ft3skipbytes(fid,8 * self.ncells * self.nph)

    # Skip pressure: (nz+2) * (ny+2) * (nx+2)
    ft3skipbytes(fid,8 * self.ncells)

    # Note: Skipping the velocity fields is easier than actually 
    # reading them, since the staggered velocity requires 1 cell less 
    # in the direction of the flow compared to the other directions, 
    # but this missing cell is still present in the ft3 file (as a 
    # dummy), to make sure that the field is still of size (nx+2)*(ny+2)*(nz+2).
    # So in case anyone wants to actually store the velocity fields, 
    # refer to the original ft3 file format to properly accommodate 
    # this!
    
    # Skip x-vel: (nz+2) * (ny+2) * (nx+2)
    ft3skipbytes(fid,8 * self.ncells)

    # Skip y-vel: nph * (nz+2) * (ny+2) * (nx+2)
    ft3skipbytes(fid,8 * self.ncells)

    # Skip z-vel: nph * (nz+2) * (ny+2) * (nx+2)
    ft3skipbytes(fid,8 * self.ncells)

    # We arrived at the bubble mesh definitions!
    for i in range(self.neli):
      nmar = ft3int(fid)
      npos = ft3int(fid)

      # Get the point positions from the file, reshape in a 3*npos array
      pointpos = np.reshape(
        np.fromfile(fid, dtype=float, count=npos * 3), (npos,3))

      # The connectivity and point numbers are stored 
      # alternatingly for each marker, e.g. for marker M
      # (connected_marker[M][0], point[M][0], connected_marker
      # [M][1], point[M][1], connected_marker[M][2], point[M]
      # [2]) --- The following first organizes these into 2 
      # columns (1 for connected markers, the next for the 
      # points), slices them into separate arrays, which are then
      # reshaped to hold 3 markers/points respectively for each 
      # marker.
      connmrk = np.reshape(
        np.fromfile(fid, dtype=np.int32, count=nmar * 3 * 2), (nmar*3,2))

      conn = np.reshape(connmrk[:,0],(nmar,3))
      points = np.reshape(connmrk[:,1],(nmar,3))

      # Add the bubble as raw data to the array
      self.Bubbles.append(Bubble(pointpos,conn,points))

  def getBubble(self,bubbleNumber):
    # Check if bubble number is within range, return arrays to position, marker and connectivity
    test = 0

  def plotBubblePoints(self,bubbleNumber):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
      self.Bubbles[bubbleNumber].positions[:,0],
      self.Bubbles[bubbleNumber].positions[:,1],
      self.Bubbles[bubbleNumber].positions[:,2])
    plt.show()

  def createSTLFromBubble(self,bubbleNumber,save=True):
    # numpy-stl is a Python library that can create STL files from numpy arrays
    from stl import mesh

    # Set up the mesh
    bubbleMesh = mesh.Mesh(np.zeros(
      self.Bubbles[bubbleNumber].points.shape[0], 
      dtype=mesh.Mesh.dtype))

    # Insert points for each face, and their position
    for i, f in enumerate(self.Bubbles[bubbleNumber].points):
      for j in range(3):
        bubbleMesh.vectors[i][j] = self.Bubbles[bubbleNumber].positions[f[j],:]

    # Assign correct time to file
    file_name = "F{}_bubble_{}.stl".format(self.time,bubbleNumber)
    file_name = os.path.join(self.sname, file_name)

    # Save the stl file
    if save:
      bubbleMesh.save(file_name)

    return bubbleMesh

def stl_volume(stl_mesh):
    volume = 0.0
    for i in range(len(stl_mesh.vectors)):
        v0, v1, v2 = stl_mesh.vectors[i]
        # Compute the signed volume of the tetrahedron
        v = np.dot(np.cross(v0, v1), v2) / 6.0
        volume += v
    return abs(volume)

def stl_area(stl_mesh):
    area = 0.0
    for i in range(len(stl_mesh.vectors)):
        v0, v1, v2 = stl_mesh.vectors[i]
        # Compute the area of the triangle
        a = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0
        area += a
    return area

# Redundant due to wrong calculations
def stl_sauter_mean_diameter(stl_mesh):
    # Calculate the Sauter mean diameter of the bubble
    # This is the volume divided by the surface area
    volume = stl_volume(stl_mesh)
    area = stl_area(stl_mesh)
    if area == 0:
        return 0.0
    return (6 * volume) / area

def volume_diameter(stl_mesh) -> float:
    # Calculate the volume of the bubble and convert to diameter
    volume = stl_volume(stl_mesh)
    diameter = (6 * volume / np.pi) ** (1/3)
    return diameter

def get_diameter(bubble_num: int, infile: Ft3file) -> int:
    bubbleMesh = infile.createSTLFromBubble(bubble_num, save=False)
    diameter = volume_diameter(bubbleMesh)
    return diameter * 1000 # Convert to millimeters

def get_folder_name(file: str, folder: str = "") -> str:
  # This function is called to get the folder name for the STL files
  fname = os.path.join(folder, file)
  infile = Ft3file(fname, "")
  infile.loadFT3()

  # Calculate the diameter of the bubble using multiprocessing
  with Pool() as p:
    partial_func = partial(get_diameter, infile=infile)
    diameters = p.map(partial_func, range(len(infile.Bubbles)))
  diameter = int(np.round(np.mean(diameters), 0))

  # Get the gas holdup of the bubble
  # Round the gas holdup to the nearest integer of 5
  volume = 1/6 * np.pi * (diameter / 1000) ** 3 * infile.neli
  gas_holdup = volume / infile.volume_grid
  gas_holdup = int(np.round(gas_holdup * 100 / 5, 0) * 5)

  folder_name = f"{infile.neli}x{diameter}mm_eps{gas_holdup:02d}"
  print("Folder name for file:", folder_name)
  return folder_name

def find_folder_name(folder_in: str, dir_out: str) -> str:
  # Find the first FT3 file in the folder to determine the output folder name
  for file in sorted(os.listdir(folder_in)):
    if ".ft3" not in file:
      continue

    folder_name = get_folder_name(file, folder_in)
    folder_out = os.path.join(dir_out, folder_name)
    print("Output folder: {}".format(folder_out))

    os.makedirs(folder_out, exist_ok=True)
    break

  return folder_out

def run(file: str, map: str, save: str):
    # This function is called when the script is run
    # It creates an instance of the Ft3file class and loads the FT3 file
    # Then it creates STL files for each bubble in the file
    if ".ft3" not in file:
        return

    os.makedirs(save, exist_ok=True)
    fname = os.path.join(map, file)

    infile = Ft3file(fname,save)
    infile.loadFT3()

    for i in range(len(infile.Bubbles)):
        infile.createSTLFromBubble(i)

if __name__ == "__main__":      
  # Set input and output folders
  root = os.path.dirname(os.path.dirname(__file__))
  folder_in = os.path.join(root, "convert_dir")
  dir_out = os.path.join(root, "bubbles_stl")

  # Loop over different folders
  # folder_in = os.path.join(dir_in, folder)

  print('Working in folder: {}'.format(folder_in))

    # Make folder for storing STL files
  folder_out = None

  ft3_files = [file for file in os.listdir(folder_in) if ".ft3" in file]
  if not ft3_files:
      raise RuntimeError(f"No .ft3 files found in {folder_in}")
    
  folder_out = find_folder_name(folder_in, dir_out)  # creates it + returns path
  print(f"[INFO] Saving STL files into: {folder_out}")
    # Apply multiprocessing to run the function on all files in the folder
    # This is done to speed up the process of creating STL files from FT3 files
    # Create a partial function to pass the folder_in and folder_out
  part_func = partial(run, map=folder_in, save=dir_out)

    # Loop over different files
  with Pool() as p:
      p.map(part_func, os.listdir(folder_in))
