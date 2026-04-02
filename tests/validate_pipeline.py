#!/usr/bin/env python3
"""
validate_pipeline.py
Processes a single FT3 file using the SAME Python logic as the production
pipeline (FT3_2_STL_2.py + stl_2_sph_harm_new.py) and compares the
spherical harmonic weights against the C++ converter's CSV output.

Usage:
  python3 validate_pipeline.py <ft3_file> <cpp_csv> [--lmax 14]

Example:
  ./converter --input test_input --output test_output --lmax 14
  python3 validate_pipeline.py F28310.ft3 test_output/32x6mm_eps10.csv
"""

import argparse
import struct
import numpy as np
import scipy.special
import pandas as pd
import sys
from collections import namedtuple

Bubble = namedtuple('Bubble', ['positions', 'connectivity', 'points'])

# ── FT3 reader (mirrors FT3_2_STL_2.py) ────────────────────────────────

def ft3int(fid):
    return int.from_bytes(fid.read(4), "little", signed=True)

def ft3double(fid):
    return struct.unpack('d', fid.read(8))[0]

def ft3skip(fid, n):
    fid.read(n)


def load_ft3(fname):
    fid = open(fname, "rb")
    cycle = ft3int(fid)

    ft3skip(fid, 4)
    time_val = ft3double(fid)
    time_rounded = round(time_val * 1e5)
    ft3skip(fid, 24)

    nx = ft3int(fid); ft3skip(fid, 4)
    ny = ft3int(fid); ft3skip(fid, 4)
    nz = ft3int(fid); ft3skip(fid, 4)

    dx = ft3double(fid)
    dy = ft3double(fid)
    dz = ft3double(fid)

    nph = ft3int(fid); ft3skip(fid, 4)
    neli = ft3int(fid)

    ft3skip(fid, 300)
    ncells = (nz+2) * (ny+2) * (nx+2)
    ft3skip(fid, 8 * ncells * nph)
    ft3skip(fid, 8 * ncells)
    ft3skip(fid, 8 * ncells)
    ft3skip(fid, 8 * ncells)
    ft3skip(fid, 8 * ncells)

    bubbles = []
    for i in range(neli):
        nmar = ft3int(fid)
        npos = ft3int(fid)
        pointpos = np.reshape(np.fromfile(fid, dtype=float, count=npos*3), (npos, 3))
        connmrk = np.reshape(np.fromfile(fid, dtype=np.int32, count=nmar*3*2), (nmar*3, 2))
        points = np.reshape(connmrk[:, 1], (nmar, 3))
        bubbles.append(Bubble(pointpos, connmrk[:, 0].reshape(nmar, 3), points))

    fid.close()
    return {
        'cycle': cycle, 'time': time_val, 'time_rounded': time_rounded,
        'nx': nx, 'ny': ny, 'nz': nz, 'dx': dx, 'dy': dy, 'dz': dz,
        'nph': nph, 'neli': neli, 'bubbles': bubbles
    }


# ── Real SH (matches sph_harm_functions.py) ─────────────────────────────

def real_sph_harm_matrix(theta, phi, l_max):
    k = (l_max + 1) ** 2
    sph = np.zeros((len(theta), k))
    for j in range(k):
        l = int(np.floor(np.sqrt(j)))
        m = j - l**2 - l
        Y_lm = scipy.special.sph_harm(m, l, theta, phi)
        if m < 0:
            sph[:, j] = np.sqrt(2) * Y_lm.imag
        elif m == 0:
            sph[:, j] = Y_lm.real
        else:
            sph[:, j] = np.sqrt(2) * Y_lm.real
    return sph


# ── Bounding-box center (matches PyVista PolyData.center) ───────────────

def bbox_center(positions):
    return (positions.min(axis=0) + positions.max(axis=0)) / 2.0


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ft3_file', help='Path to FT3 file')
    parser.add_argument('cpp_csv', help='Path to C++ converter CSV output')
    parser.add_argument('--lmax', type=int, default=14)
    args = parser.parse_args()

    l_max = args.lmax
    num_orbs = (l_max + 1) ** 2

    # Load and process FT3
    data = load_ft3(args.ft3_file)
    print(f"FT3: cycle={data['cycle']}, time={data['time']}, neli={data['neli']}")

    py_weights = []
    py_centers = []

    for bi in range(data['neli']):
        bub = data['bubbles'][bi]
        center = bbox_center(bub.positions)
        py_centers.append(center)

        # Center positions
        pts = bub.positions - center
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
        theta = np.arctan2(pts[:, 1], pts[:, 0])
        phi = np.arccos(np.clip(pts[:, 2] / np.clip(r, 1e-15, None), -1, 1))

        # Filter degenerate (r≈0) vertices
        mask = r > 1e-20
        r_f, theta_f, phi_f = r[mask], theta[mask], phi[mask]

        # Build SH matrix and solve
        A = real_sph_harm_matrix(theta_f, phi_f, l_max)
        weights, _, _, _ = np.linalg.lstsq(A, r_f, rcond=None)
        py_weights.append(weights)

    # Load C++ output
    df_cpp = pd.read_csv(args.cpp_csv, index_col=0)
    print(f"C++ CSV: {len(df_cpp)} rows")

    # Compare
    orb_cols = [f'orb_{j}' for j in range(num_orbs)]
    max_pos_diff = 0.0
    max_weight_diff = 0.0
    n_fail = 0

    for bi in range(data['neli']):
        row = df_cpp[df_cpp['bub_num'] == bi].iloc[0]

        # Position comparison
        cpp_pos = np.array([row['pos_x'], row['pos_y'], row['pos_z']])
        py_pos = py_centers[bi]
        pos_diff = np.max(np.abs(cpp_pos - py_pos))
        max_pos_diff = max(max_pos_diff, pos_diff)

        # Weight comparison
        cpp_w = np.array([row[c] for c in orb_cols])
        py_w = py_weights[bi]
        w_diff = np.max(np.abs(cpp_w - py_w))
        max_weight_diff = max(max_weight_diff, w_diff)

        if w_diff > 1e-5:
            n_fail += 1
            worst_j = np.argmax(np.abs(cpp_w - py_w))
            l_w = int(np.floor(np.sqrt(worst_j)))
            m_w = worst_j - l_w**2 - l_w
            print(f"  Bubble {bi}: max weight diff = {w_diff:.2e} at orb_{worst_j} (l={l_w}, m={m_w})")
            print(f"    Python: {py_w[worst_j]:.15e}")
            print(f"    C++:    {cpp_w[worst_j]:.15e}")

    print(f"\nResults for {data['neli']} bubbles:")
    print(f"  Max position diff:  {max_pos_diff:.2e}")
    print(f"  Max weight diff:    {max_weight_diff:.2e}")

    TOLERANCE = 1e-5
    if max_weight_diff < TOLERANCE and max_pos_diff < TOLERANCE:
        print(f"\n✓ PASS: All values match within {TOLERANCE}")
        return 0
    else:
        print(f"\n✗ FAIL: {n_fail}/{data['neli']} bubbles have weight diffs >= {TOLERANCE}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
