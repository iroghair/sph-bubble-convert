#!/usr/bin/env python3
"""
validate_sph_harm.py
Compare real spherical harmonic values computed in C++ vs Python (scipy).

Usage:
  1.  make validate_sph && ./validate_sph   # → writes /tmp/cpp_sph_values.csv
  2.  python3 validate_sph_harm.py          # reads that CSV and compares

Asserts all values match to 5 decimal places (atol=1e-5).
"""

import numpy as np
import scipy.special
import sys
import os

# ── Same test points as validate_sph.cpp ───────────────────────────────
theta_az = np.array([0.0, 0.5, 1.2, 2.0, np.pi, 0.1, 2.5, 6.0])
phi_pol  = np.array([0.0, 0.3, 2.1, 1.5, np.pi/4, 2.9, 0.01, 1.0])

L_MAX    = 14
num_orbs = (L_MAX + 1) ** 2
N        = len(theta_az)


def real_sph_harm_py(theta, phi, l_max):
    """Compute real SH matrix matching sph_harm_functions.py :: sph_harm_fast."""
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


# ── Compute Python reference values ─────────────────────────────────────
Y_py = real_sph_harm_py(theta_az, phi_pol, L_MAX)

# ── Load C++ values ─────────────────────────────────────────────────────
cpp_csv = "/tmp/cpp_sph_values.csv"
if not os.path.exists(cpp_csv):
    print(f"ERROR: {cpp_csv} not found. Run:  make validate_sph && ./validate_sph")
    sys.exit(1)

import csv
with open(cpp_csv) as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

Y_cpp = np.zeros((N, num_orbs))
for i, row in enumerate(rows):
    # columns: theta_az, phi_pol, Y_0, Y_1, ...
    for j in range(num_orbs):
        Y_cpp[i, j] = float(row[2 + j])

# ── Compare ──────────────────────────────────────────────────────────────
diff = np.abs(Y_py - Y_cpp)
max_diff = diff.max()
mean_diff = diff.mean()

print(f"Comparison of {N} test points × {num_orbs} orbitals")
print(f"  Max  abs diff:  {max_diff:.2e}")
print(f"  Mean abs diff:  {mean_diff:.2e}")

# Find worst cases
worst = np.unravel_index(np.argmax(diff), diff.shape)
l_worst = int(np.floor(np.sqrt(worst[1])))
m_worst = worst[1] - l_worst**2 - l_worst
print(f"  Worst orbital:  j={worst[1]} (l={l_worst}, m={m_worst}) at point {worst[0]}")
print(f"    Python: {Y_py[worst]:.15e}")
print(f"    C++:    {Y_cpp[worst]:.15e}")

TOLERANCE = 1e-5
if max_diff < TOLERANCE:
    print(f"\n✓ PASS: All values match within {TOLERANCE}")
    sys.exit(0)
else:
    n_fail = np.sum(diff >= TOLERANCE)
    print(f"\n✗ FAIL: {n_fail}/{diff.size} values differ by >= {TOLERANCE}")
    sys.exit(1)
