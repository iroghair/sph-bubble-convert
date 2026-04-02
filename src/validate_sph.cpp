// ============================================================================
//  validate_sph.cpp
//  C++ counterpart to validate_sph_py.py.
//  Computes real spherical harmonics at the same fixed test points and writes
//  /tmp/cpp_sph_values.csv so the two outputs can be diffed numerically.
//
//  Build:  make validate_sph
//  Run:    ./validate_sph
//  Compare: python3 /tmp/compare_sph.py
// ============================================================================

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

#include "sph_harm_math.hpp"

int main()
{
    // ── Same test points as validate_sph_py.py ────────────────────────────
    const std::vector<double> theta_az = {
        0.0,   0.5,   1.2,   2.0,
        M_PI,  0.1,   2.5,   6.0
    };
    const std::vector<double> phi_pol = {
        0.0,   0.3,   2.1,   1.5,
        M_PI/4, 2.9,  0.01,  1.0
    };

    const int L_MAX    = 14;
    const int num_orbs = (L_MAX + 1) * (L_MAX + 1);
    const int N        = static_cast<int>(theta_az.size());

    const Eigen::MatrixXd Y = build_sph_matrix(theta_az, phi_pol, L_MAX);

    // ── Write CSV ─────────────────────────────────────────────────────────
    const std::string out_path = "/tmp/cpp_sph_values.csv";
    std::ofstream f(out_path);
    if (!f) { std::cerr << "Cannot open " << out_path << "\n"; return 1; }

    f << std::setprecision(15) << std::scientific;

    // Header
    f << "theta_az,phi_pol";
    for (int j = 0; j < num_orbs; ++j) f << ",Y_" << j;
    f << "\n";

    for (int k = 0; k < N; ++k) {
        f << theta_az[k] << "," << phi_pol[k];
        for (int j = 0; j < num_orbs; ++j) f << "," << Y(k, j);
        f << "\n";
    }

    std::cout << "Written " << out_path << " (" << N << " rows, "
              << num_orbs << " orbitals)\n";

    // Quick sanity: Y_0^0 should equal 1/sqrt(4π) ≈ 0.28209479 at every point
    const double expected_Y00 = 1.0 / std::sqrt(4.0 * M_PI);
    std::cout << "Y_0^0 values (expected " << expected_Y00 << "):\n";
    for (int k = 0; k < N; ++k)
        std::cout << "  row " << k << ": " << Y(k, 0) << "\n";

    return 0;
}
