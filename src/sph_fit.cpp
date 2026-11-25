// Spherical harmonic fit of one closed surface, see sph_fit.h

#include "sph_fit.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "sph_harm_math.hpp"

int sph_num_weights(int l_max) { return (l_max + 1) * (l_max + 1); }

static int fit(const double* xyz, int n, int l_max, LegendreMethod method,
               double* centre, double* weights)
{
    // Centre of the bounding box (= PyVista PolyData.center)
    for (int d = 0; d < 3; ++d) {
        double lo = 1e300, hi = -1e300;
        for (int i = 0; i < n; ++i) {
            lo = std::min(lo, xyz[3 * i + d]);
            hi = std::max(hi, xyz[3 * i + d]);
        }
        centre[d] = (lo + hi) / 2.0;
    }

    std::vector<double> theta_az, phi_pol, r_vals;
    theta_az.reserve(n);
    phi_pol .reserve(n);
    r_vals  .reserve(n);
    for (int i = 0; i < n; ++i) {
        const auto sph = cart2sph(xyz[3 * i]     - centre[0],
                                  xyz[3 * i + 1] - centre[1],
                                  xyz[3 * i + 2] - centre[2]);
        if (sph.r < 1e-20) continue;   // point at the centre: no direction
        theta_az.push_back(sph.theta_az);
        phi_pol .push_back(sph.phi_pol);
        r_vals  .push_back(sph.r);
    }

    const int num = sph_num_weights(l_max);
    if (static_cast<int>(r_vals.size()) < num) {
        std::fill(weights, weights + num, 0.0);
        return SPH_TOO_FEW_POINTS;
    }

    const Eigen::MatrixXd A = build_sph_matrix(theta_az, phi_pol, l_max, method);
    const Eigen::VectorXd b =
        Eigen::Map<const Eigen::VectorXd>(r_vals.data(), r_vals.size());
    const Eigen::VectorXd w = lstsq(A, b);
    std::copy(w.data(), w.data() + num, weights);
    return SPH_OK;
}

int sph_fit(const double* xyz, int n, int l_max, int legendre,
            double centre[3], double* weights)
{
    if (n < 0 || (n > 0 && !xyz) || !centre || !weights ||
        l_max < 0 || l_max > SPH_MAX_LMAX)
        return SPH_BAD_INPUT;
    if (legendre != SPH_LEGENDRE_RECURRENCE && legendre != SPH_LEGENDRE_STD)
        return SPH_BAD_INPUT;
    for (long i = 0; i < 3L * n; ++i)
        if (!std::isfinite(xyz[i])) return SPH_BAD_INPUT;
    try {
        return fit(xyz, n, l_max,
                   legendre == SPH_LEGENDRE_STD ? LegendreMethod::STD
                                                : LegendreMethod::RECURRENCE,
                   centre, weights);
    } catch (...) {
        return SPH_FAILED;
    }
}
