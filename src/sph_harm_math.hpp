#pragma once
// ============================================================================
//  sph_harm_math.hpp
//  Real spherical harmonic basis and Eigen least-squares solver.
//
//  Convention exactly mirrors scipy.special.sph_harm(m, l, theta_az, phi_pol):
//    theta_az : azimuthal angle (longitude), 0..2π
//    phi_pol  : polar angle / colatitude   , 0..π
//
//  Orbital ordering (j = 0..num_orbs-1):
//    l = floor(sqrt(j))
//    m = j - l² - l        ( m runs from -l to +l )
//  → matches sph_harm_functions.py indexing exactly.
// ============================================================================

#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/SVD>

// ----------------------------------------------------------------------------
//  Legendre computation method selector
// ----------------------------------------------------------------------------
enum class LegendreMethod {
    STD,        // Use std::sph_legendre from <cmath> (C++17 special functions)
    RECURRENCE  // Use local recurrence-relation implementation
};

// ----------------------------------------------------------------------------
//  Y_l^m — one real spherical harmonic value at (theta_az, phi_pol)
//
//  Uses std::sph_legendre only (reference / scalar implementation).
//  Derivation:
//   std::sph_legendre(l, |m|, φ) = (−1)^|m| √((2l+1)/4π · (l−|m|)!/(l+|m|)!) P_l^|m|(cosφ)
//                                 = Re( Y_l^|m|(φ, ψ=0) )   [scipy Condon–Shortley convention]
//
//   m > 0 : Re(Y_l^m)        = sph_legendre(l,m,φ) · cos(m·θ)
//   m = 0 : Y_l^0            = sph_legendre(l,0,φ)
//   m < 0 : Im(Y_l^{-|m|})  = (−1)^(|m|+1) · sph_legendre(l,|m|,φ) · sin(|m|·θ)
//            [from scipy: Y_l^{-m} = (−1)^m · conj(Y_l^m)]
// ----------------------------------------------------------------------------
inline double real_Y_lm(int l, int m, double theta_az, double phi_pol)
{
    const unsigned abs_m = static_cast<unsigned>(std::abs(m));
    // std::sph_legendre takes the polar angle directly (not its cosine).
    const double leg = std::sph_legendre(static_cast<unsigned>(l), abs_m, phi_pol);

    if (m < 0) {
        // sign = (−1)^(|m|+1) : +1 for odd |m|, −1 for even |m|
        const double sign = (abs_m % 2u == 0u) ? -1.0 : 1.0;
        return sign * M_SQRT2 * leg * std::sin(abs_m * theta_az);
    } else if (m == 0) {
        return leg;
    } else {
        return M_SQRT2 * leg * std::cos(static_cast<unsigned>(m) * theta_az);
    }
}

// ============================================================================
//  Build design matrix  A  of shape  (N, (l_max+1)²)
//  Each row k holds  [ Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, … ]  at point k.
//
//  Two back-ends selectable via LegendreMethod:
//    STD        — calls std::sph_legendre per (l,m) pair (reference)
//    RECURRENCE — uses recurrence relation (faster, same results)
// ============================================================================

// --- Shared helpers ---------------------------------------------------------

struct OrbInfo { int l; int m; unsigned abs_m; double sign; };

inline std::vector<OrbInfo> make_orb_table(int l_max)
{
    const int num_orbs = (l_max + 1) * (l_max + 1);
    std::vector<OrbInfo> orbs(num_orbs);
    for (int j = 0; j < num_orbs; ++j) {
        int l = static_cast<int>(std::floor(std::sqrt(static_cast<double>(j))));
        int m = j - l * l - l;
        unsigned am = static_cast<unsigned>(std::abs(m));
        double s = (m < 0) ? ((am % 2u == 0u) ? -1.0 : 1.0) : 0.0;
        orbs[j] = {l, m, am, s};
    }
    return orbs;
}

// Fill a row of A from precomputed Legendre values (leg[l][abs_m]) and
// trig multiples cos(m*θ), sin(m*θ).
inline void fill_row(Eigen::MatrixXd& A, int k,
                     const std::vector<OrbInfo>& orbs,
                     const std::vector<std::vector<double>>& leg,
                     const std::vector<double>& cos_mt,
                     const std::vector<double>& sin_mt)
{
    const int num_orbs = static_cast<int>(orbs.size());
    for (int j = 0; j < num_orbs; ++j) {
        const auto& o = orbs[j];
        const double lv = leg[o.l][o.abs_m];
        if (o.m < 0)
            A(k, j) = o.sign * M_SQRT2 * lv * sin_mt[o.abs_m];
        else if (o.m == 0)
            A(k, j) = lv;
        else
            A(k, j) = M_SQRT2 * lv * cos_mt[o.abs_m];
    }
}

// Precompute cos(m*θ) and sin(m*θ) using Chebyshev recurrence.
inline void trig_multiples(double theta, int l_max,
                           std::vector<double>& cos_mt,
                           std::vector<double>& sin_mt)
{
    cos_mt[0] = 1.0; sin_mt[0] = 0.0;
    if (l_max >= 1) {
        cos_mt[1] = std::cos(theta);
        sin_mt[1] = std::sin(theta);
    }
    for (int m = 2; m <= l_max; ++m) {
        cos_mt[m] = 2.0 * cos_mt[1] * cos_mt[m-1] - cos_mt[m-2];
        sin_mt[m] = 2.0 * cos_mt[1] * sin_mt[m-1] - sin_mt[m-2];
    }
}

// --- Back-end: std::sph_legendre -------------------------------------------
//   Calls the C++17 standard library per (l, |m|) pair at each point.

namespace detail {

inline void compute_legendre_std(
    double phi,
    int l_max,
    const std::vector<OrbInfo>& orbs,
    std::vector<std::vector<double>>& leg)
{
    // Fill only the (l, |m|) pairs that actually appear.
    // Since orbs covers all (l,m) with l = 0..l_max, we just iterate l,m.
    for (int l = 0; l <= l_max; ++l)
        for (int m = 0; m <= l; ++m)
            leg[l][m] = std::sph_legendre(static_cast<unsigned>(l),
                                          static_cast<unsigned>(m), phi);
}

} // namespace detail

// --- Back-end: recurrence relation -----------------------------------------

// Precompute normalization factors: N_l^m = sqrt((2l+1)/(4π) · (l-m)!/(l+m)!)
// Combined with Condon-Shortley phase, this matches std::sph_legendre exactly.
struct SHNormTable {
    // norm[l][m] for m = 0..l,  l = 0..l_max
    std::vector<std::vector<double>> norm;
    int l_max;

    explicit SHNormTable(int lmax) : l_max(lmax) {
        norm.resize(lmax + 1);
        for (int l = 0; l <= lmax; ++l) {
            norm[l].resize(l + 1);
            for (int m = 0; m <= l; ++m) {
                double ratio = 1.0;
                for (int k = l - m + 1; k <= l + m; ++k)
                    ratio *= k;
                norm[l][m] = std::sqrt((2.0 * l + 1.0) / (4.0 * M_PI) / ratio);
            }
        }
    }
};

// Compute all sph_legendre(l, m, phi) for a single point via recurrence.
// Stores results in leg[l][m] for m = 0..l.
inline void compute_legendre_recurrence(
    double phi,
    int l_max,
    const SHNormTable& ntab,
    std::vector<std::vector<double>>& leg)
{
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);

    // Sectoral: P_m^m
    double pmm = 1.0;
    leg[0][0] = ntab.norm[0][0] * pmm;

    for (int m = 1; m <= l_max; ++m) {
        pmm *= -(2 * m - 1) * sin_phi;
        leg[m][m] = ntab.norm[m][m] * pmm;
    }

    // Tesseral: upward recurrence in l using normalized form
    //   sph_leg(l,m) = a*cos_phi*sph_leg(l-1,m) - b*sph_leg(l-2,m)
    for (int m = 0; m <= l_max; ++m) {
        if (m + 1 <= l_max) {
            double a = std::sqrt((4.0*(m+1)*(m+1) - 1.0) /
                                 ((m+1.0)*(m+1.0) - m*m));
            leg[m+1][m] = a * cos_phi * leg[m][m];
        }
        for (int l = m + 2; l <= l_max; ++l) {
            double l2 = static_cast<double>(l) * l;
            double m2 = static_cast<double>(m) * m;
            double a = std::sqrt((4.0 * l2 - 1.0) / (l2 - m2));
            double b = std::sqrt(((2.0*l + 1.0) * ((l-1.0)*(l-1.0) - m2)) /
                                 ((2.0*l - 3.0) * (l2 - m2)));
            leg[l][m] = a * cos_phi * leg[l-1][m] - b * leg[l-2][m];
        }
    }
}

// --- Public interface -------------------------------------------------------

inline Eigen::MatrixXd build_sph_matrix(
    const std::vector<double>& theta_az,
    const std::vector<double>& phi_pol,
    int l_max,
    LegendreMethod method = LegendreMethod::RECURRENCE)
{
    const int N        = static_cast<int>(theta_az.size());
    const int num_orbs = (l_max + 1) * (l_max + 1);
    Eigen::MatrixXd A(N, num_orbs);

    const auto orbs = make_orb_table(l_max);

    // Normalization table (only needed for recurrence, but cheap to build)
    SHNormTable ntab(l_max);

    // Per-point scratch: leg[l][m] for m = 0..l
    std::vector<std::vector<double>> leg(l_max + 1);
    for (int l = 0; l <= l_max; ++l)
        leg[l].resize(l + 1);

    std::vector<double> cos_mt(l_max + 1), sin_mt(l_max + 1);

    for (int k = 0; k < N; ++k) {
        const double phi   = phi_pol[k];
        const double theta = theta_az[k];

        // Compute Legendre values using the selected method
        switch (method) {
        case LegendreMethod::STD:
            detail::compute_legendre_std(phi, l_max, orbs, leg);
            break;
        case LegendreMethod::RECURRENCE:
            compute_legendre_recurrence(phi, l_max, ntab, leg);
            break;
        }

        trig_multiples(theta, l_max, cos_mt, sin_mt);
        fill_row(A, k, orbs, leg, cos_mt, sin_mt);
    }
    return A;
}

// ----------------------------------------------------------------------------
//  Least-squares solver: find x that minimises ||A·x − b||²
//  Uses Eigen BDCSVD — equivalent to numpy.linalg.lstsq(A, b, rcond=None).
// ----------------------------------------------------------------------------
inline Eigen::VectorXd lstsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b)
{
    return A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}

// ----------------------------------------------------------------------------
//  Cartesian → Spherical
//  Matches Python cart2sph: r=√(X²+Y²+Z²), θ=atan2(Y,X), φ=arccos(Z/r)
// ----------------------------------------------------------------------------
struct Spherical { double r, theta_az, phi_pol; };

inline Spherical cart2sph(double X, double Y, double Z)
{
    const double r   = std::sqrt(X*X + Y*Y + Z*Z);
    const double taz = std::atan2(Y, X);
    const double pol = (r > 1e-15) ? std::acos(std::clamp(Z / r, -1.0, 1.0)) : 0.0;
    return {r, taz, pol};
}
