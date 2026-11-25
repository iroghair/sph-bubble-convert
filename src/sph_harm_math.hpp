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

// Evaluates the basis for blocks of points (recurrence method); same arithmetic
// per point as before, but all points of a block at once, so it vectorises.
class SphBlock {
public:
    static constexpr int SIZE = 128;   // points per block: fits in L2 cache

    explicit SphBlock(int l_max)
        : l_max_(l_max), norm_(l_max), a_(index(l_max, l_max) + 1),
          b_(a_.size()), leg_(SIZE, a_.size()), cos_(SIZE, l_max + 1),
          sin_(SIZE, l_max + 1)
    {
        for (int l = 1; l <= l_max; ++l)
            for (int m = 0; m < l; ++m) {
                const double l2 = static_cast<double>(l) * l;
                const double m2 = static_cast<double>(m) * m;
                a_[index(l, m)] = std::sqrt((4.0 * l2 - 1.0) / (l2 - m2));
                if (m <= l - 2)
                    b_[index(l, m)] = std::sqrt(((2.0*l + 1.0) * ((l-1.0)*(l-1.0) - m2)) /
                                                ((2.0*l - 3.0) * (l2 - m2)));
            }
    }

    // First n rows of out (n <= SIZE): the basis at these points
    template <class Matrix>
    void fill(const double* theta, const double* phi, int n, Matrix&& out)
    {
        Eigen::Array<double, SIZE, 1> cp, sp, pmm;
        for (int k = 0; k < n; ++k) {
            cp[k] = std::cos(phi[k]);
            sp[k] = std::sin(phi[k]);
            cos_(k, 0) = 1.0;
            sin_(k, 0) = 0.0;
            if (l_max_ >= 1) {
                cos_(k, 1) = std::cos(theta[k]);
                sin_(k, 1) = std::sin(theta[k]);
            }
        }
        const auto c1 = cp.head(n);
        for (int m = 2; m <= l_max_; ++m) {
            cos_.col(m).head(n) = 2.0 * cos_.col(1).head(n) * cos_.col(m-1).head(n) - cos_.col(m-2).head(n);
            sin_.col(m).head(n) = 2.0 * cos_.col(1).head(n) * sin_.col(m-1).head(n) - sin_.col(m-2).head(n);
        }

        // Sectoral P_m^m, then upward recurrence in l
        pmm.head(n).setOnes();
        for (int m = 0; m <= l_max_; ++m) {
            if (m > 0) pmm.head(n) *= -static_cast<double>(2 * m - 1) * sp.head(n);
            leg_.col(index(m, m)).head(n) = norm_.norm[m][m] * pmm.head(n);
            if (m + 1 <= l_max_)
                leg_.col(index(m+1, m)).head(n) = a_[index(m+1, m)] * c1 * leg_.col(index(m, m)).head(n);
            for (int l = m + 2; l <= l_max_; ++l)
                leg_.col(index(l, m)).head(n) =
                    a_[index(l, m)] * c1 * leg_.col(index(l-1, m)).head(n)
                    - b_[index(l, m)] * leg_.col(index(l-2, m)).head(n);
        }

        for (int l = 0; l <= l_max_; ++l) {
            const int c = l * l + l;
            out.col(c).head(n) = leg_.col(index(l, 0)).head(n).matrix();
            for (int m = 1; m <= l; ++m) {
                const double s = (m % 2 == 0 ? -1.0 : 1.0) * M_SQRT2;
                out.col(c + m).head(n) = (M_SQRT2 * leg_.col(index(l, m)).head(n) * cos_.col(m).head(n)).matrix();
                out.col(c - m).head(n) = (s * leg_.col(index(l, m)).head(n) * sin_.col(m).head(n)).matrix();
            }
        }
    }

private:
    static int index(int l, int m) { return l * (l + 1) / 2 + m; }
    int l_max_;
    SHNormTable norm_;
    std::vector<double> a_, b_;
    Eigen::ArrayXXd leg_, cos_, sin_;
};

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

    if (method == LegendreMethod::RECURRENCE) {
        SphBlock block(l_max);
        for (int k = 0; k < N; k += SphBlock::SIZE) {
            const int n = std::min(SphBlock::SIZE + 0, N - k);
            block.fill(&theta_az[k], &phi_pol[k], n, A.middleRows(k, n));
        }
        return A;
    }

    const auto orbs = make_orb_table(l_max);

    // Per-point scratch: leg[l][m] for m = 0..l
    std::vector<std::vector<double>> leg(l_max + 1);
    for (int l = 0; l <= l_max; ++l)
        leg[l].resize(l + 1);

    std::vector<double> cos_mt(l_max + 1), sin_mt(l_max + 1);

    for (int k = 0; k < N; ++k) {
        const double phi   = phi_pol[k];
        const double theta = theta_az[k];

        detail::compute_legendre_std(phi, l_max, orbs, leg);
        trig_multiples(theta, l_max, cos_mt, sin_mt);
        fill_row(A, k, orbs, leg, cos_mt, sin_mt);
    }
    return A;
}

// ----------------------------------------------------------------------------
//  Least-squares solver: find x that minimises ||A·x − b||²
//  Uses Eigen BDCSVD — equivalent to numpy.linalg.lstsq(A, b, rcond=None).
// ----------------------------------------------------------------------------
inline Eigen::VectorXd lstsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                             double* rcond = nullptr)   // 1/condition number of A^T A
{
    const auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.info() != Eigen::Success)
        throw std::runtime_error("SVD did not converge");
    if (rcond) {
        const auto& sv = svd.singularValues();   // decreasing
        const double q = sv[sv.size() - 1] / sv[0];
        *rcond = q * q;
    }
    return svd.solve(b);
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
