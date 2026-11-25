// Spherical harmonic fit of one closed surface, see sph_fit.h

#include "sph_fit.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// Failed checks inside Eigen: exception (caught in sph_fit) instead of abort()
#define eigen_assert(x) \
    do { if (!(x)) throw std::logic_error("Eigen: " #x); } while (false)
#include "sph_harm_math.hpp"
#include <Eigen/Cholesky>

namespace {

struct Points { std::vector<double> theta, phi, r; };

// Basis at n <= SphBlock::SIZE points from point k
struct Basis {
    const Points& p;
    LegendreMethod method;
    int l_max;
    SphBlock block;
    Basis(const Points& pts, int lmax, LegendreMethod m)
        : p(pts), method(m), l_max(lmax), block(lmax) {}

    void rows(int k, int n, Eigen::MatrixXd& out)
    {
        if (method == LegendreMethod::RECURRENCE)
            block.fill(&p.theta[k], &p.phi[k], n, out);
        else
            out.topRows(n) = build_sph_matrix(
                std::vector<double>(p.theta.begin() + k, p.theta.begin() + k + n),
                std::vector<double>(p.phi.begin() + k, p.phi.begin() + k + n),
                l_max, method);
    }
};

// f(k, n) for all blocks of points
template <class F> void for_blocks(const Points& p, F f)
{
    const int N = static_cast<int>(p.r.size());
    for (int k = 0; k < N; k += SphBlock::SIZE) {
        const int n = N - k < SphBlock::SIZE ? N - k : SphBlock::SIZE;
        f(k, n);
    }
}

// Cholesky of A^T A w = A^T r, summed in blocks so that A is never stored.
// False if the normal matrix is ill-conditioned or the result is not finite.
bool cholesky(const Points& p, Basis& basis, Eigen::VectorXd& w, double& rcond)
{
    const int num = sph_num_weights(basis.l_max);
    Eigen::MatrixXd rows(SphBlock::SIZE, num), M = Eigen::MatrixXd::Zero(num, num);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(num);
    for_blocks(p, [&](int k, int n) {
        basis.rows(k, n, rows);
        M.selfadjointView<Eigen::Lower>().rankUpdate(rows.topRows(n).transpose());
        rhs.noalias() += rows.topRows(n).transpose() *
                         Eigen::Map<const Eigen::VectorXd>(&p.r[k], n);
    });
    const Eigen::LLT<Eigen::MatrixXd> llt(M);
    rcond = llt.info() == Eigen::Success ? llt.rcond() : 0.0;
    if (!(rcond >= SPH_RCOND_CHOLESKY)) return false;
    w = llt.solve(rhs);
    return w.allFinite();
}

int fit(const double* xyz, int n, int l_max, LegendreMethod method, int solver,
        double* centre, double* weights, sph_info& info, bool errors)
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

    Points p;
    p.theta.reserve(n);
    p.phi  .reserve(n);
    p.r    .reserve(n);
    for (int i = 0; i < n; ++i) {
        const auto sph = cart2sph(xyz[3 * i]     - centre[0],
                                  xyz[3 * i + 1] - centre[1],
                                  xyz[3 * i + 2] - centre[2]);
        if (!std::isfinite(sph.r)) return SPH_BAD_INPUT;   // overflow
        if (sph.r < 1e-20) continue;   // point at the centre: no direction
        p.theta.push_back(sph.theta_az);
        p.phi  .push_back(sph.phi_pol);
        p.r    .push_back(sph.r);
    }

    const int num = sph_num_weights(l_max);
    info.points = static_cast<int>(p.r.size());
    if (info.points < num) {
        std::fill(weights, weights + num, 0.0);
        return SPH_TOO_FEW_POINTS;
    }

    Basis basis(p, l_max, method);
    Eigen::VectorXd w;
    info.solver = 0;
    bool done = false;
    if (solver == SPH_SOLVER_AUTO) {
        try {
            done = cholesky(p, basis, w, info.rcond);
        } catch (const std::logic_error&) {
            // check inside Eigen: try SVD
        }
    }
    if (!done) {
        info.solver = 1;
        w = lstsq(build_sph_matrix(p.theta, p.phi, l_max, method),
                  Eigen::Map<const Eigen::VectorXd>(p.r.data(), info.points),
                  &info.rcond);
        if (!w.allFinite()) return SPH_FAILED;
    }

    info.rms = info.max = 0.0;
    if (errors) {
        Eigen::MatrixXd rows(SphBlock::SIZE, num);
        double sum = 0.0;
        for_blocks(p, [&](int k, int m) {
            basis.rows(k, m, rows);
            const Eigen::VectorXd e = rows.topRows(m) * w -
                                      Eigen::Map<const Eigen::VectorXd>(&p.r[k], m);
            sum += e.squaredNorm();
            info.max = std::max(info.max, e.cwiseAbs().maxCoeff());
        });
        info.rms = std::sqrt(sum / info.points);
    }
    std::copy(w.data(), w.data() + num, weights);
    return info.rcond < SPH_RCOND_SVD ? SPH_ILL_CONDITIONED : SPH_OK;
}

} // namespace

int sph_num_weights(int l_max) { return (l_max + 1) * (l_max + 1); }

const char* sph_status_text(int status)
{
    switch (status) {
    case SPH_OK:             return "ok";
    case SPH_ILL_CONDITIONED: return "ill-conditioned, weights not reliable";
    case SPH_TOO_FEW_POINTS: return "too few points";
    case SPH_BAD_INPUT:      return "input wrong or not finite";
    case SPH_FAILED:         return "fit failed";
    }
    return "unknown status";
}

int sph_fit(const double* xyz, int n, int l_max, int legendre, int solver,
            double centre[3], double* weights, sph_info* info)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (info) *info = {0, 0, 0.0, nan, nan};
    if (!centre || !weights || l_max < 0 || l_max > SPH_MAX_LMAX)
        return SPH_BAD_INPUT;

    int status = SPH_OK;
    if (n < 0 || (n > 0 && !xyz) ||
        (legendre != SPH_LEGENDRE_RECURRENCE && legendre != SPH_LEGENDRE_STD) ||
        (solver != SPH_SOLVER_AUTO && solver != SPH_SOLVER_SVD))
        status = SPH_BAD_INPUT;
    for (long i = 0; status == SPH_OK && i < 3L * n; ++i)
        if (!std::isfinite(xyz[i])) status = SPH_BAD_INPUT;

    if (status == SPH_OK) {
        sph_info result = {0, 0, 0.0, nan, nan};
        try {
            status = fit(xyz, n, l_max,
                         legendre == SPH_LEGENDRE_STD ? LegendreMethod::STD
                                                      : LegendreMethod::RECURRENCE,
                         solver, centre, weights, result, info != nullptr);
        } catch (...) {
            status = SPH_FAILED;
        }
        if (info) *info = result;
    }
    if (status == SPH_BAD_INPUT) std::fill(centre, centre + 3, nan);
    if (status == SPH_BAD_INPUT || status == SPH_FAILED)
        std::fill(weights, weights + sph_num_weights(l_max), nan);
    return status;
}
