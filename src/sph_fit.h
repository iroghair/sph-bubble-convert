/* Spherical harmonic fit of one closed surface: r(theta, phi) = sum w_j Y_j.
 * C interface of libsphfit.a; used by bin/converter and by other programs.
 * Link with a C++ compiler, or add -lstdc++ -lm.
 *
 * No global state: may be called from several threads at once. Starts no
 * threads, throws no exceptions, prints nothing and never stops the program:
 * the caller decides what to do with the returned status.
 *
 * Order of the weights: j = l*l + l + m, l = 0..l_max, m = -l..l.
 * Basis (Y_l^m of scipy.special.sph_harm, with Condon-Shortley phase):
 *   m < 0: sqrt(2) Im(Y_l^m)   (the negative m, not |m|)
 *   m = 0: Y_l^0
 *   m > 0: sqrt(2) Re(Y_l^m)
 */
#ifndef SPH_FIT_H
#define SPH_FIT_H

#ifdef __cplusplus
extern "C" {
#endif

#define SPH_MAX_LMAX 85 /* factorials overflow above this */

enum sph_status {
    SPH_OK = 0,         /* weights are valid (see sph_info.solver) */
    SPH_ILL_CONDITIONED,/* the points do not determine the weights: SVD weights
                           are given, but may be far too large */
    SPH_TOO_FEW_POINTS, /* fewer points than weights; centre set, weights 0 */
    SPH_BAD_INPUT,      /* l_max, legendre or solver out of range, NULL, or
                           coordinates not finite; outputs NaN */
    SPH_FAILED          /* also SVD failed, or out of memory; weights NaN */
};

/* Associated Legendre functions: local recurrence (default) or C++17 library */
enum sph_legendre { SPH_LEGENDRE_RECURRENCE = 0, SPH_LEGENDRE_STD = 1 };

/* Least squares: Cholesky of the normal equations, with SVD if the points do
 * not determine all weights well (default); or always SVD (about 12x slower) */
enum sph_solver { SPH_SOLVER_AUTO = 0, SPH_SOLVER_SVD = 1 };

/* How the fit went; valid with SPH_OK and SPH_ILL_CONDITIONED */
struct sph_info {
    int points;       /* points used */
    int solver;       /* 0: Cholesky, 1: SVD */
    double rcond;     /* normal matrix: 1/condition number. Below
                         SPH_RCOND_CHOLESKY SVD is used (error of Cholesky is
                         then up to about 1e-10), below SPH_RCOND_SVD the
                         status is SPH_ILL_CONDITIONED */
    double rms, max;  /* |r of the fit - r of the point|, same unit as xyz;
                         large for a surface that r(theta, phi) cannot describe */
};
#define SPH_RCOND_CHOLESKY 1e-6
#define SPH_RCOND_SVD 1e-12

/* Number of weights: (l_max + 1)^2 */
int sph_num_weights(int l_max);

/* Text for a status */
const char *sph_status_text(int status);

/* Fit n points, xyz = x0 y0 z0 x1 y1 z1 ... (not changed).
 * centre[3]: centre of the bounding box, origin of the fit, in the
 *            coordinates of xyz (no window shift added).
 * weights:   sph_num_weights(l_max) values.
 * info:      may be NULL (saves computing the fit error).
 * Returns an enum sph_status. */
int sph_fit(const double *xyz, int n, int l_max, int legendre, int solver,
            double centre[3], double *weights, struct sph_info *info);

#ifdef __cplusplus
}
#endif

#endif
