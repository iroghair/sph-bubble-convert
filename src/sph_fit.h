/* Spherical harmonic fit of one closed surface: r(theta, phi) = sum w_j Y_j.
 * C interface of libsphfit.a; used by bin/converter and by other programs.
 * Link with a C++ compiler, or add -lstdc++ -lm.
 *
 * No global state: may be called from several threads at once. Starts no
 * threads, throws no exceptions and prints nothing.
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
    SPH_OK = 0,
    SPH_TOO_FEW_POINTS, /* fewer points than weights; centre set, weights 0 */
    SPH_BAD_INPUT,      /* l_max or method out of range, NULL, or not finite */
    SPH_FAILED          /* for example out of memory */
};

/* Associated Legendre functions: local recurrence (default) or C++17 library */
enum sph_legendre { SPH_LEGENDRE_RECURRENCE = 0, SPH_LEGENDRE_STD = 1 };

/* Number of weights: (l_max + 1)^2 */
int sph_num_weights(int l_max);

/* Fit n points, xyz = x0 y0 z0 x1 y1 z1 ... (not changed).
 * centre[3]: centre of the bounding box, origin of the fit, in the
 *            coordinates of xyz (no window shift added).
 * weights:   sph_num_weights(l_max) values.
 * Returns an enum sph_status; with SPH_BAD_INPUT or SPH_FAILED the outputs are
 * not valid. */
int sph_fit(const double *xyz, int n, int l_max, int legendre,
            double centre[3], double *weights);

#ifdef __cplusplus
}
#endif

#endif
