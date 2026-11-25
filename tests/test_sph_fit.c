/* Test of the C interface: known shapes, wrong input, several threads. */
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include "sph_fit.h"

#define NT 24
#define NP 48
#define N (NT * NP)
#define LMAX 6
#define NW ((LMAX + 1) * (LMAX + 1))

static const double pi = 3.14159265358979323846;
static int bad = 0;

static void check(int ok, const char *what)
{
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    bad += !ok;
}

/* r = 1 + sum of three known terms, each with the sign of sph_fit.h; point set
 * and terms are symmetric in the origin, so the bounding box is centred there */
static void shape(double *xyz, const double *c)
{
    int i, j, k = 0;
    for (i = 0; i < NT; i++)
        for (j = 0; j < NP; j++, k++) {
            double p = pi * (i + 0.5) / NT, t = 2 * pi * (j + 0.5) / NP, s = sin(p);
            double r = 1
                - 0.08 * 0.25 * sqrt(15 / pi) * s * s * sin(2 * t)         /* l=2, m=-2 */
                - 0.05 * 0.5 * sqrt(15 / pi) * s * cos(p) * sin(t)         /* l=2, m=-1 */
                + 0.03 * 0.25 * sqrt(5 / pi) * (3 * cos(p) * cos(p) - 1);  /* l=2, m=0 */
            xyz[3 * k]     = c[0] + r * s * cos(t);
            xyz[3 * k + 1] = c[1] + r * s * sin(t);
            xyz[3 * k + 2] = c[2] + r * cos(p);
        }
}

static double worst(const double *w, double w2)
{
    double e = 0;
    int j;
    for (j = 0; j < NW; j++) {
        double x = j == 0 ? sqrt(4 * pi) : j == 4 ? w2 : j == 5 ? 0.05 : j == 6 ? 0.03 : 0;
        if (fabs(w[j] - x) > e) e = fabs(w[j] - x);
    }
    return e;
}

static double pts[3 * N], ref[NW], ref_centre[3];

static void *same(void *unused)
{
    double c[3], w[NW];
    int i;
    (void)unused;
    for (i = 0; i < 20; i++)
        if (sph_fit(pts, N, LMAX, SPH_LEGENDRE_RECURRENCE, SPH_SOLVER_AUTO, c, w, NULL) != SPH_OK ||
            memcmp(w, ref, sizeof w) || memcmp(c, ref_centre, sizeof c))
            return (void *)1;
    return NULL;
}

int main(void)
{
    const double centre[3] = {0.3, -0.2, 1.7};
    double c[3], w[NW], v[NW], copy[3 * N];
    struct sph_info info;
    double pmax;
    int cap = 1, used[2] = {0, 0};
    pthread_t th[8];
    void *r, *any = NULL;
    int i, n0;

    shape(pts, centre);
    memcpy(copy, pts, sizeof pts);
    check(sph_num_weights(LMAX) == NW, "number of weights");
    check(sph_fit(pts, N, LMAX, SPH_LEGENDRE_STD, SPH_SOLVER_AUTO, c, w, NULL) == SPH_OK && worst(w, 0.08) < 1e-13, "known shape, library Legendre");
    check(sph_fit(pts, N, LMAX, SPH_LEGENDRE_RECURRENCE, SPH_SOLVER_AUTO, c, w, NULL) == SPH_OK && worst(w, 0.08) < 1e-13, "known shape, recurrence");
    check(sph_fit(pts, N, LMAX, 0, SPH_SOLVER_AUTO, c, w, &info) == SPH_OK && info.solver == 0 && info.points == N &&
          info.rcond > 1e-3 && info.rms < 1e-14 && info.max < 1e-13, "information: Cholesky used, fit error 0");
    check(sph_fit(pts, N, LMAX, 0, SPH_SOLVER_SVD, c, v, &info) == SPH_OK && info.solver == 1 && worst(v, 0.08) < 1e-13, "known shape, SVD");
    check(worst(w, -0.08) > 0.1, "opposite sign of m = -2 is detected");
    check(fabs(c[0] - 0.3) + fabs(c[1] + 0.2) + fabs(c[2] - 1.7) < 1e-14, "centre");
    check(!memcmp(copy, pts, sizeof pts), "input not changed");
    memcpy(ref, w, sizeof w);
    memcpy(ref_centre, c, sizeof c);

    for (i = 0; i < 8; i++) if (pthread_create(&th[i], NULL, same, NULL)) return 2;
    for (i = 0; i < 8; i++) { pthread_join(th[i], &r); if (r) any = r; }
    check(!any, "8 threads give the same result");

    /* 1% spike at one point: the fit error must show it */
    memcpy(copy, pts, sizeof pts);
    for (i = 0; i < 3; i++) copy[3 * 500 + i] = c[i] + 1.01 * (pts[3 * 500 + i] - c[i]);
    check(sph_fit(copy, N, LMAX, 0, 0, c, v, &info) == SPH_OK && info.max > 0.005 && info.max < 0.011, "fit error of a spike");
    /* all points in a small range of directions: Cholesky is not possible */
    for (i = 0; i < N; i++) {
        double x = (i % NP) / (double)NP - 0.5, y = (i / NP) / (double)NT - 0.5;
        copy[3 * i] = 10 + 0.01 * x; copy[3 * i + 1] = 0.01 * y; copy[3 * i + 2] = 0;
    }
    copy[0] = -10; copy[1] = copy[2] = 0;
    n0 = sph_fit(copy, N, LMAX, 0, 0, c, v, &info);
    check(n0 == SPH_ILL_CONDITIONED && info.solver == 1, "points in one direction: reported as ill-conditioned");
    for (i = 0; i < N; i++) { copy[3 * i] = i; copy[3 * i + 1] = copy[3 * i + 2] = 0; }
    check(sph_fit(copy, N, LMAX, 0, 0, c, v, &info) == SPH_ILL_CONDITIONED, "points on one line: reported as ill-conditioned");
    /* part of a sphere missing, except 6 points on the axes that fix the centre */
    for (pmax = 1.0; pmax < 3.05; pmax += 0.1) {
        for (i = 0; i < N; i++) {
            double p = pmax * ((i / NP) + 0.5) / NT, t = 2 * pi * ((i % NP) + 0.5) / NP;
            copy[3 * i] = sin(p) * cos(t); copy[3 * i + 1] = sin(p) * sin(t); copy[3 * i + 2] = cos(p);
        }
        for (i = 0; i < 18; i++) copy[i] = 0;
        for (i = 0; i < 6; i++) copy[3 * i + i / 2] = i % 2 ? 1 : -1;
        n0 = sph_fit(copy, N, LMAX, 0, 0, c, v, &info);
        cap = cap && n0 == SPH_OK && fabs(v[0] - sqrt(4 * pi)) < 1e-11 &&
              info.solver == (info.rcond < SPH_RCOND_CHOLESKY);
        used[info.solver]++;
    }
    check(cap && used[0] && used[1], "unit sphere with a gap: SVD fallback when needed, same result");
    for (i = 0; i < 3 * N; i++) copy[i] = 1e200 * pts[i];
    check(sph_fit(copy, N, LMAX, 0, 0, c, v, &info) == SPH_BAD_INPUT && v[0] != v[0], "overflow: weights NaN");
    memcpy(copy, pts, sizeof pts);

    w[0] = 1;
    check(sph_fit(pts, NW - 1, LMAX, 0, 0, c, w, NULL) == SPH_TOO_FEW_POINTS && w[0] == 0, "too few points");
    check(sph_fit(NULL, 0, LMAX, 0, 0, c, w, NULL) == SPH_TOO_FEW_POINTS, "no points");
    n0 = sph_fit(pts, N, -1, 0, 0, c, w, NULL) == SPH_BAD_INPUT && sph_fit(pts, N, SPH_MAX_LMAX + 1, 0, 0, c, w, NULL) == SPH_BAD_INPUT;
    check(n0, "l_max out of range");
    check(sph_fit(pts, N, LMAX, 7, 0, c, w, NULL) == SPH_BAD_INPUT, "unknown Legendre method");
    check(sph_fit(pts, N, LMAX, 0, 7, c, w, NULL) == SPH_BAD_INPUT, "unknown solver");
    copy[10] = NAN;
    check(sph_fit(copy, N, LMAX, 0, 0, c, w, NULL) == SPH_BAD_INPUT && w[0] != w[0] && c[0] != c[0], "NaN coordinate: outputs NaN");
    puts(bad ? "FAILED" : "PASS");
    return bad != 0;
}
