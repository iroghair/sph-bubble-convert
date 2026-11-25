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
        if (sph_fit(pts, N, LMAX, SPH_LEGENDRE_RECURRENCE, c, w) != SPH_OK ||
            memcmp(w, ref, sizeof w) || memcmp(c, ref_centre, sizeof c))
            return (void *)1;
    return NULL;
}

int main(void)
{
    const double centre[3] = {0.3, -0.2, 1.7};
    double c[3], w[NW], copy[3 * N];
    pthread_t th[8];
    void *r, *any = NULL;
    int i, n0;

    shape(pts, centre);
    memcpy(copy, pts, sizeof pts);
    check(sph_num_weights(LMAX) == NW, "number of weights");
    check(sph_fit(pts, N, LMAX, SPH_LEGENDRE_STD, c, w) == SPH_OK && worst(w, 0.08) < 1e-13, "known shape, library Legendre");
    check(sph_fit(pts, N, LMAX, SPH_LEGENDRE_RECURRENCE, c, w) == SPH_OK && worst(w, 0.08) < 1e-13, "known shape, recurrence");
    check(worst(w, -0.08) > 0.1, "opposite sign of m = -2 is detected");
    check(fabs(c[0] - 0.3) + fabs(c[1] + 0.2) + fabs(c[2] - 1.7) < 1e-14, "centre");
    check(!memcmp(copy, pts, sizeof pts), "input not changed");
    memcpy(ref, w, sizeof w);
    memcpy(ref_centre, c, sizeof c);

    for (i = 0; i < 8; i++) if (pthread_create(&th[i], NULL, same, NULL)) return 2;
    for (i = 0; i < 8; i++) { pthread_join(th[i], &r); if (r) any = r; }
    check(!any, "8 threads give the same result");

    w[0] = 1;
    check(sph_fit(pts, NW - 1, LMAX, 0, c, w) == SPH_TOO_FEW_POINTS && w[0] == 0, "too few points");
    check(sph_fit(NULL, 0, LMAX, 0, c, w) == SPH_TOO_FEW_POINTS, "no points");
    n0 = sph_fit(pts, N, -1, 0, c, w) == SPH_BAD_INPUT && sph_fit(pts, N, SPH_MAX_LMAX + 1, 0, c, w) == SPH_BAD_INPUT;
    check(n0, "l_max out of range");
    check(sph_fit(pts, N, LMAX, 7, c, w) == SPH_BAD_INPUT, "unknown Legendre method");
    copy[10] = NAN;
    check(sph_fit(copy, N, LMAX, 0, c, w) == SPH_BAD_INPUT, "NaN coordinate");
    puts(bad ? "FAILED" : "PASS");
    return bad != 0;
}
