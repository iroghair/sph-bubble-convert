#pragma once
// ============================================================================
//  ft3_reader.hpp
//  Binary FT3 Front-Tracking file parser.
//  Parsing logic mirrors FT3_2_STL_2.py :: Ft3file::loadFT3() exactly.
// ============================================================================

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

// ---- Data structures -------------------------------------------------------

struct Bubble {
    // unique vertex positions, shape (npos, 3)
    std::vector<std::array<double, 3>> positions;
    // triangular face vertex indices, shape (nmar, 3) — indices into positions[]
    std::vector<std::array<int32_t, 3>> face_pts;
};

struct Ft3Frame {
    int    cycle;
    double time;   // raw time in seconds (NOT rounded)
    int    nx, ny, nz;
    double dx, dy, dz;
    int    nph, neli;
    std::vector<Bubble> bubbles;
};

// ---- Low-level I/O helpers -------------------------------------------------

static inline int32_t ft3_int(std::ifstream& f)
{
    int32_t v;
    f.read(reinterpret_cast<char*>(&v), 4);
    return v;
}

static inline double ft3_double(std::ifstream& f)
{
    double v;
    f.read(reinterpret_cast<char*>(&v), 8);
    return v;
}

static inline void ft3_skip(std::ifstream& f, std::streamsize n)
{
    f.seekg(n, std::ios::cur);
}

// ---- Main loader -----------------------------------------------------------

inline Ft3Frame load_ft3(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open FT3 file: " + path);

    Ft3Frame fr;

    // Header: cycle, dummy, time, originshift (x,y,z = 3×8 bytes)
    fr.cycle = ft3_int(f);
    ft3_skip(f, 4);               // dummy int
    fr.time  = ft3_double(f);
    ft3_skip(f, 24);              // originshift: 3 doubles

    // Grid dimensions
    fr.nx = ft3_int(f);  ft3_skip(f, 4);
    fr.ny = ft3_int(f);  ft3_skip(f, 4);
    fr.nz = ft3_int(f);  ft3_skip(f, 4);

    // Cell sizes
    fr.dx = ft3_double(f);
    fr.dy = ft3_double(f);
    fr.dz = ft3_double(f);

    fr.nph  = ft3_int(f);  ft3_skip(f, 4);
    fr.neli = ft3_int(f);

    // Remaining header fields (7*4 + 4*8 + 4*4 + 28*8 = 300 bytes)
    ft3_skip(f, 300);

    // Skip field arrays — all on the extended (nz+2)*(ny+2)*(nx+2) grid
    const int64_t ncells =
        static_cast<int64_t>(fr.nz + 2) *
        static_cast<int64_t>(fr.ny + 2) *
        static_cast<int64_t>(fr.nx + 2);

    ft3_skip(f, 8 * ncells * fr.nph);  // phase fractions
    ft3_skip(f, 8 * ncells);            // pressure
    ft3_skip(f, 8 * ncells);            // x-velocity  (staggered, but stored full size)
    ft3_skip(f, 8 * ncells);            // y-velocity
    ft3_skip(f, 8 * ncells);            // z-velocity

    // ----- Bubble mesh blocks -----------------------------------------------
    fr.bubbles.resize(fr.neli);
    for (int i = 0; i < fr.neli; ++i) {
        const int32_t nmar = ft3_int(f);   // number of markers (= triangular faces)
        const int32_t npos = ft3_int(f);   // number of unique vertex positions

        Bubble& b = fr.bubbles[i];

        // Bulk read positions: npos * 3 doubles (matches numpy.fromfile dtype=float)
        b.positions.resize(npos);
        f.read(reinterpret_cast<char*>(b.positions.data()),
               static_cast<std::streamsize>(npos) * 3 * sizeof(double));

        // Bulk read connectivity: nmar * 3 * 2 int32s
        // Layout: [conn_marker, point_idx] × 3 per face, repeated nmar times
        std::vector<int32_t> raw_conn(nmar * 6);
        f.read(reinterpret_cast<char*>(raw_conn.data()),
               static_cast<std::streamsize>(nmar) * 6 * sizeof(int32_t));
        b.face_pts.resize(nmar);
        for (int32_t m = 0; m < nmar; ++m) {
            b.face_pts[m][0] = raw_conn[m * 6 + 1];
            b.face_pts[m][1] = raw_conn[m * 6 + 3];
            b.face_pts[m][2] = raw_conn[m * 6 + 5];
        }
    }

    return fr;
}

// ---- Geometry helpers ------------------------------------------------------
// These replicate stl_volume() and stl_area() from FT3_2_STL_2.py

struct Vec3 { double x, y, z; };

static inline Vec3  v3_sub (Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline Vec3  v3_cross(Vec3 a, Vec3 b)
{
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}
static inline double v3_dot (Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline double v3_norm(Vec3 a)          { return std::sqrt(v3_dot(a, a)); }

// Returns (volume [m³], surface_area [m²])
inline std::pair<double,double> mesh_volume_area(const Bubble& b)
{
    double vol = 0.0, area = 0.0;
    for (const auto& tri : b.face_pts) {
        const auto& p0 = b.positions[tri[0]];
        const auto& p1 = b.positions[tri[1]];
        const auto& p2 = b.positions[tri[2]];
        Vec3 v0{p0[0], p0[1], p0[2]};
        Vec3 v1{p1[0], p1[1], p1[2]};
        Vec3 v2{p2[0], p2[1], p2[2]};
        // Signed tetrahedron volume from origin
        vol  += v3_dot(v3_cross(v0, v1), v2) / 6.0;
        // Triangle area
        area += v3_norm(v3_cross(v3_sub(v1, v0), v3_sub(v2, v0))) / 2.0;
    }
    return { std::abs(vol), area };
}

// Bounding-box center — matches PyVista PolyData.center
inline std::array<double,3> bbox_center(const Bubble& b)
{
    double xlo =  1e300, xhi = -1e300;
    double ylo =  1e300, yhi = -1e300;
    double zlo =  1e300, zhi = -1e300;
    for (const auto& p : b.positions) {
        xlo = std::min(xlo, p[0]);  xhi = std::max(xhi, p[0]);
        ylo = std::min(ylo, p[1]);  yhi = std::max(yhi, p[1]);
        zlo = std::min(zlo, p[2]);  zhi = std::max(zhi, p[2]);
    }
    return { (xlo+xhi)/2.0, (ylo+yhi)/2.0, (zlo+zhi)/2.0 };
}

// Volume-equivalent diameter [m]
inline double volume_diameter(double volume)
{
    return std::cbrt(6.0 * volume / M_PI);
}

// Output folder name: {neli}x{diam_mm}mm_eps{holdup_pct}
// Mirrors FT3_2_STL_2.py :: get_folder_name()
inline std::string make_folder_name(const Ft3Frame& fr)
{
    double sum_d = 0.0;
    for (const auto& b : fr.bubbles) {
        auto [vol, area] = mesh_volume_area(b);
        sum_d += volume_diameter(vol) * 1000.0;  // mm
    }
    const int diam_mm   = static_cast<int>(std::round(sum_d / fr.neli));
    const double V_grid = static_cast<double>(fr.nx) * fr.dx *
                          static_cast<double>(fr.ny) * fr.dy *
                          static_cast<double>(fr.nz) * fr.dz;
    // Use ideal sphere volume from rounded diameter (matches Python)
    const double vol_ideal = (1.0/6.0) * M_PI * std::pow(diam_mm / 1000.0, 3) * fr.neli;
    const double holdup = vol_ideal / V_grid;
    const int    eps    = static_cast<int>(std::round(holdup * 100.0 / 5.0)) * 5;
    return std::to_string(fr.neli) + "x" + std::to_string(diam_mm) +
           "mm_eps" + (eps < 10 ? "0" : "") + std::to_string(eps);
}
