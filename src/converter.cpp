// ============================================================================
//  converter.cpp
//  FT3 → Spherical Harmonic weights CSV converter.
//
//  Replaces the two-step Python pipeline:
//    FT3_2_STL_2.py  (FT3 → STL)
//    stl_2_sph_harm_new.py  (STL → CSV with SH weights)
//
//  Usage:
//    ./converter --input <dir> --output <dir> [--lmax 14] [--legendre std|recurrence] [--threads N]
//
//  Input  directory: flat folder containing *.ft3 files (same as convert_dir)
//  Output directory: CSVs written here as <folder_name>.csv
//
//  Output CSV columns (matches stl_2_sph_harm_new.py):
//    ,id,stl,sim,bub_num,time [s],pos_x,pos_y,pos_z,
//    vel_x,vel_y,vel_z,l_max,orb_0,...,orb_{(lmax+1)^2-1}
// ============================================================================

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ft3_reader.hpp"
#include "sph_harm_math.hpp"

namespace fs = std::filesystem;

// ============================================================================
//  CLI argument parsing
// ============================================================================
struct Config {
    std::string     input_dir;
    std::string     output_dir;
    int             l_max = 14;
    LegendreMethod  legendre = LegendreMethod::RECURRENCE;
    int             threads = 0;  // 0 = let OpenMP decide
};

static Config parse_args(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--input"  || arg == "-i") && i + 1 < argc)
            cfg.input_dir  = argv[++i];
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc)
            cfg.output_dir = argv[++i];
        else if ((arg == "--lmax"   || arg == "-l") && i + 1 < argc)
            cfg.l_max = std::stoi(argv[++i]);
        else if (arg == "--legendre" && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "std")
                cfg.legendre = LegendreMethod::STD;
            else if (val == "recurrence")
                cfg.legendre = LegendreMethod::RECURRENCE;
            else {
                std::cerr << "Unknown legendre method: " << val
                          << " (use 'std' or 'recurrence')\n";
                std::exit(1);
            }
        }
        else if ((arg == "--threads" || arg == "-t") && i + 1 < argc)
            cfg.threads = std::stoi(argv[++i]);
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Usage: converter --input <dir> --output <dir> "
                         "[--lmax 14] [--legendre std|recurrence] [--threads N]\n";
            std::exit(1);
        }
    }
    if (cfg.input_dir.empty() || cfg.output_dir.empty()) {
        std::cerr << "Usage: converter --input <dir> --output <dir> "
                     "[--lmax 14] [--legendre std|recurrence] [--threads N]\n";
        std::exit(1);
    }
    return cfg;
}

// ============================================================================
//  Per-bubble SH fit result — columns match stl_2_sph_harm_new.py output
// ============================================================================
struct BubbleResult {
    std::string id;        // e.g. "32x6mm_eps10_0"
    std::string stl;       // virtual STL path
    std::string sim;       // folder name / sim label
    int    bub_num;        // bubble index within frame
    double time_s;         // time in seconds (round(raw*1e5)*1e-5)
    double pos_x, pos_y, pos_z;   // bounding-box center (= PyVista center)
    double vel_x, vel_y, vel_z;   // forward-difference velocity (filled later)
    int    l_max;
    std::vector<double> weights;   // (l_max+1)^2 elements
};

// ============================================================================
//  Process a single FT3 frame → vector of BubbleResult (velocities = 0)
// ============================================================================
static std::vector<BubbleResult> process_frame(
    const Ft3Frame& fr, int l_max,
    const std::string& folder_name,
    LegendreMethod legendre_method = LegendreMethod::RECURRENCE)
{
    const int num_orbs = (l_max + 1) * (l_max + 1);
    const int time_rounded = static_cast<int>(std::round(fr.time * 1e5));
    const double time_s = time_rounded * 1e-5;

    // Preallocate result slots so threads can write independently
    std::vector<BubbleResult> results(fr.neli);

    #pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < fr.neli; ++bi) {
        const Bubble& bub = fr.bubbles[bi];

        // --- 1. Bounding-box center (= PyVista PolyData.center) ---------------
        auto ctr = bbox_center(bub);

        // --- 2. Translate vertices to origin ----------------------------------
        std::vector<double> theta_az, phi_pol, r_vals;
        theta_az.reserve(bub.positions.size());
        phi_pol .reserve(bub.positions.size());
        r_vals  .reserve(bub.positions.size());

        for (const auto& p : bub.positions) {
            const double X = p[0] - ctr[0];
            const double Y = p[1] - ctr[1];
            const double Z = p[2] - ctr[2];
            auto sph = cart2sph(X, Y, Z);
            // Skip degenerate vertices (at the exact centroid)
            if (sph.r < 1e-20) continue;
            theta_az.push_back(sph.theta_az);
            phi_pol .push_back(sph.phi_pol);
            r_vals  .push_back(sph.r);
        }

        // Guard against degenerate bubbles
        if (static_cast<int>(r_vals.size()) < num_orbs) {
            #pragma omp critical
            std::cerr << "[WARN] bubble " << bi
                      << " at t=" << fr.time
                      << " has too few vertices (" << r_vals.size()
                      << "), skipping SH fit.\n";
            // Still record with zero weights
            BubbleResult& res = results[bi];
            res.id = folder_name + "_" + std::to_string(bi);
            res.stl = "data/bubbles_stl/" + folder_name + "/F" +
                       std::to_string(time_rounded) + "_bubble_" +
                       std::to_string(bi) + ".stl";
            res.sim = folder_name;
            res.bub_num = bi;
            res.time_s = time_s;
            res.pos_x = ctr[0]; res.pos_y = ctr[1]; res.pos_z = ctr[2];
            res.vel_x = res.vel_y = res.vel_z = 0.0;
            res.l_max = l_max;
            res.weights.assign(num_orbs, 0.0);
            continue;
        }

        // --- 3. Build SH design matrix and solve lstsq ------------------------
        const Eigen::MatrixXd A = build_sph_matrix(theta_az, phi_pol,
                                                    l_max, legendre_method);
        const int N = static_cast<int>(r_vals.size());
        Eigen::VectorXd b(N);
        for (int k = 0; k < N; ++k) b(k) = r_vals[k];

        const Eigen::VectorXd w = lstsq(A, b);

        // --- 4. Store result --------------------------------------------------
        BubbleResult& res = results[bi];
        res.id = folder_name + "_" + std::to_string(bi);
        res.stl = "data/bubbles_stl/" + folder_name + "/F" +
                   std::to_string(time_rounded) + "_bubble_" +
                   std::to_string(bi) + ".stl";
        res.sim = folder_name;
        res.bub_num = bi;
        res.time_s = time_s;
        res.pos_x = ctr[0]; res.pos_y = ctr[1]; res.pos_z = ctr[2];
        res.vel_x = res.vel_y = res.vel_z = 0.0;
        res.l_max = l_max;
        res.weights.resize(num_orbs);
        for (int j = 0; j < num_orbs; ++j) res.weights[j] = w(j);
    }
    return results;
}

// ============================================================================
//  Compute velocities via forward difference, grouped by bub_num.
//  Matches stl_2_sph_harm_new.py :: get_velocities():
//    vel(t) = (pos(t+1) - pos(t)) / (time(t+1) - time(t))
//    last timestep gets vel = 0 (fillna(0))
// ============================================================================
static void fill_velocities(std::vector<BubbleResult>& records)
{
    // Group record indices by bub_num
    std::map<int, std::vector<size_t>> groups;
    for (size_t i = 0; i < records.size(); ++i)
        groups[records[i].bub_num].push_back(i);

    for (auto& [bub_num, indices] : groups) {
        // Sort by time within group
        std::sort(indices.begin(), indices.end(),
                  [&](size_t a, size_t b) {
                      return records[a].time_s < records[b].time_s;
                  });

        for (size_t j = 0; j + 1 < indices.size(); ++j) {
            auto& cur  = records[indices[j]];
            auto& next = records[indices[j + 1]];
            const double dt = next.time_s - cur.time_s;
            if (dt > 0.0) {
                cur.vel_x = (next.pos_x - cur.pos_x) / dt;
                cur.vel_y = (next.pos_y - cur.pos_y) / dt;
                cur.vel_z = (next.pos_z - cur.pos_z) / dt;
            }
        }
        // Last entry in each group stays at vel = 0 (initialized)
    }
}

// ============================================================================
//  Write results to CSV — matches pandas to_csv() output format
// ============================================================================
static void write_csv(
    const std::string& path,
    const std::vector<BubbleResult>& records,
    int l_max)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write CSV: " + path);

    const int num_orbs = (l_max + 1) * (l_max + 1);

    // Header: leading comma = pandas unnamed index column
    f << ",id,stl,sim,bub_num,time [s],pos_x,pos_y,pos_z,"
         "vel_x,vel_y,vel_z,l_max";
    for (int j = 0; j < num_orbs; ++j) f << ",orb_" << j;
    f << "\n";

    f << std::setprecision(15);

    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        f << i << ','
          << r.id << ','
          << r.stl << ','
          << r.sim << ','
          << r.bub_num << ','
          << r.time_s << ','
          << r.pos_x << ',' << r.pos_y << ',' << r.pos_z << ','
          << r.vel_x << ',' << r.vel_y << ',' << r.vel_z << ','
          << r.l_max;
        for (int j = 0; j < num_orbs; ++j) f << ',' << r.weights[j];
        f << '\n';
    }
}

// ============================================================================
//  main
// ============================================================================
int main(int argc, char** argv)
{
    const Config cfg = parse_args(argc, argv);

    // Configure OpenMP thread count
#ifdef _OPENMP
    if (cfg.threads > 0)
        omp_set_num_threads(cfg.threads);
    std::cout << "OpenMP enabled: using " << omp_get_max_threads() << " threads\n";
#else
    if (cfg.threads > 0)
        std::cerr << "[WARN] --threads ignored (compiled without OpenMP)\n";
    std::cout << "OpenMP disabled: single-threaded\n";
#endif

    std::cout << "Legendre method: "
              << (cfg.legendre == LegendreMethod::STD ? "std" : "recurrence")
              << "\n";

    // Collect .ft3 files, sort by name for time-ordering
    std::vector<fs::path> ft3_files;
    for (const auto& entry : fs::directory_iterator(cfg.input_dir)) {
        if (entry.path().extension() == ".ft3")
            ft3_files.push_back(entry.path());
    }
    if (ft3_files.empty()) {
        std::cerr << "No .ft3 files found in: " << cfg.input_dir << "\n";
        return 1;
    }
    std::sort(ft3_files.begin(), ft3_files.end());

    std::cout << "Found " << ft3_files.size() << " .ft3 files in "
              << cfg.input_dir << "\n";

    // Determine output folder name from first file
    std::string folder_name;
    std::cout << "Loading first file to determine output folder name…\n";
    {
        auto first = load_ft3(ft3_files[0].string());
        folder_name = make_folder_name(first);
    }
    std::cout << "Output label: " << folder_name << "\n";

    fs::create_directories(cfg.output_dir);
    const std::string csv_path =
        (fs::path(cfg.output_dir) / (folder_name + ".csv")).string();

    // Process all frames — collect into a flat records vector
    std::vector<BubbleResult> all_records;
    all_records.reserve(ft3_files.size() * 32);  // estimated

    const auto t_start = std::chrono::steady_clock::now();

    for (const auto& fp : ft3_files) {
        std::cout << "  Processing: " << fp.filename().string() << " … " << std::flush;
        const auto t0 = std::chrono::steady_clock::now();

        Ft3Frame fr = load_ft3(fp.string());

        auto frame_results = process_frame(fr, cfg.l_max, folder_name,
                                             cfg.legendre);
        all_records.insert(all_records.end(),
                           std::make_move_iterator(frame_results.begin()),
                           std::make_move_iterator(frame_results.end()));

        const auto t1  = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << fr.neli << " bubbles  (" << std::fixed << std::setprecision(1)
                  << ms << " ms)\n";
    }

    // Fill velocities (forward difference grouped by bub_num)
    fill_velocities(all_records);

    // Write CSV
    write_csv(csv_path, all_records, cfg.l_max);

    const auto t_end = std::chrono::steady_clock::now();
    const double total_s = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nDone. Output: " << csv_path << "\n";
    std::cout << "Total wall time: " << std::fixed << std::setprecision(2)
              << total_s << " s\n";

    return 0;
}
