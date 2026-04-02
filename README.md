# FT3 → Spherical Harmonic Converter

A high-performance C++ pipeline that reads Front-Tracking (FT3) binary simulation files, extracts bubble surface meshes, and computes real spherical harmonic weight decompositions. Replaces a multi-step Python workflow with a single compiled binary, achieving over 10× speedup while producing numerically identical results.

## Table of Contents

1. [Design and Purpose](#design-and-purpose)
2. [Compilation, Validation and Running](#compilation-validation-and-running)
3. [File Descriptions](#file-descriptions)
4. [Future Work](#future-work)

---

## Design and Purpose

### Background

Multiphase CFD simulations using the Front-Tracking method produce `.ft3` binary snapshot files at each timestep. Each file contains the Eulerian grid data (pressure, velocity, phase fractions) and Lagrangian bubble surface meshes (vertex positions and triangular connectivity). Characterising bubble shapes requires decomposing each surface into real spherical harmonic (SH) coefficients up to a configurable order $l_{\max}$.

### Original Python Pipeline

The original workflow involved two sequential Python scripts:

1. **`FT3_2_STL_2.py`** — Parsed each `.ft3` file, extracted per-bubble triangle meshes, and wrote them to disk as individual `.stl` files.
2. **`stl_2_sph_harm_new.py`** — Read each `.stl` file back from disk, converted vertex coordinates to spherical form, built the SH design matrix using `scipy.special.sph_harm`, solved an overdetermined least-squares system via `numpy.linalg.lstsq`, and accumulated results into a CSV.

This approach suffered from heavy I/O overhead (thousands of intermediate STL files), slow per-bubble Python loops, and the inability to parallelise across bubbles.

### C++ Replacement

The converter eliminates these bottlenecks by:

- **Skipping intermediate files entirely** — FT3 binary data is parsed directly into memory and processed without writing STL files to disk.
- **Using optimised numerics** — Real SH basis functions are computed via a recurrence relation for associated Legendre polynomials (or optionally via `std::sph_legendre`). Least-squares fitting uses Eigen's BDCSVD solver.
- **Parallelising across bubbles** — Each bubble's SH fit is independent; OpenMP distributes them across available cores.

### Mathematical Convention

The real spherical harmonic basis follows the same convention as `scipy.special.sph_harm`:

$$
Y_l^m(\theta, \phi) = \begin{cases}
\sqrt{2} \cdot \text{Im}(Y_l^{|m|}) & m < 0 \\
Y_l^0 & m = 0 \\
\sqrt{2} \cdot \text{Re}(Y_l^m) & m > 0
\end{cases}
$$

where $\theta$ is the azimuthal angle (longitude, $0 \ldots 2\pi$) and $\phi$ is the polar angle (colatitude, $0 \ldots \pi$).

Orbital indexing follows a flat scheme: for index $j = 0, 1, \ldots, (l_{\max}+1)^2 - 1$:

$$
l = \lfloor \sqrt{j} \rfloor, \quad m = j - l^2 - l
$$

For $l_{\max} = 14$, this yields 225 orbital weight coefficients per bubble.

### Output Format

The output CSV matches the format produced by the original Python pipeline, with columns:

| Column | Description |
|--------|-------------|
| (index) | Row number |
| `id` | Unique identifier: `{folder}_{bubble_index}` |
| `stl` | Virtual STL path (for compatibility with downstream scripts) |
| `sim` | Simulation label derived from grid and bubble properties |
| `bub_num` | Bubble index within the frame (0-based) |
| `time [s]` | Simulation time, rounded to $10^{-5}$ s |
| `pos_x`, `pos_y`, `pos_z` | Bounding-box centre of the bubble mesh |
| `vel_x`, `vel_y`, `vel_z` | Forward-difference velocity (0 for last timestep) |
| `l_max` | Maximum SH degree used |
| `orb_0` … `orb_224` | Real spherical harmonic weight coefficients |

---

## Compilation, Validation and Running

### Prerequisites

- **Compiler**: GCC with C++17 support (tested with `g++ 11+`)
- **Eigen 3**: Header-only linear algebra library (`sudo apt install libeigen3-dev`)
- **OpenMP**: For multi-threaded bubble processing (included with GCC)
- **Python 3** (for validation only): `numpy`, `scipy`, `pandas`

### Compilation

Build both the converter and the SH validation binary:

```bash
make all
```

This produces two executables in `bin/`:

- `bin/converter` — the main FT3-to-CSV pipeline
- `bin/validate_sph` — standalone SH math validation tool

To build only the converter:

```bash
make bin/converter
```

To clean build artefacts:

```bash
make clean
```

### Validation

#### 1. Spherical Harmonic Math Validation

Verifies that the C++ real SH implementation matches `scipy.special.sph_harm` at fixed test points for all 225 orbitals ($l_{\max} = 14$):

```bash
# Generate C++ reference values
./bin/validate_sph

# Generate Python reference values and compare
python3 tests/validate_sph_harm.py
```

Expected output: maximum absolute difference on the order of $10^{-15}$ (machine epsilon).

#### 2. Full Pipeline Validation

Compares the complete C++ pipeline output against the Python pipeline for a given FT3 file:

```bash
# Run C++ converter on test data
mkdir -p test_output
./bin/converter --input input --output test_output --lmax 14

# Compare against Python pipeline
python3 tests/validate_pipeline.py input/F28310.ft3 test_output/32x6mm_eps10.csv --lmax 14
```

Expected output: position and weight differences on the order of $10^{-16}$ (PASS).

### Running

#### Basic Usage

```bash
./bin/converter --input <input_dir> --output <output_dir> [options]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | (required) | Directory containing `.ft3` files |
| `--output` | `-o` | (required) | Directory for output CSV(s) |
| `--lmax` | `-l` | `14` | Maximum spherical harmonic degree |
| `--legendre` | | `recurrence` | Legendre computation method: `std` or `recurrence` |
| `--threads` | `-t` | (auto) | Number of OpenMP threads (0 = system default) |

#### Examples

Process a directory of FT3 files with default settings:

```bash
./bin/converter --input /path/to/ft3_files --output /path/to/output
```

Use the C++17 standard library Legendre functions and limit to 8 threads:

```bash
./bin/converter -i data/ -o results/ --legendre std --threads 8
```

Use a lower SH order for faster (but less detailed) decomposition:

```bash
./bin/converter -i data/ -o results/ --lmax 6
```

#### Integration with `script.sh`

The `scripts/script.sh` shell script provides a polling loop that watches for new `.ft3` files, moves them to a conversion directory, runs the C++ converter, and cleans up. Edit the paths in `scripts/script.sh` to match your simulation directory layout.

---

## File Descriptions

### C++ Source Files (`src/`)

| File | Description |
|------|-------------|
| `src/converter.cpp` | Main application. Parses CLI arguments, iterates over `.ft3` files, calls `process_frame()` for SH fitting (parallelised with OpenMP), computes inter-frame velocities via forward differencing, and writes the output CSV. |
| `src/ft3_reader.hpp` | Header-only FT3 binary parser. Reads the file header (cycle, time, grid dimensions), skips Eulerian field arrays, and extracts per-bubble vertex positions and triangular face connectivity via bulk reads. Also provides geometry helpers: bounding-box centre, mesh volume/area (signed tetrahedron method), volume-equivalent diameter, and the folder-naming convention. |
| `src/sph_harm_math.hpp` | Header-only spherical harmonic library. Provides: the `real_Y_lm()` scalar function; `build_sph_matrix()` to construct the $N \times (l_{\max}+1)^2$ design matrix; `lstsq()` for Eigen BDCSVD least-squares solving; `cart2sph()` for coordinate conversion. Supports two Legendre back-ends selectable via `LegendreMethod` enum — `STD` (delegates to `std::sph_legendre`) and `RECURRENCE` (local implementation using sectoral/tesseral recurrence with precomputed normalisation). |
| `src/validate_sph.cpp` | Standalone validation program. Evaluates all 225 real SH basis functions at 8 fixed test points and writes the results to `/tmp/cpp_sph_values.csv` for comparison against Python. |
| `Makefile` | Build rules for `bin/converter` and `bin/validate_sph`. Flags: `-O3 -std=c++17 -fopenmp -I/usr/include/eigen3`. |

### Test / Validation Files (`tests/`)

| File | Description |
|------|-------------|
| `tests/validate_pipeline.py` | End-to-end validation script. Independently re-implements the Python FT3 parsing and SH fitting pipeline, then compares bubble positions and all 225 orbital weights against the C++ CSV output. Reports per-bubble discrepancies exceeding a $10^{-5}$ tolerance. |
| `tests/validate_sph_harm.py` | SH math validation companion. Computes `scipy.special.sph_harm` at the same test points as `validate_sph.cpp` and reports maximum differences. |

### Python Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `scripts/FT3_2_STL_2.py` | Original Python FT3→STL converter (superseded by `converter`). |
| `scripts/stl_2_sph_harm_new.py` | Original Python STL→SH CSV pipeline (superseded by `converter`). |
| `scripts/sph_harm_functions.py` | Python SH helper functions used by the original pipeline. |

### Other Files

| File | Description |
|------|-------------|
| `scripts/script.sh` | Bash polling loop for production use. Watches for `.ft3` files, moves them to a conversion directory, invokes the C++ converter, and cleans up. |
| `input/F28310.ft3` | Sample FT3 file (cycle 28310) used for testing. |

---

## Future Work

* [ ] Point Cloud Input Support
* [ ] Configuration File
* [ ] GUI for Visualising Input and Output Shapes
