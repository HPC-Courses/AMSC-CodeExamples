# Thrust sparse matrices

`apsc::gpu::SparseMatrix` is a header-only sparse matrix library. Include
`<SparseMatrix.hpp>` and configure Thrust for the desired execution backend.
The implementation is in `namespace apsc::gpu` and does not require Eigen. Tests use
Eigen as an independent numerical reference.

```cpp
namespace apsc::gpu {
template<class Scalar, int Options = ColMajor, class StorageIndex = std::size_t>
class SparseMatrix;
}
```

The template parameter order and storage-order option values match
[`Eigen::SparseMatrix`](https://eigen.tuxfamily.org/dox/classEigen_1_1SparseMatrix.html).
You can pass `Eigen::RowMajor` or `Eigen::ColMajor` if Eigen is included. This is
not an Eigen expression-template replacement or a complete copy of its API.
Supported scalars are floating-point types and `thrust::complex<T>` with
floating-point `T`; `std::complex` and integer scalars are not supported.
`Index` and the default `StorageIndex` are `std::size_t`; unlike Eigen,
`StorageIndex` must be an unsigned integral type other than `bool`.
Dimensions and stored-entry counts must fit both `StorageIndex` and the signed
`std::ptrdiff_t` iterator range, including the extra outer end offset; oversized
inputs raise `std::length_error`.

The public types and utilities (`SparseMatrix`, `Triplet`, `DiagonalView`,
`TransposeView`, `RowMajor`, `ColMajor`, and free operators) live in
`apsc::gpu`. NVIDIA's containers and algorithms remain in `thrust`, so
ordinary references such as `thrust::device_vector` are unambiguous, including
inside `namespace apsc`. No leading global `::` qualification is needed.
The header is `include/SparseMatrix.hpp`; the CMake target remains
`apsc::MatrixVector`.

## Folder contents

| Path | Purpose |
| --- | --- |
| `include/SparseMatrix.hpp` | Header-only matrix, triplet type, views, and device handle |
| `examples/sparse_example.cpp` | Assembly, multiplication, norms, and writable views |
| `tests/sparse_tests.cpp` | Numerical/API checks against Eigen and CUDA memory checks |
| `benchmarks/sparse_benchmark.cpp` | Validated timings for one selected Thrust backend |
| `benchmarks/compare_backends.py` | Builds CPU and CUDA versions, runs them, and computes speedups |
| `benchmarks/results/rtx4060-cpp/` | Recorded serial-CPU/RTX-4060 measurements, logs, and metadata |
| `CMakeLists.txt` | Interface library and optional examples, tests, and benchmark targets |

## Build and test

From the repository root, using your GPU's compute capability 8.9:

```sh
cmake -S MatrixVector -B MatrixVector/build-gpu \
  -DAPSC_THRUST_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build MatrixVector/build-gpu --parallel 2
ctest --test-dir MatrixVector/build-gpu --output-on-failure
./MatrixVector/build-gpu/sparse_example
```

The default backend is CUDA, with architecture `native` (local GPU detection).
The architecture setting is clearly marked near the top of `CMakeLists.txt`.
Use a numeric capability without the dot, or quote a list such as
`'-DCMAKE_CUDA_ARCHITECTURES=86;89'`. GPU detection and execution need access to
the NVIDIA driver, which may require running outside an agent sandbox.

For a CPU-only build:

```sh
cmake -S MatrixVector -B MatrixVector/build-cpu \
  -DAPSC_THRUST_BACKEND=CPP -DCMAKE_BUILD_TYPE=Release
cmake --build MatrixVector/build-cpu --parallel 2
ctest --test-dir MatrixVector/build-cpu --output-on-failure
```

`OMP` and `TBB` are also supported CPU backends, requiring OpenMP and TBB
respectively. **Only the CUDA backend places the arrays in GPU memory.**
With CPP/OMP/TBB, `thrust::device_vector` uses the selected CPU backend's memory.
Use separate build directories when changing backend.

Dependencies: CMake 3.24+, C++20, Thrust/CUB/libcu++ (CCCL), and CUDA Toolkit for
the CUDA backend. CMake searches the default Linux Toolkit installation;
otherwise set `CCCL_ROOT`, `Thrust_DIR`, or `CMAKE_PREFIX_PATH`. Eigen3 is needed
only when `APSC_BUILD_TESTS=ON`. Standalone builds enable tests and examples by
default; disable with `-DAPSC_BUILD_TESTS=OFF` / `-DAPSC_BUILD_EXAMPLES=OFF`.

To use it in another project:

```cmake
# Set APSC_THRUST_BACKEND and CMAKE_CUDA_ARCHITECTURES before adding the directory.
add_subdirectory(path/to/MatrixVector)
target_link_libraries(my_program PRIVATE apsc::MatrixVector)
```

Tests and examples default to off when added as a subdirectory. CUDA consumers
must compile translation units that instantiate numerical operations as CUDA
(e.g. `.cu` files, or CMake source property `LANGUAGE CUDA`). The library uses
Thrust algorithms, requires no compiled library, no Eigen headers, and no
extended-lambda or separable-compilation flags.

## CPU versus GPU benchmark

The benchmark uses **the same `apsc::gpu::SparseMatrix` implementation** in separate
Release builds. The default host baseline is serial Thrust (`CPP`), not Eigen.
Select `OMP` or `TBB` to compare against parallel CPU execution. Python 3 is
required only for the comparison driver; the C++ executable has no Eigen
requirement.

Run from the repository root:

```sh
python3 MatrixVector/benchmarks/compare_backends.py --architecture 89
```

This configures and builds `sparse_benchmark` for CPP and CUDA, then runs both
with identical data. Build files go into `MatrixVector/build-benchmark/`; reports
go into `MatrixVector/benchmark-results/`. A failed correctness check, failed
process, or mismatched workload causes the comparison to fail.

For a parallel CPU comparison and a larger workload:

```sh
python3 MatrixVector/benchmarks/compare_backends.py \
  --architecture 89 --cpu-backend OMP --threads 8 \
  --sizes 1000,10000,100000,1000000 --entries-per-row 9 \
  --iterations 20 --samples 5 --warmup 3 --layout both \
  --output MatrixVector/benchmark-results-omp
```

`--threads` sets `OMP_NUM_THREADS`; it does not limit TBB. If omitted, OpenMP
uses its runtime default/environment. `--jobs` controls compilation parallelism,
not benchmark threads. Other useful options are `--cccl-root /path/to/cccl`,
`--build-root /path/to/build`, and `--output /path/to/results`. Each output
directory is overwritten by a new run, so use distinct directories to retain
comparisons. Run `--help` for the full option list. For multi-configuration
CMake generators the script explicitly builds and runs the Release configuration.

### Build and run the benchmark manually

```sh
cmake -S MatrixVector -B MatrixVector/build-benchmark-cpu \
  -DCMAKE_BUILD_TYPE=Release -DAPSC_THRUST_BACKEND=CPP \
  -DAPSC_BUILD_BENCHMARKS=ON -DAPSC_BUILD_TESTS=OFF -DAPSC_BUILD_EXAMPLES=OFF
cmake --build MatrixVector/build-benchmark-cpu --target sparse_benchmark --parallel 2
./MatrixVector/build-benchmark-cpu/sparse_benchmark --csv cpu.csv

cmake -S MatrixVector -B MatrixVector/build-benchmark-gpu \
  -DCMAKE_BUILD_TYPE=Release -DAPSC_THRUST_BACKEND=CUDA \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DAPSC_BUILD_BENCHMARKS=ON -DAPSC_BUILD_TESTS=OFF -DAPSC_BUILD_EXAMPLES=OFF
cmake --build MatrixVector/build-benchmark-gpu --target sparse_benchmark --parallel 2
./MatrixVector/build-benchmark-gpu/sparse_benchmark --csv gpu.csv
```

The executable accepts `--sizes`, `--entries-per-row`, `--iterations`,
`--samples`, `--warmup`, `--layout csr|csc|both`, and `--csv`. Without `--csv`,
CSV is printed to stdout; progress goes to stderr. All counts must be positive.
Benchmarks are intentionally not registered as automatic CTest performance
thresholds: timings depend on the hardware and current system load.

### What is measured

Defaults: square double-precision matrices of order 1,000, 10,000, and 100,000;
9 distinct nonzeros per row (clamped to the matrix order); CSR and CSC;
3 warmup calls; 5 samples of 20 calls per operation. Columns are generated
with a deterministic coprime stride to exercise indirect vector accesses.
The same entries and input vectors are used in both processes. This workload
has uniform row lengths; it does not represent every sparsity pattern.

| CSV operation | Work included |
| --- | --- |
| `spmv` | `A*x` with matrix and vector already resident |
| `transpose_spmv` | `A.transpose()*x` with resident data |
| `diagonal` | Materialize the diagonal into a vector |
| `frobenius_norm`, `max_norm` | Corresponding scalar matrix norms |
| `l1_norm`, `infinity_norm`, `entrywise_l1_norm` | Column-sum, row-sum, and entrywise norms |
| `spmv_vector_roundtrip` | Copy input from host, multiply, copy result to host; matrix remains resident |

Timing uses `std::chrono::steady_clock` and explicit CUDA synchronization before
and after each sample. CUDA initialization, input generation, and reference
calculations are excluded. Each sample reports elapsed time divided by the
number of calls; the summary uses the median of those sample means. Minimum
and maximum sample means are retained. These are wall-clock **API timings**:
result allocation, temporary buffers, library synchronization, and scalar
returns to the host are included. They are not isolated GPU kernel timings.
CPU compiler barriers and observable, validated outputs prevent benchmark work
from being optimized away.

Every operation is checked against an independent serial reference before
sampling and after every sample; full vector comparisons occur outside timed
regions. The round-trip operation includes vector copies on CPU too. Matrix
assembly (including host sorting and initial GPU upload) and the initial input
copy are reported separately as `assembly_ms` and `input_copy_ms`. These are
single setup measurements, repeated as metadata on each row for that matrix;
they are not summed into the resident operation timings or speedups. The
round-trip case does **not** include uploading the matrix for every multiply.

Results:

* `cpp.csv` (or `omp.csv` / `tbb.csv`) and `cuda.csv`: raw per-operation timings,
  sample ranges, validation checksums, and setup costs.
* `comparison.csv`: CPU and GPU median times and `speedup_cpu_over_gpu`.
* `report.md`: readable table; speedup greater than 1 favors the GPU.
* `metadata.json`: hardware, timestamp, options, thread environment, and status.
* `*-configure.log`, `*-build.log`, `*-run.log`: reproducibility and diagnostic logs.

A serial CPU baseline must not be interpreted as the fastest available host
implementation. Small matrices can favor the CPU because allocation, launch,
and synchronization costs dominate. Layout matters: CSR ordinary products and
CSC transpose products use a direct gather; the opposite paths use sorting and
reduction. Repeat measurements on an otherwise idle machine, with consistent
power and thermal conditions, before drawing performance conclusions.

### Recorded results on this machine

The [full report](benchmarks/results/rtx4060-cpp/report.md) and
[raw comparison](benchmarks/results/rtx4060-cpp/comparison.csv) were generated
on 2026-10-23 using an Intel Core Ultra 7 165H (serial CPP backend) and NVIDIA
GeForce RTX 4060 Laptop GPU (architecture 89). All 54 operation/workload cases
passed independent numerical validation on each backend.

For CSR ordinary multiplication, 9 entries per row:

| Rows | Nonzeros | CPU ms | Resident GPU ms | Resident speedup | Vector round-trip speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 9,000 | 0.003783 | 0.014833 | 0.26x | 0.11x |
| 10,000 | 90,000 | 0.041016 | 0.017052 | 2.41x | 0.73x |
| 100,000 | 900,000 | 0.590800 | 0.051470 | 11.48x | 1.49x |

These measurements show the difference between keeping data resident and moving
vectors on every call. They are one recorded run, not guaranteed speedups.
The larger CSC ordinary-product speedups in the full report also reflect the
library's more expensive sort/reduce path; CSR remains the faster layout for
ordinary matrix-vector multiplication.

## Basic use

```cpp
#include <SparseMatrix.hpp>
#include <vector>

apsc::gpu::SparseMatrix<double, apsc::gpu::RowMajor> A(3, 3);
std::vector<apsc::gpu::Triplet<double>> entries{
    {0, 0, 4}, {0, 1, -1}, {1, 0, -1}, {1, 1, 4}, {2, 2, 2}};
A.setFromTriplets(entries.begin(), entries.end());
thrust::device_vector<double> x{1, 2, 3};
auto y = A * x;                    // [2, 7, 6], stored on the device
A.multiply(x, y);                 // output argument; x and y may alias
A(2, 2) = 5;                      // writable host proxy
const double value = A.coeff(0, 1);
auto diagonal = A.diagonal();     // non-owning, no allocation
diagonal[0] += 1;                 // modifies A
auto d = diagonal.eval();         // independent device_vector copy
auto dy = diagonal * x;           // elementwise diagonal-vector product
auto At = A.transpose();          // non-owning, non-conjugating transpose
auto z = At * x;
auto B = At.eval();               // independent matrix in the opposite layout
```

Triplets may be in any order; duplicates are summed and exact zeros removed.
Any iterator value type with integral `row()` and `col()` coordinates and a
`value()` works, including
`Eigen::Triplet` with signed indices. Coordinates are checked before conversion
to unsigned storage; negative or out-of-bounds coordinates are rejected.
Assembly sorts on the host and uploads the three compressed
buffers once. There is no persistent host mirror. Device triplet iterators also
work, but transfer entries individually; host iterators are preferable.

## Storage and access

* `ColMajor` (0, default) uses compressed sparse columns (CSC).
* `RowMajor` (1) uses compressed sparse rows (CSR).
* `outerIndices()`, `innerIndices()`, and `values()` return const references to
  the three `thrust::device_vector` buffers. Outer offsets have length
  `outerSize()+1`, start at zero, and end at `nonZeros()`. Inner indices are
  strictly increasing within each segment.
* `outerIndexPtr()` and `innerIndexPtr()` expose read-only device pointers;
  `valuePtr()` exposes a device pointer to the coefficients (writable for a
  mutable matrix). Never dereference CUDA device pointers on the CPU.
* `deviceView()` returns a small read-only handle suitable for capture in a
  Thrust device functor. Its `coeff(r,c)` and `operator()(r,c)` perform an
  allocation-free binary search on the device. Coordinates must be in bounds.

Host `coeff`, `coeffRef`, and `operator()` check bounds. Reads return scalar
copies; writes use a proxy because GPU memory cannot be returned as `Scalar&`.
[Thrust host-side element access](https://nvidia.github.io/cccl/unstable/thrust/api/classthrust_1_1device__reference.html)
involves transfers and can synchronize. Prefer batched algorithms in performance
critical code. Existing entries update in place. Inserting an absent entry via
`setCoeff` or a proxy rebuilds the matrix on the host: use `setFromTriplets` for
bulk construction. Assigning zero to an existing entry retains its slot until
`prune(tolerance)` removes entries whose magnitude is at most the tolerance.

`resize` discards entries; `setZero` preserves dimensions; `setIdentity` also
works on rectangular matrices. `reserve` reserves coefficient/index capacity,
but does not turn individual insertions into a fast assembly strategy.
`makeCompressed()` is a no-op because storage is always compressed.
`toHostTriplets()` explicitly exports a host copy. Copying a matrix performs a
deep device-to-device copy; moving leaves the source as a valid 0-by-0 matrix.
`sum()`, in-place scalar multiplication, free scalar multiplication, and `swap`
are also provided. Scalar factors in the free operators have the matrix's
scalar type (e.g. `2.0 * A` for a double matrix).

## Norm definitions

All return `RealScalar`, including for complex matrices. Missing entries are
zero. All norms of an empty/zero matrix are zero.

| Method | Definition |
| --- | --- |
| `frobeniusNorm()`, `norm()` | square root of the sum of squared magnitudes |
| `maxNorm()` | largest absolute entry |
| `l1Norm()` | induced matrix 1-norm: maximum absolute column sum |
| `infinityNorm()` | induced infinity norm: maximum absolute row sum |
| `entrywiseL1Norm()` | sum of magnitudes of all entries |

The Frobenius implementation scales by the largest magnitude before squaring
to avoid intermediate overflow/underflow. Norms propagate NaN values; an infinite
entry yields an infinite Frobenius norm unless NaN is also present. Operations
on complex matrices use magnitudes for norms and ordinary scalar multiplication
for products; transpose does **not** conjugate values.

## Views, lifetime, and execution cost

Diagonal and transpose views refer to the parent matrix object. They reflect
later edits, including structural edits, and must not outlive the parent.
Views of a const matrix are read-only. View creation from matrix temporaries is
disabled. A view of a mutable matrix remains writable even if the view object
itself is const, as with a mutable span. Moving/swapping/resizing the parent
changes what its existing views observe. `eval()` creates independent storage.
Device handles and raw pointers have stricter lifetimes: discard them after
any structural change, assignment, move, swap, or destruction. Concurrent
mutation and access require synchronization by the caller; proxies are not
atomic.

Numerical work is dispatched through Thrust. CSR `A*x` and CSC `A.transpose()*x`
use one parallel work item per output segment and O(nnz + output-size) work.
The opposite layouts compute contributions, sort by output index, reduce, and
scatter, using O(nnz) temporary storage in addition to the output. Work includes
O(nnz log outer-size) segment lookups, sorting (O(nnz log nnz) as a general
comparison-sort upper bound), reduction, and output initialization. This avoids floating-point atomics and supports complex
values. The same strategy is used for norms in the non-contiguous direction.
For repeated ordinary matrix-vector products, prefer CSR. Highly unequal row
lengths may cause load imbalance; this teaching implementation does not replace
a tuned cuSPARSE solver. Scalar norm results return to the host; no asynchronous
stream API is exposed.

Tests compare against Eigen and cover both layouts, 32-/64-bit indices, complex
arithmetic, duplicate assembly, rectangular/empty matrices, mutation, views,
invalid input, aliasing, and scaling. CUDA tests also verify that all three
compressed buffers report device memory and exercise the raw device handle.

Validation on 2026-10-23: the test suite passed with CPP, OpenMP, TBB, and CUDA
using the installed CCCL 3.4.3 package. CUDA tests ran on the NVIDIA GeForce
RTX 4060 Laptop GPU with architecture 89, CUDA compiler 13.4, and driver
595.99.02. The example returned `A*x = 2 7 6` and, after its diagonal update,
`A^T*x = 2 7 15`.
