# Thrust teaching examples and sparse linear algebra

This directory collects C++ examples for learning Thrust, experimenting with
CPU and NVIDIA GPU execution, and applying parallel algorithms to numerical
computing. The projects are built independently; there is no single top-level
build target.

**A Note** The ompilation of the code in these folders do not follow the general makefiles of the other examples. This is becouse the makefile and (when present) the CMakeLists.txt files are adapted fo a gpu enabled environment using the thrust library. Athe the end of this file you find indication on where
to find thrust documentation.

| Path | Contents |
| --- | --- |
| [`simple/`](simple/) | Introductory example using `thrust::device_vector` and `thrust::inner_product` to compute a sum of `x[i] * y[i]^2`. Includes comments on device functors and CUDA lambdas, plus an nvcc Makefile. |
| [`dotProdZip/`](dotProdZip/README.md) | Pairwise dot products of 3D vectors using zip iterators and structure-of-arrays storage. Its Makefile supports serial CPU and CUDA backends. |
| [`heattransfer/`](heattransfer/) | Numerical solution of the temperature distribution in a one-dimensional bar, with Thrust operations, parameter-file readers, a Thomas tridiagonal solver, sample inputs, and a plotting script. Its Makefile provides GPU and host builds. |
| [`thust_examples/`](thust_examples/README.md) | Examples extracted from NVIDIA's Thrust sources, with a standalone CMake build for individual examples or the whole collection. Supports CPP, OpenMP, TBB, and CUDA. The directory name is intentionally written here as it exists: `thust_examples`. |
| [`thust_examples/cuda/`](thust_examples/cuda/) | CUDA-specific examples, including streams, asynchronous reduction, pointer wrapping, and temporary allocation. |
| [`thust_examples/cpp_integration/`](thust_examples/cpp_integration/README) | Example linking separate C++ and CUDA translation units, integrated into the standalone example build. |
| [`MatrixVector/`](MatrixVector/README.md) | Header-only `apsc::gpu::SparseMatrix` library with Eigen-compatible template parameters, CSR/CSC storage in Thrust vectors, matrix-vector products, norms, and diagonal/transpose views. Includes examples, correctness tests, and CPU/GPU benchmarks. |
| [`GetPot`](GetPot) | Bundled header for command-line and parameter-file parsing, used by the heat-transfer example. |
| `cccl-main.zip` | Local archive of the CUDA Core Compute Libraries sources; it can supply compatible Thrust, CUB, and libcu++ headers for the extracted examples. |


## Getting started

Start with `simple/` for a small reduction example, then `dotProdZip/` for zip
iterators. Explore `thust_examples/` for more algorithms and iterator patterns.
Use `heattransfer/` and `MatrixVector/` for numerical applications, and the
`MatrixVector` benchmarks to investigate performance and transfer overhead.

### Requirements and backend selection

* A C++ compiler and compatible CCCL headers (Thrust, CUB, and libcu++).
* CUDA Toolkit and nvcc for CUDA builds, plus an NVIDIA GPU and working driver
  for GPU execution.
* GNU Make for `simple`, `dotProdZip`, and `heattransfer`; CMake 3.24 or newer
  for `thust_examples` and `MatrixVector`.
* `MatrixVector`, `simple`, and `dotProdZip` use C++20; the other CMake projects use C++17.
* `MatrixVector` correctness tests additionally require Eigen3. Its benchmark
  comparison script requires Python 3.8 or newer.
* OpenMP or TBB support is needed only when selecting those CPU backends.

With a CPU Thrust backend, `thrust::device_vector` uses CPU memory: its name does
not by itself guarantee GPU storage. Select CUDA for device execution. Use
separate CMake build directories for different backends.

GPU architecture **89** corresponds to the RTX 4060 Laptop GPU used for the
local tests. CMake accepts `-DCMAKE_CUDA_ARCHITECTURES=89`; nvcc Makefiles use
`sm_89`. Change this setting for your GPU. The CMake projects also support
`native` detection when the GPU is accessible during configuration.

### Build a standalone Thrust example

Run these commands from this directory:

```sh
# CPU version; build just the norm example.
cmake -S thust_examples -B build-examples-cpu -DTHRUST_BACKEND=CPP
cmake --build build-examples-cpu --target norm
./build-examples-cpu/bin/norm

# GPU version; build all CUDA-compatible examples.
cmake -S thust_examples -B build-examples-gpu \
  -DTHRUST_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-examples-gpu --parallel 2
```

If your installed headers are too old for the extracted examples, unpack the
local archive and configure with `-DCCCL_ROOT=/path/to/cccl-main`. See the
[standalone examples README](thust_examples/README.md) for dependency setup,
selected-example builds, and architecture overrides.

### Build and benchmark the sparse matrix library

```sh
cmake -S MatrixVector -B build-matrix-gpu \
  -DCMAKE_BUILD_TYPE=Release -DAPSC_THRUST_BACKEND=CUDA \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-matrix-gpu --parallel 2
ctest --test-dir build-matrix-gpu --output-on-failure

# Build and compare optimized CPU and GPU benchmarks.
python3 MatrixVector/benchmarks/compare_backends.py --architecture 89
```

The [MatrixVector README](MatrixVector/README.md) describes the API, optional
build targets, test dependencies, benchmark methodology, and recorded results.

### Makefile-based examples

```sh
make -C simple
./simple/main

make -C dotProdZip BACKEND=gpu CUDA_ARCH=sm_89
./dotProdZip/dot_products_with_zip
# Alternatively: make -C dotProdZip BACKEND=cpu

make -C heattransfer gpu GETPOT_DIR=.. CUDA_ARCH=sm_89
# Alternatively: make -C heattransfer host GETPOT_DIR=..
```

Run the heat-transfer executable from `heattransfer/` so it can find its default
`parameters.pot` input. Its host target produces `main_host`; its GPU target
produces `main`. Check `JSON_DIR`, `CUDA_HOME`, and other dependency paths in
its Makefile for your installation. The current CPU target is named **`host`**.
For `simple`, the GPU architecture is set directly in its Makefile flags.

## Where to find Thrust documentation

Thrust is now maintained within NVIDIA's **CUDA Core Compute Libraries (CCCL)**,
alongside CUB and libcu++. The former standalone Thrust repository is archived;
use the [CCCL repository](https://github.com/NVIDIA/cccl) for current sources,
release information, and installation guidance.

* **[Official Thrust documentation](https://nvidia.github.io/cccl/unstable/thrust/):**
  start here for the overview, containers, algorithms, iterators, function
  objects, and execution policies. Use the navigation and search to locate
  APIs such as `device_vector`, `transform`, `reduce`, and `zip_iterator`.
* **[Thrust API reference](https://nvidia.github.io/cccl/unstable/thrust/api/index.html):**
  detailed class and function declarations for looking up signatures and behavior.
* **[Thrust CMake integration](https://github.com/NVIDIA/cccl/blob/main/lib/cmake/thrust/README.md):**
  explains `find_package(Thrust)`, `thrust_create_target`, dependency discovery,
  and selecting CUDA, CPP, OpenMP, or TBB.
* **[Upstream Thrust examples](https://github.com/NVIDIA/cccl/tree/main/thrust/examples):**
  source examples for learning individual algorithms and iterator techniques;
  the local `thust_examples/` collection was extracted from these sources.
* **[CCCL documentation portal](https://nvidia.github.io/cccl/):**
  entry point for the related CUB and libcu++ documentation as well as Thrust.

The `/unstable/` documentation describes the development version. Choose
release-specific documentation when available, or consult the corresponding
release tag in the CCCL repository, to match your installed headers. Examples
from `main` can use features absent from an older CUDA Toolkit.

For a learning sequence, read about host/device vectors and transfers first,
then `transform` and reductions, followed by fancy iterators and execution
policies. The local examples and their comments provide runnable companions
to the API documentation.
