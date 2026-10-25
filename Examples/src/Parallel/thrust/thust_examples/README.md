# Standalone Thrust teaching examples

This examples contains thrust examples extracted form the NVIDIA the CCCL 
developer kit.
Requirements: CMake 3.24+, a C++17 compiler, and compatible CCCL headers
(Thrust, CUB, and libcu++, nurmally distributed with Cuda). Older Toolkit headers 
may lack APIs used by the newer examples. The complete
set has been compiled with the accompanying `cccl-main.zip` snapshot (Thrust
3.6.0.0). No dependencies are downloaded automatically.

Commands below run from this directory. Use separate build directories for
CPU and GPU configurations. Executables are placed in `build-*/bin` (with an
additional configuration directory for multi-configuration generators).

## Dependency setup

Use an installed Thrust CMake package, or extract the accompanying archive:

```sh
unzip ../cccl-main.zip -d ../dependencies
cmake -S . -B build-cpu -DCCCL_ROOT=../dependencies/cccl-main -DTHRUST_BACKEND=CPP
```

`CCCL_ROOT` accepts an unpacked CCCL source tree or installation prefix.
Alternatively, set `Thrust_DIR` to the directory containing
`thrust-config.cmake`, or use `CMAKE_PREFIX_PATH`. The default Linux CUDA
installation at `/usr/local/cuda` is also searched. CPU builds need CCCL
headers but do not need nvcc, a GPU, or a CUDA driver.

## Build all examples or just one

```sh
# All examples supported by the configured backend:
cmake --build build-cpu --parallel 4
# Equivalent explicit aggregate target:
cmake --build build-cpu --target examples --parallel 4

# Build and run a single example from the same configuration:
cmake --build build-cpu --target norm
./build-cpu/bin/norm
```

For a configuration containing only selected examples:

```sh
cmake -S . -B build-selected -DCCCL_ROOT=../dependencies/cccl-main \
  -DTHRUST_BACKEND=CPP '-DTHRUST_EXAMPLES=norm;saxpy;cpp_integration'
cmake --build build-selected --parallel 4
```

The default `THRUST_EXAMPLES=all` includes every supported example. Target names
match source basenames; CUDA-specific targets have a `cuda_` prefix, such as
`cuda_async_reduce`. `cpp_integration` links `host.cpp` and `device.cu` into
one executable and works with either backend. Invalid or unavailable names
produce a configuration error. Building does not run the programs: some are
substantial computations or accept input.

## CPU backends

| `THRUST_BACKEND` | Execution | Extra requirement |
| --- | --- | --- |
| `CPP` (default) | Serial CPU | None |
| `OMP` | Parallel CPU using OpenMP | Compiler OpenMP support |
| `TBB` | Parallel CPU using TBB | Installed TBB development package |
| `CUDA` | NVIDIA GPU | CUDA Toolkit, nvcc, compatible host compiler |

For example, configure `build-omp` with `-DTHRUST_BACKEND=OMP` and the same
dependency arguments. Even `thrust::device_vector` uses CPU memory with a CPU
backend. Files in `cuda/` explicitly use CUDA and are omitted from CPU builds.

## GPU build and architecture selection

**GPU architecture selection is isolated in `cmake/GpuArchitecture.cmake`.**
The default is `native`: CMake detects the GPU visible on the build machine.
This requires a working NVIDIA driver at configuration time. You need to address -DCCCL_ROOT
to the CCCL toolkit, unless it is installed in /usr/local/cuda/. You can use the toolkint provided
in `ccl-main.zip`

```sh
cmake -S . -B build-gpu -DCCCL_ROOT=../dependencies/cccl-main \
  -DTHRUST_BACKEND=CUDA
cmake --build build-gpu --parallel 4
./build-gpu/bin/norm
```

To target a specific GPU, override `CMAKE_CUDA_ARCHITECTURES`. Use its compute
capability without the dot: for example, capability 8.6 becomes `86`.

```sh
# Explicit architecture also allows compilation on a machine without a GPU:
cmake -S . -B build-gpu -DCCCL_ROOT=../dependencies/cccl-main \
  -DTHRUST_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=86

# Build for several GPU architectures (quote the semicolon-separated list):
cmake -S . -B build-multi-gpu -DCCCL_ROOT=../dependencies/cccl-main \
  -DTHRUST_BACKEND=CUDA '-DCMAKE_CUDA_ARCHITECTURES=80;86;89'
```

On a working driver installation, inspect the GPU with
`nvidia-smi --query-gpu=name,compute_cap --format=csv`, or consult
[NVIDIA's compute capability tables](https://developer.nvidia.com/cuda-gpus).
Choose architectures supported by your installed CUDA Toolkit. `CUDAARCHS`
can also set the initial architecture through the environment. Cached settings
persist; pass a new `-D` value to change them. GPU execution still requires a
compatible GPU and working driver, even when compilation uses an explicit
architecture. To select nvcc explicitly on the first configure, use
`-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.

The build enables CUDA extended lambdas and separable compilation for the
examples that need them. CPU builds compile generated C++ wrappers around the
original `.cu` sources, preserving the teaching code.

Upstream references:
[examples](https://github.com/NVIDIA/cccl/tree/main/thrust/examples) and
[Thrust CMake integration](https://github.com/NVIDIA/cccl/blob/main/lib/cmake/thrust/README.md).

## Validation

Using the accompanying CCCL snapshot, all 53 CPU examples compiled with each
of CPP, OMP, and TBB; all 60 CUDA examples compiled with CUDA 13.4 targeting
architecture 86. Representative CPU programs were also run.

GPU validation on 2026-10-23: all 60 CUDA executables were rebuilt for
architecture 89 and ran successfully (exit code 0, assertions enabled) on an
NVIDIA GeForce RTX 4060 Laptop GPU, compute capability 8.9, with driver
595.99.02. Automatic `native` detection was also verified by building and
running `norm`, which returned the expected value 5.47723. These are execution
smoke tests, including assertions supplied by the examples, rather than an
independent numerical validation of every output. GPU access required running
outside the agent sandbox; the initial driver failure was a sandbox restriction.
