# Parallel Matrix-Vector Product with MPI

Matrix-vector multiplication is one of the classic examples used to introduce
distributed linear algebra.

In this folder, the matrix is partitioned by rows across MPI processes, while
the vector is replicated where needed. The example also measures the time spent
in the different stages of the computation.

The code shows the use of several collective operations, including:

- broadcast
- scatter
- gather
- barriers for synchronization

It also highlights a practical issue: scattering the matrix can be expensive,
and storing the full matrix on the root process is memory-intensive. In real
applications one often tries to build local matrix blocks directly on each
process instead of assembling everything on rank 0 first.

The default build uses MPI only. An optional hybrid MPI/OpenMP build lets each
MPI process use OpenMP threads for local work.

## Building and Running

Run these commands from this folder. The Makefile includes the shared
`Examples/Makefile.inc` and the local `Makefile.inc`, which selects `mpic++`.

Build and run the default version without OpenMP:

```sh
make
mpirun -n 8 ./main_matrix_vector_product -r 1000 -c 1000
```

Build and run the hybrid version with OpenMP enabled through `-fopenmp`:

```sh
make openmp
mpirun -n 8 --bind-to none ./main_matrix_vector_product_openmp -r 1000 -c 1000 -t 2
```

Here, `-n 8` selects eight MPI processes and `-t 2` requests two OpenMP threads
per process. The `-t` option controls parallel execution only in the OpenMP
build. Matrix rows (`-r`) and columns (`-c`) both default to 1000; the thread
count (`-t`) defaults to 1. Use `-v` to print the matrix, vector and result, or
`-h` to display the synopsis.

For the hybrid build, use Open MPI's `--bind-to none` unless you explicitly
allocate multiple cores to each rank. Depending on its version and launch
configuration, Open MPI may bind a rank to a single core. Its OpenMP threads
inherit that CPU affinity and can then compete for the same core, preventing
useful scaling when `-t` increases. `--bind-to none` disables MPI process
binding, allowing the operating system to schedule the threads across the
available CPUs; it does not itself pin individual threads or reserve cores.

For example, compare thread counts on one rank with:

```sh
mpirun -n 1 --bind-to none ./main_matrix_vector_product_openmp -r 1000 -c 1000 -t 1
mpirun -n 1 --bind-to none ./main_matrix_vector_product_openmp -r 1000 -c 1000 -t 4
```

Choose the number of ranks and threads to fit the available CPU resources.
For controlled placement, an alternative is to allocate multiple cores per
rank, for example `--map-by slot:PE=4 --bind-to core` with `-t 4`, and configure
OpenMP thread affinity as needed. Use `--report-bindings` to inspect MPI's
placement. See the [Open MPI mpirun documentation](https://www.open-mpi.org/doc/v4.1/man1/mpirun.1.php)
for details. Removing restrictive binding enables parallel execution, but
small workloads and shared memory bandwidth can still limit speedup.

The two builds use separate object files and executables, so they can coexist
without cleaning between builds. `make clean` removes both versions and their
object files.

## What You Learn Here

- a classic distributed matrix-vector product pattern
- the use of collective MPI communication
- the cost of setup and data movement in distributed linear algebra
- an example of hybrid parallel paradigm
