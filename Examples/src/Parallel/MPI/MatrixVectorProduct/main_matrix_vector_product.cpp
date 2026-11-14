#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#include <omp.h>
#pragma GCC diagnostic pop

#include "GetPot"
#include <../../Utilities/partitioner.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>
// #include <mdspan> if you want to use mdspan, which is a C++23 feature that
// allows for multi-dimensional array views
//  See https://en.cppreference.com/w/cpp/experimental/mdspan for more
//  information.
static double c_start{0.};
inline void
tic()
{
  c_start = MPI_Wtime();
}
inline double
toc()
{
  return MPI_Wtime() - c_start;
}
inline void
printout(double const c_diff, const std::string &x)
{
  std::cout << x << c_diff << " [s]" << std::endl;
}

void
printHelp()
{
  std::cout
    << "Usage: mpirun -n <processes> ./main_matrix_vector_product [options]\n"
    << "  -r, --rows <number>     Matrix rows (default: 1000)\n"
    << "  -c, --cols <number>     Matrix columns (default: 1000)\n"
    << "  -t, --threads <number>  OpenMP threads (default: 1)\n"
    << "  -v                     Print matrix, vector and result\n"
    << "  -h, --help             Print this synopsis and exit\n";
}

/**
 * Parallel matrix-vector product.
 *
 * Data are generated randomly by rank 0 and then broadcasted. Each rank is
 * assigned a sub-block of the input matrix (split by rows, without overlap) and
 * computes the local matrix-vector product. Finally, the results are collected
 * back by rank 0, which prints them to output.
 *
 * This example makes use of hybrid shared/distributed parallelization through
 * OpenMP and MPI.
 *
 * Running, e.g., on a matrix of size 10^4 x 10^3 should show good scalability
 * properties.
 *
 * The bottleneck in the computational time is constructing the matrix on rank 0
 * and scattering it to other ranks: in real applications each rank would build
 * its own "local" matrix.
 */
int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  int mpi_rank;
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  int mpi_size;
  MPI_Comm_size(mpi_comm, &mpi_size);

  // Set to true to print matrix, vector and result.
  // only for debug purposes, as printing large matrices and vectors is not
  // practical.
  GetPot     options(argc, argv);
  const bool print = options.search("-v");

  unsigned int n_rows;
  unsigned int n_cols;

  std::vector<double> matrix;
  std::vector<double> rhs;
  std::vector<double> result;

  int num_threads;
  int input_status = 0;
  if(mpi_rank == 0)
    {
      if(options.search(2, "-h", "--help"))
        {
          printHelp();
          input_status = 1;
        }
      else
        {
          const int rows = options.follow(1000, 2, "-r", "--rows");
          const int cols = options.follow(1000, 2, "-c", "--cols");
          num_threads = options.follow(1, 2, "-t", "--threads");
          if(rows < 1 || cols < 1 || num_threads < 1)
            {
              std::cerr << "ERROR: Rows, columns and threads must be positive."
                        << std::endl;
              input_status = 2;
            }
          else
            {
              n_rows = rows;
              n_cols = cols;
              std::cout << "Number of processes: " << mpi_size << std::endl;
            }
        }
    }

  MPI_Bcast(&input_status, 1, MPI_INT, 0, mpi_comm);
  if(input_status != 0)
    {
      MPI_Finalize();
      return input_status == 1 ? 0 : 1;
    }

  MPI_Bcast(&n_rows, 1, MPI_UNSIGNED, 0, mpi_comm);
  MPI_Bcast(&n_cols, 1, MPI_UNSIGNED, 0, mpi_comm);
  MPI_Bcast(&num_threads, 1, MPI_INT, 0, mpi_comm);

  rhs.resize(n_cols);
  if(mpi_rank == 0)
    result.resize(n_rows);

  // Here I use the partitioner utilities to compute the counts and
  // displacements for each rank, which will be used in the MPI_Scatterv and
  // MPI_Gatherv calls. The partitioner takes care of distributing the rows
  // of the matrix evenly among the ranks, taking into account any remainder
  // if the number of rows is not divisible by the number of ranks.
  apsc::MatrixPartitioner<apsc::GroupedPartitioner, apsc::ORDERINGTYPE::ROWWISE>
                           partitionMatrix(n_rows, n_cols, mpi_size);
  apsc::GroupedPartitioner partitionVector(mpi_size, n_rows);
  const auto               n_rows_local =
    partitionVector.last(mpi_rank) - partitionVector.first(mpi_rank);
  std::cout << "Number of rows on rank " << mpi_rank << ": " << n_rows_local
            << std::endl;

  std::vector<double> matrix_local(n_rows_local * n_cols);

  std::vector<double> result_local(n_rows_local, 0.0);

  std::vector<int>    send_counts;
  std::vector<int>    send_start_idx;
  std::vector<int>    recv_counts;
  std::vector<int>    recv_start_idx;
  std::vector<double> timings(mpi_size, 0.0);
  if(mpi_rank == 0)
    {
      tic();
      std::random_device rd;
      // generate a seed sequnce, one for each thread, to ensure that each
      // thread has a different seed
      std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
      std::vector<std::uint32_t> seeds(num_threads);
      seed.generate(seeds.begin(), seeds.end());

      // Generate matrix.
      matrix.resize(n_rows * n_cols);
      // Work sharing covers all entries even if fewer threads are created.
#pragma omp parallel shared(matrix, seeds) num_threads(num_threads)
      {
        // initialize a random number generator for each thread with a different
        // seed
#if defined(_OPENMP)
        auto thread_num = omp_get_thread_num();
#else
        auto thread_num = 0;
#endif
        std::mt19937                           engine(seeds[thread_num]);
        std::uniform_real_distribution<double> rand(0, 1);
#pragma omp for schedule(static)
        for(std::size_t i = 0; i < matrix.size(); ++i)
          matrix[i] = rand(engine);
      }
      if(print)
        {
          std::cout << "Matrix:" << std::endl;
          for(unsigned int i = 0; i < n_rows; ++i)
            {
              for(unsigned int j = 0; j < n_cols; ++j)
                std::cout << std::setw(12) << matrix[i * n_cols + j];

              std::cout << std::endl;
            }
          std::cout << std::endl;
        }

        // Generate rhs filling it with ones in parallel using OpenMP.
#pragma omp parallel for shared(rhs) num_threads(num_threads) schedule(static)
      for(std::size_t i = 0; i < rhs.size(); ++i)
        rhs[i] = 1.0;

      if(print)
        {
          std::cout << "RHS:" << std::endl;
          for(const auto &v : rhs)
            {
              std::cout << std::setw(12) << v;
            }
          std::cout << std::endl << std::endl;
        }
      std::cout << std::endl;
      printout(toc(), "Assemble matrix: time elapsed on rank " +
                        std::to_string(mpi_rank) + ": ");
      // counts and displacements are in the scope of rank 0,
      // as they are only needed for the MPI_Scatterv and MPI_Gatherv calls.
      std::tie(send_counts, send_start_idx) =
        apsc::counts_and_displacements(partitionMatrix);
      std::tie(recv_counts, recv_start_idx) =
        apsc::counts_and_displacements(partitionVector);
    }

  tic();
  MPI_Bcast(rhs.data(), rhs.size(), MPI_DOUBLE, 0, mpi_comm);

  MPI_Scatterv(matrix.data(), send_counts.data(), send_start_idx.data(),
               MPI_DOUBLE, matrix_local.data(), matrix_local.size(), MPI_DOUBLE,
               0, mpi_comm);

  timings[mpi_rank] = toc();
  MPI_Barrier(mpi_comm);
  for(int rank = 0; rank < mpi_size; ++rank)
    {
      if(rank == mpi_rank && mpi_rank == 0)
        std::cout << std::endl;
      MPI_Barrier(mpi_comm);

      if(rank == mpi_rank)
        {
          printout(timings[mpi_rank], "Scatter: time elapsed on rank " +
                                        std::to_string(mpi_rank) + ": ");
        }

      MPI_Barrier(mpi_comm);
    }
    // To make measuring meaningful I need to make sure that all ranks have
    // initialised OpenMP, otherwise, only rank 0 will have initialised OpenMP
    // and the timings will be skewed. This is a known issue with OpenMP and
    // MPI. This modification is however needed only for timings. In a real code
    // you may ignore it, as the first call to an OpenMP parallel region will
    // initialize OpenMP on all ranks.

    // Warm up OpenMP on every MPI rank.
#if defined(_OPENMP)
  int warmup = 0;
#pragma omp parallel num_threads(num_threads) shared(warmup)
  {
#pragma omp atomic update
    warmup += omp_get_thread_num() + 1;
  }
#endif
  // Align ranks before starting the measurement.
  MPI_Barrier(mpi_comm);

  // mdspan is a C++23 feature that allows for
  // multi-dimensional array views without copying data creates a mdspan view of
  // the local matrix with the specified extents and row-major layout
  // std::mdspan<double, std::extents<std::size_t, std::size_t>,
  // std::layout_right> matrix_local_2d(matrix_local.data(), n_rows_local,
  // n_cols);
  tic();
#pragma omp parallel for shared(result_local) num_threads(num_threads)
  for(std::size_t i = 0; i < n_rows_local; ++i)
    {
      for(std::size_t j = 0; j < n_cols; ++j)
        {
          // result_local[i] += matrix_local_2d(i, j) * rhs[j]; // if using
          // mdspan
          result_local[i] += matrix_local[i * n_cols + j] * rhs[j];
        }
    }
  double time = toc();
  // gather timeings from all ranks and print them on rank 0
  MPI_Gather(&time, 1, MPI_DOUBLE, timings.data(), 1, MPI_DOUBLE, 0, mpi_comm);
  MPI_Barrier(mpi_comm);

  if(mpi_rank == 0)
    {
      for(int rank = 0; rank < mpi_size; ++rank)
        {
          printout(timings[rank],
                   "Matrix-vector product: time elapsed on rank " +
                     std::to_string(rank) + ": ");
        }
    }

  tic();
  MPI_Gatherv(result_local.data(), result_local.size(), MPI_DOUBLE,
              result.data(), recv_counts.data(), recv_start_idx.data(),
              MPI_DOUBLE, 0, mpi_comm);

  timings[mpi_rank] = toc();
  if(print && mpi_rank == 0)
    {
      std::cout << std::endl << "Result:" << std::endl;

      for(const auto &v : result)
        std::cout << std::setw(12) << v;

      std::cout << std::endl << std::endl;
    }

  MPI_Barrier(mpi_comm);
  // This is a trick only to make the output more readable. It avoids garbled
  // output. It does not guarantee that the output respact rank order.
  // In a real code you may ignore it, unless output  it is important for you.
  // If also the order is important, you need to collect data on a single rank
  // and print them from there, or use MPI I/O.
  for(int rank = 0; rank < mpi_size; ++rank)
    {
      if(rank == mpi_rank && mpi_rank == 0)
        std::cout << std::endl;
      MPI_Barrier(mpi_comm);

      if(rank == mpi_rank)
        {
          printout(timings[mpi_rank], "Gather: time elapsed on rank " +
                                        std::to_string(mpi_rank) + ": ");
        }

      MPI_Barrier(mpi_comm);
    }

  MPI_Finalize();

  return 0;
}
