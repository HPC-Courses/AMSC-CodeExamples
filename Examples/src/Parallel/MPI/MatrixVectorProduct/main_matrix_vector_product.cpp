#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#include <omp.h>
#pragma GCC diagnostic pop

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
static double c_start, c_diff;
#define tic() c_start = MPI_Wtime();
#define toc(x)                                       \
  {                                                  \
    c_diff = MPI_Wtime() - c_start;                  \
    std::cout << x << c_diff << " [s]" << std::endl; \
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

  if(mpi_rank == 0)
    std::cout << "Number of processes: " << mpi_size << std::endl;

  // Set to true to print matrix, vector and result.
  // only for debug purposes, as printing large matrices and vectors is not
  // practical.
  const bool print = false;

  unsigned int n_rows;
  unsigned int n_cols;

  std::vector<double> matrix;
  std::vector<double> rhs;
  std::vector<double> result;

  int num_threads;
  if(mpi_rank == 0)
    {
      std::cout << std::endl << "Enter the number of matrix rows:" << std::endl;
      std::cin >> n_rows;

      if(n_rows < 1)
        {
          std::cerr << "ERROR: Number of rows should be greater than 1."
                    << std::endl;

          return 1;
        }

      std::cout << "Enter the number of matrix columns:" << std::endl;
      std::cin >> n_cols;

      if(n_cols < 1)
        {
          std::cerr << "ERROR: Number of columns should be greater than 1."
                    << std::endl;

          return 1;
        }

      std::cout << std::endl;

      std::cout << "Enter the number of openMP threads:" << std::endl;
      std::cout << "(the actual number used can be different to to hardware "
                   "limitations)\n";
      std::cin >> num_threads;
    }

  MPI_Bcast(&n_rows, 1, MPI_INT, 0, mpi_comm);
  MPI_Bcast(&n_cols, 1, MPI_INT, 0, mpi_comm);
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

  std::vector<int> send_counts;
  std::vector<int> send_start_idx;
  std::vector<int> recv_counts;
  std::vector<int> recv_start_idx;
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
        auto         thread_num = omp_get_thread_num();
        std::mt19937 engine(seeds[thread_num]);
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
      toc("Assemble matrix: time elapsed on rank " + std::to_string(mpi_rank) +
          ": ");
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

  MPI_Barrier(mpi_comm);
  for(int rank = 0; rank < mpi_size; ++rank)
    {
      if(rank == mpi_rank && mpi_rank == 0)
        std::cout << std::endl;
      MPI_Barrier(mpi_comm);

      if(rank == mpi_rank)
        {
          toc("Scatter: time elapsed on rank " + std::to_string(mpi_rank) +
              ": ");
        }

      MPI_Barrier(mpi_comm);
    }
  // an use of mdspan to access the local matrix as a 2D array
  // mdspan is a C++23 feature that allows for multi-dimensional array views
  // without copying data creates a mdspan view of the local matrix with the
  // specified extents and row-major layout
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

  MPI_Barrier(mpi_comm);
  for(int rank = 0; rank < mpi_size; ++rank)
    {
      if(rank == mpi_rank && mpi_rank == 0)
        std::cout << std::endl;
      MPI_Barrier(mpi_comm);

      if(rank == mpi_rank)
        {
          toc("Matrix-vector product: time elapsed on rank " +
              std::to_string(mpi_rank) + ": ");
        }

      MPI_Barrier(mpi_comm);
    }

  tic();

  MPI_Gatherv(result_local.data(), result_local.size(), MPI_DOUBLE,
              result.data(), recv_counts.data(), recv_start_idx.data(),
              MPI_DOUBLE, 0, mpi_comm);

  if(print && mpi_rank == 0)
    {
      std::cout << std::endl << "Result:" << std::endl;

      for(const auto &v : result)
        std::cout << std::setw(12) << v;

      std::cout << std::endl << std::endl;
    }

  MPI_Barrier(mpi_comm);
  for(int rank = 0; rank < mpi_size; ++rank)
    {
      if(rank == mpi_rank && mpi_rank == 0)
        std::cout << std::endl;
      MPI_Barrier(mpi_comm);

      if(rank == mpi_rank)
        {
          toc("Gather: time elapsed on rank " + std::to_string(mpi_rank) +
              ": ");
        }

      MPI_Barrier(mpi_comm);
    }

  MPI_Finalize();

  return 0;
}
