#include <SparseMatrix.hpp>
#include <iostream>
#include <vector>

int main() {
  // RowMajor selects CSR. Omitting it selects Eigen-compatible column-major CSC.
  apsc::gpu::SparseMatrix<double, apsc::gpu::RowMajor> A(3, 3);
  std::vector<apsc::gpu::Triplet<double>> entries{
    {0, 0, 4}, {0, 1, -1}, {1, 0, -1}, {1, 1, 4}, {2, 2, 2}};
  A.setFromTriplets(entries.begin(), entries.end());
  thrust::device_vector<double> x{1, 2, 3};
  thrust::host_vector<double> y = A * x;
  std::cout << "A*x =";
  for (double v : y) std::cout << ' ' << v;
  std::cout << "\nFrobenius norm = " << A.frobeniusNorm()
            << "\nMaximum absolute entry = " << A.maxNorm()
            << "\nMatrix 1-norm = " << A.l1Norm() << '\n';

  auto diagonal = A.diagonal(); // non-owning; does not copy matrix data
  diagonal[2] = 5;             // host-side coefficient update
  auto transposed = A.transpose();
  thrust::host_vector<double> z = transposed * x;
  std::cout << "A^T*x after diagonal update =";
  for (double v : z) std::cout << ' ' << v;
  std::cout << '\n';
}
