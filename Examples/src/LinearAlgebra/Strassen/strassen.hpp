#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

/*!
 * \file strassen.hpp
 * \brief Hybrid Strassen/Eigen dense matrix multiplication.
 */

namespace apsc
{
namespace detail
{
//! Convenience alias for a dynamic dense Eigen matrix.
template <typename T>
using DynamicMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

/*!
 * \brief Non-owning read-only view used inside the recursive kernel.
 *
 * The recursive kernel receives owned dense matrices and quadrant blocks of
 * those matrices. This fixed `Ref` type accepts both without copying the
 * blocks, while avoiding unbounded template instantiations from recursively
 * nesting `Block<Block<...>>` expression types.
 */
template <typename T>
using ConstMatrixView =
  Eigen::Ref<const DynamicMatrix<T>, 0, Eigen::OuterStride<Eigen::Dynamic>>;

/*!
 * \brief Computes a dense matrix product with Eigen's optimized kernel.
 *
 * This is the fallback path used whenever recursion is not beneficial.
 */
template <typename T>
[[nodiscard]] inline DynamicMatrix<T>
gemm(const ConstMatrixView<T> &A, const ConstMatrixView<T> &B)
{
  DynamicMatrix<T> C(A.rows(), B.cols());
  C.noalias() = A * B;
  return C;
}

/*!
 * \brief Returns the square size needed to embed a compatible product.
 *
 * Strassen's standard identities are written for square blocks. Rectangular
 * products are therefore embedded in a square zero-padded product and cropped
 * back to the requested shape.
 */
template <typename MatrixA, typename MatrixB>
[[nodiscard]] inline Eigen::Index
square_extent(const MatrixA &A, const MatrixB &B)
{
  return std::max(A.rows(), std::max(A.cols(), B.cols()));
}

//! Returns n itself when even, otherwise the next even integer.
[[nodiscard]] inline Eigen::Index
even_extent(Eigen::Index n)
{
  return n + (n % 2);
}

/*!
 * \brief Decides whether a product is large enough for Strassen recursion.
 *
 * Odd or rectangular shapes are handled by zero-padding before the recursive
 * square kernel is called. Very small products still use Eigen's direct
 * product to avoid recursive overhead.
 */
template <typename MatrixA, typename MatrixB>
[[nodiscard]] inline bool
use_strassen(const MatrixA &A, const MatrixB &B, Eigen::Index cutoff)
{
  const Eigen::Index extent = square_extent(A, B);
  return A.cols() == B.rows() && extent > cutoff && extent > 1;
}

/*!
 * \brief Decides whether a square subproblem should be recursively split.
 *
 * The dimension may be odd; \ref strassen_impl pads odd square subproblems by
 * one row and one column before forming equal-sized quadrants.
 */
template <typename MatrixA, typename MatrixB>
[[nodiscard]] inline bool
use_square_strassen(const MatrixA &A, const MatrixB &B, Eigen::Index cutoff)
{
  return A.rows() == A.cols() && B.rows() == B.cols() && A.rows() == B.rows() &&
         A.rows() > cutoff && A.rows() > 1;
}

/*!
 * \brief Recursive implementation of Strassen multiplication.
 *
 * \param A Left square matrix, passed either as an owned dense matrix or as a
 * matrix-block view.
 * \param B Right square matrix, passed either as an owned dense matrix or as a
 * matrix-block view.
 * \param cutoff Recursion threshold below which Eigen's direct product is
 * used.
 * \return The matrix product \f$AB\f$.
 */
template <typename T>
[[nodiscard]] DynamicMatrix<T>
strassen_impl(const ConstMatrixView<T> &A, const ConstMatrixView<T> &B,
              Eigen::Index cutoff)
{
  // For small or unsuitable problems, Eigen's blocked GEMM is the better
  // kernel and avoids recursive overhead.
  if(!use_square_strassen(A, B, cutoff))
    {
      return gemm<T>(A, B);
    }

  using Matrix = DynamicMatrix<T>;
  const Eigen::Index n = A.rows();

  // Standard Strassen block formulas require equal-sized quadrants. For odd
  // square subproblems, pad by one row and column and crop after recursion.
  if(n % 2 != 0)
    {
      const Eigen::Index paddedSize = n + 1;
      Matrix             paddedA = Matrix::Zero(paddedSize, paddedSize);
      Matrix             paddedB = Matrix::Zero(paddedSize, paddedSize);

      paddedA.topLeftCorner(n, n) = A;
      paddedB.topLeftCorner(n, n) = B;

      Matrix paddedC = strassen_impl<T>(paddedA, paddedB, cutoff);
      return paddedC.topLeftCorner(n, n);
    }

  const Eigen::Index h = n / 2;

  // Split both operands into four equally-sized quadrant views.
  const auto A11 = A.topLeftCorner(h, h);
  const auto A12 = A.topRightCorner(h, h);
  const auto A21 = A.bottomLeftCorner(h, h);
  const auto A22 = A.bottomRightCorner(h, h);

  const auto B11 = B.topLeftCorner(h, h);
  const auto B12 = B.topRightCorner(h, h);
  const auto B21 = B.bottomLeftCorner(h, h);
  const auto B22 = B.bottomRightCorner(h, h);

  // Reusable evaluated temporaries for the linear combinations required by the
  // seven Strassen products. Keeping these sums lazy would recompute them while
  // the recursive calls split their operands again.
  Matrix T1(h, h), T2(h, h);

  // These are the seven recursive products that replace the eight products of
  // the classical 2x2 block formula.
  T1.noalias() = A11 + A22;
  T2.noalias() = B11 + B22;
  Matrix M1 = strassen_impl<T>(T1, T2, cutoff);

  T1.noalias() = A21 + A22;
  Matrix M2 = strassen_impl<T>(T1, B11, cutoff);

  T2.noalias() = B12 - B22;
  Matrix M3 = strassen_impl<T>(A11, T2, cutoff);

  T2.noalias() = B21 - B11;
  Matrix M4 = strassen_impl<T>(A22, T2, cutoff);

  T1.noalias() = A11 + A12;
  Matrix M5 = strassen_impl<T>(T1, B22, cutoff);

  T1.noalias() = A21 - A11;
  T2.noalias() = B11 + B12;
  Matrix M6 = strassen_impl<T>(T1, T2, cutoff);

  T1.noalias() = A12 - A22;
  T2.noalias() = B21 + B22;
  Matrix M7 = strassen_impl<T>(T1, T2, cutoff);

  // Reassemble the four quadrants of C from the seven Strassen products.
  Matrix C(n, n);
  C.topLeftCorner(h, h).noalias() = M1 + M4 - M5 + M7;
  C.topRightCorner(h, h).noalias() = M3 + M5;
  C.bottomLeftCorner(h, h).noalias() = M2 + M4;
  C.bottomRightCorner(h, h).noalias() = M1 - M2 + M3 + M6;

  return C;
}

/*!
 * \brief Embeds any compatible product into a square Strassen product.
 *
 * The recursive kernel only sees square matrices. This wrapper zero-pads the
 * operands when their shape is rectangular or odd-sized, then crops the
 * resulting product back to \f$A.rows() \times B.cols()\f$.
 */
template <typename T>
[[nodiscard]] DynamicMatrix<T>
strassen_with_padding(const DynamicMatrix<T> &A, const DynamicMatrix<T> &B,
                      Eigen::Index cutoff)
{
  if(!use_strassen(A, B, cutoff))
    {
      return gemm<T>(A, B);
    }

  using Matrix = DynamicMatrix<T>;

  if(A.rows() == A.cols() && B.rows() == B.cols() && A.rows() == B.rows())
    {
      return strassen_impl<T>(A, B, cutoff);
    }

  const Eigen::Index paddedSize = even_extent(square_extent(A, B));

  Matrix paddedA = Matrix::Zero(paddedSize, paddedSize);
  Matrix paddedB = Matrix::Zero(paddedSize, paddedSize);

  paddedA.topLeftCorner(A.rows(), A.cols()) = A;
  paddedB.topLeftCorner(B.rows(), B.cols()) = B;

  Matrix paddedC = strassen_impl<T>(paddedA, paddedB, cutoff);
  return paddedC.topLeftCorner(A.rows(), B.cols());
}
} // namespace detail

/*!
 * \brief Multiplies two dense Eigen matrices with a hybrid Strassen strategy.
 *
 * The function accepts generic Eigen matrix expressions, evaluates them once
 * into owned dense matrices, and then either:
 *
 * - dispatches directly to Eigen's optimized product, or
 * - applies Strassen recursion when the padded square problem is larger than
 *   the chosen cutoff.
 *
 * Inside the recursive kernel, quadrant operands are passed as non-owning
 * Eigen::Ref views. Only the Strassen linear combinations are materialized,
 * because keeping those sums lazy would recompute them across deeper recursive
 * splits.
 *
 * \tparam DerivedA Eigen expression type for the left operand.
 * \tparam DerivedB Eigen expression type for the right operand.
 * \tparam Scalar Scalar type stored in the matrices.
 * \param A Left matrix.
 * \param B Right matrix.
 * \param cutoff Recursion threshold. Below this size, Eigen's direct product
 * is used.
 * \return The matrix product \f$AB\f$.
 *
 * \throw std::invalid_argument if the matrix dimensions are incompatible.
 */
template <
  typename DerivedA, typename DerivedB,
  typename Scalar =
    std::common_type_t<typename DerivedA::Scalar, typename DerivedB::Scalar>>
[[nodiscard]] inline detail::DynamicMatrix<Scalar>
strassen(const Eigen::MatrixBase<DerivedA> &A,
         const Eigen::MatrixBase<DerivedB> &B, Eigen::Index cutoff = 256)
{
  static_assert(std::is_same_v<typename DerivedA::Scalar, Scalar> &&
                  std::is_same_v<typename DerivedB::Scalar, Scalar>,
                "strassen requires both operands to have the same scalar type");

  if(A.cols() != B.rows())
    {
      throw std::invalid_argument("strassen: incompatible matrix dimensions");
    }

  using Matrix = detail::DynamicMatrix<Scalar>;

  // Evaluate arbitrary user expressions once. Recursive calls then operate on
  // these owned matrices and on non-owning block views into them.
  const Matrix lhs = A.eval();
  const Matrix rhs = B.eval();

  return detail::strassen_with_padding(lhs, rhs, cutoff);
}
} // namespace apsc
