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
 * \brief Computes a dense matrix product with Eigen's optimized kernel.
 *
 * This is the fallback path used whenever recursion is not beneficial.
 */
template <typename T>
[[nodiscard]] inline DynamicMatrix<T>
gemm(const DynamicMatrix<T> &A, const DynamicMatrix<T> &B)
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
template <typename T>
[[nodiscard]] inline Eigen::Index
square_extent(const DynamicMatrix<T> &A, const DynamicMatrix<T> &B)
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
template <typename T>
[[nodiscard]] inline bool
use_strassen(const DynamicMatrix<T> &A, const DynamicMatrix<T> &B,
             Eigen::Index cutoff)
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
template <typename T>
[[nodiscard]] inline bool
use_square_strassen(const DynamicMatrix<T> &A, const DynamicMatrix<T> &B,
                    Eigen::Index cutoff)
{
  return A.rows() == A.cols() && B.rows() == B.cols() && A.rows() == B.rows() &&
         A.rows() > cutoff && A.rows() > 1;
}

/*!
 * \brief Recursive implementation of Strassen multiplication.
 *
 * \param A Left dense matrix, already evaluated into contiguous storage.
 * \param B Right dense matrix, already evaluated into contiguous storage.
 * \param cutoff Recursion threshold below which Eigen's direct product is
 * used.
 * \return The matrix product \f$AB\f$.
 */
template <typename T>
[[nodiscard]] DynamicMatrix<T>
strassen_impl(const DynamicMatrix<T> &A, const DynamicMatrix<T> &B,
              Eigen::Index cutoff)
{
  // For small or unsuitable problems, Eigen's blocked GEMM is the better
  // kernel and avoids recursive overhead.
  if(!use_square_strassen(A, B, cutoff))
    {
      return gemm(A, B);
    }

  using Matrix = DynamicMatrix<T>;
  const Eigen::Index n = A.rows();

  if(n % 2 != 0)
    {
      const Eigen::Index paddedSize = n + 1;
      Matrix             paddedA = Matrix::Zero(paddedSize, paddedSize);
      Matrix             paddedB = Matrix::Zero(paddedSize, paddedSize);

      paddedA.topLeftCorner(n, n) = A;
      paddedB.topLeftCorner(n, n) = B;

      Matrix paddedC = strassen_impl(paddedA, paddedB, cutoff);
      return paddedC.topLeftCorner(n, n);
    }

  const Eigen::Index h = n / 2;

  // Split both operands into four equally-sized quadrants.
  auto A11 = A.topLeftCorner(h, h);
  auto A12 = A.topRightCorner(h, h);
  auto A21 = A.bottomLeftCorner(h, h);
  auto A22 = A.bottomRightCorner(h, h);

  auto B11 = B.topLeftCorner(h, h);
  auto B12 = B.topRightCorner(h, h);
  auto B21 = B.bottomLeftCorner(h, h);
  auto B22 = B.bottomRightCorner(h, h);

  // Reusable temporaries for the linear combinations required by the seven
  // Strassen products.
  Matrix T1(h, h), T2(h, h);

  // These are the seven recursive products that replace the eight products of
  // the classical 2x2 block formula.
  T1.noalias() = A11 + A22;
  T2.noalias() = B11 + B22;
  Matrix M1 = strassen_impl(T1, T2, cutoff);

  T1.noalias() = A21 + A22;
  Matrix M2 = strassen_impl(T1, B11.eval(), cutoff);

  T2.noalias() = B12 - B22;
  Matrix M3 = strassen_impl(A11.eval(), T2, cutoff);

  T2.noalias() = B21 - B11;
  Matrix M4 = strassen_impl(A22.eval(), T2, cutoff);

  T1.noalias() = A11 + A12;
  Matrix M5 = strassen_impl(T1, B22.eval(), cutoff);

  T1.noalias() = A21 - A11;
  T2.noalias() = B11 + B12;
  Matrix M6 = strassen_impl(T1, T2, cutoff);

  T1.noalias() = A12 - A22;
  T2.noalias() = B21 + B22;
  Matrix M7 = strassen_impl(T1, T2, cutoff);

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
      return gemm(A, B);
    }

  using Matrix = DynamicMatrix<T>;

  if(A.rows() == A.cols() && B.rows() == B.cols() && A.rows() == B.rows())
    {
      return strassen_impl(A, B, cutoff);
    }

  const Eigen::Index paddedSize = even_extent(square_extent(A, B));

  Matrix paddedA = Matrix::Zero(paddedSize, paddedSize);
  Matrix paddedB = Matrix::Zero(paddedSize, paddedSize);

  paddedA.topLeftCorner(A.rows(), A.cols()) = A;
  paddedB.topLeftCorner(B.rows(), B.cols()) = B;

  Matrix paddedC = strassen_impl(paddedA, paddedB, cutoff);
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

  // Evaluate once up front so recursive levels work on contiguous owned data
  // instead of on generic Eigen expressions.
  const Matrix lhs = A.eval();
  const Matrix rhs = B.eval();

  return detail::strassen_with_padding(lhs, rhs, cutoff);
}
} // namespace apsc
