#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

/*!
 * \file strassen.hpp
 * \brief Hybrid Strassen/Eigen dense matrix multiplication.
 *
 * Design notes
 * ------------
 * This implementation differs from a textbook Strassen in three ways that
 * matter for performance on top of Eigen:
 *
 *  1. The recursive kernel uses an **out-parameter** signature, writing the
 *     product directly into a caller-supplied block of the destination. The
 *     seven Strassen sub-products are therefore stored straight into the four
 *     quadrants of C, with no per-product temporary matrix.
 *
 *  2. Per recursion level we keep only **two scratch buffers** (one for an
 *     A-side combination, one for a B-side combination) plus one extra buffer
 *     to hold the sub-product whose value must be combined with another
 *     quadrant. That is O(n^2) extra memory per level, vs. ~9 n^2 in the
 *     textbook version.
 *
 *  3. Odd square sub-problems are handled by **peeling** the last row/column
 *     instead of zero-padding to (n+1) x (n+1). Peeling does one Strassen call
 *     of size (n-1) plus three small GEMMs and one rank-1 update, which is
 *     considerably cheaper than re-padding at every odd level.
 */

namespace apsc
{
namespace detail
{
//! Convenience alias for a dynamic dense Eigen matrix.
template <typename T>
using DynamicMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

/*!
 * \brief Non-owning read-only view of a dense matrix or sub-block.
 *
 * Accepts both owned matrices and Block expressions without copying, while
 * keeping the recursive kernel's signature monomorphic in the scalar type.
 */
template <typename T>
using ConstMatrixView =
  Eigen::Ref<const DynamicMatrix<T>, 0, Eigen::OuterStride<Eigen::Dynamic>>;

/*!
 * \brief Non-owning writable view of a dense matrix or sub-block.
 */
template <typename T>
using MatrixView =
  Eigen::Ref<DynamicMatrix<T>, 0, Eigen::OuterStride<Eigen::Dynamic>>;

/*!
 * \brief Computes C = A * B with Eigen's optimized kernel.
 *
 * Writes directly into the caller's destination block: no allocation, no copy.
 */
template <typename T>
inline void
gemm_into(const ConstMatrixView<T> &A, const ConstMatrixView<T> &B,
         MatrixView<T> C)
{
  C.noalias() = A * B;
}

/*!
 * \brief Returns the largest dimension of the (possibly rectangular) product.
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
 * \brief Decides whether a top-level product is large enough for Strassen.
 */
template <typename MatrixA, typename MatrixB>
[[nodiscard]] inline bool
use_strassen(const MatrixA &A, const MatrixB &B, Eigen::Index cutoff)
{
  const Eigen::Index extent = square_extent(A, B);
  return A.cols() == B.rows() && extent > cutoff && extent > 1;
}

// Forward declaration: the recursive square kernel.
template <typename T>
void strassen_square(const ConstMatrixView<T> &A,
                     const ConstMatrixView<T> &B,
                     MatrixView<T> C, Eigen::Index cutoff);

/*!
 * \brief Handles odd-size square problems by peeling the last row and column.
 *
 * Decompose A and B as
 *
 *     A = [ A0  a ]    B = [ B0  b ]
 *         [ a^T α ]        [ c^T β ]
 *
 * with A0, B0 of size (n-1) x (n-1). Then
 *
 *     C0 = A0*B0 + a c^T            (Strassen on the even part + rank-1)
 *     C top-right    = A0*b + β a   (matvec)
 *     C bottom-left  = a^T*B0 + α c^T
 *     C bottom-right = a^T*b + α β
 *
 * This is much cheaper than allocating two (n+1) x (n+1) padded matrices.
 */
template <typename T>
inline void
strassen_peel_odd(const ConstMatrixView<T> &A, const ConstMatrixView<T> &B,
                  MatrixView<T> C, Eigen::Index cutoff)
{
  const Eigen::Index n = A.rows();
  const Eigen::Index m = n - 1;

  const auto A0 = A.topLeftCorner(m, m);
  const auto a_col = A.topRightCorner(m, 1);          // last column of A (top part)
  const auto a_row = A.bottomLeftCorner(1, m);        // last row of A (left part)
  const T    alpha = A(m, m);

  const auto B0 = B.topLeftCorner(m, m);
  const auto b_col = B.topRightCorner(m, 1);          // last column of B (top part)
  const auto c_row = B.bottomLeftCorner(1, m);        // last row of B (left part)
  const T    beta  = B(m, m);

  // Recurse on the even (n-1)-sized core, writing directly into C's top-left.
  auto C0 = C.topLeftCorner(m, m);
  strassen_square<T>(A0, B0, C0, cutoff);

  // Rank-1 correction: C0 += a_col * c_row.
  C0.noalias() += a_col * c_row;

  // Last column of C (top part):  A0 * b_col + beta * a_col.
  C.topRightCorner(m, 1).noalias() = A0 * b_col + beta * a_col;

  // Last row of C (left part):    a_row * B0 + alpha * c_row.
  C.bottomLeftCorner(1, m).noalias() = a_row * B0 + alpha * c_row;

  // Bottom-right scalar: a_row * b_col + alpha * beta  (1x1).
  C(m, m) = (a_row * b_col).value() + alpha * beta;
}

/*!
 * \brief Recursive Strassen kernel for square matrices, writing into C.
 *
 * Uses two scratch matrices (TA, TB) for A-side and B-side linear
 * combinations and one extra buffer (M) for sub-products that need to be
 * combined with another quadrant. The seven Strassen products are computed
 * in an order that minimises live temporaries.
 *
 * Strassen's identities (writing C = A * B in 2x2 block form):
 *
 *   M1 = (A11+A22)(B11+B22)
 *   M2 = (A21+A22) B11
 *   M3 = A11 (B12-B22)
 *   M4 = A22 (B21-B11)
 *   M5 = (A11+A12) B22
 *   M6 = (A21-A11)(B11+B12)
 *   M7 = (A12-A22)(B21+B22)
 *
 *   C11 = M1 + M4 - M5 + M7
 *   C12 = M3 + M5
 *   C21 = M2 + M4
 *   C22 = M1 - M2 + M3 + M6
 *
 * We schedule them so that each Mk is written either directly into a quadrant
 * of C (when only that quadrant needs it) or into the small buffer M.
 */
template <typename T>
void
strassen_square(const ConstMatrixView<T> &A, const ConstMatrixView<T> &B,
                MatrixView<T> C, Eigen::Index cutoff)
{
  const Eigen::Index n = A.rows();

  // Base case: hand off to Eigen's tuned GEMM, writing into the destination.
  if(n <= cutoff || n <= 1)
    {
      gemm_into<T>(A, B, C);
      return;
    }

  // Odd dimension: peel one row/column instead of repadding.
  if(n % 2 != 0)
    {
      strassen_peel_odd<T>(A, B, C, cutoff);
      return;
    }

  using Matrix = DynamicMatrix<T>;
  const Eigen::Index h = n / 2;

  const auto A11 = A.topLeftCorner(h, h);
  const auto A12 = A.topRightCorner(h, h);
  const auto A21 = A.bottomLeftCorner(h, h);
  const auto A22 = A.bottomRightCorner(h, h);

  const auto B11 = B.topLeftCorner(h, h);
  const auto B12 = B.topRightCorner(h, h);
  const auto B21 = B.bottomLeftCorner(h, h);
  const auto B22 = B.bottomRightCorner(h, h);

  auto C11 = C.topLeftCorner(h, h);
  auto C12 = C.topRightCorner(h, h);
  auto C21 = C.bottomLeftCorner(h, h);
  auto C22 = C.bottomRightCorner(h, h);

  // Two reusable scratch buffers for the A- and B-side sums; one buffer to
  // hold a Strassen product whose value is needed in two quadrants.
  Matrix TA(h, h), TB(h, h), M(h, h);

  // -- M1 = (A11+A22)(B11+B22). Used in C11 and C22, so store in M.
  TA.noalias() = A11 + A22;
  TB.noalias() = B11 + B22;
  strassen_square<T>(TA, TB, M, cutoff);
  C11 = M;        // accumulate later: C11 += M4 - M5 + M7
  C22 = M;        // accumulate later: C22 += -M2 + M3 + M6

  // -- M2 = (A21+A22) B11. Used in C21 (+) and C22 (-).
  TA.noalias() = A21 + A22;
  strassen_square<T>(TA, B11, M, cutoff);
  C21 = M;
  C22.noalias() -= M;

  // -- M3 = A11 (B12-B22). Used in C12 (+) and C22 (+).
  TB.noalias() = B12 - B22;
  strassen_square<T>(A11, TB, M, cutoff);
  C12 = M;
  C22.noalias() += M;

  // -- M4 = A22 (B21-B11). Used in C11 (+) and C21 (+).
  TB.noalias() = B21 - B11;
  strassen_square<T>(A22, TB, M, cutoff);
  C11.noalias() += M;
  C21.noalias() += M;

  // -- M5 = (A11+A12) B22. Used in C11 (-) and C12 (+).
  TA.noalias() = A11 + A12;
  strassen_square<T>(TA, B22, M, cutoff);
  C11.noalias() -= M;
  C12.noalias() += M;

  // -- M6 = (A21-A11)(B11+B12). Used in C22 (+) only -> write directly.
  TA.noalias() = A21 - A11;
  TB.noalias() = B11 + B12;
  // Need to accumulate into C22; reuse M as scratch then add.
  strassen_square<T>(TA, TB, M, cutoff);
  C22.noalias() += M;

  // -- M7 = (A12-A22)(B21+B22). Used in C11 (+) only -> write directly.
  TA.noalias() = A12 - A22;
  TB.noalias() = B21 + B22;
  strassen_square<T>(TA, TB, M, cutoff);
  C11.noalias() += M;
}

/*!
 * \brief Top-level dispatcher. Handles rectangular shapes by single-shot
 *        zero-padding to a square even size, then calls the recursive kernel.
 *
 * Padding happens at most once (here), never inside recursion.
 */
template <typename T>
[[nodiscard]] DynamicMatrix<T>
strassen_dispatch(const DynamicMatrix<T> &A, const DynamicMatrix<T> &B,
                  Eigen::Index cutoff)
{
  using Matrix = DynamicMatrix<T>;

  if(!use_strassen(A, B, cutoff))
    {
      Matrix C(A.rows(), B.cols());
      C.noalias() = A * B;
      return C;
    }

  // Already square, equal-sized: call the kernel directly, no padding.
  if(A.rows() == A.cols() && B.rows() == B.cols() && A.rows() == B.rows())
    {
      Matrix C(A.rows(), B.cols());
      strassen_square<T>(A, B, C, cutoff);
      return C;
    }

  // Rectangular: pad once to a common even square size.
  const Eigen::Index paddedSize = even_extent(square_extent(A, B));

  Matrix paddedA = Matrix::Zero(paddedSize, paddedSize);
  Matrix paddedB = Matrix::Zero(paddedSize, paddedSize);
  Matrix paddedC(paddedSize, paddedSize);

  paddedA.topLeftCorner(A.rows(), A.cols()) = A;
  paddedB.topLeftCorner(B.rows(), B.cols()) = B;

  strassen_square<T>(paddedA, paddedB, paddedC, cutoff);
  return paddedC.topLeftCorner(A.rows(), B.cols());
}
} // namespace detail

/*!
 * \brief Multiplies two dense Eigen matrices with a hybrid Strassen strategy.
 *
 * \tparam DerivedA Eigen expression type for the left operand.
 * \tparam DerivedB Eigen expression type for the right operand.
 * \param A Left matrix (any compatible Eigen expression).
 * \param B Right matrix (any compatible Eigen expression).
 * \param cutoff Recursion threshold. At or below this size, Eigen's direct
 *               product is used. Defaults to 1024, which is roughly the
 *               break-even with Eigen's tuned GEMM on modern CPUs for double.
 * \return The matrix product A * B.
 *
 * \throw std::invalid_argument if matrix dimensions are incompatible.
 */
template <
  typename DerivedA, typename DerivedB,
  typename Scalar =
    std::common_type_t<typename DerivedA::Scalar, typename DerivedB::Scalar>>
[[nodiscard]] inline detail::DynamicMatrix<Scalar>
strassen(const Eigen::MatrixBase<DerivedA> &A,
         const Eigen::MatrixBase<DerivedB> &B, Eigen::Index cutoff = 1024)
{
  static_assert(std::is_same_v<typename DerivedA::Scalar, Scalar> &&
                  std::is_same_v<typename DerivedB::Scalar, Scalar>,
                "strassen requires both operands to have the same scalar type");

  if(A.cols() != B.rows())
    {
      throw std::invalid_argument("strassen: incompatible matrix dimensions");
    }

  using Matrix = detail::DynamicMatrix<Scalar>;

  // Empty product: return correctly-shaped zero matrix.
  if(A.rows() == 0 || B.cols() == 0 || A.cols() == 0)
    {
      return Matrix::Zero(A.rows(), B.cols());
    }

  // Evaluate user expressions exactly once.
  const Matrix lhs = A.eval();
  const Matrix rhs = B.eval();

  return detail::strassen_dispatch<Scalar>(lhs, rhs, cutoff);
}
} // namespace apsc
