#pragma once

// A compressed sparse matrix with device-resident storage and host-side assembly.
// No Eigen dependency: the template order and storage-option values match Eigen.
#include <thrust/binary_search.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <thrust/execution_policy.h>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <concepts>

// This is needed to have the cpu counterparts of the device algorithms available for testing.
#if defined(__CUDACC__)
#  define APSC_SPARSE_HD __host__ __device__
#else
#  define APSC_SPARSE_HD
#endif

namespace apsc::gpu {
  // I use constant expressions instead of a enumerator,
inline constexpr int ColMajor = 0;
inline constexpr int RowMajor = 1;

/*!
@brief A triplet is a single nonzero entry in a sparse matrix, represented by its row and column indices and its value.
@tparam Scalar_ The type of the stored value.
@tparam StorageIndex_ The type of the row and column indices. It must be an unsigned integral type.
*/
template<class Scalar_, std::unsigned_integral StorageIndex_ = std::size_t>
class Triplet {
public:
  using Scalar = Scalar_;
  using StorageIndex = StorageIndex_;
  static_assert(!std::is_same_v<StorageIndex, bool>, "StorageIndex must not be bool");
  //!@brief Construct a new Triplet object.
  APSC_SPARSE_HD Triplet() {}
  //!@brief Construct a new Triplet object with the given row, column, and value.
  APSC_SPARSE_HD Triplet(StorageIndex row, StorageIndex col, Scalar value)
      : row_(row), col_(col), value_(value) {}
  //!@brief Set the row, column, and value of this triplet.
  APSC_SPARSE_HD void set(StorageIndex row, StorageIndex col, Scalar value) {
    row_ = row; col_ = col; value_ = value;
  }
  APSC_SPARSE_HD StorageIndex row() const { return row_; }
  APSC_SPARSE_HD StorageIndex col() const { return col_; }
  APSC_SPARSE_HD Scalar value() const { return value_; }
private:
  StorageIndex row_ = 0, col_ = 0;
  Scalar value_{};
};

// In this subnamespace we have implementation details that are not part of the public API.
namespace detail {
//! traits for scalar types used in sparse matrices. 
// The real type is the same as the scalar type for floating-point scalars, 
// and is the underlying floating-point type for complex scalars.
template<class T> struct ScalarTraits {
  static constexpr bool supported = std::is_floating_point_v<T>;
  using Real = T;
  APSC_SPARSE_HD static Real abs(T x) { return x < T(0) ? -x : x; }
};
//! Specialization for thrust::complex<T> where T is a floating-point type.
template<class T> struct ScalarTraits<thrust::complex<T>> {
  static constexpr bool supported = std::is_floating_point_v<T>;
  using Real = T;
  APSC_SPARSE_HD static Real abs(thrust::complex<T> x) { return thrust::abs(x); }
};
//! Functor to compute the absolute value of a scalar, using ScalarTraits.
template<class T> struct Absolute {
  APSC_SPARSE_HD auto operator()(T x) const { return ScalarTraits<T>::abs(x); }
};
//! Functor to compute the square of a scalar, using ScalarTraits.
template<class T> struct Square {
  APSC_SPARSE_HD auto operator()(T x) const { return x * x; }
};
//! Functor to compute the scaled square of a scalar, using ScalarTraits.
template<class T> struct ScaledSquare {
  using R = typename ScalarTraits<T>::Real;
  R scale;
  APSC_SPARSE_HD R operator()(T x) const {
    const R a = ScalarTraits<T>::abs(x) / scale;
    return a * a;
  }
};
//! Unlike an ordinary comparison-based maximum, retain NaNs in norm reductions.
template<class R> struct MaximumMagnitude {
  APSC_SPARSE_HD R operator()(R a, R b) const {
    if (a != a) return a;
    if (b != b) return b;
    return a < b ? b : a;
  }
};
//! Functor to scale a scalar by a given factor, using ScalarTraits.
template<class T> struct Scale {
  T factor;
  APSC_SPARSE_HD T operator()(T x) const { return x * factor; }
};

/*!
 * \brief A read-only, trivially copyable device handle.
 * \details It never owns device memory.
 * \tparam S The scalar type.
 * \tparam I The index type.
 * \tparam RowWise Whether the matrix is stored row-wise.
 * \note Binary searches are possible because each compressed segment is sorted.
 */
template<class S, std::unsigned_integral I, bool RowWise>
struct CompressedView {
  const I* outer;
  const I* inner;
  const S* values;
  std::size_t rows, cols;
  APSC_SPARSE_HD S coeff(std::size_t row, std::size_t col) const {
    const auto o = RowWise ? row : col;
    const I target = static_cast<I>(RowWise ? col : row);
    I lo = outer[o], hi = outer[o + 1];
    const I end = hi;
    while (lo < hi) {
      const I mid = lo + (hi - lo) / 2;
      if (inner[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return lo < end && inner[lo] == target ? values[lo] : S{};
  }
  APSC_SPARSE_HD S operator()(std::size_t r, std::size_t c) const {
    return coeff(r, c);
  }
};
/*!
 * \brief A functor to compute the product of a segment of a sparse matrix with a vector.
  * \tparam S The scalar type.
  * \tparam I The index type.
  * \note This functor is used in the multiplyImpl function of the SparseMatrix class
 */
template<class S, std::unsigned_integral I> struct SegmentProduct {
  const I* outer;
  const I* inner;
  const S* values;
  const S* x;
  APSC_SPARSE_HD S operator()(std::size_t o) const {
    S result{};
    for (I p = outer[o]; p < outer[o + 1]; ++p) result += values[p] * x[inner[p]];
    return result;
  }
};

// Find the outer segment containing a stored entry, including empty segments.
/*!
 * \brief Find the outer segment containing a stored entry.
 * \tparam I The index type.
 * \param outer The outer array.
 * \param count The number of outer segments (the outer array has count + 1 elements).
 * \param entry The entry to search for.
 * \return The index of the outer segment containing the entry.
 */
template<std::unsigned_integral I>
APSC_SPARSE_HD I segmentOf(const I* outer, std::size_t count, I entry) {
  std::size_t lo = 0u, hi = count + 1u;
  while (lo < hi) {
    const auto mid = lo + (hi - lo) / 2;
    if (outer[mid] <= entry) lo = mid + 1u;
    else hi = mid;
  }
  return static_cast<I>(lo - 1u);
}
/*!
 * \brief A functor to compute the product of a single entry in a sparse matrix with a vector.
 * \tparam S The scalar type.
 * \tparam I The index type.
 * \note This functor is used in the multiplyImpl function of the SparseMatrix class.
 */
template<class S, std::unsigned_integral I> struct EntryProduct {
  const I* outer;
  const S* values;
  const S* x;
  std::size_t count;
  APSC_SPARSE_HD S operator()(std::size_t p) const {
    return values[p] * x[segmentOf(outer, count, static_cast<I>(p))];
  }
};

/*!
 * \brief A functor to compute the absolute sum of a segment of a sparse matrix.
 * \tparam S The scalar type.
 * \tparam I The index type.
 * \note This functor is used in the multiplyImpl function of the SparseMatrix class.
 */
template<class S, std::unsigned_integral I> struct SegmentAbsoluteSum {
  const I* outer;
  const S* values;
  using R = typename ScalarTraits<S>::Real;
  /*!
   * \brief Compute the absolute sum of a segment of the sparse matrix.
   * \param o The index of the segment.
   * \return The absolute sum of the segment.
   */
  APSC_SPARSE_HD R operator()(std::size_t o) const {
    R result{};
    for (I p = outer[o]; p < outer[o + 1u]; ++p) result += ScalarTraits<S>::abs(values[p]);
    return result;
  }
};

/*!
 * \brief A functor to compute the diagonal entry of a sparse matrix.
 * \tparam View The view type.
 * \note This functor is used in the multiplyImpl function of the SparseMatrix class.
 */
template<class View> struct DiagonalEntry {
  View view;
  /*! \brief Compute the diagonal entry of the sparse matrix.
   * \param i The index of the diagonal entry.
   * \return The diagonal entry of the sparse matrix.
   */
  APSC_SPARSE_HD auto operator()(std::size_t i) const { return view.coeff(i, i); }
};
} // namespace detail

// Two forward declarations to support diagonal() and transpose() member functions.
template<class Matrix> class DiagonalView;
template<class Matrix> class TransposeView;

/*! \brief Always-compressed sparse matrix: CSC (ColMajor) or CSR (RowMajor).
 * \details All three buffers are thrust::device_vector. With the CUDA backend they live
 * on the GPU. CPU Thrust backends are useful for teaching and testing.
 * Scalar is a floating-point type or thrust::complex<floating-point>.
 * Host member calls are not safe to invoke concurrently with matrix mutation.
 * \addtogroup SparseMatrix
 * \tparam Scalar_ The type of the stored value.
 * \tparam Options_ The storage order: ColMajor (0) or RowMajor (1).
 * \tparam StorageIndex_ The type of the row and column indices. It must be an unsigned integral type.
 */
template<class Scalar_, int Options_ = ColMajor, std::unsigned_integral StorageIndex_ = std::size_t>
class SparseMatrix {
public:
  using Scalar = Scalar_;
  using StorageIndex = StorageIndex_;
  using Index = std::size_t;
  using RealScalar = typename detail::ScalarTraits<Scalar>::Real;
  using Vector = thrust::device_vector<Scalar>;
  using IndexVector = thrust::device_vector<StorageIndex>;
  using TripletType = Triplet<Scalar, StorageIndex>;
  static constexpr int Options = Options_;
  static constexpr bool IsRowMajor = (Options == RowMajor);
  using DeviceView = detail::CompressedView<Scalar, StorageIndex, IsRowMajor>;
  static_assert(Options == ColMajor || Options == RowMajor, "Use ColMajor (0) or RowMajor (1)");
  static_assert(!std::is_same_v<StorageIndex, bool>, "StorageIndex must not be bool");
  static_assert(detail::ScalarTraits<Scalar>::supported,
                "Use a real floating-point scalar or thrust::complex of one");

  SparseMatrix() : outer_(1, StorageIndex(0)) {}
  SparseMatrix(Index rows, Index cols) : SparseMatrix() { resize(rows, cols); }
  SparseMatrix(const SparseMatrix&) = default; //!< device-to-device deep copy
  SparseMatrix& operator=(const SparseMatrix&) = default; //!< device-to-device deep copy assignment
  SparseMatrix(SparseMatrix&& other) : SparseMatrix() { swap(other); } //!< device-to-device move
  //! Move the buffers, leaving the source empty. Self-move preserves the matrix.
  SparseMatrix& operator=(SparseMatrix&& other) {
    if (this != &other) {
      SparseMatrix moved(std::move(other));
      swap(moved);
    }
    return *this;
  }

  Index rows() const noexcept { return rows_; }//!< The number of rows in the matrix.
  Index cols() const noexcept { return cols_; }//!< The number of columns in the matrix.
  Index outerSize() const noexcept { return IsRowMajor ? rows_ : cols_; }//!< The number of outer segments (rows for RowMajor, columns for ColMajor).
  Index innerSize() const noexcept { return IsRowMajor ? cols_ : rows_; }//!< The number of inner segments (columns for RowMajor, rows for ColMajor).
  Index nonZeros() const noexcept { return static_cast<Index>(values_.size()); }//!< The number of non-zero entries in the matrix.
  constexpr bool isCompressed() const noexcept { return true; }//!< The matrix is always stored in compressed format.
  void makeCompressed() noexcept {} // invariant, not a deferred operation


  /*!
    * \brief Resize the matrix to the given number of rows and columns.
   * \param rows The number of rows in the matrix.
   * \param cols The number of columns in the matrix.
   * \note Resizing discards coefficients. Dimensions exceeding storage/iterator limits are errors.
   */
  void resize(Index rows, Index cols) {
    checkDimension(rows); checkDimension(cols);
    const Index outerSize = IsRowMajor ? rows : cols;
    // The offsets include an extra end sentinel; its length must also fit iterator differences.
    if (outerSize == static_cast<Index>(std::numeric_limits<std::ptrdiff_t>::max()))
      throw std::length_error("Outer dimension leaves no room for the end offset");
    IndexVector outer(outerSize + 1, StorageIndex(0));
    values_.clear(); inner_.clear(); outer_.swap(outer);
    rows_ = rows; cols_ = cols;
  }
  /*!
    * \brief Set all coefficients to zero.
    * \note This operation is in-place and does not change the matrix dimensions.
    */
  void setZero() {
    IndexVector outer(static_cast<std::size_t>(outerSize()) + 1, StorageIndex(0));
    values_.clear(); inner_.clear(); outer_.swap(outer);
  }
  /*! \brief Reserve space for the given number of non-zero entries.
   * \param count The number of non-zero entries to reserve space for.
   * \note Reserving space does not change the matrix dimensions or the number of non-zero entries.
   */
  void reserve(Index count) {
    checkDimension(count);
    values_.reserve(static_cast<std::size_t>(count));
    inner_.reserve(static_cast<std::size_t>(count));
  }
  /*! \brief Swap the contents of this matrix with another matrix.
   * \param other The matrix to swap with.
   */
  void swap(SparseMatrix& other) noexcept {
    std::swap(rows_, other.rows_); std::swap(cols_, other.cols_);
    outer_.swap(other.outer_); inner_.swap(other.inner_); values_.swap(other.values_);
  }

  /*!
   * \brief Assemble from iterators over objects with row(), col(), value() (including Eigen::Triplet).
   * \details Duplicates are summed; exact zeros are dropped. Assembly
   * sorts on the host and uploads once. Validation precedes matrix replacement.
   * \note Prefer this bulk operation over repeated single-coefficient insertion.
   * \tparam Iterator An input iterator type whose value_type has row(), col(), and value() methods.
   * \param first The beginning of the range of triplet-like objects.
   * \param last The end of the range of triplet-like objects.
   */
  template<class Iterator>
  void setFromTriplets(Iterator first, Iterator last) {
    std::vector<TripletType> entries;
    const auto outOfRange = [](auto coordinate, Index extent) {
      using Coordinate = decltype(coordinate);
      if constexpr (std::is_signed_v<Coordinate>) {
        if (coordinate < 0) return true;
      }
      return static_cast<std::make_unsigned_t<Coordinate>>(coordinate) >= extent;
    };
    for (; first != last; ++first) {
      const typename std::iterator_traits<Iterator>::value_type t = *first;
      // Check before narrowing a wider input index to StorageIndex.
      if (outOfRange(t.row(), rows_) || outOfRange(t.col(), cols_))
        throw std::out_of_range("Triplet coordinate outside matrix");
      entries.emplace_back(static_cast<StorageIndex>(t.row()), static_cast<StorageIndex>(t.col()),
                           static_cast<Scalar>(t.value()));
    }
    // Sort by outer index, then inner index. Stable sort preserves the order of duplicates for summation.
    auto outer = [](const TripletType& t) { return IsRowMajor ? t.row() : t.col(); };
    auto inner = [](const TripletType& t) { return IsRowMajor ? t.col() : t.row(); };
    std::stable_sort(entries.begin(), entries.end(), [&](const auto& a, const auto& b) {
      return outer(a) != outer(b) ? outer(a) < outer(b) : inner(a) < inner(b);
    });
    std::vector<StorageIndex> offsets(static_cast<std::size_t>(outerSize()) + 1, 0), indices;
    std::vector<Scalar> values;
    // Sum duplicates and drop exact zeros. Count outer segments for offsets.
    for (std::size_t p = 0; p < entries.size();) {
      const auto t = entries[p++];
      Scalar value = t.value();
      while (p < entries.size() && outer(entries[p]) == outer(t) && inner(entries[p]) == inner(t))
        value += entries[p++].value();
      if (value == Scalar{}) continue;
      if (values.size() >= maxStoredCount()) throw std::length_error("Too many nonzeros for index type");
      ++offsets[static_cast<std::size_t>(outer(t)) + 1];
      indices.push_back(inner(t)); values.push_back(value);
    }
    // Convert offsets to cumulative sum. The first entry is always zero.
    for (std::size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
    IndexVector newOuter(offsets.begin(), offsets.end()), newInner(indices.begin(), indices.end());
    Vector newValues(values.begin(), values.end());
    outer_.swap(newOuter); inner_.swap(newInner); values_.swap(newValues);
  }

  /*! \brief Export for inspection/assembly. This explicitly transfers all buffers to CPU.
    * \return A std::vector of TripletType objects representing the non-zero entries of the matrix.
    * \note This operation is not efficient for large matrices, as it involves transferring data from device to host.
    */ 
  std::vector<TripletType> toHostTriplets() const {
    thrust::host_vector<StorageIndex> outer = outer_, inner = inner_;
    thrust::host_vector<Scalar> values = values_;
    std::vector<TripletType> result;
    result.reserve(values.size());
    for (Index o = 0; o < outerSize(); ++o)
      for (StorageIndex p = outer[o]; p < outer[o + 1]; ++p)
        result.emplace_back(IsRowMajor ? static_cast<StorageIndex>(o) : inner[p],
                            IsRowMajor ? inner[p] : static_cast<StorageIndex>(o), values[p]);
    return result;
  }

  /*! \brief Get the value of a coefficient.
    * \details Host access returns a scalar copy, synchronizing/transferring from the GPU.
    * \param row The row index of the coefficient.
    * \param col The column index of the coefficient.
    * \return The value of the coefficient at the specified position.
    */
  Scalar coeff(Index row, Index col) const {
    const Index p = findPosition(row, col);
    return p == nonZeros() ? Scalar{} : Scalar(values_[p]);
  }


 /*! \brief Access a coefficient by its row and column indices.
    * \details This is a convenience function that calls coeff() internally.
    * \param row The row index of the coefficient.
    * \param col The column index of the coefficient.
    * \return The value of the coefficient at the specified position.
    */
  Scalar operator()(Index row, Index col) const { return coeff(row, col); }

  /*!
   * \brief Internal class to provide a proxy for accessing and modifying coefficients in the sparse matrix.
   * \details This class allows for convenient access to coefficients in the sparse matrix, enabling both
   * reading and writing of values. It is used in conjunction with the coeffRef() function
   * to provide a reference-like interface for coefficients.
   */
  class CoefficientProxy {
  public:
    operator Scalar() const { return matrix_->coeff(row_, col_); }
    CoefficientProxy& operator=(Scalar value) { matrix_->setCoeff(row_, col_, value); return *this; }
    CoefficientProxy& operator=(const CoefficientProxy& value) { return *this = Scalar(value); }
    CoefficientProxy& operator+=(Scalar v) { return *this = Scalar(*this) + v; }
    CoefficientProxy& operator-=(Scalar v) { return *this = Scalar(*this) - v; }
    CoefficientProxy& operator*=(Scalar v) { return *this = Scalar(*this) * v; }
    CoefficientProxy& operator/=(Scalar v) { return *this = Scalar(*this) / v; }
  private:
    friend class SparseMatrix;
    CoefficientProxy(SparseMatrix& m, Index r, Index c) : matrix_(&m), row_(r), col_(c) {}
    SparseMatrix* matrix_;
    Index row_, col_;
  };
  /*!
   * \brief Get a reference to a coefficient.
   * \details This function returns a proxy object that allows for convenient access to a coefficient in the sparse matrix.
   * \param row The row index of the coefficient.
   * \param col The column index of the coefficient.
   * \return A proxy object that can be used to access and modify the coefficient.
   */
  CoefficientProxy coeffRef(Index row, Index col) {
    checkCoordinate(row, col); return CoefficientProxy(*this, row, col);
  }
  /*!
   * \brief Access a coefficient by its row and column indices.
   * \details This is a convenience function that calls coeffRef() internally.
   * \param row The row index of the coefficient.
   * \param col The column index of the coefficient.
   * \return A proxy object that can be used to access and modify the coefficient.
   */
  CoefficientProxy operator()(Index row, Index col) { return coeffRef(row, col); }

  /*!
  \brief Existing entries update in place. 
  \details Inserting a new entry rebuilds on CPU. Assigning zero to an existing entry preserves its slot until prune().
  \param row The row index of the coefficient.
  \param col The column index of the coefficient.
  \param value The new value to set for the coefficient.
  */
  void setCoeff(Index row, Index col, Scalar value) {
    const Index p = findPosition(row, col);
    if (p != nonZeros()) { values_[p] = value; return; }
    if (value == Scalar{}) return;
    auto entries = toHostTriplets();
    entries.emplace_back(static_cast<StorageIndex>(row), static_cast<StorageIndex>(col), value);
    setFromTriplets(entries.begin(), entries.end());
  }
  /*!
  \brief Remove |a_ij| <= tolerance. Structural editing is a host-side operation.
  \param tolerance The threshold below which entries are considered zero and removed.
  \details This function modifies the sparse matrix in place, removing entries whose absolute value is less
  than or equal to the specified tolerance. It is a host-side operation that rebuilds the matrix structure.
  */
  void prune(RealScalar tolerance = RealScalar(0)) {
    if (!(tolerance >= RealScalar(0))) throw std::invalid_argument("Invalid pruning tolerance");
    auto entries = toHostTriplets();
    entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const auto& t) {
      return detail::ScalarTraits<Scalar>::abs(t.value()) <= tolerance;
    }), entries.end());
    setFromTriplets(entries.begin(), entries.end());
  }
  /*!
  \brief Set the matrix to the identity matrix.
  \details This function sets the sparse matrix to the identity matrix by setting all diagonal entries to 1 and all off-diagonal entries to 0.
  */
  void setIdentity() {
    std::vector<TripletType> entries;
    for (Index i = 0; i < std::min(rows_, cols_); ++i)
      entries.emplace_back(static_cast<StorageIndex>(i), static_cast<StorageIndex>(i), Scalar(1));
    setFromTriplets(entries.begin(), entries.end());
  }

  /// Read-only structural access protects compressed-format invariants.
  /*!
  \brief Get the outer indices of the sparse matrix.
  \return A const reference to the outer indices vector.
  */
  const IndexVector& outerIndices() const noexcept { return outer_; }
  ///! \brief Get the inner indices of the sparse matrix.
  const IndexVector& innerIndices() const noexcept { return inner_; }
  ///! \brief Get the values of the sparse matrix.
  const Vector& values() const noexcept { return values_; }
  ///! \brief Get a pointer to the outer indices of the sparse matrix.
  const StorageIndex* outerIndexPtr() const { return thrust::raw_pointer_cast(outer_.data()); }
  ///! \brief Get a pointer to the inner indices of the sparse matrix.
  const StorageIndex* innerIndexPtr() const { return thrust::raw_pointer_cast(inner_.data()); }
  ///! \brief Get a pointer to the values of the sparse matrix.
  const Scalar* valuePtr() const { return thrust::raw_pointer_cast(values_.data()); }
  ///! \brief Get a writable pointer to the values of the sparse matrix.
  Scalar* valuePtr() { return thrust::raw_pointer_cast(values_.data()); }

  /*!
  \brief Get a device view of the sparse matrix.
  \details This function returns a device view of the sparse matrix, which can be used in
  device functors for efficient computations on the GPU. 
  The device view provides access to the outer indices, inner indices, and values of the sparse matrix.
  */
  DeviceView deviceView() const {
    return {outerIndexPtr(), innerIndexPtr(), valuePtr(), rows_, cols_};
  }
/*!
  \brief Multiply the sparse matrix with a vector.
  \details This function multiplies the sparse matrix with a vector and returns the result.
  */
  template<class Allocator>
  Vector multiply(const thrust::device_vector<Scalar, Allocator>& x) const { return multiplyImpl(x, false); }
  
  /*!
  \brief Multiply the sparse matrix with a vector using the * operator.
  \details This function multiplies the sparse matrix with a vector and returns the result.
  */
  template<class Allocator>
  Vector operator*(const thrust::device_vector<Scalar, Allocator>& x) const { return multiply(x); }
  
  /*!
  \brief Multiply the transpose of the sparse matrix with a vector.
  \details This function multiplies the transpose of the sparse matrix with a vector and returns the result.
  */
  template<class Allocator>
  Vector transposeMultiply(const thrust::device_vector<Scalar, Allocator>& x) const { return multiplyImpl(x, true); }
  /*!
  \brief Multiply the sparse matrix with a vector.
  \details This function multiplies the sparse matrix with a vector and returns the result.
  */
  void multiply(const Vector& x, Vector& y) const { Vector result = multiply(x); y.swap(result); }
  /*!
  \brief Compute the sum of all entries in the sparse matrix.
  \details This function computes the sum of all entries in the sparse matrix and returns the result.
  */
  Scalar sum() const { return thrust::reduce(values_.begin(), values_.end(), Scalar{}, thrust::plus<Scalar>()); }
  
  /*!
  \brief Scale the sparse matrix by a factor.
  \details This function scales the sparse matrix by a factor and returns a reference to the modified matrix.
  */
  SparseMatrix& operator*=(Scalar factor) {
    thrust::transform(values_.begin(), values_.end(), values_.begin(), detail::Scale<Scalar>{factor});
    return *this;
  }  
  /*!
  \brief Compute the maximum absolute value of any entry in the sparse matrix.
  \details This function computes the maximum absolute value of any entry in the sparse matrix and returns the result.
  */
  RealScalar maxNorm() const {
    return thrust::transform_reduce(values_.begin(), values_.end(), detail::Absolute<Scalar>{},
                                    RealScalar(0), detail::MaximumMagnitude<RealScalar>{});
  }
  /*!
  \brief Compute the Frobenius norm of the sparse matrix.
  \details This function computes the Frobenius norm of the sparse matrix and returns the result.
  \note Scaled sum of squares avoids overflow when squaring large finite entries.
  */
  RealScalar frobeniusNorm() const {
    const RealScalar scale = maxNorm();
    if (scale == RealScalar(0) || !std::isfinite(scale)) return scale;
    const RealScalar squares = thrust::transform_reduce(values_.begin(), values_.end(),
      detail::ScaledSquare<Scalar>{scale}, RealScalar(0), thrust::plus<RealScalar>());
    return scale * std::sqrt(squares);
  }
  RealScalar norm() const { return frobeniusNorm(); }
  /*!
  \brief Compute the entrywise L1 norm of the sparse matrix.
  \details This function computes the entrywise L1 norm of the sparse matrix and returns the result.
  */
  RealScalar entrywiseL1Norm() const {
    return thrust::transform_reduce(values_.begin(), values_.end(), detail::Absolute<Scalar>{},
                                    RealScalar(0), thrust::plus<RealScalar>());
  }
  /*!
  \brief Compute the L1 norm of the sparse matrix.
  \details This function computes the L1 norm of the sparse matrix and returns the result.
  \note This is the maximum absolute sum of the columns.
  */
  RealScalar l1Norm() const { return maximumAbsoluteSum(false); } // max column sum
  
  /*!
  \brief Compute the infinity norm of the sparse matrix.
  \details This function computes the infinity norm of the sparse matrix and returns the result.
  \note This is the maximum absolute sum of the rows.
  */
  RealScalar infinityNorm() const { return maximumAbsoluteSum(true); } // max row sum

  /*!
  \brief Get the diagonal of the sparse matrix.
  \details This function returns the diagonal of the sparse matrix.
  */
  DiagonalView<SparseMatrix> diagonal() & { return DiagonalView<SparseMatrix>(*this); }
  /*!
  \brief Get the diagonal of the sparse matrix (const version).
  \details This function returns the diagonal of the sparse matrix.
  \note This is a const version of the diagonal function.
  */
  DiagonalView<const SparseMatrix> diagonal() const & { return DiagonalView<const SparseMatrix>(*this); }
  /*!
  \brief Get the diagonal of the sparse matrix (rvalue version).
  \details This function returns the diagonal of the sparse matrix.
  \note This is an rvalue version of the diagonal function.
  */
  DiagonalView<SparseMatrix> diagonal() && = delete;
  
  /*!
  \brief Get the diagonal of the sparse matrix (const rvalue version).
  \details This function returns the diagonal of the sparse matrix.
  \note This is a const rvalue version of the diagonal function.
  */
  DiagonalView<const SparseMatrix> diagonal() const && = delete;
  
  /*!
  \brief Get the transpose of the sparse matrix.
  \details This function returns the transpose of the sparse matrix.
  */
  TransposeView<SparseMatrix> transpose() & { return TransposeView<SparseMatrix>(*this); }
  /*!
  \brief Get the transpose of the sparse matrix (const version).
  \details This function returns the transpose of the sparse matrix.
  \note This is a const version of the transpose function.
  */
  TransposeView<const SparseMatrix> transpose() const & { return TransposeView<const SparseMatrix>(*this); }
  /*!
  \brief Get the transpose of the sparse matrix (rvalue version).
  \details This function returns the transpose of the sparse matrix.
  \note This is an rvalue version of the transpose function.
  */
  TransposeView<SparseMatrix> transpose() && = delete;
  /*!
  \brief Get the transpose of the sparse matrix (const rvalue version).
  \details This function returns the transpose of the sparse matrix.
  \note This is a const rvalue version of the transpose function.
  */
  TransposeView<const SparseMatrix> transpose() const && = delete;

private:
  template<class> friend class TransposeView;
  template<class, int, std::unsigned_integral> friend class SparseMatrix;
  Index rows_ = 0, cols_ = 0;
  IndexVector outer_, inner_;
  Vector values_;
//! \brief Get the maximum number of non-zero entries that can be stored in the sparse matrix.
  static std::size_t maxStoredCount() {
    // Coordinates/counts are unsigned, but Thrust/STL iterator differences are signed.
    return static_cast<std::size_t>(std::min<std::uintmax_t>(
      static_cast<std::uintmax_t>(std::numeric_limits<StorageIndex>::max()),
      static_cast<std::uintmax_t>(std::numeric_limits<std::ptrdiff_t>::max())));
  }
  //! \brief Check if the given dimension is valid for the sparse matrix.
  static void checkDimension(Index n) {
    if (n > maxStoredCount())
      throw std::length_error("Dimension/count exceeds index type");
  }
  //! \brief Check if the given coordinate is within the bounds of the sparse matrix.
  void checkCoordinate(Index row, Index col) const {
    if (row >= rows_ || col >= cols_)
      throw std::out_of_range("Sparse matrix coordinate out of range");
  }
  //! \brief Find the position of a coefficient, or return nonZeros() if absent.
  Index findPosition(Index row, Index col) const {
    checkCoordinate(row, col);
    const Index o = IsRowMajor ? row : col;
    const auto target = static_cast<StorageIndex>(IsRowMajor ? col : row);
    const StorageIndex start = outer_[o], end = outer_[o + 1];
    if (start == end) return nonZeros();
    const auto it = thrust::lower_bound(inner_.begin() + start, inner_.begin() + end, target);
    if (it == inner_.begin() + end || StorageIndex(*it) != target) return nonZeros();
    return static_cast<Index>(it - inner_.begin());
  }

  //! \brief Compute the sum of the elements in each segment of the sparse matrix.
  //! \details This function computes the sum of the elements in each segment of the sparse matrix.
  //! \note Sorting and segmented reduction implement scatter-style products without
  //! floating-point atomics, including complex scalars and arbitrary index sizes.
  template<class T>
  thrust::device_vector<T> scatterSum(thrust::device_vector<T> contributions, Index size) const {
    thrust::device_vector<T> result(static_cast<std::size_t>(size), T{});
    if (values_.empty()) return result;
    IndexVector keys = inner_, unique(keys.size());
    thrust::device_vector<T> sums(keys.size());
    thrust::stable_sort_by_key(keys.begin(), keys.end(), contributions.begin());
    const auto ends = thrust::reduce_by_key(keys.begin(), keys.end(), contributions.begin(), unique.begin(), sums.begin());
    thrust::scatter(sums.begin(), ends.second, unique.begin(), result.begin());
    return result;
  }
  //! \brief Multiply the sparse matrix with a vector.
  //! \details This function multiplies the sparse matrix with a vector.
  template<class Allocator>
  Vector multiplyImpl(const thrust::device_vector<Scalar, Allocator>& x, bool transposed) const {
    const Index input = transposed ? rows_ : cols_, output = transposed ? cols_ : rows_;
    if (x.size() != static_cast<std::size_t>(input)) throw std::invalid_argument("Matrix/vector dimension mismatch");
    if (values_.empty()) return Vector(static_cast<std::size_t>(output), Scalar{});
    const auto* xp = thrust::raw_pointer_cast(x.data());
    if (IsRowMajor != transposed) {
      Vector result(static_cast<std::size_t>(output));
      auto first = thrust::make_counting_iterator<Index>(0);
      thrust::transform(first, first + output, result.begin(),
        detail::SegmentProduct<Scalar, StorageIndex>{outerIndexPtr(), innerIndexPtr(), valuePtr(), xp});
      return result;
    }
    Vector contributions(values_.size());
    auto first = thrust::make_counting_iterator<Index>(0);
    thrust::transform(first, first + nonZeros(), contributions.begin(),
      detail::EntryProduct<Scalar, StorageIndex>{outerIndexPtr(), valuePtr(), xp, outerSize()});
    return scatterSum(std::move(contributions), output);
  }
  //! \brief Compute the maximum absolute sum of the elements in the sparse matrix.
  //! \details This function computes the maximum absolute sum of the elements in the sparse matrix.
  RealScalar maximumAbsoluteSum(bool rows) const {
    if (values_.empty()) return RealScalar(0);
    if (rows == IsRowMajor) {
      auto first = thrust::make_counting_iterator<Index>(0);
      return thrust::transform_reduce(thrust::device, first, first + outerSize(),
        detail::SegmentAbsoluteSum<Scalar, StorageIndex>{outerIndexPtr(), valuePtr()},
        RealScalar(0), detail::MaximumMagnitude<RealScalar>{});
    }
    thrust::device_vector<RealScalar> magnitudes(values_.size());
    thrust::transform(values_.begin(), values_.end(), magnitudes.begin(), detail::Absolute<Scalar>{});
    auto sums = scatterSum(std::move(magnitudes), innerSize());
    return thrust::reduce(sums.begin(), sums.end(), RealScalar(0), detail::MaximumMagnitude<RealScalar>{});
  }
};

//! \brief Lazy main-diagonal vector. 
//! \details This is a lazy view of the main diagonal of the matrix.
//! \note Changes to the parent remain visible.
//! \tparam Matrix The type of the parent matrix.
template<class Matrix>
class DiagonalView {
  using Plain = std::remove_const_t<Matrix>;
public:
  using Scalar = typename Plain::Scalar;
  using Index = typename Plain::Index;
  using Vector = typename Plain::Vector;
  //! \brief Construct a DiagonalView from a matrix.
  explicit DiagonalView(Matrix& matrix) : matrix_(&matrix) {}
  //! \brief Get the size of the diagonal (the minimum of rows and columns).
  Index size() const { return std::min(matrix_->rows(), matrix_->cols()); }
  //! \brief Get the coefficient at the given index on the diagonal.
  Scalar coeff(Index i) const { check(i); return matrix_->coeff(i, i); }
  //! \brief Access the coefficient at the given index on the diagonal.
  decltype(auto) operator[](Index i) const { check(i); return (*matrix_)(i, i); }
  //! \brief Access the coefficient at the given index on the diagonal.
  decltype(auto) operator()(Index i) const { return (*this)[i]; }
  //! \brief evaluate the diagonal as a vector.
  Vector eval() const {
    Vector result(static_cast<std::size_t>(size()));
    auto first = thrust::make_counting_iterator<Index>(0);
    thrust::transform(first, first + size(), result.begin(),
      detail::DiagonalEntry<typename Plain::DeviceView>{matrix_->deviceView()});
    return result;
  }
  //! \brief Elementwise multiplication of the diagonal by a vector of matching size.
  template<class Allocator>
  Vector operator*(const thrust::device_vector<Scalar, Allocator>& x) const {
    if (x.size() != static_cast<std::size_t>(size())) throw std::invalid_argument("Diagonal/vector dimension mismatch");
    Vector result = eval();
    thrust::transform(result.begin(), result.end(), x.begin(), result.begin(), thrust::multiplies<Scalar>());
    return result;
  }
private:
  Matrix* matrix_;
  void check(Index i) const {
    if (i >= size()) throw std::out_of_range("Diagonal index out of range");
  }
};

/*! \brief Non-conjugating transpose. CSR(A) is CSC(A^T), with identical buffers.
 *  \details This host view stores only a parent pointer and does not allocate/copy data.
 *  \tparam Matrix The type of the parent matrix.
 */
template<class Matrix>
class TransposeView {
  using Plain = std::remove_const_t<Matrix>;
public:
  using Scalar = typename Plain::Scalar;
  using StorageIndex = typename Plain::StorageIndex;
  using Index = typename Plain::Index;
  using RealScalar = typename Plain::RealScalar;
  using Vector = typename Plain::Vector;
  static constexpr int Options = Plain::Options ^ RowMajor;
  static constexpr bool IsRowMajor = !Plain::IsRowMajor;
  using Evaluated = SparseMatrix<Scalar, Options, StorageIndex>;
  using DeviceView = typename Evaluated::DeviceView;
  explicit TransposeView(Matrix& matrix) : matrix_(&matrix) {}
  Index rows() const { return matrix_->cols(); }//!< The number of rows in the transposed view (original matrix's columns).
  Index cols() const { return matrix_->rows(); }//!< The number of columns in the transposed view (original matrix's rows).
  Index nonZeros() const { return matrix_->nonZeros(); }//!< The number of non-zero entries in the transposed view (same as original matrix).
  Index outerSize() const { return matrix_->outerSize(); }//!< The number of outer segments in the transposed view (same as original matrix).
  Index innerSize() const { return matrix_->innerSize(); }//!< The number of inner segments in the transposed view (same as original matrix).
  constexpr bool isCompressed() const { return true; }//!< The transposed view is always compressed.
  /*!
   * \brief Get the coefficient at the specified position.
   * \param r The row index.
   * \param c The column index.
   * \return The coefficient at the specified position.
   */
  Scalar coeff(Index r, Index c) const { return matrix_->coeff(c, r); }
  
  /*!
   * \brief Get the coefficient at the specified position.
   * \param r The row index.
   * \param c The column index.
   * \return The coefficient at the specified position.
   */
  /*!
   * \brief Get the coefficient at the specified position.
   * \param r The row index.
   * \param c The column index.
   * \return The coefficient at the specified position.
   */
  decltype(auto) operator()(Index r, Index c) const { return (*matrix_)(c, r); }
  
  /*!
   * \brief Multiply the transposed matrix by a vector.
   * \param x The vector to multiply by.
   * \tparam Allocator The allocator type for the vector.
   * \return The result of the multiplication.
   */
  template<class Allocator>
  Vector multiply(const thrust::device_vector<Scalar, Allocator>& x) const { return matrix_->transposeMultiply(x); }
  /*!
   * \brief Multiply the matrix by a vector.
   * \param x The vector to multiply by.
   * \tparam Allocator The allocator type for the vector.
   * \return The result of the multiplication.
   */
  template<class Allocator>
  Vector operator*(const thrust::device_vector<Scalar, Allocator>& x) const { return this->multiply(x); }
  /*!
   * \brief Multiply the matrix by a host vector.
   * \param x The vector to multiply by.
   * \param y The vector to store the result in.
   */
  void multiply(const Vector& x, Vector& y) const { Vector result = multiply(x); y.swap(result); }
  
  /*!
   * \brief Get the transpose of the matrix.
   * \return The transpose of the matrix.
   */
  Matrix& transpose() const { return *matrix_; }
  
  /*!
   * \brief Get the diagonal of the matrix.
   * \return The diagonal of the matrix.
   */
  DiagonalView<Matrix> diagonal() const { return DiagonalView<Matrix>(*matrix_); }
  
  /*!
   * \brief Get the device view of the matrix.
   * \return The device view of the matrix.
   */
  DeviceView deviceView() const {
    return {matrix_->outerIndexPtr(), matrix_->innerIndexPtr(), matrix_->valuePtr(), rows(), cols()};
  }
  /*!
   * \brief Materialize an independent device-to-device copy in the opposite layout.
   * \return The evaluated matrix.
   */
  Evaluated eval() const {
    Evaluated result;
    result.rows_ = rows(); result.cols_ = cols();
    result.outer_ = matrix_->outer_; result.inner_ = matrix_->inner_; result.values_ = matrix_->values_;
    return result;
  }
  //! \brief Compute the sum of all entries in the matrix.
  Scalar sum() const { return matrix_->sum(); }
  RealScalar norm() const { return matrix_->norm(); } //!< The Frobenius norm of the matrix.
  RealScalar frobeniusNorm() const { return matrix_->frobeniusNorm(); }//!< The Frobenius norm of the matrix.
  RealScalar maxNorm() const { return matrix_->maxNorm(); }//!< The maximum absolute value of any entry in the matrix.
  RealScalar entrywiseL1Norm() const { return matrix_->entrywiseL1Norm(); }//!< The entrywise L1 norm of the matrix.
  RealScalar l1Norm() const { return matrix_->infinityNorm(); }//!< The L1 norm of the matrix (maximum absolute column sum).
  RealScalar infinityNorm() const { return matrix_->l1Norm(); }//!< The infinity norm of the matrix (maximum absolute row sum).
private:
  Matrix* matrix_;
};

template<class S, int O, class I>
void swap(SparseMatrix<S, O, I>& a, SparseMatrix<S, O, I>& b) noexcept { a.swap(b); }
template<class S, int O, class I>
SparseMatrix<S, O, I> operator*(SparseMatrix<S, O, I> matrix, S factor) { matrix *= factor; return matrix; }
template<class S, int O, class I>
SparseMatrix<S, O, I> operator*(S factor, SparseMatrix<S, O, I> matrix) { matrix *= factor; return matrix; }
} // namespace apsc::gpu

#undef APSC_SPARSE_HD
