# The `apsc::gpu::SparseMatrix` class

This guide describes the implementation in [include/SparseMatrix.hpp](include/SparseMatrix.hpp).
It explains the data representation, public operations, and execution costs, with
examples that use the current unsigned-index API. Build and benchmark commands
are also collected in [README.md](README.md).

## 1. Purpose and execution model

`SparseMatrix` represents a rectangular matrix whose unspecified coefficients
are zero. It stores coefficients in compressed sparse row (CSR) or compressed
sparse column (CSC) format and uses NVIDIA Thrust containers and algorithms for
numerical operations. The library is header-only and requires C++20.

Its main capabilities are assembly from triplets, individual coefficient access
and modification, matrix-vector products, scalar scaling, matrix norms, and
writable diagonal and transpose views. The implementation has no Eigen
dependency; the test suite uses Eigen as an independent numerical reference.

The selected Thrust backend determines where data lives and where numerical
algorithms execute:

| `APSC_THRUST_BACKEND` | Storage and execution |
| --- | --- |
| `CUDA` | GPU memory and CUDA execution |
| `CPP` | Host memory and serial CPU execution |
| `OMP` | Host memory and OpenMP execution |
| `TBB` | Host memory and TBB execution |

The container type remains `thrust::device_vector` for all four configurations.
Its name alone does not imply GPU allocation. Matrix assembly, insertion, and
pruning perform work on the host even with the CUDA backend. Numerical results
that are vectors remain in the selected backend's storage until explicitly
copied to the host; scalar results are returned to the calling host code.

## 2. Template parameters and public types

The class declaration is:

```cpp
template<class Scalar_, int Options_ = apsc::gpu::ColMajor,
         std::unsigned_integral StorageIndex_ = std::size_t>
class SparseMatrix;
```

The declaration belongs to `namespace apsc::gpu`.

| Parameter | Meaning and constraints |
| --- | --- |
| `Scalar_` | Coefficient type: a floating-point type or `thrust::complex<T>` with floating-point `T`. Integer coefficients and `std::complex` are unsupported. |
| `Options_` | `apsc::gpu::ColMajor` (`0`, the default) selects CSC; `apsc::gpu::RowMajor` (`1`) selects CSR. |
| `StorageIndex_` | Unsigned integral type other than `bool`, defaulting to `std::size_t`. Used for both inner coordinates and compressed offsets. |

The parameter order and storage-order option values match Eigen's sparse matrix
interface. The index requirement and default differ: this class uses unsigned
storage indices. `Eigen::RowMajor` and `Eigen::ColMajor` can be passed as options
when Eigen is included.

For example:

```cpp
using DefaultMatrix = apsc::gpu::SparseMatrix<double>;  // CSC, size_t storage
using CompactMatrix = apsc::gpu::SparseMatrix<
    double, apsc::gpu::RowMajor, std::uint32_t>;
using ComplexMatrix = apsc::gpu::SparseMatrix<
    thrust::complex<double>, apsc::gpu::ColMajor>;
```

Using 32-bit storage indices can reduce index-buffer memory compared with 64-bit
indices, provided that the dimensions **and stored-entry count** fit the type.

| Public member type or constant | Meaning |
| --- | --- |
| `Scalar` | Coefficient type |
| `RealScalar` | Same type for real coefficients; underlying real type for complex coefficients |
| `Index` | Always `std::size_t`; used for dimensions, sizes, and host coordinate arguments |
| `StorageIndex` | The selected compressed-storage index type |
| `Vector` | `thrust::device_vector<Scalar>` |
| `IndexVector` | `thrust::device_vector<StorageIndex>` |
| `TripletType` | `apsc::gpu::Triplet<Scalar, StorageIndex>` |
| `DeviceView` | Small read-only handle for accessing the compressed buffers in device code |
| `Options`, `IsRowMajor` | Compile-time storage-order information |

Unsigned coordinates still need bounds checks. Dimensions and stored counts must
fit `StorageIndex` and the signed `std::ptrdiff_t` iterator-difference range.
The outer buffer additionally needs space for one end offset. These restrictions
are checked before constructing oversized storage; available memory and
allocator limits can impose smaller practical limits.

Use zero-based indices. Validate signed dimensions before converting them to
`size_t`: unsigned conversion does not make negative input meaningful. External
signed triplet coordinates are checked for negativity and bounds before being
narrowed to `StorageIndex`.

## 3. A complete first example

Consider the following 3-by-4 matrix:

```text
    [ 4  0 -1  0 ]
A = [ 0  2  0  0 ]
    [ 3  0  0  5 ]
```

The program below assembles it from unordered entries. Two entries at `(1,1)`
sum to `2`, and two entries at `(1,3)` cancel. Only five entries remain stored.

```cpp
#include <SparseMatrix.hpp>
#include <thrust/host_vector.h>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

using Matrix = apsc::gpu::SparseMatrix<
    double, apsc::gpu::RowMajor, std::uint32_t>;

Matrix makeExample() {
  Matrix A(3, 4);
  std::vector<Matrix::TripletType> entries{
      {2, 3, 5.0}, {0, 2, -1.0}, {1, 1, 1.25}, {2, 0, 3.0},
      {0, 0, 4.0}, {1, 1, 0.75}, {1, 3, 7.0}, {1, 3, -7.0}};
  A.setFromTriplets(entries.begin(), entries.end());
  return A;
}

int main() {
  Matrix A = makeExample();
  Matrix::Vector x{1.0, 2.0, 3.0, 4.0};
  Matrix::Vector y = A * x;
  thrust::host_vector<double> hostY = y;

  std::cout << "shape: " << A.rows() << " x " << A.cols() << '\n';
  std::cout << "stored entries: " << A.nonZeros() << '\n';
  std::cout << "A*x:";
  for (double value : hostY) std::cout << ' ' << value;
  std::cout << "\n1-norm: " << A.l1Norm() << '\n';
}
```

Expected output:

```text
shape: 3 x 4
stored entries: 5
A*x: 1 4 23
1-norm: 7
```

The first product entry is `4*1 - 1*3 = 1`; the last is `3*1 + 5*4 = 23`.
The `hostY` assignment performs the explicit result transfer when using CUDA.

Unless stated otherwise, subsequent snippets are independent examples to place
inside `main()`, using the headers, `Matrix` alias, and `makeExample()` above.
Each starts with a fresh matrix, so changes in one example do not affect another.

## 4. How compressed storage works

Three buffers represent the matrix:

| Buffer accessor | Contents | Length |
| --- | --- | --- |
| `outerIndices()` | Start offset of each row or column, plus a final end offset | `outerSize() + 1` |
| `innerIndices()` | Column coordinates in CSR; row coordinates in CSC | `nonZeros()` |
| `values()` | Coefficients in the same order as the inner indices | `nonZeros()` |

An outer segment `o` occupies the half-open range
`[outer[o], outer[o+1])`. An empty segment has equal start and end offsets.
The first offset is zero and the final offset equals `nonZeros()`. Inner
coordinates are strictly increasing within each segment.

For the example matrix in CSR form:

```text
outerIndices = [0, 2, 3, 5]
innerIndices = [0, 2, 1, 0, 3]
values       = [4,-1, 2, 3, 5]
```

Row 0 occupies positions 0 and 1: columns 0 and 2 contain `4` and `-1`.
Row 1 occupies position 2, and row 2 occupies positions 3 and 4.

The same matrix assembled in CSC form has:

```text
outerIndices = [0, 2, 3, 4, 5]
innerIndices = [0, 2, 1, 0, 2]
values       = [4, 3, 2,-1, 5]
```

| Query | CSR | CSC |
| --- | --- | --- |
| `outerSize()` | Number of rows | Number of columns |
| `innerSize()` | Number of columns | Number of rows |

Ignoring spare capacity and container overhead, compressed storage uses roughly
`nnz * (sizeof(Scalar) + sizeof(StorageIndex)) + (outerSize()+1) * sizeof(StorageIndex)`
bytes. Storage order is part of the matrix's type and cannot be changed in place.
To reassemble the same mathematical matrix in another layout, export its
triplets and assemble a matrix with the desired option.

The matrix is always compressed: `isCompressed()` returns `true`, and
`makeCompressed()` is a no-op. Compressed does not mean every stored value is
nonzero: subsequent coefficient writes or scaling can leave explicit zeros.

## 5. Construction, assembly, and export

`SparseMatrix()` creates a valid 0-by-0 matrix. `SparseMatrix(rows, cols)` creates
an all-zero matrix of the requested shape. A zero matrix still has an outer
offset buffer, but its coefficient and inner-index buffers are empty.

`Triplet<Scalar, StorageIndex>` holds one `(row, column, value)` record. It has a
zero-initializing default constructor, a three-argument constructor,
`row()`, `col()`, and `value()` accessors, and `set(row, col, value)`.

`setFromTriplets(first, last)` **replaces** the current coefficients without
changing the shape. The iterator's value type must provide integral row and
column coordinates and a value convertible to `Scalar`. This includes the
library's triplets and signed-index `Eigen::Triplet` objects.

Assembly proceeds as follows:

1. Read and validate coordinates before converting to the storage index type.
2. Stable-sort on the host by outer coordinate and then inner coordinate.
3. Sum duplicate coordinates, retaining their input order for summation.
4. Remove coefficients whose accumulated value is exactly zero.
5. Form cumulative outer offsets and upload the new compressed buffers.

New buffers replace the original buffers only after assembly succeeds. Invalid
coordinates and excess stored counts therefore leave the matrix unchanged.
An empty input range removes all entries while preserving the dimensions.

Use host triplet iterators for bulk assembly. Device iterators are accepted, but
the host assembly loop reads their entries individually and can incur many
small transfers. There is no persistent host mirror of the matrix.

`toHostTriplets()` returns an independent `std::vector<TripletType>` containing
the stored entries, in compressed traversal order. It transfers all buffers to
the host when using CUDA, and includes any explicitly stored zeros.

```cpp
Matrix A = makeExample();
auto entries = A.toHostTriplets();

apsc::gpu::SparseMatrix<double, apsc::gpu::ColMajor, std::uint32_t>
    columnMajor(A.rows(), A.cols());
columnMajor.setFromTriplets(entries.begin(), entries.end());
// columnMajor represents A in CSC form; it is not A's transpose.
```

## 6. Reading and modifying coefficients

`coeff(row, col)` returns a `Scalar` copy. An in-bounds location with no stored
entry returns zero. The const overload of `operator()(row, col)` behaves the
same way. Both check coordinates.

For a mutable matrix, `operator()` and `coeffRef()` return a
`CoefficientProxy`. The proxy supports conversion to `Scalar`, assignment, and
`+=`, `-=`, `*=`, `/=`. It records the matrix object and coordinates and performs
the corresponding read or write when used.

```cpp
Matrix A = makeExample();
double missing = A.coeff(0, 1);    // 0; no insertion
double snapshot = A(0, 0);        // Scalar copy: 4
auto proxy = A(0, 0);             // CoefficientProxy referring to A(0,0)
proxy += 2.0;                    // A(0,0) becomes 6; snapshot remains 4
A.coeffRef(1, 1) *= 3.0;          // Existing coefficient becomes 6
A.setCoeff(1, 2, 8.0);            // Inserts a previously absent coefficient
A(2, 0) = A(0, 0);               // Copies the value 6
```

Use `double value = A(r,c)` or `auto value = A.coeff(r,c)` when you want a value
snapshot. `auto value = A(r,c)` on a mutable matrix instead keeps a proxy. A
proxy is not a `Scalar&`, must not outlive its matrix, and is not an atomic
update mechanism.

Updating a stored coefficient changes its value in place. Inserting an absent
nonzero coefficient exports and rebuilds the matrix on the host. Individual
host accesses can transfer data and synchronize with the GPU, so use bulk
assembly and vector operations for large workloads.

`nonZeros()` reports the number of **stored slots**. Assigning zero to an
existing coefficient retains its slot until pruning or reassembly removes it.
Assigning zero to an absent coefficient does not create a slot.

```cpp
Matrix A = makeExample();
A(0, 2) = 0.0;
auto storedBeforePruning = A.nonZeros();  // 5
A.prune();
auto storedAfterPruning = A.nonZeros();   // 4
A(1, 3) = 1e-9;
A.prune(1e-8);                           // Removes the small inserted value
```

`prune(tolerance)` removes entries with magnitude at most the tolerance and
reassembles the result. The default tolerance is zero. A negative or NaN
tolerance raises `std::invalid_argument`.

## 7. Matrix-vector products

For an `m`-by-`n` matrix, the supported products are:

| Expression | Input length | Output length | Meaning |
| --- | ---: | ---: | --- |
| `A * x` or `A.multiply(x)` | `n` | `m` | Ordinary matrix-vector product |
| `A.transposeMultiply(z)` | `m` | `n` | Product with the ordinary transpose |
| `A.transpose() * z` | `m` | `n` | Same transpose product through a view |
| `A.diagonal() * d` | `min(m,n)` | `min(m,n)` | Elementwise product with the main diagonal |

Inputs are `thrust::device_vector` objects with the same scalar type as the
matrix. The return-by-value methods accept a custom input-vector allocator;
their results use the matrix's `Vector` type. Copy a host vector into a device
vector before multiplication.

```cpp
Matrix A = makeExample();
Matrix::Vector x{1.0, 2.0, 3.0, 4.0};
Matrix::Vector z{1.0, 2.0, 3.0};
auto y = A.multiply(x);               // [1, 4, 23]
auto t = A.transposeMultiply(z);      // [13, 4, -1, 15]

Matrix::Vector output;
A.multiply(x, output);               // output becomes [1, 4, 23]
A.multiply(x, x);                    // Allowed: x becomes a length-3 result
```

The output-argument overload accepts `const Vector& x, Vector& y`. It builds
a temporary result and swaps it into `y`, so input and output may alias. It does
not promise to reuse the output allocation. The transpose view provides the
same output-argument form. A dimension mismatch raises `std::invalid_argument`.

Zero and empty dimensions are supported. For example, a 3-by-0 matrix times an
empty vector yields three zeros; a 0-by-4 matrix times a length-4 vector yields
an empty result.

## 8. Sums, scaling, and norms

`sum()` adds the coefficients algebraically and returns `Scalar`. `A *= factor`
scales all stored values in place. The free expressions `A * factor` and
`factor * A` return a scaled matrix; when passed an lvalue matrix they start
from a deep copy. These operations preserve the stored sparsity pattern, even
when the factor is zero.

```cpp
Matrix A = makeExample();
double total = A.sum();               // 13
auto B = 2.0 * A;                     // B(0,0) == 8; A(0,0) == 4
A *= 0.5;                            // A(0,0) == 2
```

Use a factor of the matrix's scalar type in the free operators: for a double
matrix, write `2.0 * A`. A literal such as `2 * A` conflicts with template
deduction between `int` and `double`.

All norms return `RealScalar`. In the following definitions, `abs` denotes
absolute value for real coefficients and magnitude for complex coefficients.

| Method | Definition | Value for the example matrix |
| --- | --- | ---: |
| `frobeniusNorm()`, `norm()` | `sqrt(sum over i,j of abs(a_ij)^2)` | `sqrt(55)`, approximately `7.416198487` |
| `maxNorm()` | Largest coefficient magnitude | `5` |
| `l1Norm()` | Maximum absolute column sum | `7` |
| `infinityNorm()` | Maximum absolute row sum | `8` |
| `entrywiseL1Norm()` | Sum of all coefficient magnitudes | `15` |

For example, column sums of magnitudes are `[7,2,1,5]`, whereas row sums are
`[5,2,8]`. Thus `l1Norm()` is a different quantity from `entrywiseL1Norm()`.
Empty matrices and matrices whose values are all zero have zero norms.

The Frobenius calculation first finds the largest magnitude `s` and, for
finite nonzero `s`, computes `s * sqrt(sum over i,j of (abs(a_ij)/s)^2)`. Scaling protects
the intermediate squares from overflow and underflow; it cannot make an
unrepresentable final result finite. Norm reductions propagate NaNs. An
infinite entry produces an infinite Frobenius norm unless a NaN is also present.

## 9. The diagonal view

`A.diagonal()` returns a lightweight `DiagonalView` referring to the main
diagonal. Its size is `min(A.rows(), A.cols())`. Creating the view allocates no
coefficient buffer and copies no matrix data.

| View operation | Effect |
| --- | --- |
| `size()` | Diagonal length |
| `coeff(i)` | Read a scalar value |
| `operator[](i)`, `operator()(i)` | Read a value or access a writable proxy, depending on the parent matrix |
| `eval()` | Materialize an independent `Vector` |
| `operator*(vector)` | Return the elementwise diagonal-vector product |

```cpp
Matrix A = makeExample();
auto diagonal = A.diagonal();
auto saved = diagonal.eval();             // [4, 2, 0], independent vector
diagonal[2] = 7.0;                        // Inserts A(2,2)
double current = diagonal.coeff(2);       // 7
Matrix::Vector weights{1.0, 2.0, 3.0};
auto weighted = diagonal * weights;       // [4, 4, 21]
// saved still contains [4, 2, 0].
```

`eval()` returns a vector, not a diagonal sparse matrix. Changes to the parent,
including later insertions or resizing, are visible through an existing view.

## 10. The transpose view

`A.transpose()` returns a lightweight `TransposeView`. It exchanges the logical
row and column coordinates, so `T.coeff(r,c)` reads `A.coeff(c,r)`. The view's
storage-order flag is the opposite of the parent's: the CSR representation of
`A` is already a CSC representation of `A` transposed.

```cpp
Matrix A = makeExample();
auto T = A.transpose();                   // Shape 4 x 3; CSC view of CSR A
Matrix::Vector z{1.0, 2.0, 3.0};
auto y = T * z;                          // [13, 4, -1, 15]
auto independent = T.eval();             // Independent 4 x 3 CSC matrix
T(3, 2) = 9.0;                           // Changes A(2,3) from 5 to 9
double previous = independent.coeff(3,2); // Still 5
Matrix& original = T.transpose();        // Reference to A
```

`eval()` copies the compressed buffers into an independent matrix of the
opposite layout and swapped dimensions; it does not sort or reassemble them.
It materializes the transpose, whereas exporting and reassembling triplets in
another layout preserves the original mathematical matrix.

The view supports shape and stored-count queries, coefficient access,
multiplication (including the output-argument overload), `diagonal()`,
`deviceView()`, `sum()`, and the same norms as the matrix. Transposition swaps
`l1Norm()` with `infinityNorm()`; the other listed norms are unchanged.

This is an **ordinary, non-conjugating transpose**, also for complex matrices.
For example, the following independent snippet uses the original headers but
defines its own complex matrix:

```cpp
using C = thrust::complex<double>;
apsc::gpu::SparseMatrix<C, apsc::gpu::RowMajor> A(1, 2);
std::vector<apsc::gpu::Triplet<C>> entries{
    {0, 0, C(3.0, 4.0)}, {0, 1, C(0.0, -2.0)}};
A.setFromTriplets(entries.begin(), entries.end());
thrust::device_vector<C> x{C(1.0, 0.0)};
thrust::host_vector<C> y = A.transpose() * x;
// y is [3+4i, -2i]; the imaginary parts are not conjugated.
double magnitudeNorm = A.norm();           // sqrt(29)
```

## 11. Ownership, constness, and lifetime

Matrices own their compressed buffers. Copy construction and copy assignment
perform deep copies, including device-to-device copies with CUDA. A move
transfers ownership and leaves the source as a valid 0-by-0 matrix; self-move
assignment preserves the matrix. `swap(other)` and the free `apsc::gpu::swap`
exchange dimensions and buffers.

```cpp
Matrix A = makeExample();
Matrix B = A;                            // Independent buffers
A(0, 0) = 10.0;                          // B(0,0) is still 4
Matrix C = std::move(B);                  // C has B's former contents
// B.rows() == 0, B.cols() == 0, B.nonZeros() == 0
B.resize(2, 2);                          // The moved-from object is reusable
B.setIdentity();
```

Include `<utility>` when using `std::move` in application code.

Diagonal and transpose views do not own buffers. They refer to a matrix
**object**, so moving, assigning, swapping, or resizing that object changes
what an existing view observes. The parent must outlive the view. Calling
`diagonal()` or `transpose()` on a temporary matrix is disabled to prevent an
immediately dangling view.

Constness comes from the parent. A view created from a `const Matrix&` is
read-only. A view created from a mutable matrix remains writable even when the
view variable itself is declared `const`.

Raw pointers and device handles instead refer to particular buffers. Refresh
them after structural edits, operations that may reallocate (including
`reserve`), assignment, move, or swap. Discard them when the owning matrix is
destroyed.
Concurrent mutation and access require caller-managed synchronization.

## 12. Resizing, clearing, and capacity

| Method | Behavior |
| --- | --- |
| `resize(rows, cols)` | Set a new shape and discard all coefficients, even if the shape is unchanged |
| `setZero()` | Remove all stored entries while preserving the shape |
| `setIdentity()` | Replace the entries by ones on the main diagonal; works for rectangular matrices |
| `reserve(count)` | Reserve capacity in coefficient and inner-index buffers without changing shape or stored count |
| `prune(tolerance)` | Remove small entries through host reassembly |

```cpp
Matrix A(2, 4);
A.setIdentity();                         // A(0,0) = A(1,1) = 1; two entries
A.setZero();                             // Still 2 x 4; no stored entries
A.resize(3, 3);                          // All-zero 3 x 3 matrix
A.reserve(9);                            // Still no stored entries
```

`reserve()` does not create entries or introduce an uncompressed insertion
mode. Bulk assembly constructs fresh buffers, and inserting one missing
coefficient still rebuilds the matrix. Reserving capacity therefore does not
turn repeated individual insertions into an efficient assembly strategy.

## 13. Buffer inspection and device access

`outerIndices()`, `innerIndices()`, and `values()` return const references to
the three `thrust::device_vector` buffers. Use explicit host copies to inspect
them in bulk:

```cpp
Matrix A = makeExample();
thrust::host_vector<Matrix::StorageIndex> outer = A.outerIndices();
thrust::host_vector<Matrix::StorageIndex> inner = A.innerIndices();
thrust::host_vector<double> values = A.values();
// CSR arrays: outer=[0,2,3,5], inner=[0,2,1,0,3], values=[4,-1,2,3,5].
```

`outerIndexPtr()` and `innerIndexPtr()` expose read-only raw structural
pointers. `valuePtr()` exposes a writable coefficient pointer for a mutable
matrix and a read-only pointer for a const matrix. Modifying values through
the writable pointer does not alter the compressed structure or remove zeros.
With CUDA these are device pointers and must not be dereferenced by host code.

`deviceView()` packages the pointers and dimensions into a small, trivially
copyable, read-only handle. Its `coeff(row,col)` and `operator()(row,col)` use
binary search within a compressed segment and return zero for a missing
in-bounds entry. They perform **no coordinate validation**. The handle's
dimensions are data members named `rows` and `cols`.

For example, define this functor at namespace scope, before `main()`:

```cpp
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

struct ReadFirstColumn {
  Matrix::DeviceView view;
#if defined(__CUDACC__)
  __host__ __device__
#endif
  double operator()(std::size_t row) const {
    return view.coeff(row, 0);
  }
};
```

Then use it inside `main()`:

```cpp
Matrix A = makeExample();
Matrix::Vector firstColumn(A.rows());
auto first = thrust::make_counting_iterator<std::size_t>(0);
thrust::transform(thrust::device, first, first + A.rows(), firstColumn.begin(),
                  ReadFirstColumn{A.deviceView()});
// firstColumn contains [4, 0, 3]. All accessed coordinates are in bounds.
```

The explicit `thrust::device` policy dispatches the counting-iterator operation
to the configured device backend. Capture the handle in a device functor rather
than copying the owning `SparseMatrix` into one. Host convenience methods such
as `A.coeff()` manage host-side operations and are not device-callable methods.

## 14. Validation and errors

| Condition | Result |
| --- | --- |
| Unsupported scalar, signed storage index, `bool` storage index, or invalid layout | Compile-time rejection |
| Host coefficient or diagonal coordinate outside the shape | `std::out_of_range` |
| Negative or out-of-bounds input triplet coordinate | `std::out_of_range` |
| Dimension/count exceeds storage or iterator limits | `std::length_error` |
| Product input length does not match its required dimension | `std::invalid_argument` |
| Negative or NaN pruning tolerance | `std::invalid_argument` |

Backend allocation and execution errors may also propagate from Thrust or CUDA.
These checks concern the host API; callers must ensure device-handle coordinates
are valid. A `size_t` API argument has already undergone any signed-to-unsigned
conversion before the method receives it. A negative input such as `-1` becomes
a large unsigned value and is rejected by dimension or coordinate range checks.

## 15. Algorithm choices and performance

CSR stores each row contiguously, so ordinary multiplication computes each
output row by gathering its coefficients and input-vector elements. CSC gives
the analogous direct path for transpose multiplication. These paths use one
work item per output segment and `O(nnz + output-size)` total work.

For CSC ordinary multiplication and CSR transpose multiplication, the code
forms individual contributions, sorts them by output coordinate, reduces equal
coordinates, and scatters the sums into a zero-initialized result. This uses
`O(nnz)` temporary storage. It also performs binary searches to associate
entries with outer segments; comparison sorting has an `O(nnz log nnz)` general
upper bound. This strategy supports complex coefficients without floating-point
atomics.

The induced norms follow the same layout distinction: CSR gives direct row
sums for `infinityNorm()`, and CSC gives direct column sums for `l1Norm()`.
Summation in the other direction uses sorting and reduction. Diagonal
materialization performs a lookup for each diagonal coordinate.

Choose CSR when ordinary products dominate, and CSC when transpose products
dominate. Assemble in batches, keep matrix and vectors resident across repeated
operations, and copy to the host only when the result is needed there. Highly
unequal segment lengths can cause load imbalance. These implementation choices
are useful for understanding sparse algorithms; they do not provide the full
functionality or tuning of a dedicated sparse solver library.

The API has no explicit asynchronous stream interface. Timings of these methods
include allocations and synchronization where required, rather than only
isolated kernel execution. See [benchmarks/sparse_benchmark.cpp](benchmarks/sparse_benchmark.cpp)
and the benchmark discussion in [README.md](README.md) for the measured workloads.

## 16. Building and using the examples

Dependencies are CMake 3.24 or later, a C++20 compiler, and Thrust/CUB/libcu++
(CCCL). CUDA execution additionally needs the CUDA Toolkit and a compatible
GPU/driver. OpenMP and TBB need their respective runtimes. Eigen is required
only for the provided tests.

From the repository root, build and run the existing CPU example and tests:

```sh
cmake -S MatrixVector -B MatrixVector/build-cpu \
  -DAPSC_THRUST_BACKEND=CPP -DCMAKE_BUILD_TYPE=Release
cmake --build MatrixVector/build-cpu --parallel 2
ctest --test-dir MatrixVector/build-cpu --output-on-failure
./MatrixVector/build-cpu/sparse_example
```

The existing example uses its own 3-by-3 matrix. To run the complete 3-by-4
program from this guide, save it as `demo.cpp` in a separate project directory
and use the following `CMakeLists.txt`, replacing the library path:

```cmake
cmake_minimum_required(VERSION 3.24)
project(SparseDemo LANGUAGES CXX)

set(APSC_THRUST_BACKEND CPP CACHE STRING "Thrust backend")
if(APSC_THRUST_BACKEND STREQUAL "CUDA")
  enable_language(CUDA)
endif()

add_subdirectory(/absolute/path/to/MatrixVector matrix-vector)
add_executable(sparse_demo demo.cpp)
target_link_libraries(sparse_demo PRIVATE apsc::MatrixVector)

if(APSC_THRUST_BACKEND STREQUAL "CUDA")
  set_source_files_properties(demo.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(sparse_demo PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
endif()
```

The interface target supplies include directories, Thrust configuration, and
the C++20 requirement. Library tests and examples default to disabled when
included as a subdirectory. Configure the standalone demonstration with:

```sh
cmake -S . -B build-cpu -DAPSC_THRUST_BACKEND=CPP
cmake --build build-cpu --parallel 2
./build-cpu/sparse_demo
```

For a CUDA build, use a separate directory and the appropriate architecture:

```sh
cmake -S . -B build-cuda -DAPSC_THRUST_BACKEND=CUDA \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --parallel 2
./build-cuda/sparse_demo
```

Here `89` is an example for compute capability 8.9. CUDA translation units that
instantiate the numerical operations must be compiled by the CUDA compiler,
using a `.cu` file or CMake's `LANGUAGE CUDA` property as above.

## 17. Scope and further reading

The current class provides sparse matrix storage, assembly, access, matvec,
norms, scalar scaling, and the two views described here. It does not implement
matrix addition, sparse matrix-matrix multiplication, a conjugate transpose,
linear system solvers, or a general expression-template interface. A transpose
view exposes the listed operations rather than every mutator of `SparseMatrix`.

For further examples and implementation details, see:

- [SparseMatrix.hpp](include/SparseMatrix.hpp): the complete implementation and Doxygen comments.
- [sparse_example.cpp](examples/sparse_example.cpp): a small executable example.
- [sparse_tests.cpp](tests/sparse_tests.cpp): real and complex arithmetic, both layouts, unsigned index limits, views, mutation, and Eigen comparisons.
- [README.md](README.md): backend configuration, integration, benchmarks, and recorded validation information.
