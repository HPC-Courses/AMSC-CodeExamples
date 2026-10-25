#include <SparseMatrix.hpp>
#include <Eigen/Sparse>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>

namespace {
template<class View> struct ReadDeviceCoefficient {
  View view;
#if defined(__CUDACC__)
  __host__ __device__
#endif
  double operator()(std::size_t i) const { return view(i / view.cols, i % view.cols); }
};
void require(bool condition, const std::string& what) {
  if (!condition) throw std::runtime_error(what);
}
void near(double a, double b, const std::string& what, double tolerance = 1e-10) {
  require(std::abs(a-b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)}), what);
}
template<class Exception, class F> void throws(F f, const std::string& what) {
  try { f(); } catch (const Exception&) { return; }
  throw std::runtime_error("Missing exception: " + what);
}
template<class V> void compare(const V& actual, const Eigen::VectorXd& expected, const std::string& what) {
  thrust::host_vector<double> host = actual;
  require(host.size() == static_cast<std::size_t>(expected.size()), what + " size");
  for (Eigen::Index i = 0; i < expected.size(); ++i) near(host[i], expected[i], what);
}

template<int Order, class Index> void realTests() {
  using Matrix = apsc::gpu::SparseMatrix<double, Order, Index>;
  // Eigen's storage indices are signed; assembly accepts these external triplets.
  using ReferenceIndex = std::make_signed_t<Index>;
  using Reference = Eigen::SparseMatrix<double, Order, ReferenceIndex>;
  static_assert(Matrix::Options == Reference::Options);
  static_assert(std::is_same_v<typename Matrix::Index, std::size_t>);
  // An unsorted rectangular matrix with duplicates, cancellation, and empty
  // interior/boundary rows and columns exercises both gather and scatter paths.
  std::vector<Eigen::Triplet<double, ReferenceIndex>> triplets{
    {3, 4, 2}, {1, 1, 3}, {1, 1, -1}, {3, 2, -4}, {2, 4, 5},
    {4, 1, 1}, {4, 1, -1}, {1, 4, -2}, {0, 0, 0}};
  Matrix a(6, 7);
  a.setFromTriplets(triplets.begin(), triplets.end());
  Reference reference(6, 7);
  reference.setFromTriplets(triplets.begin(), triplets.end());
  Eigen::MatrixXd dense(reference);
  // Exercise the raw handle inside a device-dispatched Thrust algorithm.
  thrust::device_vector<double> deviceCoefficients(42);
  auto count = thrust::make_counting_iterator<std::size_t>(0);
  thrust::transform(thrust::device, count, count+42, deviceCoefficients.begin(),
                   ReadDeviceCoefficient<typename Matrix::DeviceView>{a.deviceView()});
  thrust::host_vector<double> hostCoefficients = deviceCoefficients;
  for(int r=0;r<6;++r) for(int c=0;c<7;++c)
    near(hostCoefficients[r*7+c],dense(r,c),"device handle");
#if defined(__CUDACC__)
  for(const void* pointer : {static_cast<const void*>(a.outerIndexPtr()),
                            static_cast<const void*>(a.innerIndexPtr()),
                            static_cast<const void*>(a.valuePtr())}) {
    cudaPointerAttributes attributes{};
    require(cudaPointerGetAttributes(&attributes,pointer)==cudaSuccess,"CUDA pointer attributes");
    require(attributes.type==cudaMemoryTypeDevice,"compressed buffers reside on GPU");
  }
#endif
  require(a.nonZeros() == 5, "duplicate combination and zero removal");
  const Matrix& ca = a;
  for (int r=0; r<6; ++r) for (int c=0; c<7; ++c)
    near(ca(r,c), dense(r,c), "coefficient lookup");
  thrust::device_vector<double> x{1,2,3,4,5,6,7}, xt{1,2,3,4,5,6};
  Eigen::VectorXd hx(7), ht(6);
  hx << 1,2,3,4,5,6,7; ht << 1,2,3,4,5,6;
  compare(a*x, dense*hx, "matvec");
  compare(a.transpose()*xt, dense.transpose()*ht, "transpose matvec");
  compare(a.transposeMultiply(xt), dense.transpose()*ht, "transposeMultiply");
  auto alias = x;
  a.multiply(alias, alias);
  compare(alias, dense*hx, "aliasing output");
  near(a.frobeniusNorm(), dense.norm(), "Frobenius");
  near(a.maxNorm(), dense.cwiseAbs().maxCoeff(), "maximum");
  near(a.l1Norm(), dense.cwiseAbs().colwise().sum().maxCoeff(), "matrix L1");
  near(a.infinityNorm(), dense.cwiseAbs().rowwise().sum().maxCoeff(), "infinity");
  near(a.entrywiseL1Norm(), dense.cwiseAbs().sum(), "entrywise L1");
  near(a.sum(), dense.sum(), "sum");
  near(a.transpose().l1Norm(), a.infinityNorm(), "transpose L1");
  near(a.transpose().infinityNorm(), a.l1Norm(), "transpose infinity");
  compare(a.diagonal().eval(), dense.diagonal(), "diagonal eval");
  compare(a.diagonal()*xt, dense.diagonal().cwiseProduct(ht), "diagonal product");
  const auto constTranspose = ca.transpose();
  static_assert(std::is_same_v<decltype(constTranspose(0,0)), double>);
  using TransposeDeviceView = typename decltype(constTranspose)::DeviceView;
  thrust::transform(thrust::device, count, count+42, deviceCoefficients.begin(),
                   ReadDeviceCoefficient<TransposeDeviceView>{constTranspose.deviceView()});
  hostCoefficients = deviceCoefficients;
  for(int r=0;r<7;++r) for(int c=0;c<6;++c)
    near(hostCoefficients[r*6+c],dense(c,r),"transpose device handle");
  auto evaluated = ca.transpose().eval();
  static_assert(decltype(evaluated)::IsRowMajor != Matrix::IsRowMajor);
  compare(evaluated*xt, dense.transpose()*ht, "transpose eval");

  // Both views are live and writable; materialization and matrix copies are deep.
  auto diagonal = a.diagonal(); auto transpose = a.transpose();
  a(1,1) = 8;
  near(diagonal.coeff(1), 8, "live diagonal");
  transpose(4,3) = 9;
  near(a.coeff(3,4), 9, "transpose write");
  diagonal[0] = 3; // insertion into an empty compressed segment
  a(0,1) = a(0,0); // proxy assignment copies the value, not proxy coordinates
  a(0,1) += 2;
  near(a.coeff(0,1), 5, "proxy compound assignment");
  auto copy = a;
  a(0,0) = 10;
  near(copy.coeff(0,0), 3, "deep copy");
  Matrix assigned(1,1);
  assigned(0,0) = -2;
  assigned = copy;
  copy(1,1) = 12;
  require(assigned.rows()==6 && assigned.cols()==7, "copy assignment dimensions");
  near(assigned.coeff(1,1), 8, "copy assignment deep copy");
  Matrix destination(2,2);
  destination.setIdentity();
  destination = std::move(assigned);
  require(assigned.rows()==0 && assigned.cols()==0 && assigned.nonZeros()==0,
          "move assignment empties source");
  require(assigned.outerIndices().size()==1 && assigned.outerIndices()[0]==0,
          "move assignment preserves empty compressed storage");
  require((assigned*thrust::device_vector<double>{}).empty(), "moved-from multiplication");
  near(destination.coeff(0,0), 3, "move assignment value");
  Matrix& self = destination;
  destination = std::move(self);
  near(destination.coeff(0,0), 3, "self-move assignment");
  near(evaluated.coeff(4,3), 2, "transpose eval deep copy");
  require(&transpose.transpose() == &a, "double transpose");
  near(transpose.diagonal().coeff(0), 10, "transpose diagonal");
  const auto constDiagonal = ca.diagonal();
  static_assert(std::is_same_v<decltype(constDiagonal[0]), double>);
  auto moved = std::move(copy);
  require(copy.rows()==0 && copy.cols()==0 && copy.nonZeros()==0, "valid moved-from matrix");
  near(moved.coeff(0,0), 3, "move value");
  moved *= 2;
  near(moved.coeff(0,0), 6, "scaling");
  auto scaled = 0.5 * moved;
  near(scaled.coeff(0,0), 3, "free scaling");
  const auto nnz = a.nonZeros();
  a(0,1) = 0;
  require(a.nonZeros()==nnz, "explicit zero storage");
  a.prune();
  require(a.nonZeros()==nnz-1, "pruning");
  a.setIdentity();
  require(a.nonZeros()==6, "rectangular identity");
  a.setZero();
  require(a.nonZeros()==0 && a.rows()==6 && a.cols()==7, "setZero dimensions");
  compare(a*x, Eigen::VectorXd::Zero(6), "zero matrix matvec");
  a.resize(0,7);
  require((a*x).empty(), "zero rows");
  a.resize(6,0);
  compare(a*thrust::device_vector<double>{}, Eigen::VectorXd::Zero(6), "zero columns");
  require((a.transpose()*xt).empty(), "transpose zero columns");
  require(a.diagonal().eval().empty(), "empty diagonal");
  near(a.norm(),0,"empty Frobenius"); near(a.maxNorm(),0,"empty max");
  near(a.l1Norm(),0,"empty L1"); near(a.infinityNorm(),0,"empty infinity");
  Matrix empty;
  require((empty*thrust::device_vector<double>{}).empty(),"default matrix");
  const auto negativeIndex = static_cast<typename Matrix::Index>(-1);
  throws<std::length_error>([&]{ a.resize(negativeIndex,2); },"negative dimension converted to size_t");
  throws<std::out_of_range>([&]{ (void)ca.coeff(6,0); },"coefficient bounds");
  throws<std::out_of_range>([&]{ (void)a.diagonal()[0]; },"diagonal bounds");
  throws<std::invalid_argument>([&]{ (void)(a*x); },"matvec dimension");
  throws<std::invalid_argument>([&]{ (void)(a.transpose()*x); },"transpose dimension");
  throws<std::invalid_argument>([&]{ (void)(a.diagonal()*x); },"diagonal dimension");
  throws<std::invalid_argument>([&]{ a.prune(-1); },"prune tolerance");
  a.resize(6,7); a.setIdentity();
  std::vector<apsc::gpu::Triplet<double, std::uint64_t>> invalid{{0, 10000000000ULL, 1}};
  throws<std::out_of_range>([&]{ a.setFromTriplets(invalid.begin(),invalid.end()); },"wide invalid triplet");
  require(a.nonZeros()==6,"failed assembly preserves matrix");
  std::vector<Eigen::Triplet<double, long long>> negative{{-1, 0, 1}};
  throws<std::out_of_range>([&]{ a.setFromTriplets(negative.begin(),negative.end()); },
                            "negative external triplet row");
  negative = {{0, -1, 1}};
  throws<std::out_of_range>([&]{ a.setFromTriplets(negative.begin(),negative.end()); },
                            "negative external triplet column");
  require(a.nonZeros()==6,"negative triplet preserves matrix");
  throws<std::out_of_range>([&]{ (void)a.coeff(negativeIndex,0); },"negative row converted to size_t");
  throws<std::out_of_range>([&]{ a.setCoeff(0,negativeIndex,1); },"negative column converted to size_t");
  throws<std::out_of_range>([&]{ (void)a.diagonal()[negativeIndex]; },"negative diagonal index");

  // A second independent randomized reference covers unequal segment lengths.
  std::mt19937 random(42);
  std::uniform_real_distribution<double> value(-3,3);
  triplets.clear();
  for(int k=0;k<300;++k) triplets.emplace_back(random()%23,random()%17,value(random));
  a.resize(23,17); a.setFromTriplets(triplets.begin(),triplets.end());
  reference.resize(23,17); reference.setFromTriplets(triplets.begin(),triplets.end());
  dense = Eigen::MatrixXd(reference);
  thrust::device_vector<double> ones(17,1), onesT(23,1);
  compare(a*ones,dense*Eigen::VectorXd::Ones(17),"random matvec");
  compare(a.transpose()*onesT,dense.transpose()*Eigen::VectorXd::Ones(23),"random transpose");
  near(a.norm(),dense.norm(),"random norm");
  near(a.l1Norm(),dense.cwiseAbs().colwise().sum().maxCoeff(),"random L1");
  near(a.infinityNorm(),dense.cwiseAbs().rowwise().sum().maxCoeff(),"random infinity");
  thrust::host_vector<Index> offsets=a.outerIndices(), indices=a.innerIndices();
  require(offsets.front()==0 && offsets.back()==a.nonZeros(),"compressed endpoints");
  for(std::size_t o=0;o+1<offsets.size();++o)
    for(Index p=offsets[o]+1;p<offsets[o+1];++p) require(indices[p-1]<indices[p],"sorted unique segment");
}

template<int Order> void unsignedIndexTests() {
  using Matrix = apsc::gpu::SparseMatrix<double, Order>;
  static_assert(std::is_same_v<typename Matrix::StorageIndex, std::size_t>);
  static_assert(std::is_copy_assignable_v<Matrix>);
  Matrix a(4,4);
  // Missing entries in both empty and occupied segments must read as zero,
  // including positions before, between, and after stored inner indices.
  a(1,1) = 2;
  a(1,3) = 4;
  a(3,1) = 6;
  for (std::size_t r=0; r<4; ++r) for (std::size_t c=0; c<4; ++c) {
    if ((r==1 && (c==1 || c==3)) || (r==3 && c==1)) continue;
    near(a.coeff(r,c), 0, "missing coefficient");
  }
  a.setCoeff(0,0,0);
  a.setCoeff(1,2,0);
  require(a.nonZeros()==3, "assigning absent zero preserves structure");
  a(0,0) = 7;
  a(1,0) = 8;
  a(1,2) = 9;
  a(0,1) = 10;
  a(2,1) = 11;
  require(a.nonZeros()==8, "insertions into empty and occupied segments");
  near(a.coeff(0,0),7,"empty segment insertion");
  near(a.coeff(1,2),9,"occupied row insertion");
  near(a.coeff(2,1),11,"occupied column insertion");

  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  const std::size_t differenceMaximum = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  for (std::size_t invalid : {maximum, differenceMaximum + 1}) {
    throws<std::length_error>([&]{ a.resize(invalid,1); }, "row dimension overflow");
    throws<std::length_error>([&]{ a.resize(1,invalid); }, "column dimension overflow");
    throws<std::length_error>([&]{ a.reserve(invalid); }, "reserved count overflow");
    require(a.rows()==4 && a.cols()==4 && a.nonZeros()==8, "overflow preserves matrix");
  }
  throws<std::length_error>([&]{
    a.resize(Order==apsc::gpu::RowMajor ? differenceMaximum : 1,
             Order==apsc::gpu::ColMajor ? differenceMaximum : 1);
  }, "outer sentinel exceeds iterator range");
  throws<std::out_of_range>([&]{ (void)a.coeff(maximum,0); }, "SIZE_MAX coordinate");

  // An 8-bit storage index makes the upper bound affordable to exercise.
  using Narrow = apsc::gpu::SparseMatrix<double,Order,std::uint8_t>;
  throws<std::length_error>([]{ Narrow invalid(256,1); }, "narrow row overflow");
  throws<std::length_error>([]{ Narrow invalid(1,256); }, "narrow column overflow");
  Narrow narrow(16,16);
  std::vector<apsc::gpu::Triplet<double>> entries;
  for (std::size_t r=0; r<16; ++r) for (std::size_t c=0; c<16; ++c)
    if (r!=15 || c!=15) entries.emplace_back(r,c,1);
  narrow.setFromTriplets(entries.begin(),entries.end());
  require(narrow.nonZeros()==255 && narrow.outerIndices().back()==255,
          "maximum narrow nonzero count");
  Eigen::VectorXd expected = Eigen::VectorXd::Constant(16,16);
  expected[15] = 15;
  thrust::device_vector<double> ones(16,1);
  compare(narrow*ones,expected,"narrow maximum matvec");
  compare(narrow.transpose()*ones,expected,"narrow maximum transpose matvec");
  near(narrow.l1Norm(),16,"narrow maximum L1 norm");
  near(narrow.infinityNorm(),16,"narrow maximum infinity norm");
  entries.emplace_back(15,15,1);
  throws<std::length_error>([&]{ narrow.setFromTriplets(entries.begin(),entries.end()); },
                            "nonzero count overflow");
  require(narrow.nonZeros()==255,"overflow preserves original nonzeros");
  near(narrow.coeff(15,15),0,"overflow preserves absent coefficient");
  throws<std::length_error>([&]{ narrow(15,15)=1; }, "insertion nonzero count overflow");
  require(narrow.nonZeros()==255,"insertion overflow preserves original nonzeros");
}

template<int Order> void complexTests() {
  using C = thrust::complex<double>;
  apsc::gpu::SparseMatrix<C,Order> a(2,3);
  std::vector<apsc::gpu::Triplet<C>> t{{0,0,C(3,4)},{0,2,C(0,-2)},{1,1,C(-1,0)}};
  a.setFromTriplets(t.begin(),t.end());
  thrust::device_vector<C> x{C(1,1),C(2,-1),C(3,0)};
  thrust::host_vector<C> y=a*x;
  near(thrust::abs(y[0]-C(-1,1)),0,"complex product row0");
  near(thrust::abs(y[1]-C(-2,1)),0,"complex product row1");
  thrust::device_vector<C> xt{C(1,1),C(2,-1)};
  thrust::host_vector<C> yt=a.transpose()*xt;
  near(thrust::abs(yt[0]-C(-1,7)),0,"nonconjugating transpose");
  near(thrust::abs(yt[2]-C(2,-2)),0,"complex transpose scatter");
  near(a.norm(),std::sqrt(30.),"complex Frobenius");
  near(a.maxNorm(),5,"complex max"); near(a.l1Norm(),5,"complex L1");
  near(a.infinityNorm(),7,"complex infinity"); near(a.entrywiseL1Norm(),8,"complex entrywise L1");
}
}
int main() {
  try {
    realTests<Eigen::ColMajor,std::uint32_t>(); realTests<Eigen::RowMajor,std::uint32_t>();
    realTests<Eigen::ColMajor,std::uint64_t>(); realTests<Eigen::RowMajor,std::uint64_t>();
    unsignedIndexTests<apsc::gpu::ColMajor>(); unsignedIndexTests<apsc::gpu::RowMajor>();
    complexTests<apsc::gpu::ColMajor>(); complexTests<apsc::gpu::RowMajor>();
    apsc::gpu::SparseMatrix<float> f(1,1); f(0,0)=2; near(f.norm(),2,"float scalar");
    apsc::gpu::SparseMatrix<double> large(1,1); large(0,0)=1e200;
    near(large.norm()/1e200,1,"overflow-safe Frobenius");
    large(0,0)=std::numeric_limits<double>::quiet_NaN();
    require(std::isnan(large.norm()) && std::isnan(large.maxNorm()) && std::isnan(large.l1Norm()),"NaN norms");
    large(0,0)=std::numeric_limits<double>::infinity();
    require(std::isinf(large.norm()),"infinite norm");
    std::cout << "All sparse matrix tests passed\n";
  } catch(const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n'; return 1;
  }
}
