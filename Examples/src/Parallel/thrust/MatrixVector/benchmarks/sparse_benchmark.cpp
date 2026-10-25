// One source, built separately for CPP/OMP/TBB and CUDA. Timings measure the
// public matrix API, including its result allocation, not just GPU kernel time.
#include <SparseMatrix.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
struct Options {
  std::vector<int> sizes{1000, 10000, 100000};
  int entries = 9, iterations = 20, samples = 5, warmup = 3;
  std::string layout = "both", csv;
};
int positive(const std::string& value) {
  std::size_t consumed = 0;
  const long long n = std::stoll(value, &consumed);
  if (consumed != value.size() || n < 1 || n > std::numeric_limits<int>::max())
    throw std::invalid_argument("Expected a positive integer: " + value);
  return static_cast<int>(n);
}
Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "sparse_benchmark [--sizes 1000,10000,100000] [--entries-per-row 9]\n"
                   "  [--iterations 20] [--samples 5] [--warmup 3]\n"
                   "  [--layout csr|csc|both] [--csv output.csv]\n"
                   "Timings are milliseconds per API call. Without --csv, CSV goes to stdout.\n";
      std::exit(0);
    }
    if (++i == argc) throw std::invalid_argument("Missing value for " + key);
    const std::string value = argv[i];
    if (key == "--sizes") {
      o.sizes.clear(); std::stringstream stream(value); std::string item;
      while (std::getline(stream, item, ',')) o.sizes.push_back(positive(item));
      if (o.sizes.empty() || value.back() == ',') throw std::invalid_argument("Empty matrix size");
      std::sort(o.sizes.begin(), o.sizes.end());
      if (std::adjacent_find(o.sizes.begin(), o.sizes.end()) != o.sizes.end())
        throw std::invalid_argument("Matrix sizes must be distinct");
    } else if (key == "--entries-per-row") o.entries = positive(value);
    else if (key == "--iterations") o.iterations = positive(value);
    else if (key == "--samples") o.samples = positive(value);
    else if (key == "--warmup") o.warmup = positive(value);
    else if (key == "--layout") o.layout = value;
    else if (key == "--csv") o.csv = value;
    else throw std::invalid_argument("Unknown option " + key);
  }
  if (o.layout != "csr" && o.layout != "csc" && o.layout != "both")
    throw std::invalid_argument("Layout must be csr, csc, or both");
  return o;
}
void synchronize() {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  const auto status = cudaDeviceSynchronize();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
#endif
}
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void near(double actual, double expected) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-9 * std::max(1.0, std::abs(expected)))
    throw std::runtime_error("Benchmark correctness check failed");
}
template<class V> double checkVector(const V& actual, const std::vector<double>& expected) {
  thrust::host_vector<double> host = actual;
  if (host.size() != expected.size()) throw std::runtime_error("Incorrect result size");
  double checksum = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    near(host[i], expected[i]); checksum += host[i];
  }
  return checksum;
}

// Input generation and the independent serial reference are never timed.
struct Problem {
  std::vector<apsc::gpu::Triplet<double>> entries;
  std::vector<double> x, product, transposed, diagonal;
  double frobenius = 0, maximum = 0, l1 = 0, infinity = 0, entrywise = 0;
  Problem(int n, int requestedEntries) : x(n), product(n, 0), transposed(n, 0), diagonal(n, 0) {
    const int perRow = std::min(n, requestedEntries);
    if (static_cast<long long>(n) * perRow > std::numeric_limits<int>::max())
      throw std::length_error("Nonzero count exceeds this benchmark's supported workload limit");
    entries.reserve(static_cast<std::size_t>(n) * perRow);
    // Coprimality guarantees distinct columns per row. Wide strides exercise
    // indirect x accesses, rather than benchmarking only a narrow diagonal band.
    long long stride = 104729;
    while (std::gcd(stride, static_cast<long long>(n)) != 1) ++stride;
    for (int i = 0; i < n; ++i) x[i] = 1.0 + (i % 97) / 97.0;
    std::vector<double> columnSums(n, 0);
    long double squares = 0;
    for (int r = 0; r < n; ++r) {
      double rowSum = 0;
      for (int k = 0; k < perRow; ++k) {
        const int c = static_cast<int>((17LL * r + stride * k) % n);
        const double value = k == 0 ? 4.0 : -1.0 / (k + 1);
        entries.emplace_back(r, c, value);
        product[r] += value * x[c]; transposed[c] += value * x[r];
        if (r == c) diagonal[r] = value;
        squares += static_cast<long double>(value) * value;
        maximum = std::max(maximum, std::abs(value));
        rowSum += std::abs(value); columnSums[c] += std::abs(value);
        entrywise += std::abs(value);
      }
      infinity = std::max(infinity, rowSum);
    }
    l1 = *std::max_element(columnSums.begin(), columnSums.end());
    frobenius = static_cast<double>(std::sqrt(squares));
  }
};

struct Timing { double median, minimum, maximum, checksum; };
template<class Operation, class Check>
Timing measure(const Options& options, Operation operation, Check check) {
  operation(); synchronize(); check(); // validate before collecting any samples
  for (int i = 0; i < options.warmup; ++i) operation();
  synchronize();
  std::vector<double> samples;
  double checksum = 0;
  for (int sample = 0; sample < options.samples; ++sample) {
    synchronize();
    const auto start = Clock::now();
    for (int i = 0; i < options.iterations; ++i) {
      // Keep the compiler from hoisting pure CPU reductions out of the loop.
      std::atomic_signal_fence(std::memory_order_seq_cst);
      operation();
    }
    synchronize(); // include completion, not just asynchronous launch overhead
    samples.push_back(elapsed(start) / options.iterations);
    checksum = check(); // full correctness check and result transfer outside timing
  }
  std::sort(samples.begin(), samples.end());
  const auto middle = samples.size() / 2;
  const double median = samples.size() % 2 ? samples[middle] : (samples[middle-1] + samples[middle]) / 2;
  return {median, samples.front(), samples.back(), checksum};
}

template<int Layout> void run(const Options& o, int n, std::ostream& csv) {
  using Matrix = apsc::gpu::SparseMatrix<double, Layout>;
  const Problem p(n, o.entries);
  synchronize();
  auto start = Clock::now();
  Matrix a(n, n);
  a.setFromTriplets(p.entries.begin(), p.entries.end());
  synchronize();
  const double assemblyMs = elapsed(start); // host sorting + allocation + upload
  start = Clock::now();
  thrust::device_vector<double> x(p.x.begin(), p.x.end());
  synchronize();
  const double inputMs = elapsed(start);
  thrust::device_vector<double> result;
  thrust::host_vector<double> hostResult;
  double scalar = 0;
  const char* layout = Layout == apsc::gpu::RowMajor ? "csr" : "csc";
  auto record = [&](const char* name, auto operation, auto check) {
    const Timing timing = measure(o, operation, check);
    csv << APSC_BENCHMARK_BACKEND << ',' << APSC_BENCHMARK_BUILD_TYPE << ',' << layout << ','
        << n << ',' << a.nonZeros() << ',' << name << ',' << o.iterations << ',' << o.samples << ','
        << timing.median << ',' << timing.minimum << ',' << timing.maximum << ',' << timing.checksum << ','
        << assemblyMs << ',' << inputMs << '\n';
    if (!csv) throw std::runtime_error("Cannot write benchmark CSV");
    std::cerr << APSC_BENCHMARK_BACKEND << ' ' << layout << " n=" << n << ' ' << name
              << ": " << timing.median << " ms\n";
  };
  record("spmv", [&]{ result = a * x; }, [&]{ return checkVector(result, p.product); });
  record("transpose_spmv", [&]{ result = a.transpose() * x; }, [&]{ return checkVector(result, p.transposed); });
  record("diagonal", [&]{ result = a.diagonal().eval(); }, [&]{ return checkVector(result, p.diagonal); });
  auto norm = [&](const char* name, auto operation, double expected) {
    record(name, [&]{ scalar = operation(); }, [&]{ near(scalar, expected); return scalar; });
  };
  norm("frobenius_norm", [&]{ return a.frobeniusNorm(); }, p.frobenius);
  norm("max_norm", [&]{ return a.maxNorm(); }, p.maximum);
  norm("l1_norm", [&]{ return a.l1Norm(); }, p.l1);
  norm("infinity_norm", [&]{ return a.infinityNorm(); }, p.infinity);
  norm("entrywise_l1_norm", [&]{ return a.entrywiseL1Norm(); }, p.entrywise);
  // Matrix stays resident; vectors cross the host/device boundary on each call.
  // For CPU backends the same code measures the equivalent host copies.
  record("spmv_vector_roundtrip", [&]{
    thrust::device_vector<double> input(p.x.begin(), p.x.end());
    auto output = a * input;
    hostResult = output;
  }, [&]{ return checkVector(hostResult, p.product); });
}
} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse(argc, argv);
    std::cerr << "Backend: " << APSC_BENCHMARK_BACKEND << "; build: " << APSC_BENCHMARK_BUILD_TYPE << '\n';
    if (std::string(APSC_BENCHMARK_BUILD_TYPE) != "Release")
      std::cerr << "WARNING: use a Release build for performance comparisons.\n";
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    std::cerr << "GPU: " << properties.name << "; compute capability " << properties.major << '.' << properties.minor << '\n';
#endif
    synchronize(); // initialize CUDA before any reported timing
    std::ofstream file;
    if (!options.csv.empty()) {
      file.open(options.csv);
      if (!file) throw std::runtime_error("Cannot open " + options.csv);
    }
    std::ostream& csv = options.csv.empty() ? std::cout : file;
    csv << std::setprecision(12)
        << "backend,build,layout,rows,nnz,operation,iterations,samples,median_ms,min_ms,max_ms,checksum,assembly_ms,input_copy_ms\n";
    for (int n : options.sizes) {
      if (options.layout != "csc") run<apsc::gpu::RowMajor>(options, n, csv);
      if (options.layout != "csr") run<apsc::gpu::ColMajor>(options, n, csv);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Benchmark failed: " << e.what() << '\n'; return 1;
  }
}
