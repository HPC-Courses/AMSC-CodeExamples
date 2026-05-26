#include "strassen.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using Matrix = Eigen::MatrixXd;
using Clock = std::chrono::steady_clock;

struct Options
{
  std::vector<Eigen::Index> sizes{64, 128, 256, 512, 1024, 2048};
  std::vector<Eigen::Index> cutoffs{32, 64, 128, 256, 512};
  int                       repetitions = 5;
  unsigned int              seed = 12345U;
};

struct TimingStats
{
  double minMs = 0.0;
  double medianMs = 0.0;
  double maxMs = 0.0;
};

struct BenchmarkRow
{
  Eigen::Index size = 0;
  std::string  method;
  Eigen::Index cutoff = -1;
  TimingStats  stats;
  double       speedup = 0.0;
  double       relError = 0.0;
};

[[nodiscard]] std::vector<Eigen::Index>
parseList(const std::string &text)
{
  std::vector<Eigen::Index> values;
  std::stringstream         input(text);
  std::string               item;

  while(std::getline(input, item, ','))
    {
      if(item.empty())
        {
          continue;
        }
      values.push_back(static_cast<Eigen::Index>(std::stoll(item)));
    }

  return values;
}

void
printUsage(const char *program)
{
  std::cout << "Usage: " << program
            << " [--sizes=64,128,256] [--cutoffs=32,64,128] [--repetitions=5]"
               " [--seed=12345]\n";
}

[[nodiscard]] Options
parseOptions(int argc, char **argv)
{
  Options options;

  for(int i = 1; i < argc; ++i)
    {
      const std::string arg(argv[i]);
      if(arg == "--help" || arg == "-h")
        {
          printUsage(argv[0]);
          std::exit(0);
        }
      if(arg.rfind("--sizes=", 0) == 0)
        {
          options.sizes = parseList(arg.substr(8));
          continue;
        }
      if(arg.rfind("--cutoffs=", 0) == 0)
        {
          options.cutoffs = parseList(arg.substr(10));
          continue;
        }
      if(arg.rfind("--repetitions=", 0) == 0)
        {
          options.repetitions = std::stoi(arg.substr(14));
          continue;
        }
      if(arg.rfind("--seed=", 0) == 0)
        {
          options.seed = static_cast<unsigned int>(std::stoul(arg.substr(7)));
          continue;
        }
      throw std::invalid_argument("Unknown option: " + arg);
    }

  if(options.sizes.empty())
    {
      throw std::invalid_argument("At least one matrix size is required");
    }
  if(options.cutoffs.empty())
    {
      throw std::invalid_argument("At least one cutoff is required");
    }
  if(options.repetitions <= 0)
    {
      throw std::invalid_argument("repetitions must be positive");
    }

  std::sort(options.sizes.begin(), options.sizes.end());
  options.sizes.erase(std::unique(options.sizes.begin(), options.sizes.end()),
                      options.sizes.end());
  std::sort(options.cutoffs.begin(), options.cutoffs.end());
  options.cutoffs.erase(
    std::unique(options.cutoffs.begin(), options.cutoffs.end()),
    options.cutoffs.end());

  return options;
}

void
printConfiguration(const Options &options)
{
  std::cout << "# Strassen vs Eigen benchmark\n";
  std::cout << "# format: csv\n";
  std::cout << "# repetitions = " << options.repetitions << '\n';
  std::cout << "# seed        = " << options.seed << '\n';
  std::cout << "# sizes       =";
  for(const auto n : options.sizes)
    {
      std::cout << ' ' << n;
    }
  std::cout << '\n';
  std::cout << "# cutoffs     =";
  for(const auto cutoff : options.cutoffs)
    {
      std::cout << ' ' << cutoff;
    }
  std::cout << '\n';
  std::cout << "size,method,cutoff,min_ms,median_ms,max_ms,speedup,rel_error\n";
}

template <typename Multiply>
[[nodiscard]] TimingStats
timeKernel(const Matrix &A, const Matrix &B, int repetitions, Multiply multiply,
           double &checksum)
{
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repetitions));

  for(int r = 0; r < repetitions; ++r)
    {
      const auto start = Clock::now();
      Matrix     C = multiply(A, B);
      const auto stop = Clock::now();
      checksum += C(0, 0);
      samples.push_back(
        std::chrono::duration<double, std::milli>(stop - start).count());
    }

  std::sort(samples.begin(), samples.end());
  return TimingStats{samples.front(), samples[samples.size() / 2],
                     samples.back()};
}

[[nodiscard]] double
relativeError(const Matrix &reference, const Matrix &candidate)
{
  const auto refNorm = reference.norm();
  const auto diffNorm = (reference - candidate).norm();
  if(refNorm == 0.0)
    {
      return diffNorm;
    }
  return diffNorm / refNorm;
}

void
printRow(const BenchmarkRow &row)
{
  std::cout << row.size << ',' << row.method << ',';
  if(row.cutoff < 0)
    {
      std::cout << "NA";
    }
  else
    {
      std::cout << row.cutoff;
    }
  std::cout << ',' << std::fixed << std::setprecision(6) << row.stats.minMs
            << ',' << row.stats.medianMs << ',' << row.stats.maxMs << ','
            << row.speedup << ',' << std::scientific << std::setprecision(6)
            << row.relError << std::defaultfloat << '\n';
}
} // namespace

int
main(int argc, char **argv)
try
  {
    const auto options = parseOptions(argc, argv);
    printConfiguration(options);

    double checksum = 0.0;

    for(const auto n : options.sizes)
      {
        std::srand(static_cast<unsigned int>(options.seed + n));
        const Matrix A = Matrix::Random(n, n);
        const Matrix B = Matrix::Random(n, n);

        Matrix warmup(n, n);
        warmup.noalias() = A * B;
        checksum += warmup(0, 0);

        const Matrix reference = [&]() {
          Matrix C(n, n);
          C.noalias() = A * B;
          return C;
        }();

        const TimingStats eigenStats = timeKernel(
          A, B, options.repetitions,
          [](const Matrix &lhs, const Matrix &rhs) {
            Matrix C(lhs.rows(), rhs.cols());
            C.noalias() = lhs * rhs;
            return C;
          },
          checksum);

        printRow(
          BenchmarkRow{n, "Eigen", -1, eigenStats, 1.0, 0.0});

        for(const auto cutoff : options.cutoffs)
          {
            const TimingStats strassenStats = timeKernel(
              A, B, options.repetitions,
              [cutoff](const Matrix &lhs, const Matrix &rhs) {
                return apsc::strassen(lhs, rhs, cutoff);
              },
              checksum);

            const Matrix candidate = apsc::strassen(A, B, cutoff);
            const double error = relativeError(reference, candidate);
            const double speedup = (strassenStats.medianMs > 0.0)
                                     ? (eigenStats.medianMs /
                                        strassenStats.medianMs)
                                     : 0.0;

            printRow(BenchmarkRow{n, "Strassen", cutoff, strassenStats,
                                  speedup, error});
          }
      }

    std::cout << "# checksum " << std::setprecision(17) << checksum << '\n';
    return 0;
  }
catch(const std::exception &e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
