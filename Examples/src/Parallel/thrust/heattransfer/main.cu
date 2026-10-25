// HeatExchange/main.cpp
// device_vector.h pulls in Thrust's GPU-resident container.
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// inner_product.h provides the two-range fused transform+reduce algorithm.
#include <thrust/inner_product.h>
// functional.h provides ready-made functors like thrust::plus/thrust::multiplies.
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <thrust/system/cuda/execution_policy.h>
#endif
#include <limits>
#include "GetPot"      // for command line and input file processing
#ifdef GNUPLOT          // compiled with -DGNUPLOT
#include "gnuplot-iostream.hpp" // interface with gnuplot
#endif
#include "readParameters.hpp" // for reading parameters
#include "thomas.hpp" // for Thomas algorithm. You need to have the file thomas.hpp
					  // in the include path
#include <cmath>      // (for sqrt)
#include <iostream>   // input output
#include <tuple>      // I am using tuples to pass data to gnuplot iostream
#include <vector>     // For vectors
#include <fstream>    // For file output (ofstream)
// write directive to test if I am using at least c++20
#if __cplusplus > 201703L
#include <format> // Available in C++20 and later. For formatted output
#endif
/*!
  @file main.cpp
  @brief Temperature distribution in a 1D bar.

  @details
  We solve  \f$ -T^{\prime\prime}(x)+act*(T(x)-T_e)=0, 0<x<L \f$ with
  boundary conditions \f$ T(0)=To; T^\prime(L)=0\f$

 **************************************************
  Linear finite elements
  Iterative resolution by Gauss Siedel or
  Direct solve by Thomas algorithm
 **************************************************

  Example adapted by Luca Formaggia from  a code found in
  "Simulation numerique an C++" di I. Danaila, F. Hecht e
  O. Pironneau.
 */
//! a helper function. Prints a help message
void
printHelp()
{
  std::cout
    << "USAGE: main [-h] [-v] -p parameterFile (default: parameters.pot)"
    << std::endl;
  std::cout << "-h this help" << std::endl;
  std::cout << "-v verbose output" << std::endl;
}

//! main program
int
main(int argc, char **argv)
{
  using namespace std; // avoid std::
  int    status{0};    // final program status
  bool   jsonfile = false;
  GetPot cl(argc, argv);
  if(cl.search(2, "-h", "--help")) // help required
    {
      printHelp();
      return 0;
    }
  // check if we want verbosity
  bool verbose = cl.search(1, "-v");
  // Get file with parameter values
  string filename = cl.follow("parameters.pot", "-p");// Look if -p option is there
  // if not use default parameters.pot
  auto   pos = filename.find(".json");// look for substring ".json" in filename
  if(pos != std::string::npos)
    {
      jsonfile = true;
      std::cout << "Json input file\n";
    }
  else
    {
      jsonfile = false;
      std::cout << "Getpot input file\n";
    }
  cout << "Reading parameters from " << filename << std::endl;
  parameters param;
  if(jsonfile)
    param = readParameters_json(filename, verbose);
  else
    param = readParameters(filename, verbose);

  
  // REVIEW: a1/a2 are unchecked: zero passes validation and divides by zero
  // in act. Reject nonfinite inputs and invalid iteration limits/tolerances too.
  if(param.M <= 0 || param.L <= 0 || param.k <= 0) // Add more checks as appropriate
    {
      cerr << "Error: Invalid or missing parameters in input file." << endl;
      return 1;
    }

#if __cplusplus < 201703L
  // This version is perfectly fine and
  // works also if you compile with C++17+, but since C++17 you
  // may make things simpler
  // Transfer parameters to local variables, to avoid having towrite every time
  // param.xx. I use references to save memory (not really an issue here, it is
  // just to show a possible  use of references)
  const int &itermax = param.itermax; // max number of iteration for
                                      // Gauss-Siedel
  const double &toler = param.toler;  // Tolerance for stopping criterion
  // Here I use auto (remember that you need const and & if you want constant
  // references)
  const auto &L = param.L;   // Bar length
  const auto &a1 = param.a1; // First longitudinal dimension
  const auto &a2 = param.a2; //  Second longitudinal dimension
  const auto &To = param.To; // Dirichlet condition
  const auto &Te = param.Te; // External temperature (Centigrades)
  const auto &k = param.k;   // Thermal conductivity
  const auto &hc = param.hc; // Convection coefficient
  const auto &M = param.M;   // Number of grid elements
  const auto &solverType =
    param.solverType; // 1 Gauss siedel, !=1 direct method
#else
  // C++17 onwards version. This version works only with at least C++17
  // A oneliner! This is called structured bindings. It works because parameter
  // class is an aggregate!
  // This stuff is just to avoid writing param.xx all the time!

  const auto &[itermax, toler, L, a1, a2, To, Te, k, hc, M, solverType] = param;

#endif

  // Match main_old.cpp: keep coefficients, solution buffers, and reductions
  // in double precision. At M=3000, float rounds 2+h*h*act to exactly 2,
  // dropping the reaction term from the Jacobi equation.
  const double act = 2. * (a1 + a2) * hc * L * L / (k * a1 * a2);
  const double h = 1.0 / M; // Floating-point division, not integer division.
  thrust::host_vector<double> theta(M + 1, 0.0);
  const double deltaT = Te == 0 ? To : (To - Te) / Te;
  if(solverType == 1)
    {
      // Parallel Jacobi: every row reads the old buffer and writes the new one.
      // Reject invalid stopping settings before launching any iterations.
      if(itermax <= 0 || !std::isfinite(toler) || toler < 0)
        {
          cerr << "Error: itermax must be positive and toler finite/nonnegative.\n";
          return 1;
        }
      thrust::device_vector<double> d_theta(M + 1);
      thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(M + 1),
        d_theta.begin(),
        [=] __device__(int m) { return (1. - m * h) * deltaT; });
      // Both buffers keep the fixed Dirichlet value at index zero.
      thrust::device_vector<double> d_theta_new = d_theta;

      // A host-visible reduction synchronizes the GPU and needs workspace.
      // Check every 50 sweeps, and always at itermax. This may do up to 49
      // extra sweeps after convergence, but uses the same update-norm tolerance.
      constexpr int checkEvery = 50;
      const int lastNode = M; // Capture a value, not a C++17 structured binding.
      int iter = 0;
      double epsilon = std::numeric_limits<double>::infinity();
      bool converged = false;
      while(iter < itermax)
        {
          // Refresh pointers after each buffer swap; capture pointers, not vectors.
          const double *oldTheta = thrust::raw_pointer_cast(d_theta.data());
          double *newTheta = thrust::raw_pointer_cast(d_theta_new.data());
          auto update = [=] __device__(int m) -> double {
            // Include the insulated endpoint in the same GPU operation. Its
            // Jacobi update reads the previous neighbor, avoiding a data race.
            const double value = m == lastNode ? oldTheta[lastNode - 1] :
              (oldTheta[m - 1] + oldTheta[m + 1]) / (2.0 + h * h * act);
            newTheta[m] = value;
            const double diff = value - oldTheta[m];
            return diff * diff;
          };
          ++iter;
          const bool check = iter % checkEvery == 0 || iter == itermax;
          if(check)
            epsilon = thrust::transform_reduce(
              thrust::counting_iterator<int>(1),
              thrust::counting_iterator<int>(M + 1), update,
              0.0, thrust::plus<double>());
          else
            {
              // No scalar transfers or reduction on ordinary sweeps. CUDA's
              // default stream orders the launches, so a host wait is unnecessary.
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
              thrust::for_each(thrust::cuda::par_nosync,
#else
              thrust::for_each(
#endif
                thrust::counting_iterator<int>(1),
                thrust::counting_iterator<int>(M + 1), update);
            }
          // Exchange ownership in O(1), instead of copying M+1 values per sweep.
          // Queued kernels retain valid pointers; both allocations stay alive.
          d_theta.swap(d_theta_new);
          if(check)
            {
              if(!std::isfinite(epsilon))
                break; // NaN/Inf must never be reported as convergence.
              // Double precision avoids the premature single-precision stagnation, but
              // an update norm is still not a bound on the solution error.
              // Fine meshes require many sweeps; compare with the direct solve.
              converged = std::sqrt(epsilon) <= toler;
              if(converged)
                break;
            }
        }

      if(converged)
        cout << "M=" << M << "  Convergence in " << iter << " iterations\n";
      else
        {
          cerr << "NOT CONVERGING in " << iter << " iterations "
               << "||dx||=" << std::sqrt(epsilon) << endl;
          status = 1;
        }
      // The final checked sweep has completed before the single host copy.
      thrust::copy(d_theta.begin(), d_theta.end(), theta.begin());

    }
  else
    {
      // Thomas algorithm. I need to build on the device the vector a, b,c that define the
      // tridiagonal matrix
          // Thomas algorithm. I need to build the vector a, b,c that define the
      // tridiagonal matrix
      std::vector<double> a(
        M + 1, 1.); // a vector filled with 1. using the constructor
      // to fill vectors with values you can also do like this
      std::vector<double> b;                     // an empty vector
      b.assign(M + 1, -1. / (2. + h * h * act)); // fill with values
      std::vector<double> c = b;                 // The matrix is symmetric
      b.back() = -1;                             // correction for Neumann bc
      c.front() = 0;                             // correction for Dirichlet bc c[0]=0
      std::vector<double> source(
        M + 1, 0.); // The rhs term. all zero since it is an homogeneous problem
      source.front() = (To - Te) / Te; // correction for Dirichlet bc.
      theta = apsc::thomasSolve(a, b, c, source);
  }
  if (Te !=0)
  {
    for(auto &t : theta)
      t = Te * (1 + t);
  }
  // Analytic solution

  thrust::host_vector<double> thetaa(M + 1);

  // REVIEW: cosh still overflows for sufficiently large act, even in double.
  // For example hc=200 with the default geometry gives NaN at m=0 (inf/inf),
  // where the exact temperature must be To. Use a stable exponential ratio.
  for(int m = 0; m <= M; m++)
    thetaa[m] =
      Te + (To - Te) * cosh(sqrt(static_cast<double>(act)) * static_cast<double>(1 - m * h)) / cosh(sqrt(static_cast<double>(act)));

  // writing results with format
  // x_i u_h(x_i) u(x_i) and launch gnuplot

  thrust::host_vector<double> coor(M + 1);

  cout << "Result file: result.dat" << endl;
  ofstream f("result.dat");
  // In gnuplot lines beginning with # are comments
  // \t writes a tab
  f << "#node coordinate\tcomputed solution\texact solution" << std::endl;
#if __cplusplus > 201703L
  // I show the new format facility
  for(int m = 0; m <= M; m++)
    {
      f << std::format("{:<18.15f}\t\t{:<18.15f}\t\t{:<18.15f}\n", m * h * L,
                       theta[m], thetaa[m]);
      coor[m] = m * h * L;
    }
#else
  for(int m = 0; m <= M; m++)
    {
      f.setf(std::ios::left, std::ios::adjustfield);
      f.width(18);
      f.precision(15);
      f << m * h * L << "\t\t" << theta[m] << "\t\t" << thetaa[m] << "\n";
      coor[m] = m * h * L;
    }
#endif
    // If you have gnuplot iostream you get the plot on the screen
#ifdef GNUPLOT
  Gnuplot gp; // gnuplot iostream! Plots solution on the screen
  // It may not work on virtual machines. Take it out in that case
  // Using temporary files (another nice use of tie)
  // Comment this statement if you are not using gnuplot iostream
  // to plot the solution directly on the terminal
  gp << "plot" << gp.file1d(std::tie(coor, theta)) << "w lp lw 2 title 'uh',"
     << gp.file1d(std::tie(coor, thetaa)) << "w l lw 2 title 'uex'"
     << std::endl;
#endif
  f.close();
  return status;
}
