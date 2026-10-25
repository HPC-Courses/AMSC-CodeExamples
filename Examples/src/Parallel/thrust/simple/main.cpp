// device_vector.h pulls in Thrust's GPU-resident container.
#include <thrust/device_vector.h>
// inner_product.h provides the two-range fused transform+reduce algorithm.
#include <thrust/inner_product.h>
// functional.h provides ready-made functors like thrust::plus/thrust::multiplies.
#include <thrust/functional.h>
#include <iostream>

// Named functor (not a lambda) so nvcc can reliably mangle/instantiate it as a
// CUDA kernel template argument. __host__ __device__ lets it be called from
// both host and device code; here it only ever runs on the device via inner_product.
//
// A lambda also works here, but it must be explicitly annotated with
// __host__ __device__ (or __device__) right after the capture list, e.g.:
//   [] __host__ __device__ (double a, double b) { return a * b * b; }
// This also requires nvcc's --extended-lambda flag (see Makefile). Plain,
// unannotated lambdas compile but fail deep inside CUB/Thrust's device code
// with confusing "template argument is invalid" errors, since nvcc has no way
// to know the closure should be callable on the device.
struct mul_sq {
    __host__ __device__ double operator()(double a, double b) const {
        return a * b * b;
    }
};

int main() {
    const int n = 1'000'000;

    // thrust::device_vector allocates memory on the GPU and copies the fill
    // value there; unlike std::vector, its storage lives in device memory,
    // not host RAM.
    thrust::device_vector<double> x(n, 2.0);
    thrust::device_vector<double> y(n, 3.0);

    // Instead of a functor you may use a lambda (see above for caveats about __host__ __device__).
    // you need to add the option --extended-lambda to nvcc command line to compile this code.
    auto lam=[] __host__ __device__ (double a, double b) { return a * b * b; };
    // inner_product computes: init `binary_op1` (result_0 `binary_op1` result_1 `binary_op1` ...)
    // where result_i = binary_op2(x[i], y[i]).
    // Conceptually: sum = 0.0; for i in [0, n): sum = sum + mul_sq(x[i], y[i]);
    // The two ranges [x.begin(), x.end()) and [y.begin(), y.begin()+n) must be
    // the same length; only the first range needs both begin() and end().
    // Thrust dispatches this as a single fused GPU kernel (zip + transform + reduce),
    // so the elementwise products are never materialized as a separate array.
    double dot = thrust::inner_product(
        x.begin(), x.end(), y.begin(),
        0.0,                           // init value for the reduction
        thrust::plus<double>{},       // binary_op1: combines results across elements (the "reduce" step)
        lam                         // binary_op2: x[i] * y[i]^2 (the "transform" step)
    );

    // Result lives on the host now (inner_product returns a plain double,
    // copied back from the device reduction); safe to print directly.
    std::cout << dot << '\n';  // 2.0 * 3.0^2 * 1,000,000 = 18,000,000
}