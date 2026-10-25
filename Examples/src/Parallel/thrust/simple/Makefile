CXX=nvcc

# -O3               : optimize the generated host and device code.
# -std=c++20        : language standard, must match on both compile and link.
# -arch=sm_89       : target GPU compute capability (Ada, e.g. RTX 40-series).
# -I/usr/local/cuda/include : find CUDA/Thrust/CCCL headers.
# -x cu             : force nvcc to compile main.cpp as CUDA (.cu) source; without
#                     this, nvcc treats .cpp as host-only C++ and Thrust's
#                     device_vector algorithms fail with "unimplemented for this
#                     system" since no device backend is compiled in.
# --extended-lambda : allow lambdas annotated __host__ __device__ to be used as
#                     device functors (see main.cpp for why the annotation itself
#                     is still required).
CXXFLAGS=-O3 -std=c++20 -arch=sm_89 -I/usr/local/cuda/include -x cu --extended-lambda

# Link flags: no -x cu here, since the link step only sees main.o, not source;
# applying -x cu to the linker input would make nvcc misinterpret the object
# file as a source file to recompile.
LDFLAGS=-O3 -std=c++20 -arch=sm_89

main: main.o
	$(CXX) $(LDFLAGS) -o main main.o
# Explicit compile/link recipes below: make's built-in .o->executable rule
# links with $(CC) (plain cc), which doesn't pull in libstdc++/CUDA runtime
# symbols and fails with undefined references. Using $(CXX)=nvcc explicitly
# for both steps avoids that.
main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c -o main.o main.cpp
clean:
	$(RM) main main.o
