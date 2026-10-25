# GPU ARCHITECTURE SELECTION
# native detects the GPU visible on the build machine (requires a working driver).
# Override on the command line, especially for cross-compilation/headless hosts:
#   -DCMAKE_CUDA_ARCHITECTURES=86
# or for several GPU generations:
#   '-DCMAKE_CUDA_ARCHITECTURES=80;86;89'
# Use the compute capability without the dot: 8.6 -> 86.
# CMake also honors the CUDAARCHS environment variable on the first configure.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
    "NVIDIA GPU compute capabilities (e.g. 86 or 80;86), or native for local detection")
endif()
message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES} (CUDAARCHS='$ENV{CUDAARCHS}')")
message(STATUS "Override GPU selection with -DCMAKE_CUDA_ARCHITECTURES=<compute capability without dot>")
