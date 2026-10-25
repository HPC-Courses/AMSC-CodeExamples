cmake_minimum_required(VERSION 3.24)
project(ThrustTeachingExamples LANGUAGES CXX)

# CPP is a serial CPU backend; OMP and TBB are parallel CPU backends.
set(THRUST_BACKEND "CPP" CACHE STRING "Execution backend: CPP, OMP, TBB, or CUDA")
set_property(CACHE THRUST_BACKEND PROPERTY STRINGS CPP OMP TBB CUDA)
string(TOUPPER "${THRUST_BACKEND}" THRUST_BACKEND)
if(NOT THRUST_BACKEND MATCHES "^(CPP|OMP|TBB|CUDA)$")
  message(FATAL_ERROR "THRUST_BACKEND must be CPP, OMP, TBB, or CUDA")
endif()

if(THRUST_BACKEND STREQUAL "CUDA")
  include(cmake/GpuArchitecture.cmake)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# Use an installed CCCL package, or point at an unpacked CCCL source tree.
set(CCCL_ROOT "" CACHE PATH "CCCL source tree or installation prefix")
find_package(Thrust CONFIG REQUIRED HINTS
  "${CCCL_ROOT}/lib/cmake/thrust"
  "${CCCL_ROOT}/thrust/thrust/cmake"
  "/usr/local/cuda/lib64/cmake/thrust")
thrust_create_target(teaching_thrust HOST CPP DEVICE ${THRUST_BACKEND})

set(THRUST_EXAMPLES "all" CACHE STRING
  "Examples to configure: all, or semicolon-separated target names (e.g. norm;saxpy)")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
include(cmake/AddExample.cmake)
add_custom_target(examples)

file(GLOB sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach(source IN LISTS sources)
  get_filename_component(name "${source}" NAME_WE)
  add_teaching_example(${name} "${source}")
endforeach()
add_subdirectory(cpp_integration)
if(THRUST_BACKEND STREQUAL "CUDA")
  add_subdirectory(cuda)
else()
  message(STATUS "CUDA-only examples omitted for ${THRUST_BACKEND}")
endif()

get_property(available GLOBAL PROPERTY teaching_available)
if(NOT THRUST_EXAMPLES STREQUAL "all")
  foreach(name IN LISTS THRUST_EXAMPLES)
    if(NOT name IN_LIST available)
      message(FATAL_ERROR "Unknown or unavailable example '${name}' for ${THRUST_BACKEND}. Available: ${available}")
    endif()
  endforeach()
endif()
message(STATUS "Thrust backend: ${THRUST_BACKEND}; examples: ${THRUST_EXAMPLES}")
