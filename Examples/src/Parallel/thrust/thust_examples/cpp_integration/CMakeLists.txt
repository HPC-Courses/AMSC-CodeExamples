set(host_source "${CMAKE_CURRENT_SOURCE_DIR}/host.cpp")
set(device_source "${CMAKE_CURRENT_SOURCE_DIR}/device.cu")
if(THRUST_BACKEND STREQUAL "CUDA")
  # Recent CCCL versions put the CUDA architecture list in their ABI namespace.
  # An ordinary C++ compiler does not know that list. Give both translation
  # units the same private wrapper namespace so host_vector has matching types.
  # Generated wrappers preserve the original teaching sources and compilers.
  foreach(part IN ITEMS host device)
    if(part STREQUAL "host")
      set(extension cpp)
    else()
      set(extension cu)
    endif()
    set(wrapper "${CMAKE_CURRENT_BINARY_DIR}/${part}_wrapped.${extension}")
    file(WRITE "${wrapper}"
      "#define THRUST_CUB_WRAPPED_NAMESPACE teaching_integration\nnamespace teaching_integration {}\nusing namespace teaching_integration;\n#include \"${CMAKE_CURRENT_SOURCE_DIR}/${part}.${extension}\"\n")
    set(${part}_source "${wrapper}")
  endforeach()
endif()
add_teaching_example(cpp_integration "${host_source}" "${device_source}")
if(TARGET cpp_integration AND THRUST_BACKEND STREQUAL "CUDA")
  # The ordinary C++ translation unit also needs the CUDA runtime headers.
  target_link_libraries(cpp_integration PRIVATE CUDA::cudart)
endif()
