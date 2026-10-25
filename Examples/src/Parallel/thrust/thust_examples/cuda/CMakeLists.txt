# These programs explicitly use the CUDA runtime and have no CPU equivalent.
file(GLOB sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach(source IN LISTS sources)
  get_filename_component(name "${source}" NAME_WE)
  add_teaching_example(cuda_${name} "${source}")
endforeach()
