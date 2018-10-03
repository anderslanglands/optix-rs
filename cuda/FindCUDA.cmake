include(FindPackageHandleStandardArgs)
message("Trying to find CUDA at CUDA_ROOT: $ENV{CUDA_ROOT}")

find_path(CUDA_INCLUDE_DIR
  NAMES cuda.h
  PATHS ${CUDA_ROOT}/include $ENV{CUDA_ROOT}/include)

find_library(CUDA_cudart_LIBRARY
  NAMES cudart
  PATHS ${CUDA_ROOT}/lib64 $ENV{CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib $ENV{CUDA_ROOT}/lib
  )

find_program(CUDA_NVCC_EXECUTABLE
  NAMES nvcc
  PATHS ${CUDA_ROOT}/bin $ENV{CUDA_ROOT}/bin
  )

add_library(CUDA::cudart SHARED IMPORTED)
set_target_properties(CUDA::cudart PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIR}
    IMPORTED_LOCATION ${CUDA_cudart_LIBRARY}
)

find_package_handle_standard_args(CUDA
  REQUIRED_VARS CUDA_INCLUDE_DIR CUDA_cudart_LIBRARY CUDA_NVCC_EXECUTABLE
  FAIL_MESSAGE "Set CUDA_ROOT to point to your CUDA installation\n"
  )
