include(FindPackageHandleStandardArgs)
message("Trying to find OptiX at OPTIX_ROOT: $ENV{OPTIX_ROOT}")

find_path(OptiX_INCLUDE_DIR
  NAMES optix.h
  PATHS ${OPTIX_ROOT}/include $ENV{OPTIX_ROOT}/include)

find_library(OptiX_optix_LIBRARY
  NAMES optix
  PATHS ${OPTIX_ROOT}/lib64 $ENV{OPTIX_ROOT}/lib64 ${OPTIX_ROOT}/lib $ENV{OPTIX_ROOT}/lib
  )

add_library(OptiX::optix SHARED IMPORTED)
set_target_properties(OptiX::optix PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${OptiX_INCLUDE_DIR}
    IMPORTED_LOCATION ${OptiX_optix_LIBRARY}
    IMPORTED_LINK_INTERFACE_LIBRARIES "CUDA::cudart"
)

find_package_handle_standard_args(OptiX
  REQUIRED_VARS OptiX_INCLUDE_DIR OptiX_optix_LIBRARY
  FAIL_MESSAGE "Set OPTIX_ROOT to point to your OptiX installation\n"
  )
