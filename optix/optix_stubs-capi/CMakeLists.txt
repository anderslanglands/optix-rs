cmake_minimum_required(VERSION 3.5)
project(optix_stubs-capi)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(optix_stubs-capi STATIC 
    optix_stubs.c
)

target_include_directories(optix_stubs-capi PUBLIC ${INC_OPTIX} ${INC_CUDA})
# target_link_libraries(optix_stubs-capi PUBLIC ${LIB_CUDA}/lib/libosdCPU.so ${LIB_OPTIX}/lib/libosdGPU.so)
install(TARGETS optix_stubs-capi DESTINATION ${CMAKE_INSTALL_PREFIX})
