# Generate PTX files
function(nvcuda_compile_ptx)
  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED_FILES)
  set(multiValueArgs NVCC_OPTIONS SOURCES)
  cmake_parse_arguments(NVCUDA_COMPILE_PTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  # Custom build rule to generate ptx files from cuda files
  foreach(input ${NVCUDA_COMPILE_PTX_SOURCES} )
    get_filename_component(input_we ${input} NAME_WE )
    
    # generate the *.ptx files inside "ptx" folder inside the executable's output directory.
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${input_we}.ptx" )

    list(APPEND PTX_FILES ${output} )

    add_custom_command(
      OUTPUT  ${output}
      DEPENDS ${input}
      COMMAND ${CUDA_NVCC_EXECUTABLE} --machine=64 --ptx ${NVCUDA_COMPILE_PTX_NVCC_OPTIONS} ${input} -o ${output} WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endforeach( )
  
  set(${NVCUDA_COMPILE_PTX_GENERATED_FILES} ${PTX_FILES} PARENT_SCOPE)
endfunction()

macro(compile_optix_ptx SOURCES)
    get_target_property(OPTIX_INCLUDE_DIR OptiX::optix INTERFACE_INCLUDE_DIRECTORIES)
    nvcuda_compile_ptx(
        SOURCES ${SOURCES}
        TARGET_PATH ${CMAKE_CURRENT_DIR}
        GENERATED_FILES PTX_SOURCES
        NVCC_OPTIONS
            -arch=sm_70
            --use_fast_math
            --relocatable-device-code=true
            -std=c++14
            -I${OPTIX_INCLUDE_DIR}
    )

    get_filename_component(src_file ${SOURCES} NAME_WE)

    add_custom_target("${src_file}_ptx" ALL DEPENDS ${PTX_SOURCES})
    install(FILES ${PTX_SOURCES} DESTINATION ptx) 
endmacro()


