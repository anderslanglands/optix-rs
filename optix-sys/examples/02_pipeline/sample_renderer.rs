use nalgebra_glm::IVec2;

use cuda_sys::{
    cuCtxGetCurrent, cudaDeviceProp, cudaDeviceSynchronize, cudaError,
    cudaError_enum, cudaFree, cudaGetDeviceCount, cudaGetDeviceProperties,
    cudaGetErrorString, cudaGetLastError, cudaMalloc, cudaMemcpy,
    cudaMemcpyKind, cudaSetDevice, cudaStreamCreate, CUcontext, CUdeviceptr,
    CUresult, CUstream,
};

use optix_sys::{
    optixDeviceContextCreate, optixDeviceContextSetLogCallback, optixInit,
    optixLaunch, optixModuleCreateFromPTX, optixPipelineCreate,
    optixPipelineSetStackSize, optixProgramGroupCreate,
    optixSbtRecordPackHeader, OptixCompileDebugLevel,
    OptixCompileOptimizationLevel, OptixDeviceContext, OptixExceptionFlags,
    OptixModule, OptixModuleCompileOptions, OptixPipeline,
    OptixPipelineCompileOptions, OptixPipelineLinkOptions, OptixProgramGroup,
    OptixProgramGroupDesc, OptixProgramGroupDesc__bindgen_ty_1,
    OptixProgramGroupHitgroup, OptixProgramGroupKind, OptixProgramGroupOptions,
    OptixProgramGroupSingleModule, OptixResult, OptixSbtRecordAlignment,
    OptixSbtRecordHeaderSize, OptixShaderBindingTable,
    OptixTraversableGraphFlags,
};

#[repr(C)]
pub struct LaunchParams {
    frame_id: i32,
    color_buffer: *mut std::os::raw::c_uint,
    fb_size: IVec2,
}

pub struct SampleRenderer {
    // CUDA device context and stream that optix pipeline will run on as well as
    // device properties
    cuda_context: CUcontext,
    stream: CUstream,
    device_props: cudaDeviceProp,
    // The optix context that our pipeline will run in
    pipeline: OptixPipeline,
    pipeline_compile_options: OptixPipelineCompileOptions,
    pipeline_link_options: OptixPipelineLinkOptions,

    // The module that contains our device programs
    module: OptixModule,
    module_compile_options: OptixModuleCompileOptions,

    // Vector of all our program groups and the SBT built around them
    raygen_pgs: Vec<OptixProgramGroup>,
    raygen_records_buffer: CudaBuffer,
    miss_pgs: Vec<OptixProgramGroup>,
    miss_records_buffer: CudaBuffer,
    hitgroup_pgs: Vec<OptixProgramGroup>,
    hitgroup_records_buffer: CudaBuffer,
    sbt: OptixShaderBindingTable,
    launch_params: LaunchParams,
    launch_params_buffer: CudaBuffer,

    color_buffer: CudaBuffer,
}

impl SampleRenderer {
    pub fn new(fb_size: IVec2) -> SampleRenderer {
        // init optix
        unsafe {
            cudaFree(std::ptr::null_mut());
            let mut num_devices = 0i32;
            cudaGetDeviceCount(&mut num_devices as *mut i32);
            if num_devices == 0 {
                panic!("No CUDA devices found");
            }
            println!("Found {} CUDA devices", num_devices);

            let result = optixInit();
            if result != OptixResult::OPTIX_SUCCESS {
                panic!("OptiX init failed!");
            }

            println!("OptiX initialized successfully! Yay!");
        }

        unsafe {
            // create context
            let device_id = 0i32;
            let res = cudaSetDevice(device_id);
            if res != cudaError_enum::CUDA_SUCCESS as u32 {
                panic!("Could not set cuda device");
            }

            let mut stream: CUstream = std::ptr::null_mut();
            let res = cudaStreamCreate(&mut stream as *mut CUstream);

            let mut device_props =
                std::mem::MaybeUninit::<cudaDeviceProp>::uninit();
            cudaGetDeviceProperties(device_props.as_mut_ptr(), device_id);
            let device_props = device_props.assume_init();

            let name = std::ffi::CStr::from_ptr(device_props.name.as_ptr())
                .to_string_lossy()
                .into_owned();
            println!("Device name: {}", name);

            let mut cuda_context = std::mem::MaybeUninit::uninit();
            let res = cuCtxGetCurrent(cuda_context.as_mut_ptr());
            if res != cudaError_enum::CUDA_SUCCESS {
                panic!("Could not query current context");
            }
            let cuda_context = cuda_context.assume_init();

            let mut optix_context = std::mem::MaybeUninit::uninit();
            let res = optixDeviceContextCreate(
                cuda_context,
                std::ptr::null_mut(),
                optix_context.as_mut_ptr(),
            );
            if res != OptixResult::OPTIX_SUCCESS {
                panic!("Could not create optix context");
            }
            let optix_context = optix_context.assume_init();

            let res = optixDeviceContextSetLogCallback(
                optix_context,
                Some(optix_log_callback),
                std::ptr::null_mut(),
                4,
            );
            if res != OptixResult::OPTIX_SUCCESS {
                panic!("Could not set log callback");
            }

            // create module
            let module_compile_options = OptixModuleCompileOptions {
                maxRegisterCount: 100,
                optLevel: OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
                debugLevel: OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO
            };

            let launch_params_variable_name =
                std::ffi::CString::new("optixLaunchParams").unwrap();
            let pipeline_compile_options = OptixPipelineCompileOptions {
                usesMotionBlur: 0,
                traversableGraphFlags: OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
                numPayloadValues: 2,
                numAttributeValues: 2,
                exceptionFlags: OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE,
                pipelineLaunchParamsVariableName: launch_params_variable_name.as_ptr(),
            };

            let pipeline_link_options = OptixPipelineLinkOptions {
                overrideUsesMotionBlur: 0,
                maxTraceDepth: 2,
                debugLevel:
                    OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
            };

            let cuda_source = include_str!("devicePrograms.cu");
            // compile the source with nvrtc
            let ptx_code = compile_to_ptx(cuda_source);
            let ptx_code = std::ffi::CString::new(ptx_code.as_str()).unwrap();
            let ptx_bytes = ptx_code.as_bytes();

            let mut log_size = 2048;
            let mut log = vec![0u8; log_size];
            let mut module: OptixModule = std::ptr::null_mut();
            let res = optixModuleCreateFromPTX(
                optix_context,
                &module_compile_options,
                &pipeline_compile_options,
                ptx_bytes.as_ptr() as *const i8,
                ptx_bytes.len(),
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut module,
            );

            if res != OptixResult::OPTIX_SUCCESS {
                panic!("Could not create module");
            }

            if log_size > 1 {
                println!(
                    "LOG: {}",
                    std::ffi::CStr::from_bytes_with_nul(&log[0..log_size])
                        .unwrap()
                        .to_string_lossy()
                );
            }

            // create raygen programs
            let entryFunctionName =
                std::ffi::CString::new("__raygen__renderFrame").unwrap();
            let pg_options = OptixProgramGroupOptions { placeholder: 0 };
            let pg_desc = OptixProgramGroupDesc {
                kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                    raygen: OptixProgramGroupSingleModule {
                        module,
                        entryFunctionName: entryFunctionName.as_ptr(),
                    },
                },
                flags: 0,
            };

            log_size = 2048;
            let mut raygen_pg: OptixProgramGroup = std::ptr::null_mut();
            let res = optixProgramGroupCreate(
                optix_context,
                &pg_desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut raygen_pg,
            );

            if res != OptixResult::OPTIX_SUCCESS {
                panic!("Could not create module");
            }

            if log_size > 1 {
                println!(
                    "LOG: {}",
                    std::ffi::CStr::from_bytes_with_nul(&log[0..log_size])
                        .unwrap()
                        .to_string_lossy()
                );
            }

            let raygen_pgs = vec![raygen_pg];

            // create miss programs
            let entryFunctionName =
                std::ffi::CString::new("__miss__radiance").unwrap();
            let pg_options = OptixProgramGroupOptions { placeholder: 0 };
            let pg_desc = OptixProgramGroupDesc {
                kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
                __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                    miss: OptixProgramGroupSingleModule {
                        module,
                        entryFunctionName: entryFunctionName.as_ptr(),
                    },
                },
                flags: 0,
            };

            log_size = 2048;
            let mut miss_pg: OptixProgramGroup = std::ptr::null_mut();
            let res = optixProgramGroupCreate(
                optix_context,
                &pg_desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut miss_pg,
            );

            if res != OptixResult::OPTIX_SUCCESS {
                panic!("Could not create module");
            }

            if log_size > 1 {
                println!(
                    "LOG: {}",
                    std::ffi::CStr::from_bytes_with_nul(&log[0..log_size])
                        .unwrap()
                        .to_string_lossy()
                );
            }

            let miss_pgs = vec![miss_pg];

            // create hitgroup programs
            let entry_ch =
                std::ffi::CString::new("__closesthit__radiance").unwrap();
            let entry_ah =
                std::ffi::CString::new("__anyhit__radiance").unwrap();
            let pg_options = OptixProgramGroupOptions { placeholder: 0 };
            let pg_desc = OptixProgramGroupDesc {
                kind: OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                __bindgen_anon_1: OptixProgramGroupDesc__bindgen_ty_1 {
                    hitgroup: OptixProgramGroupHitgroup {
                        moduleCH: module,
                        entryFunctionNameCH: entry_ch.as_ptr(),
                        moduleAH: module,
                        entryFunctionNameAH: entry_ah.as_ptr(),
                        moduleIS: std::ptr::null_mut(),
                        entryFunctionNameIS: std::ptr::null(),
                    },
                },
                flags: 0,
            };

            log_size = 2048;
            let mut hitgroup_pg: OptixProgramGroup = std::ptr::null_mut();
            let res = optixProgramGroupCreate(
                optix_context,
                &pg_desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut hitgroup_pg,
            );

            if res != OptixResult::OPTIX_SUCCESS {
                panic!("Could not create module");
            }

            if log_size > 1 {
                println!(
                    "LOG: {}",
                    std::ffi::CStr::from_bytes_with_nul(&log[0..log_size])
                        .unwrap()
                        .to_string_lossy()
                );
            }

            let hitgroup_pgs = vec![hitgroup_pg];

            // create pipeline
            let program_groups =
                vec![raygen_pgs[0], miss_pgs[0], hitgroup_pgs[0]];
            log_size = 2048;
            let mut pipeline: OptixPipeline = std::ptr::null_mut();
            let res = optixPipelineCreate(
                optix_context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups.as_ptr(),
                program_groups.len() as u32,
                log.as_mut_ptr() as *mut i8,
                &mut log_size,
                &mut pipeline,
            );
            if res != OptixResult::OPTIX_SUCCESS {
                panic!("Could not create pipeline");
            }

            if log_size > 1 {
                println!(
                    "LOG: {}",
                    std::ffi::CStr::from_bytes_with_nul(&log[0..log_size])
                        .unwrap()
                        .to_string_lossy()
                );
            }

            let res = optixPipelineSetStackSize(
                pipeline,
                // direct stack size for direct callables invoked for IS or AH
                2 * 1024,
                // direct stack size for direct callables invoked from RG, MS or CH
                2 * 1024,
                // continuation stack size
                2 * 1024,
                // maximum depth of a traversable graph passed to trace
                3,
            );
            if res != OptixResult::OPTIX_SUCCESS {
                panic!("Could not set pipeline stack sizes");
            }

            // build SBT
            let mut raygen_records = Vec::with_capacity(raygen_pgs.len());
            for rg_pg in &raygen_pgs {
                let mut rec = RaygenRecord {
                    header: [0; OptixSbtRecordHeaderSize],
                    data: std::ptr::null_mut(),
                };
                let res = optixSbtRecordPackHeader(
                    *rg_pg,
                    &mut rec as *mut RaygenRecord as *mut std::os::raw::c_void,
                );
                if res != OptixResult::OPTIX_SUCCESS {
                    panic!("Could not raygen record header");
                }

                raygen_records.push(rec);
            }
            let raygen_records_buffer = CudaBuffer::with_data(&raygen_records);
            let raygenRecord = raygen_records_buffer.d_ptr as CUdeviceptr;

            let mut miss_records = Vec::with_capacity(miss_pgs.len());
            for miss_pg in &miss_pgs {
                let mut rec = MissRecord {
                    header: [0; OptixSbtRecordHeaderSize],
                    data: std::ptr::null_mut(),
                };
                let res = optixSbtRecordPackHeader(
                    *miss_pg,
                    &mut rec as *mut MissRecord as *mut std::os::raw::c_void,
                );
                if res != OptixResult::OPTIX_SUCCESS {
                    panic!("Could not miss record header");
                }

                miss_records.push(rec);
            }
            let miss_records_buffer = CudaBuffer::with_data(&miss_records);
            let missRecordBase = miss_records_buffer.d_ptr as CUdeviceptr;
            let missRecordStrideInBytes =
                std::mem::size_of::<MissRecord>() as u32;
            let missRecordCount = miss_records.len() as u32;

            let mut hitgroup_records = Vec::with_capacity(hitgroup_pgs.len());
            // we don't actually have any objects in this example, but let's
            // create a dummy one so the SBT doesn't have any null pointers
            // (which the sanity checks in compilation would compain about)
            let num_objects = 1;
            for i in 0..num_objects {
                let mut rec = HitgroupRecord {
                    header: [0; OptixSbtRecordHeaderSize],
                    object_id: i,
                };
                let object_type = 0;
                let hitgroup_pg = hitgroup_pgs[object_type];
                let res = optixSbtRecordPackHeader(
                    hitgroup_pg,
                    &mut rec as *mut HitgroupRecord
                        as *mut std::os::raw::c_void,
                );
                if res != OptixResult::OPTIX_SUCCESS {
                    panic!("Could not hitgroup record header");
                }

                hitgroup_records.push(rec);
            }
            let hitgroup_records_buffer =
                CudaBuffer::with_data(&hitgroup_records);
            let hitgroupRecordBase =
                hitgroup_records_buffer.d_ptr as CUdeviceptr;
            let hitgroupRecordStrideInBytes =
                std::mem::size_of::<HitgroupRecord>() as u32;
            let hitgroupRecordCount = hitgroup_records.len() as u32;

            let sbt = OptixShaderBindingTable {
                raygenRecord,
                exceptionRecord: 0,
                missRecordBase,
                missRecordStrideInBytes,
                missRecordCount,
                hitgroupRecordBase,
                hitgroupRecordStrideInBytes,
                hitgroupRecordCount,
                callablesRecordBase: 0,
                callablesRecordStrideInBytes: 0,
                callablesRecordCount: 0,
            };

            let color_buffer = CudaBuffer::new(
                (fb_size.x * fb_size.y) as usize * std::mem::size_of::<u32>(),
            );

            let mut launch_params = LaunchParams {
                frame_id: 0,
                color_buffer: color_buffer.d_ptr as *mut u32,
                fb_size,
            };

            let launch_params_buffer =
                CudaBuffer::with_data(std::slice::from_ref(&launch_params));

            println!("Setup complete.");

            SampleRenderer {
                cuda_context,
                stream,
                device_props,
                pipeline,
                pipeline_compile_options,
                pipeline_link_options,
                module,
                module_compile_options,
                raygen_pgs,
                raygen_records_buffer,
                miss_pgs,
                miss_records_buffer,
                hitgroup_pgs,
                hitgroup_records_buffer,
                sbt,
                launch_params,
                launch_params_buffer,
                color_buffer,
            }
        }
    }

    pub fn render(&mut self) {
        unsafe {
            self.launch_params_buffer
                .upload(std::slice::from_ref(&self.launch_params));
            self.launch_params.frame_id += 1;

            let res = optixLaunch(
                self.pipeline,
                self.stream,
                self.launch_params_buffer.d_ptr as CUdeviceptr,
                self.launch_params_buffer.size_in_bytes,
                &self.sbt,
                self.launch_params.fb_size.x as u32,
                self.launch_params.fb_size.y as u32,
                1,
            );
            if res != OptixResult::OPTIX_SUCCESS {
                panic!("Launch failed!");
            }

            cudaDeviceSynchronize();
            let res = cudaGetLastError();
            if res != cudaError::cudaSuccess {
                let err_str = std::ffi::CStr::from_ptr(cudaGetErrorString(res))
                    .to_string_lossy();
                panic!("Sync failure: {}", err_str);
            }
        }
    }
}

fn compile_to_ptx(src: &str) -> String {
    use cuda::nvrtc::Program;

    let optix_root = std::env::var("OPTIX_ROOT")
        .expect("OPTIX_ROOT not found. You must set OPTIX_ROOT either as an environment variable, or in build-settings.toml to point to the root of your OptiX installation.");

    let cuda_root = std::env::var("CUDA_ROOT")
        .expect("CUDA_ROOT not found. You must set CUDA_ROOT either as an environment variable, or in build-settings.toml to point to the root of your CUDA installation.");

    // Create a vector of options to pass to the compiler
    let optix_inc = format!("-I{}/include", optix_root);
    let cuda_inc = format!("-I{}/include", cuda_root);
    let source_inc = format!(
        "-I{}/examples/02_pipeline",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    );
    let common_inc = format!(
        "-I{}/examples/common",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    );

    let options = vec![
        optix_inc,
        cuda_inc,
        source_inc,
        common_inc,
        "-I/usr/include/x86_64-linux-gnu".into(),
        "-I/usr/lib/gcc/x86_64-linux-gnu/7/include".into(),
        "-arch=compute_70".to_owned(),
        "-rdc=true".to_owned(),
        "-std=c++14".to_owned(),
        "-D__x86_64".to_owned(),
        "-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1".into(),
        "-default-device".into(),
    ];

    // The program object allows us to compile the cuda source and get ptx from
    // it if successful.
    let mut prg = Program::new(src, "devicePrograms", Vec::new()).unwrap();

    match prg.compile_program(&options) {
        Err(code) => {
            panic!("{}: {}", code, prg.get_program_log().unwrap());
        }
        Ok(_) => (),
    }

    let ptx = prg.get_ptx().unwrap();
    println!("Compilation successful");
    println!("{}", ptx);
    ptx
}

#[repr(C)]
#[repr(align(16))]
struct RaygenRecord {
    #[repr(align(16))]
    header: [u8; OptixSbtRecordHeaderSize],
    data: *mut std::os::raw::c_void,
}

#[repr(C)]
#[repr(align(16))]
struct MissRecord {
    #[repr(align(16))]
    header: [u8; OptixSbtRecordHeaderSize],
    data: *mut std::os::raw::c_void,
}

#[repr(C)]
#[repr(align(16))]
struct HitgroupRecord {
    #[repr(align(16))]
    header: [u8; OptixSbtRecordHeaderSize],
    object_id: i32,
}

unsafe extern "C" fn optix_log_callback(
    level: u32,
    tag: *const std::os::raw::c_char,
    message: *const std::os::raw::c_char,
    data: *mut std::os::raw::c_void,
) {
    let tag = std::ffi::CStr::from_ptr(tag).to_string_lossy();
    let message = std::ffi::CStr::from_ptr(message).to_string_lossy();
    println!("[{:02}][{:<12}]: {}", level, tag, message);
}

struct CudaBuffer {
    pub size_in_bytes: usize,
    pub d_ptr: *mut std::os::raw::c_void,
}

impl CudaBuffer {
    pub fn new(size_in_bytes: usize) -> CudaBuffer {
        unsafe {
            let mut d_ptr = std::ptr::null_mut();
            let res = cudaMalloc(&mut d_ptr, size_in_bytes);
            if res != cudaError_enum::CUDA_SUCCESS {
                panic!(
                    "Could not allocate cuda buffer of {} bytes",
                    size_in_bytes
                );
            }
            CudaBuffer {
                size_in_bytes,
                d_ptr,
            }
        }
    }

    pub fn with_data<T>(data: &[T]) -> CudaBuffer
    where
        T: Sized,
    {
        unsafe {
            let size_in_bytes = std::mem::size_of::<T>() * data.len();
            let mut d_ptr = std::ptr::null_mut();
            let res = cudaMalloc(&mut d_ptr, size_in_bytes);
            if res != cudaError_enum::CUDA_SUCCESS {
                panic!(
                    "Could not allocate cuda buffer of {} bytes",
                    size_in_bytes
                );
            }

            let res = cudaMemcpy(
                d_ptr,
                data.as_ptr() as *const std::os::raw::c_void,
                size_in_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            if res != cudaError_enum::CUDA_SUCCESS {
                panic!("Could not upload data to device");
            }
            CudaBuffer {
                size_in_bytes,
                d_ptr,
            }
        }
    }

    pub fn upload<T>(&mut self, data: &[T]) {
        let sz = data.len() * std::mem::size_of::<T>();
        if sz != self.size_in_bytes {
            panic!("Tried to upload {} elements totalling {} bytes to a {}-byte buffer", data.len(), sz, self.size_in_bytes);
        }
        unsafe {
            let res = cudaMemcpy(
                self.d_ptr,
                data.as_ptr() as *const std::os::raw::c_void,
                self.size_in_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            if res != cudaError::cudaSuccess {
                panic!("Could not upload data to device");
            }
        }
    }
}
