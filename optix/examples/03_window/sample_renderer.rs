use nalgebra_glm::*;

type Result<T, E = Error> = std::result::Result<T, E>;

use optix::SbtRecord;
use optix_derive::sbt_record;

#[sbt_record]
pub struct RaygenRecord {
    data: *mut std::os::raw::c_void,
}

#[sbt_record]
pub struct MissRecord {
    data: *mut std::os::raw::c_void,
}

#[sbt_record]
pub struct HitgroupRecord {
    object_id: i32,
}

pub struct SampleRenderer {
    cuda_context: cuda::ContextRef,
    stream: cuda::Stream,
    device_prop: cuda::DeviceProp,

    pipeline: optix::PipelineRef,

    module: optix::ModuleRef,

    program_groups: Vec<optix::ProgramGroupRef>,
    sbt: optix::ShaderBindingTable,

    color_buffer: cuda::Buffer,
    launch_params: LaunchParams,
    launch_params_buffer: cuda::Buffer,

    ctx: optix::DeviceContext,
}

impl SampleRenderer {
    pub fn new(fb_size: IVec2) -> Result<SampleRenderer> {
        // Make sure CUDA context is initialized
        cuda::init();
        // Check that we've got available devices
        let num_devices = cuda::get_device_count();
        println!("Found {} CUDA devices", num_devices);

        // Initialize optix function table. Must be called before calling
        // any OptiX functions.
        optix::init()?;

        // Just use the first device
        cuda::set_device(0)?;

        // Create a new stream to submit work on
        let stream = cuda::Stream::new().unwrap();

        // Get the device properties
        let device_prop = cuda::get_device_properties(0)?;
        println!(
            "Running on device 0: {} {}MB",
            device_prop.name(),
            device_prop.total_global_mem() / (1024 * 1024)
        );

        // Get the context for the current thread
        let cuda_context = cuda::Context::get_current()?;

        // Create the device context and enable logging
        let mut ctx = optix::DeviceContext::create(cuda_context, None)?;
        ctx.set_log_callback(
            |level, tag, msg| println!("[{}]: {}", tag, msg),
            4,
        );

        // Create our first module
        let module_compile_options = optix::ModuleCompileOptions {
            max_register_count: 100,
            opt_level: optix::CompileOptimizationLevel::Level0,
            debug_level: optix::CompileDebugLevel::LineInfo,
        };

        let pipeline_compile_options = optix::PipelineCompileOptions {
            uses_motion_blur: false,
            traversable_graph_flags:
                optix::module::TraversableGraphFlags::AllowAny,
            num_payload_values: 2,
            num_attribute_values: 2,
            exception_flags: optix::module::ExceptionFlags::None,
            pipeline_launch_params_variable_name: "optixLaunchParams".into(),
        };

        // We just have a single program for now. Compile it from
        // source using nvrtc
        let cuda_source = include_str!("devicePrograms.cu");
        let ptx = compile_to_ptx(cuda_source);

        // Create the module
        let (module, log) = ctx.module_create_from_ptx(
            module_compile_options,
            &pipeline_compile_options,
            &ptx,
        )?;

        if !log.is_empty() {
            println!("{}", log);
        }

        // Create raygen program(s)
        let (raygen_pg, log) = ctx.program_group_create(
            optix::ProgramGroupDesc::Raygen(optix::ProgramGroupModule {
                module: module.clone(),
                entry_function_name: "__raygen__renderFrame".into(),
            }),
        )?;

        // Create miss program(s)
        let (miss_pg, log) = ctx.program_group_create(
            optix::ProgramGroupDesc::Miss(optix::ProgramGroupModule {
                module: module.clone(),
                entry_function_name: "__miss__radiance".into(),
            }),
        )?;

        // Create hitgroup programs
        let (hitgroup_pg, log) =
            ctx.program_group_create(optix::ProgramGroupDesc::Hitgroup {
                ch: Some(optix::ProgramGroupModule {
                    module: module.clone(),
                    entry_function_name: "__closesthit__radiance".into(),
                }),
                ah: Some(optix::ProgramGroupModule {
                    module: module.clone(),
                    entry_function_name: "__anyhit__radiance".into(),
                }),
                is: None,
            })?;

        // Create the pipeline
        let pipeline_link_options = optix::PipelineLinkOptions {
            max_trace_depth: 2,
            debug_level: optix::CompileDebugLevel::None,
            override_uses_motion_blur: false,
        };

        let program_groups = vec![raygen_pg, miss_pg, hitgroup_pg];
        let (mut pipeline, log) = ctx.pipeline_create(
            &pipeline_compile_options,
            pipeline_link_options,
            &program_groups,
        )?;

        ctx.pipeline_set_stack_size(
            &mut pipeline,
            // direct stack size for direct callables invoked for IS or AH
            2 * 1024,
            // direct stack size for direct callables invoked from RG, MS or CH
            2 * 1024,
            // continuation stack size
            2 * 1024,
            // maximum depth of a traversable graph passed to trace
            3,
        );

        // Build Shader Binding Table
        let mut rg_rec = RaygenRecord {
            header: optix_sys::SbtRecordHeader::default(),
            data: std::ptr::null_mut(),
        };
        rg_rec.pack(&program_groups[0]);

        let mut miss_rec = MissRecord {
            header: optix_sys::SbtRecordHeader::default(),
            data: std::ptr::null_mut(),
        };
        miss_rec.pack(&program_groups[1]);

        let mut hg_rec = HitgroupRecord {
            header: optix_sys::SbtRecordHeader::default(),
            object_id: 0,
        };
        hg_rec.pack(&program_groups[2]);

        let sbt = optix::ShaderBindingTableBuilder::new(&rg_rec)
            .miss_records(std::slice::from_ref(&miss_rec))
            .hitgroup_records(std::slice::from_ref(&hg_rec))
            .build();

        let mut color_buffer = cuda::Buffer::new(
            (fb_size.x * fb_size.y) as usize * std::mem::size_of::<Vec4>(),
        )
        .unwrap();

        let launch_params = LaunchParams {
            frame_id: 0,
            color_buffer: color_buffer.as_mut_ptr() as *mut f32,
            fb_size,
        };

        let launch_params_buffer =
            cuda::Buffer::with_data(std::slice::from_ref(&launch_params))?;

        Ok(SampleRenderer {
            cuda_context,
            stream,
            device_prop,
            pipeline,
            module,
            program_groups,
            sbt,
            color_buffer,
            launch_params,
            launch_params_buffer,
            ctx,
        })
    }

    pub fn render(&mut self) {
        self.launch_params_buffer
            .upload(std::slice::from_ref(&self.launch_params));
        self.launch_params.frame_id += 1;

        self.ctx
            .launch(
                &self.pipeline,
                &self.stream,
                &self.launch_params_buffer,
                &self.sbt,
                self.launch_params.fb_size.x as u32,
                self.launch_params.fb_size.y as u32,
                1,
            )
            .unwrap();

        // we'll want to do something clever with streams ultimately, but
        // for now do a brute-force sync
        cuda::device_synchronize().unwrap();
    }

    pub fn resize(&mut self, size: IVec2) {
        self.launch_params.fb_size = size;
        self.color_buffer = cuda::Buffer::new(
            (size.x * size.y) as usize * std::mem::size_of::<Vec4>(),
        )
        .unwrap();
        self.launch_params.color_buffer =
            self.color_buffer.as_mut_ptr() as *mut f32;
    }

    pub fn download_pixels(&self, pixels: &mut [Vec4]) -> Result<()> {
        self.color_buffer.download(pixels)?;
        Ok(())
    }
}

#[derive(Display, Debug, From)]
pub enum Error {
    #[display(fmt = "OptiX error: {}", _0)]
    OptixError(optix::Error),
    #[display(fmt = "CUDA error: {}", _0)]
    CudaError(cuda::Error),
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
        "-I{}/examples/03_window",
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
    ptx
}

#[repr(C)]
pub struct LaunchParams {
    frame_id: i32,
    color_buffer: *mut f32,
    fb_size: IVec2,
}
