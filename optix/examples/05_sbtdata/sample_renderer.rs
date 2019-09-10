use imath::*;

type Result<T, E = Error> = std::result::Result<T, E>;

use optix::{DeviceShareable, SbtRecord, SharedVariable};
use optix_derive::device_shared;

// Wrap math types in a newtype that we can share with the device
optix::wrap_copyable_for_device! {V2i32, V2i32D}
optix::wrap_copyable_for_device! {V3f32, V3f32D}

#[device_shared]
struct TriangleMeshSBTData {
    color: V3f32D,
    vertex: optix::RtBuffer,
    index: optix::RtBuffer,
}

pub struct SampleRenderer {
    cuda_context: cuda::ContextRef,
    stream: cuda::Stream,
    device_prop: cuda::DeviceProp,

    pipeline: optix::PipelineRef,

    module: optix::ModuleRef,

    program_groups: Vec<optix::ProgramGroupRef>,
    sbt: optix::ShaderBindingTable,

    launch_params: SharedVariable<LaunchParams>,

    last_set_camera: Camera,

    mesh: TriangleMesh,

    ctx: optix::DeviceContext,
}

impl SampleRenderer {
    pub fn new(
        fb_size: V2i32,
        camera: Camera,
        mesh: TriangleMesh,
    ) -> Result<SampleRenderer> {
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
                optix::module::TraversableGraphFlags::ALLOW_ANY,
            num_payload_values: 2,
            num_attribute_values: 2,
            exception_flags: optix::module::ExceptionFlags::NONE,
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

        // build accel
        // upload the model data and create the triangle array build input
        let vertex_buffer =
            optix::RtBuffer::new(&mesh.vertex, optix::BufferFormat::F32x3)
                .unwrap();
        let index_buffer =
            optix::RtBuffer::new(&mesh.index, optix::BufferFormat::I32x3)
                .unwrap();
        let build_input = optix::BuildInput::Triangle(
            optix::TriangleArray::new(
                std::slice::from_ref(&vertex_buffer),
                &index_buffer,
                optix::GeometryFlags::NONE,
            )
            .unwrap(),
        );

        // BLAS setup
        let accel_build_options = optix::AccelBuildOptions {
            build_flags: optix::BuildFlags::NONE
                | optix::BuildFlags::ALLOW_COMPACTION,
            operation: optix::BuildOperation::Build,
            motion_options: optix::MotionOptions {
                num_keys: 1,
                flags: optix::MotionFlags::NONE,
                time_begin: 0.0,
                time_end: 0.0,
            },
        };

        let blas_buffer_sizes = ctx.accel_compute_memory_usage(
            &accel_build_options,
            std::slice::from_ref(&build_input),
        )?;

        let compacted_size_buffer =
            cuda::Buffer::new(std::mem::size_of::<usize>())?;

        let compacted_size_desc = optix::AccelEmitDesc::new(
            &compacted_size_buffer,
            optix::AccelPropertyType::CompactedSize,
        );

        // allocate and execute build
        let temp_buffer =
            cuda::Buffer::new(blas_buffer_sizes[0].temp_size_in_bytes)?;
        let output_buffer =
            cuda::Buffer::new(blas_buffer_sizes[0].output_size_in_bytes)?;

        let as_handle = ctx.accel_build(
            &cuda::Stream::default(),
            &accel_build_options,
            std::slice::from_ref(&build_input),
            &temp_buffer,
            output_buffer,
            std::slice::from_ref(&compacted_size_desc),
        )?;

        // ultimately we'd want to do something more complex with async streams
        // here rather than a stop-the-world sync, but this will do for now.
        match cuda::device_synchronize() {
            Ok(_) => (),
            Err(e) => {
                println!("build sync failed");
                return Err(e.into());
            }
        };

        // now compact the acceleration structure
        let compacted_size =
            compacted_size_buffer.download_primitive::<usize>()?;

        let as_buffer = cuda::Buffer::new(compacted_size)?;
        let as_handle =
            ctx.accel_compact(&cuda::Stream::default(), as_handle, as_buffer)?;

        // sync again
        match cuda::device_synchronize() {
            Ok(_) => (),
            Err(e) => {
                println!("compact sync failed");
                return Err(e.into());
            }
        };

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
        let rg_rec =
            SbtRecord::new(0i32, std::sync::Arc::clone(&program_groups[0]));
        let miss_rec =
            SbtRecord::new(0i32, std::sync::Arc::clone(&program_groups[1]));

        let mesh_sbt_data = TriangleMeshSBTData {
            color: V3f32D(v3f32(0.8, 0.2, 0.1)),
            vertex: vertex_buffer,
            index: index_buffer,
        };

        let hg_rec = SbtRecord::new(
            mesh_sbt_data,
            std::sync::Arc::clone(&program_groups[2]),
        );

        let sbt = optix::ShaderBindingTableBuilder::new(&rg_rec)
            .miss_records(std::slice::from_ref(&miss_rec))
            .hitgroup_records(std::slice::from_ref(&hg_rec))
            .build();

        let mut color_buffer = cuda::Buffer::new(
            (fb_size.x * fb_size.y) as usize * std::mem::size_of::<V4f32>(),
        )
        .unwrap();

        let cos_fovy = 0.66f32;
        let aspect = fb_size.x as f32 / fb_size.y as f32;
        let direction = (camera.at - camera.from).normalized();
        let horizontal =
            cos_fovy * aspect * direction.cross(camera.up).normalized();
        let vertical = cos_fovy * horizontal.cross(direction).normalized();
        let launch_params = LaunchParams {
            frame: Frame {
                color_buffer,
                size: fb_size.into(),
            },
            camera: RenderCamera {
                position: camera.from.into(),
                direction: direction.into(),
                horizontal: horizontal.into(),
                vertical: vertical.into(),
            },
            traversable: as_handle,
        };

        let launch_params = SharedVariable::<LaunchParams>::new(launch_params)?;

        Ok(SampleRenderer {
            cuda_context,
            stream,
            device_prop,
            pipeline,
            module,
            program_groups,
            sbt,
            launch_params,
            last_set_camera: camera,
            mesh,
            ctx,
        })
    }

    pub fn render(&mut self) {
        self.launch_params.upload().unwrap();

        self.ctx
            .launch(
                &self.pipeline,
                &self.stream,
                &self.launch_params.variable_buffer(),
                &self.sbt,
                self.launch_params.frame.size.x as u32,
                self.launch_params.frame.size.y as u32,
                1,
            )
            .unwrap();

        // we'll want to do something clever with streams ultimately, but
        // for now do a brute-force sync
        cuda::device_synchronize().unwrap();
    }

    pub fn resize(&mut self, size: V2i32) {
        self.launch_params.frame.size = size.into();
        self.launch_params.frame.color_buffer = cuda::Buffer::new(
            (size.x * size.y) as usize * std::mem::size_of::<V4f32>(),
        )
        .unwrap();
    }

    pub fn download_pixels(&self, pixels: &mut [V4f32]) -> Result<()> {
        self.launch_params.frame.color_buffer.download(pixels)?;
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
        "-I{}/examples/05_sbtdata",
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

pub struct Camera {
    pub from: V3f32,
    pub at: V3f32,
    pub up: V3f32,
}

pub struct TriangleMesh {
    pub vertex: Vec<V3f32>,
    pub index: Vec<V3i32>,
}

impl TriangleMesh {
    pub fn new() -> TriangleMesh {
        TriangleMesh {
            vertex: Vec::new(),
            index: Vec::new(),
        }
    }

    pub fn add_cube(&mut self, center: V3f32, size: V3f32) {
        let start_index = self.vertex.len() as i32;

        self.vertex.push((v3f32(0.0, 0.0, 0.0) + center) * size);
        self.vertex.push((v3f32(1.0, 0.0, 0.0) + center) * size);
        self.vertex.push((v3f32(0.0, 1.0, 0.0) + center) * size);
        self.vertex.push((v3f32(1.0, 1.0, 0.0) + center) * size);
        self.vertex.push((v3f32(0.0, 0.0, 1.0) + center) * size);
        self.vertex.push((v3f32(1.0, 0.0, 1.0) + center) * size);
        self.vertex.push((v3f32(0.0, 1.0, 1.0) + center) * size);
        self.vertex.push((v3f32(1.0, 1.0, 1.0) + center) * size);

        const indices: [i32; 36] = [
            0, 1, 3, 2, 3, 0, 5, 7, 6, 5, 6, 4, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2,
            7, 6, 1, 5, 6, 1, 7, 3, 4, 0, 2, 4, 2, 6,
        ];

        for c in indices.chunks(3) {
            self.index.push(v3i32(
                c[0] + start_index,
                c[1] + start_index,
                c[2] + start_index,
            ));
        }
    }
}

#[device_shared]
struct RenderCamera {
    position: V3f32D,
    direction: V3f32D,
    horizontal: V3f32D,
    vertical: V3f32D,
}

#[device_shared]
struct Frame {
    color_buffer: cuda::Buffer,
    size: V2i32D,
}

#[device_shared]
pub struct LaunchParams {
    frame: Frame,
    camera: RenderCamera,
    traversable: optix::TraversableHandle,
}