pub use optix::{DeviceContext, Error, DeviceStorage};
type Result<T, E = Error> = std::result::Result<T, E>;

use ustr::ustr;

pub struct Renderer {
    stream: cu::Stream,
    launch_params: LaunchParams,
    buf_launch_params: optix::TypedBuffer<LaunchParams>,
    buf_raygen: optix::TypedBuffer<RaygenRecord>,
    buf_hitgroup: optix::TypedBuffer<HitgroupRecord>,
    buf_miss: optix::TypedBuffer<MissRecord>,
    sbt: optix::sys::OptixShaderBindingTable,
    pipeline: optix::Pipeline,
    color_buffer: optix::TypedBuffer<u32>,
}

impl Renderer {
    pub fn new(
        width: usize,
        height: usize,
    ) -> Result<Renderer, Box<dyn std::error::Error>> {
        init_optix()?;

        // create CUDA and OptiX contexts
        let device = cu::Device::get(0)?;
        let tex_align = device.get_attribute(cu::DeviceAttribute::TextureAlignment)?;
        let srf_align = device.get_attribute(cu::DeviceAttribute::SurfaceAlignment)?;
        println!("tex align: {}\nsrf align: {}", tex_align, srf_align);

        let cuda_context = device.ctx_create(
            cu::ContextFlags::SCHED_AUTO | cu::ContextFlags::MAP_HOST,
        )?;
        let stream = cu::Stream::create(cu::StreamFlags::DEFAULT)?;

        let mut ctx = optix::DeviceContext::create(&cuda_context)?;
        ctx.set_log_callback(
            |_level, tag, msg| println!("[{}]: {}", tag, msg),
            4,
        );

        // create module
        let module_compile_options = optix::ModuleCompileOptions {
            max_register_count: 50,
            opt_level: optix::CompileOptimizationLevel::Default,
            debug_level: optix::CompileDebugLevel::None,
        };

        let pipeline_compile_options = optix::PipelineCompileOptions::new()
            .uses_motion_blur(false)
            .num_attribute_values(2)
            .num_payload_values(2)
            .traversable_graph_flags(
                optix::TraversableGraphFlags::ALLOW_SINGLE_GAS,
            )
            .exception_flags(optix::ExceptionFlags::NONE)
            .pipeline_launch_params_variable_name(ustr("optixLaunchParams"));

        let ptx = include_str!(concat!(
            env!("OUT_DIR"),
            "/examples/02_pipeline/device_programs.ptx"
        ));

        let (module, _log) = ctx.module_create_from_ptx(
            &module_compile_options,
            &pipeline_compile_options,
            ptx,
        )?;

        // create raygen program
        let pgdesc_raygen = optix::ProgramGroupDesc::raygen(
            &module,
            ustr("__raygen__renderFrame"),
        );

        let (pg_raygen, _log) = ctx.program_group_create(&[pgdesc_raygen])?;

        // create miss program
        let pgdesc_miss =
            optix::ProgramGroupDesc::miss(&module, ustr("__miss__radiance"));

        let (pg_miss, _log) = ctx.program_group_create(&[pgdesc_miss])?;

        let pgdesc_hitgroup = optix::ProgramGroupDesc::hitgroup(
            Some((&module, ustr("__closesthit__radiance"))),
            Some((&module, ustr("__anyhit__radiance"))),
            None,
        );

        // create hitgroup programs
        let (pg_hitgroup, _log) =
            ctx.program_group_create(&[pgdesc_hitgroup])?;

        // create pipeline
        let mut program_groups = Vec::new();
        program_groups.extend(pg_raygen.iter().cloned());
        program_groups.extend(pg_miss.iter().cloned());
        program_groups.extend(pg_hitgroup.iter().cloned());

        let pipeline_link_options = optix::PipelineLinkOptions {
            max_trace_depth: 2,
            debug_level: optix::CompileDebugLevel::LineInfo,
        };

        let (pipeline, _log) = ctx.pipeline_create(
            &pipeline_compile_options,
            pipeline_link_options,
            &program_groups,
        )?;

        pipeline.set_stack_size(2 * 1024, 2 * 1024, 2 * 1024, 1)?;

        // create SBT
        let rec_raygen: Vec<_> = pg_raygen
            .iter()
            .map(|pg| {
                RaygenRecord::pack(0, pg).expect("failed to pack raygen record")
            })
            .collect();

        let rec_miss: Vec<_> = pg_miss
            .iter()
            .map(|pg| {
                MissRecord::pack(0, pg).expect("failed to pack miss record")
            })
            .collect();

        let num_objects = 1;
        let rec_hitgroup: Vec<_> = (0..num_objects)
            .map(|i| {
                let object_type = 0;
                let rec = HitgroupRecord::pack(
                    HitgroupSbtData { object_id: i },
                    &pg_hitgroup[object_type],
                )
                .expect("failed to pack hitgroup record");
                rec
            })
            .collect();

        let buf_raygen = optix::TypedBuffer::from_slice(&rec_raygen)?;
        let buf_miss = optix::TypedBuffer::from_slice(&rec_miss)?;
        let buf_hitgroup = optix::TypedBuffer::from_slice(&rec_hitgroup)?;

        let sbt = optix::ShaderBindingTable::new(&buf_raygen)
            .miss(&buf_miss)
            .hitgroup(&buf_hitgroup)
            .build();

        let color_buffer = optix::TypedBuffer::uninitialized(width * height)?;

        let launch_params = LaunchParams {
            frame_id: 0,
            color_buffer: color_buffer.device_ptr(),
            fb_size: Point2i {
                x: width as i32,
                y: height as i32,
            },
        };

        let buf_launch_params = optix::TypedBuffer::from_slice(&[launch_params])?;

        Ok(Renderer {
            stream,
            launch_params,
            buf_launch_params,
            buf_raygen,
            buf_hitgroup,
            buf_miss,
            sbt,
            pipeline,
            color_buffer,
        })
    }

    pub fn resize(
        &mut self,
        width: usize,
        height: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.color_buffer.resize(width * height)?;
        self.launch_params.fb_size.x = width as i32;
        self.launch_params.fb_size.y = height as i32;
        Ok(())
    }

    pub fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.buf_launch_params
            .upload(&[self.launch_params])?;
        self.launch_params.frame_id += 1;

        optix::launch(
            &self.pipeline,
            &self.stream,
            &self.buf_launch_params,
            &self.sbt,
            self.launch_params.fb_size.x as u32,
            self.launch_params.fb_size.y as u32,
            1,
        )?;

        cu::Context::synchronize()?;

        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Point2i {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct LaunchParams {
    pub frame_id: i32,
    pub color_buffer: cu::DevicePtr,
    pub fb_size: Point2i,
}

unsafe impl optix::DeviceCopy for LaunchParams {}

type RaygenRecord = optix::SbtRecord<i32>;
type MissRecord = optix::SbtRecord<i32>;
struct HitgroupSbtData {
    object_id: u32,
}
unsafe impl optix::DeviceCopy for HitgroupSbtData {}
type HitgroupRecord = optix::SbtRecord<HitgroupSbtData>;

fn init_optix() -> Result<(), Box<dyn std::error::Error>> {
    cu::init()?;
    let device_count = cu::Device::get_count()?;
    if device_count == 0 {
        panic!("No CUDA devices found!");
    }

    optix::init()?;
    Ok(())
}