use cu::{
    allocator::{DeviceFrameAllocator, Layout},
    DevicePtr,
};

pub use optix::{DeviceContext, DeviceStorage, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

use crate::{V2f32, V3f32, V3i32, V4f32};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use ustr::ustr;

pub use crate::vector::*;

static FRAME_ALLOC: Lazy<Mutex<DeviceFrameAllocator>> = Lazy::new(|| {
    Mutex::new(
        DeviceFrameAllocator::new(256 * 1024 * 1024)
            .expect("Frame allocator failed"),
    )
});

pub struct FrameAlloc;
unsafe impl cu::allocator::DeviceAllocRef for FrameAlloc {
    fn alloc(&self, layout: Layout) -> Result<DevicePtr, cu::Error> {
        FRAME_ALLOC.lock().alloc(layout)
    }

    fn alloc_with_tag(&self, layout: Layout, tag: u16) -> Result<DevicePtr, cu::Error> {
        FRAME_ALLOC.lock().alloc_with_tag(layout, tag)
    }

    fn dealloc(&self, ptr: DevicePtr) -> Result<(), cu::Error> {
        FRAME_ALLOC.lock().dealloc(ptr)
    }
}

pub struct Renderer {
    stream: cu::Stream,
    // launch_params: LaunchParams,
    // buf_launch_params: optix::TypedBuffer<LaunchParams, FrameAlloc>,
    launch_params: optix::DeviceVariable<LaunchParams, FrameAlloc>,
    buf_raygen: optix::TypedBuffer<RaygenRecord, FrameAlloc>,
    buf_hitgroup: optix::TypedBuffer<HitgroupRecord, FrameAlloc>,
    buf_miss: optix::TypedBuffer<MissRecord, FrameAlloc>,
    as_handle: optix::TraversableHandle,
    as_buffer: optix::Buffer<FrameAlloc>,
    sbt: optix::sys::OptixShaderBindingTable,
    pipeline: optix::Pipeline,
    color_buffer: optix::TypedBuffer<V4f32, FrameAlloc>,
}

impl Renderer {
    pub fn new(
        width: u32,
        height: u32,
        camera: Camera,
        mesh: TriangleMesh,
    ) -> Result<Renderer, Box<dyn std::error::Error>> {
        init_optix()?;

        // create CUDA and OptiX contexts
        let device = cu::Device::get(0)?;
        let tex_align =
            device.get_attribute(cu::DeviceAttribute::TextureAlignment)?;
        let srf_align =
            device.get_attribute(cu::DeviceAttribute::SurfaceAlignment)?;
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

        let testptr = cu::memory::mem_alloc_with_tag(64, 17)?;
        println!("testptr: {}", testptr.tag());

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
            "/examples/05_sbtdata/device_programs.ptx"
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

        // create geometry and accels
        let buf_vertex =
            optix::TypedBuffer::from_slice_in(&mesh.vertex, FrameAlloc)?;
        let buf_indices =
            optix::TypedBuffer::from_slice_in(&mesh.index, FrameAlloc)?;

        let triangle_input = optix::BuildInputTriangleArray::new(
            &[buf_vertex.device_ptr()],
            mesh.vertex.len() as u32,
            optix::VertexFormat::Float3,
            &optix::GeometryFlags::None,
        )
        .index_buffer(
            buf_indices.device_ptr(),
            mesh.index.len() as u32,
            optix::IndicesFormat::Int3,
        )
        .build();

        // blas setup
        let accel_options = optix::AccelBuildOptions::new(
            optix::BuildFlags::ALLOW_COMPACTION,
            optix::BuildOperation::Build,
        );

        let build_inputs = vec![triangle_input];

        let blas_buffer_sizes =
            ctx.accel_compute_memory_usage(&[accel_options], &build_inputs)?;

        // prepare compaction
        let temp_buffer = optix::Buffer::uninitialized_with_align_in(
            blas_buffer_sizes.temp_size_in_bytes,
            optix::ACCEL_BUFFER_BYTE_ALIGNMENT,
            FrameAlloc,
        )?;

        let output_buffer = optix::Buffer::uninitialized_with_align_in(
            blas_buffer_sizes.output_size_in_bytes,
            optix::ACCEL_BUFFER_BYTE_ALIGNMENT,
            FrameAlloc,
        )?;

        let mut compacted_size = optix::DeviceVariable::new_in(0usize, FrameAlloc)?;

        let mut properties = vec![optix::AccelEmitDesc::CompactedSize(
            compacted_size.device_ptr(),
        )];

        let as_handle = ctx.accel_build(
            &stream,
            &[accel_options],
            &build_inputs,
            &temp_buffer,
            &output_buffer,
            &mut properties,
        )?;

        cu::Context::synchronize()?;

        compacted_size.download()?;

        let as_buffer = optix::Buffer::uninitialized_with_align_in(
            *compacted_size,
            optix::ACCEL_BUFFER_BYTE_ALIGNMENT,
            FrameAlloc,
        )?;

        let as_handle = ctx.accel_compact(&stream, as_handle, &as_buffer)?;
        cu::Context::synchronize()?;

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
                    HitgroupSbtData { 
                        data: TriangleMeshSbtData {
                            color: mesh.color,
                            vertex: buf_vertex.device_ptr(),
                            index: buf_indices.device_ptr(),
                        }
                    },
                    &pg_hitgroup[object_type],
                )
                .expect("failed to pack hitgroup record");
                rec
            })
            .collect();

        let buf_raygen =
            optix::TypedBuffer::from_slice_in(&rec_raygen, FrameAlloc)?;
        let buf_miss =
            optix::TypedBuffer::from_slice_in(&rec_miss, FrameAlloc)?;
        let buf_hitgroup =
            optix::TypedBuffer::from_slice_in(&rec_hitgroup, FrameAlloc)?;

        let sbt = optix::ShaderBindingTable::new(&buf_raygen)
            .miss(&buf_miss)
            .hitgroup(&buf_hitgroup)
            .build();

        let color_buffer = optix::TypedBuffer::uninitialized_with_align_in(
            (width * height) as usize,
            16,
            FrameAlloc,
        )?;

        let cosfovy = 0.66f32;
        let aspect = width as f32 / height as f32;
        let direction = normalize(camera.at - camera.from);
        let horizontal =
            cosfovy * aspect * normalize(cross(direction, camera.up));
        let vertical = cosfovy * normalize(cross(horizontal, direction));

        let launch_params = optix::DeviceVariable::new_in(LaunchParams {
            frame: Frame {
                color_buffer: color_buffer.device_ptr(),
                size: v2i32(width as i32, height as i32),
            },
            camera: RenderCamera {
                position: camera.from,
                direction,
                horizontal,
                vertical,
            },
            traversable: as_handle,
        }, FrameAlloc)?;

        Ok(Renderer {
            stream,
            launch_params,
            buf_raygen,
            buf_hitgroup,
            buf_miss,
            as_handle,
            as_buffer,
            sbt,
            pipeline,
            color_buffer,
        })
    }

    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.color_buffer.resize((width * height) as usize)?;
        self.launch_params.frame.color_buffer = self.color_buffer.device_ptr();
        self.launch_params.frame.size.x = width as i32;
        self.launch_params.frame.size.y = height as i32;
        Ok(())
    }

    pub fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.launch_params.upload()?;

        optix::launch(
            &self.pipeline,
            &self.stream,
            &self.launch_params,
            &self.sbt,
            self.launch_params.frame.size.x as u32,
            self.launch_params.frame.size.y as u32,
            1,
        )?;

        cu::Context::synchronize()?;

        Ok(())
    }

    pub fn download_pixels(
        &self,
        slice: &mut [V4f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.color_buffer.download(slice)?;
        Ok(())
    }
}

unsafe impl optix::DeviceCopy for LaunchParams {}

type RaygenRecord = optix::SbtRecord<i32>;
type MissRecord = optix::SbtRecord<i32>;
struct HitgroupSbtData {
    data: TriangleMeshSbtData,
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

#[repr(C)]
pub struct Frame {
    color_buffer: cu::DevicePtr,
    size: V2i32,
}

#[repr(C)]
pub struct RenderCamera {
    position: V3f32,
    direction: V3f32,
    horizontal: V3f32,
    vertical: V3f32,
}

#[repr(C)]
pub struct LaunchParams {
    pub frame: Frame,
    pub camera: RenderCamera,
    pub traversable: optix::TraversableHandle,
}

pub struct Camera {
    pub from: V3f32,
    pub at: V3f32,
    pub up: V3f32,
}

#[repr(C)]
struct TriangleMeshSbtData {
    color: V3f32,
    vertex: cu::DevicePtr,
    index: cu::DevicePtr,
}

pub struct TriangleMesh {
    pub vertex: Vec<V3f32>,
    pub index: Vec<V3i32>,
    pub color: V3f32,
}

impl TriangleMesh {
    pub fn new(color: V3f32) -> TriangleMesh {
        TriangleMesh {
            vertex: Vec::new(),
            index: Vec::new(),
            color,
        }
    }

    #[cfg(feature = "cgmath")]
    pub fn add_cube(&mut self, center: V3f32, size: V3f32) {
        let start_index = self.vertex.len() as i32;

        use cgmath::ElementWise;

        self.vertex.push(
            v3f32(0.0, 0.0, 0.0).mul_element_wise(size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(1.0, 0.0, 0.0).mul_element_wise(size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(0.0, 1.0, 0.0).mul_element_wise(size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(1.0, 1.0, 0.0).mul_element_wise(size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(0.0, 0.0, 1.0).mul_element_wise(size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(1.0, 0.0, 1.0).mul_element_wise(size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(0.0, 1.0, 1.0).mul_element_wise(size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(1.0, 1.0, 1.0).mul_element_wise(size) + center - 0.5 * size,
        );

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

    #[cfg(feature = "nalgebra-glm")]
    pub fn add_cube(&mut self, center: V3f32, size: V3f32) {
        let start_index = self.vertex.len() as i32;

        self.vertex.push(
            v3f32(0.0, 0.0, 0.0).component_mul(&size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(1.0, 0.0, 0.0).component_mul(&size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(0.0, 1.0, 0.0).component_mul(&size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(1.0, 1.0, 0.0).component_mul(&size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(0.0, 0.0, 1.0).component_mul(&size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(1.0, 0.0, 1.0).component_mul(&size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(0.0, 1.0, 1.0).component_mul(&size) + center - 0.5 * size,
        );
        self.vertex.push(
            v3f32(1.0, 1.0, 1.0).component_mul(&size) + center - 0.5 * size,
        );

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

    #[cfg(not(any(feature = "cgmath", feature = "nalgebra-glm")))]
    pub fn add_cube(&mut self, center: V3f32, size: V3f32) {
        let start_index = self.vertex.len() as i32;

        self.vertex
            .push((v3f32(0.0, 0.0, 0.0)) * size + center - 0.5 * size);
        self.vertex
            .push((v3f32(1.0, 0.0, 0.0)) * size + center - 0.5 * size);
        self.vertex
            .push((v3f32(0.0, 1.0, 0.0)) * size + center - 0.5 * size);
        self.vertex
            .push((v3f32(1.0, 1.0, 0.0)) * size + center - 0.5 * size);
        self.vertex
            .push((v3f32(0.0, 0.0, 1.0)) * size + center - 0.5 * size);
        self.vertex
            .push((v3f32(1.0, 0.0, 1.0)) * size + center - 0.5 * size);
        self.vertex
            .push((v3f32(0.0, 1.0, 1.0)) * size + center - 0.5 * size);
        self.vertex
            .push((v3f32(1.0, 1.0, 1.0)) * size + center - 0.5 * size);

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
