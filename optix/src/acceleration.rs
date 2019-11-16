use super::{
    cuda::{self, Allocator, Mallocator},
    math::{Box3f32, V3f32, V3i32},
};
use optix_sys as sys;

use super::{
    buffer::{Buffer, BufferElement, BufferFormat},
    device_context::DeviceContext,
    error::Error,
    instance::Instance,
};
type Result<T, E = Error> = std::result::Result<T, E>;

use std::convert::{TryFrom, TryInto};

use std::rc::Rc;

pub enum BuildInput<'a, AllocT, V = V3f32, I = V3i32>
where
    AllocT: Allocator,
    V: BufferElement,
    I: BufferElement,
{
    Triangle(TriangleArray<'a, AllocT, V, I>),
    CustomPrimitive(CustomPrimitiveArray<'a, AllocT>),
    Instance(InstanceArray<'a, AllocT>),
}

impl<'a, AllocT, V, I> From<&BuildInput<'a, AllocT, V, I>>
    for sys::OptixBuildInput
where
    AllocT: Allocator,
    V: BufferElement,
    I: BufferElement,
{
    fn from(b: &BuildInput<'a, AllocT, V, I>) -> sys::OptixBuildInput {
        let mut input = sys::OptixBuildInputUnion::default();
        match b {
            BuildInput::Triangle(ta) => {
                let type_ =
                    sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                unsafe {
                    input.triangle_array = ta.try_into().unwrap();
                }
                sys::OptixBuildInput { type_, input }
            }
            BuildInput::Instance(ia) => {
                let type_ =
                    sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_INSTANCES;
                unsafe {
                    input.instance_array = ia.into();
                }
                sys::OptixBuildInput { type_, input }
            }
            BuildInput::CustomPrimitive(cp) => {
                let type_ = sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
                unsafe {
                    input.aabb_array = cp.into();
                }
                sys::OptixBuildInput { type_, input }
            }
        }
    }
}

pub struct TriangleArray<'a, AllocT, V, I>
where
    AllocT: Allocator,
    V: BufferElement,
    I: BufferElement,
{
    vertex_buffers: Vec<Rc<Buffer<'a, AllocT, V>>>,
    vertex_buffers_d: Vec<cuda::CUdeviceptr>,
    index_buffer: Rc<Buffer<'a, AllocT, I>>,
    flags: GeometryFlags,
}

impl<'a, AllocT, V, I> TriangleArray<'a, AllocT, V, I>
where
    AllocT: Allocator,
    V: BufferElement,
    I: BufferElement,
{
    pub fn new(
        vertex_buffers: Vec<Rc<Buffer<'a, AllocT, V>>>,
        index_buffer: Rc<Buffer<'a, AllocT, I>>,
        flags: GeometryFlags,
    ) -> Result<TriangleArray<'a, AllocT, V, I>> {
        let vertex_buffers_d: Vec<cuda::CUdeviceptr> =
            vertex_buffers.iter().map(|b| b.as_device_ptr()).collect();

        // simple sanity check to make sure the buffer shapes match
        let format = vertex_buffers[0].format();
        let count = vertex_buffers[0].len();
        if !vertex_buffers
            .iter()
            .all(|b| b.format() == format && b.len() == count)
        {
            return Err(Error::BufferShapeMismatch {
                e_format: format,
                e_count: count,
            });
        }

        Ok(TriangleArray {
            vertex_buffers,
            vertex_buffers_d,
            index_buffer,
            flags,
        })
    }
}

impl<'a, AllocT, V, I> TryFrom<&TriangleArray<'a, AllocT, V, I>>
    for sys::OptixBuildInputTriangleArray
where
    AllocT: Allocator,
    V: BufferElement,
    I: BufferElement,
{
    type Error = Error;

    #[allow(non_snake_case)]
    fn try_from(
        ta: &TriangleArray<'a, AllocT, V, I>,
    ) -> Result<sys::OptixBuildInputTriangleArray> {
        let vertexBuffers = ta.vertex_buffers_d.as_ptr();
        let numVertices = ta.vertex_buffers[0].len() as u32;
        let vertexFormat = match &ta.vertex_buffers[0].format() {
            BufferFormat::F32x2 => {
                sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT2
            }
            BufferFormat::F32x3 => {
                sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3
            }
            BufferFormat::F16x2 => {
                sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_HALF2
            }
            BufferFormat::F16x3 => {
                sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_HALF3
            }
            _ => {
                return Err(Error::IncorrectVertexBufferFormat {
                    format: ta.vertex_buffers[0].format(),
                })
            }
        };
        let indexBuffer = ta.index_buffer.as_device_ptr();
        let numIndexTriplets = ta.index_buffer.len() as u32;
        let indexFormat = match &ta.index_buffer.format() {
            BufferFormat::I32x3 => {
                sys::OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3
            }
            BufferFormat::U16x3 => {
                sys::OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3
            }
            _ => {
                return Err(Error::IncorrectIndexBufferFormat {
                    format: ta.index_buffer.format(),
                })
            }
        };

        Ok(sys::OptixBuildInputTriangleArray {
            vertexBuffers,
            numVertices,
            vertexFormat,
            vertexStrideInBytes: ta.vertex_buffers[0].format().byte_size()
                as u32,
            indexBuffer,
            numIndexTriplets,
            indexStrideInBytes: ta.index_buffer.format().byte_size() as u32,
            indexFormat,
            preTransform: 0,
            flags: &ta.flags as *const GeometryFlags as *const u32,
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0,
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
        })
    }
}

pub struct CustomPrimitiveArray<'a, AllocT = Mallocator>
where
    AllocT: Allocator,
{
    _aabb_buffers: Vec<cuda::Buffer<'a, AllocT>>,
    aabb_buffers_d: Vec<cuda::CUdeviceptr>,
    num_primitives: u32,
    flags: Box<u32>,
}

impl<'a, AllocT> CustomPrimitiveArray<'a, AllocT>
where
    AllocT: Allocator,
{
    pub fn new(
        aabbs: &[Box3f32],
        flags: GeometryFlags,
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<CustomPrimitiveArray<'a, AllocT>> {
        let num_primitives = aabbs.len();
        let buffer = unsafe {
            cuda::Buffer::with_data(
                std::slice::from_raw_parts(
                    aabbs.as_ptr() as *const sys::OptixAabb,
                    num_primitives,
                ),
                tag,
                allocator,
            )?
        };
        let aabb_buffers = vec![buffer];
        let aabb_buffers_d: Vec<_> =
            aabb_buffers.iter().map(|b| b.as_device_ptr()).collect();

        Ok(CustomPrimitiveArray {
            _aabb_buffers: aabb_buffers,
            aabb_buffers_d,
            num_primitives: num_primitives as u32,
            flags: Box::new(flags.bits()),
        })
    }
}

impl<'a, AllocT> From<&CustomPrimitiveArray<'a, AllocT>>
    for sys::OptixBuildInputCustomPrimitiveArray
where
    AllocT: Allocator,
{
    fn from(
        arr: &CustomPrimitiveArray<'a, AllocT>,
    ) -> sys::OptixBuildInputCustomPrimitiveArray {
        sys::OptixBuildInputCustomPrimitiveArray {
            aabbBuffers: arr.aabb_buffers_d.as_ptr(),
            numPrimitives: arr.num_primitives,
            strideInBytes: 0,
            flags: arr.flags.as_ref() as *const u32,
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0,
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
        }
    }
}

pub struct InstanceArray<'a, AllocT = Mallocator>
where
    AllocT: Allocator,
{
    instances: cuda::Buffer<'a, AllocT>,
    num_instances: u32,
    aabbs: Option<cuda::Buffer<'a, AllocT>>,
    num_aabbs: u32,
}

impl<'a, AllocT> InstanceArray<'a, AllocT>
where
    AllocT: Allocator,
{
    pub fn new(
        instances: &[Instance],
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<InstanceArray<'a, AllocT>> {
        let num_instances = instances.len() as u32;
        let instances = cuda::Buffer::with_data(instances, tag, allocator)?;
        Ok(InstanceArray {
            instances,
            num_instances,
            aabbs: None,
            num_aabbs: 0,
        })
    }
}

impl<'a, AllocT> From<&InstanceArray<'a, AllocT>>
    for sys::OptixBuildInputInstanceArray
where
    AllocT: Allocator,
{
    fn from(
        ia: &InstanceArray<'a, AllocT>,
    ) -> sys::OptixBuildInputInstanceArray {
        sys::OptixBuildInputInstanceArray {
            instances: ia.instances.as_device_ptr(),
            numInstances: ia.num_instances,
            aabbs: if let Some(a) = &ia.aabbs {
                a.as_device_ptr()
            } else {
                0
            },
            numAabbs: ia.num_aabbs,
        }
    }
}

bitflags! {
    pub struct GeometryFlags: u32 {
        const NONE = sys::OptixGeometryFlags::None as u32;
        const DISABLE_ANYHIT = sys::OptixGeometryFlags::DisableAnyHit as u32;
        const REQUIRE_SINGLE_ANYHIT_CALL = sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32;
    }
}

bitflags! {
    pub struct BuildFlags: u32 {
        const NONE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_NONE;
        const ALLOW_UPDATE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        const ALLOW_COMPACTION = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        const PREFER_FAST_TRACE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        const FAST_BUILD = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        const ALLOW_RANDOM_ACCESS_VERTEX = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    }
}

#[repr(u32)]
pub enum BuildOperation {
    Build = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_BUILD,
    Update = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_UPDATE,
}

bitflags! {
    pub struct MotionFlags: u16 {
        const NONE = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_NONE as u16;
        const START_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_START_VANISH as u16;
        const END_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_END_VANISH as u16;
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct MotionOptions {
    pub num_keys: u16,
    pub flags: MotionFlags,
    pub time_begin: f32,
    pub time_end: f32,
}

impl From<MotionOptions> for sys::OptixMotionOptions {
    fn from(o: MotionOptions) -> sys::OptixMotionOptions {
        sys::OptixMotionOptions {
            numKeys: o.num_keys,
            flags: o.flags.bits(),
            timeBegin: o.time_begin,
            timeEnd: o.time_end,
        }
    }
}

#[repr(C)]
pub struct AccelBuildOptions {
    pub build_flags: BuildFlags,
    pub operation: BuildOperation,
    pub motion_options: MotionOptions,
}

impl From<AccelBuildOptions> for sys::OptixAccelBuildOptions {
    fn from(o: AccelBuildOptions) -> sys::OptixAccelBuildOptions {
        sys::OptixAccelBuildOptions {
            buildFlags: o.build_flags.bits(),
            operation: o.operation as u32,
            motionOptions: o.motion_options.into(),
        }
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone)]
pub struct AccelBufferSizes {
    pub output_size_in_bytes: usize,
    pub temp_size_in_bytes: usize,
    pub temp_update_size_in_bytes: usize,
}

impl DeviceContext {
    pub fn accel_compute_memory_usage<'a, AllocT, V, I>(
        &self,
        accel_options: &AccelBuildOptions,
        build_inputs: &[BuildInput<'a, AllocT, V, I>],
    ) -> Result<Vec<AccelBufferSizes>>
    where
        AllocT: Allocator,
        V: BufferElement,
        I: BufferElement,
    {
        let mut buffer_sizes =
            vec![AccelBufferSizes::default(); build_inputs.len()];

        let build_inputs: Vec<sys::OptixBuildInput> =
            build_inputs.into_iter().map(|b| b.into()).collect();

        let res = unsafe {
            sys::optixAccelComputeMemoryUsage(
                self.ctx,
                accel_options as *const AccelBuildOptions
                    as *const sys::OptixAccelBuildOptions,
                build_inputs.as_ptr(),
                build_inputs.len() as u32,
                buffer_sizes.as_mut_ptr() as *mut sys::OptixAccelBufferSizes,
            )
        };

        if res != sys::OptixResult::OPTIX_SUCCESS {
            return Err(Error::AccelComputeMemoryUsageFailed {
                source: res.into(),
            });
        }

        Ok(buffer_sizes)
    }

    pub fn accel_build<'a, 'at, 'ao, 'ae, 'b, AllocT, V, I>(
        &self,
        stream: &cuda::Stream,
        accel_options: &AccelBuildOptions,
        build_inputs: &[BuildInput<'a, AllocT, V, I>],
        temp_buffer: &cuda::Buffer<'at, AllocT>,
        output_buffer: cuda::Buffer<'ao, AllocT>,
        emitted_properties: &[AccelEmitDesc<'ae, 'b, AllocT>],
    ) -> Result<TraversableHandle<'ao, AllocT>>
    where
        AllocT: Allocator,
        V: BufferElement,
        I: BufferElement,
    {
        let build_inputs: Vec<sys::OptixBuildInput> =
            build_inputs.into_iter().map(|b| b.into()).collect();

        let ep: Vec<sys::OptixAccelEmitDesc> = emitted_properties
            .iter()
            .map(|e| sys::OptixAccelEmitDesc {
                result: e.result.as_device_ptr(),
                type_: e.type_ as u32,
            })
            .collect();
        unsafe {
            let mut hnd = 0;

            let res = sys::optixAccelBuild(
                self.ctx,
                stream.as_sys_ptr(),
                accel_options as *const AccelBuildOptions
                    as *const sys::OptixAccelBuildOptions,
                build_inputs.as_ptr(),
                build_inputs.len() as u32,
                temp_buffer.as_device_ptr(),
                temp_buffer.byte_size(),
                output_buffer.as_device_ptr(),
                output_buffer.byte_size(),
                &mut hnd,
                ep.as_ptr(),
                ep.len() as u32,
            );

            if res != sys::OptixResult::OPTIX_SUCCESS {
                return Err(Error::AccelBuildFailed { source: res.into() });
            }

            Ok(TraversableHandle {
                hnd,
                _buffer: output_buffer,
            })
        }
    }

    pub fn accel_compact<'ai, 'ao, AllocT>(
        &self,
        stream: &cuda::Stream,
        input_handle: TraversableHandle<'ai, AllocT>,
        output_buffer: cuda::Buffer<'ao, AllocT>,
    ) -> Result<TraversableHandle<'ao, AllocT>>
    where
        AllocT: Allocator,
    {
        unsafe {
            let mut hnd = 0;
            let res = sys::optixAccelCompact(
                self.ctx,
                stream.as_sys_ptr(),
                input_handle.hnd,
                output_buffer.as_device_ptr(),
                output_buffer.byte_size(),
                &mut hnd,
            );

            if res != sys::OptixResult::OPTIX_SUCCESS {
                return Err(Error::AccelCompactFailed { source: res.into() });
            }

            Ok(TraversableHandle {
                hnd,
                _buffer: output_buffer,
            })
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum AccelPropertyType {
    CompactedSize =
        sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
    AABBs = sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_AABBS,
}

pub struct AccelEmitDesc<'a, 'b, AllocT>
where
    AllocT: Allocator,
{
    result: &'b cuda::Buffer<'a, AllocT>,
    type_: AccelPropertyType,
}

impl<'a, 'b, AllocT> AccelEmitDesc<'a, 'b, AllocT>
where
    AllocT: Allocator,
{
    pub fn new(
        result: &'b cuda::Buffer<'a, AllocT>,
        type_: AccelPropertyType,
    ) -> AccelEmitDesc<'a, 'b, AllocT> {
        AccelEmitDesc { result, type_ }
    }
}

pub struct TraversableHandle<'a, AllocT>
where
    AllocT: Allocator,
{
    pub hnd: sys::OptixTraversableHandle,
    _buffer: cuda::Buffer<'a, AllocT>,
}

impl<'a, AllocT> super::DeviceShareable for TraversableHandle<'a, AllocT>
where
    AllocT: Allocator,
{
    type Target = sys::OptixTraversableHandle;
    fn to_device(&self) -> Self::Target {
        self.hnd
    }
    fn cuda_type() -> String {
        "OptixTraversableHandle".into()
    }
}
