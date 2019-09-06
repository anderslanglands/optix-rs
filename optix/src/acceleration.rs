use optix_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

pub use super::device_context::DeviceContext;

use std::convert::{TryFrom, TryInto};
use std::ffi::{CStr, CString};

pub enum BuildInput<'b> {
    Triangle(TriangleArray<'b>),
    CustomPrimitive(CustomPrimitiveArray),
    Instance(InstanceArray),
}

impl<'b> From<&BuildInput<'b>> for sys::OptixBuildInput {
    fn from(b: &BuildInput) -> sys::OptixBuildInput {
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
            _ => unimplemented!(),
        }
    }
}

pub struct TriangleArray<'b> {
    vertex_buffers: &'b TypedBufferArray,
    index_buffer: &'b TypedBuffer,
    flags: GeometryFlags,
}

impl<'b> TriangleArray<'b> {
    pub fn new(
        vertex_buffers: &'b TypedBufferArray,
        index_buffer: &'b TypedBuffer,
        flags: GeometryFlags,
    ) -> TriangleArray<'b> {
        TriangleArray {
            vertex_buffers,
            index_buffer,
            flags,
        }
    }
}

impl<'b> TryFrom<&TriangleArray<'b>> for sys::OptixBuildInputTriangleArray {
    type Error = Error;

    #[allow(non_snake_case)]
    fn try_from(
        ta: &TriangleArray,
    ) -> Result<sys::OptixBuildInputTriangleArray> {
        let vertexBuffers = ta.vertex_buffers.buffers.as_device_ptr();
        let numVertices = ta.vertex_buffers.count as u32;
        let vertexFormat = match &ta.vertex_buffers.format {
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
                    format: ta.vertex_buffers.format,
                })
            }
        };
        let indexBuffer = ta.index_buffer.buffer.as_device_ptr();
        let numIndexTriplets = ta.index_buffer.count as u32;
        let indexFormat = match &ta.index_buffer.format {
            BufferFormat::I32x3 => {
                sys::OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3
            }
            BufferFormat::U16x3 => {
                sys::OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3
            }
            _ => {
                return Err(Error::IncorrectIndexBufferFormat {
                    format: ta.index_buffer.format,
                })
            }
        };

        Ok(sys::OptixBuildInputTriangleArray {
            vertexBuffers,
            numVertices,
            vertexFormat,
            vertexStrideInBytes: format_byte_size(ta.vertex_buffers.format)
                as u32,
            indexBuffer,
            numIndexTriplets,
            indexStrideInBytes: format_byte_size(ta.index_buffer.format) as u32,
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

pub struct CustomPrimitiveArray {}

pub struct InstanceArray {}

pub struct TypedBuffer {
    buffer: cuda::Buffer,
    count: usize,
    format: BufferFormat,
}

impl TypedBuffer {
    pub fn new<T>(data: &[T], format: BufferFormat) -> Result<TypedBuffer> {
        let buffer = cuda::Buffer::with_data(data)?;

        Ok(TypedBuffer {
            buffer,
            count: data.len(),
            format,
        })
    }

    pub fn as_ptr(&self) -> *const std::os::raw::c_void {
        self.buffer.as_ptr()
    }
}

use super::DeviceShareable;
impl DeviceShareable for TypedBuffer {
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.buffer.as_device_ptr()
    }
}

pub struct TypedBufferArray {
    buffers: cuda::BufferArray,
    count: usize,
    format: BufferFormat,
}

impl TypedBufferArray {
    pub fn new(buffers: Vec<TypedBuffer>) -> Result<TypedBufferArray> {
        let count = buffers[0].count;
        let format = buffers[0].format;
        let buffers: Vec<cuda::Buffer> =
            buffers.into_iter().map(|b| b.buffer).collect();
        let buffers = cuda::BufferArray::new(buffers);
        Ok(TypedBufferArray {
            buffers,
            count,
            format,
        })
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum BufferFormat {
    U8,
    U8x2,
    U8x3,
    U8x4,
    U16,
    U16x2,
    U16x3,
    U16x4,
    F16,
    F16x2,
    F16x3,
    F16x4,
    F32,
    F32x2,
    F32x3,
    F32x4,
    I32,
    I32x2,
    I32x3,
    I32x4,
}

fn format_byte_size(f: BufferFormat) -> usize {
    match f {
        BufferFormat::U8 => 1,
        BufferFormat::U8x2 => 2,
        BufferFormat::U8x3 => 3,
        BufferFormat::U8x4 => 4,
        BufferFormat::U16 => 2,
        BufferFormat::U16x2 => 4,
        BufferFormat::U16x3 => 6,
        BufferFormat::U16x4 => 8,
        BufferFormat::F16 => 2,
        BufferFormat::F16x2 => 4,
        BufferFormat::F16x3 => 6,
        BufferFormat::F16x4 => 8,
        BufferFormat::F32 => 4,
        BufferFormat::F32x2 => 8,
        BufferFormat::F32x3 => 12,
        BufferFormat::F32x4 => 16,
        BufferFormat::I32 => 4,
        BufferFormat::I32x2 => 8,
        BufferFormat::I32x3 => 12,
        BufferFormat::I32x4 => 16,
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
    pub fn accel_compute_memory_usage(
        &self,
        accel_options: &AccelBuildOptions,
        build_inputs: &[BuildInput],
    ) -> Result<Vec<AccelBufferSizes>> {
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
                cerr: res.into(),
            });
        }

        Ok(buffer_sizes)
    }

    pub fn accel_build(
        &self,
        stream: &cuda::Stream,
        accel_options: &AccelBuildOptions,
        build_inputs: &[BuildInput],
        temp_buffer: &cuda::Buffer,
        output_buffer: cuda::Buffer,
        emitted_properties: &[AccelEmitDesc],
    ) -> Result<TraversableHandle> {
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
                return Err(Error::AccelBuildFailed { cerr: res.into() });
            }

            Ok(TraversableHandle {
                hnd,
                buffer: output_buffer,
            })
        }
    }

    pub fn accel_compact(
        &self,
        stream: &cuda::Stream,
        input_handle: TraversableHandle,
        output_buffer: cuda::Buffer,
    ) -> Result<TraversableHandle> {
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
                return Err(Error::AccelCompactFailed { cerr: res.into() });
            }

            Ok(TraversableHandle {
                hnd,
                buffer: output_buffer,
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

pub struct AccelEmitDesc<'b> {
    result: &'b cuda::Buffer,
    type_: AccelPropertyType,
}

impl<'b> AccelEmitDesc<'b> {
    pub fn new(
        result: &'b cuda::Buffer,
        type_: AccelPropertyType,
    ) -> AccelEmitDesc {
        AccelEmitDesc { result, type_ }
    }
}

pub struct TraversableHandle {
    pub hnd: sys::OptixTraversableHandle,
    buffer: cuda::Buffer,
}

impl super::DeviceShareable for TraversableHandle {
    type Target = sys::OptixTraversableHandle;
    fn to_device(&self) -> Self::Target {
        self.hnd
    }
}
