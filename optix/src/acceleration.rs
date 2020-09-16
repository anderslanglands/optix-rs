use crate::{sys, DeviceContext, DeviceStorage, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

/// Opaque handle to a traversable acceleration structure.
/// # Safety
/// You should consider this handle to be a raw pointer, thus you can copy it
/// and it provides no tracking of lifetime or ownership. You are responsible
/// for ensuring that the device memory containing the acceleration structures
/// this handle references are alive if you try to use this handle
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TraversableHandle {
    pub(crate) inner: u64,
}

impl DeviceContext {
    /// Computes the device memory required for temporary and output buffers
    /// when building the acceleration structure. Use the returned sizes to
    /// allocate enough memory to pass to `accel_build()`
    pub fn accel_compute_memory_usage(
        &self,
        accel_options: &[AccelBuildOptions],
        build_inputs: &[sys::OptixBuildInput],
    ) -> Result<AccelBufferSizes> {
        let mut buffer_sizes = AccelBufferSizes::default();
        unsafe {
            sys::optixAccelComputeMemoryUsage(
                self.inner,
                accel_options.as_ptr() as *const _,
                build_inputs.as_ptr(),
                build_inputs.len() as u32,
                &mut buffer_sizes as *mut _ as *mut _,
            )
            .to_result()
            .map(|_| buffer_sizes)
            .map_err(|source| Error::AccelComputeMemoryUsage { source })
        }
    }

    /// Builds the acceleration structure.
    /// `temp_buffer` and `output_buffer` must be at least as large as the sizes
    /// returned by `accel_compute_memory_usage()`
    pub fn accel_build<S1: DeviceStorage, S2: DeviceStorage>(
        &self,
        stream: &cu::Stream,
        accel_options: &[AccelBuildOptions],
        build_inputs: &[sys::OptixBuildInput],
        temp_buffer: &S1,
        output_buffer: &S2,
        emitted_properties: &mut [AccelEmitDesc],
    ) -> Result<TraversableHandle> {
        let mut traversable_handle = TraversableHandle { inner: 0 };
        let properties: Vec<sys::OptixAccelEmitDesc> =
            emitted_properties.iter().map(|p| p.into()).collect();
        unsafe {
            sys::optixAccelBuild(
                self.inner,
                stream.inner(),
                accel_options.as_ptr() as *const _,
                build_inputs.as_ptr(),
                build_inputs.len() as u32,
                temp_buffer.device_ptr().ptr(),
                temp_buffer.byte_size(),
                output_buffer.device_ptr().ptr(),
                output_buffer.byte_size(),
                &mut traversable_handle as *mut _ as *mut _,
                properties.as_ptr() as *const _,
                properties.len() as u32,
            )
            .to_result()
            .map(|_| traversable_handle)
            .map_err(|source| Error::AccelBuild { source })
        }
    }

    /// Compacts the acceleration structure referenced by `input_handle`,
    /// storing the result in `output_buffer` and returning a handle to the
    /// newly compacted structure
    pub fn accel_compact<S: DeviceStorage>(
        &self,
        stream: &cu::Stream,
        input_handle: TraversableHandle,
        output_buffer: &S,
    ) -> Result<TraversableHandle> {
        let mut traversable_handle = TraversableHandle { inner: 0 };
        unsafe {
            sys::optixAccelCompact(
                self.inner,
                stream.inner(),
                input_handle.inner,
                output_buffer.device_ptr().ptr(),
                output_buffer.byte_size(),
                &mut traversable_handle as *mut _ as *mut _,
            )
            .to_result()
            .map(|_| traversable_handle)
            .map_err(|source| Error::AccelCompact { source })
        }
    }
}

bitflags::bitflags! {
    pub struct BuildFlags: u32 {
        const NONE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_NONE;
        const ALLOW_UPDATE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_UPDATE;
        const ALLOW_COMPACTION = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        const PREFER_FAST_TRACE = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        const PREFER_FAST_BUILD = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
        const ALLOW_RANDOM_VERTEX_ACCESS = sys::OptixBuildFlags_OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BuildOperation {
    Build = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_BUILD,
    Update = sys::OptixBuildOperation_OPTIX_BUILD_OPERATION_UPDATE,
}

bitflags::bitflags! {
    pub struct MotionFlags: u16 {
        const NONE = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_NONE as u16;
        const START_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_START_VANISH as u16;
        const END_VANISH = sys::OptixMotionFlags_OPTIX_MOTION_FLAG_END_VANISH as u16;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MotionOptions {
    pub num_keys: u16,
    pub flags: MotionFlags,
    pub time_begin: f32,
    pub time_end: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AccelBuildOptions {
    build_flags: BuildFlags,
    operation: BuildOperation,
    motion_options: MotionOptions,
}

impl AccelBuildOptions {
    pub fn new(build_flags: BuildFlags, operation: BuildOperation) -> Self {
        AccelBuildOptions {
            build_flags,
            operation,
            motion_options: MotionOptions {
                num_keys: 1,
                flags: MotionFlags::NONE,
                time_begin: 0.0f32,
                time_end: 1.0f32,
            },
        }
    }

    pub fn num_keys(mut self, num_keys: u16) -> Self {
        self.motion_options.num_keys = num_keys;
        self
    }

    pub fn time_interval(mut self, time_begin: f32, time_end: f32) -> Self {
        self.motion_options.time_begin = time_begin;
        self.motion_options.time_end = time_end;
        self
    }

    pub fn motion_flags(mut self, flags: MotionFlags) -> Self {
        self.motion_options.flags = flags;
        self
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct AccelBufferSizes {
    pub output_size_in_bytes: usize,
    pub temp_size_in_bytes: usize,
    pub temp_update_size_in_bytes: usize,
}

pub enum BuildInput {
    TriangleArray(BuildInputTriangleArray),
    CurveArray,
    CustomPrimitiveArray,
    InstanceArray,
}

pub enum AccelEmitDesc {
    CompactedSize(cu::DevicePtr),
    Aabbs(cu::DevicePtr),
}

impl From<&AccelEmitDesc> for sys::OptixAccelEmitDesc {
    fn from(aed: &AccelEmitDesc) -> Self {
        match aed {
            AccelEmitDesc::CompactedSize(p) => Self {
                result: p.ptr(),
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
            },
            AccelEmitDesc::Aabbs(p) => Self {
                result: p.ptr(),
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_AABBS,
            }
        }
    }
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum VertexFormat {
    None = sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_NONE as u32,
    Float3 = sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3 as u32,
    Float2 = sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT2 as u32,
    Half3 = sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_HALF3 as u32,
    Half2 = sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_HALF2 as u32,
    SNorm16 = sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_SNORM16_3 as u32,
    SNorm32 = sys::OptixVertexFormat::OPTIX_VERTEX_FORMAT_SNORM16_2 as u32,
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum IndicesFormat {
    None = sys::OptixIndicesFormat::OPTIX_INDICES_FORMAT_NONE as u32,
    Short3 =
        sys::OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 as u32,
    Int3 = sys::OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3 as u32,
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum GeometryFlags {
    None = sys::OptixGeometryFlags::None as u32,
    DisableAnyHit = sys::OptixGeometryFlags::DisableAnyHit as u32,
    RequireSingleAnyHitCall =
        sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32,
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum TransformFormat {
    None = sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_NONE,
    MatrixFloat12 =
        sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12,
}

#[repr(C)]
pub struct BuildInputTriangleArray {
    vertex_buffers: *const cu::DevicePtr,
    num_vertices: u32,
    vertex_format: VertexFormat,
    vertex_stride_in_bytes: u32,
    index_buffer: cu::DevicePtr,
    num_index_triplets: u32,
    index_format: IndicesFormat,
    index_stride_in_bytes: u32,
    pre_transform: cu::DevicePtr,
    flags: *const GeometryFlags,
    num_sbt_records: u32,
    sbt_index_offset_buffer: cu::DevicePtr,
    sbt_index_offset_size_in_bytes: u32,
    sbt_index_offset_stride_in_bytes: u32,
    primitive_index_offset: u32,
    transform_format: TransformFormat,
}

impl BuildInputTriangleArray {
    pub fn new(
        vertex_buffers: &[cu::DevicePtr],
        num_vertices: u32,
        vertex_format: VertexFormat,
        flags: &GeometryFlags,
    ) -> Self {
        BuildInputTriangleArray::new_with_stride(
            vertex_buffers,
            num_vertices,
            vertex_format,
            flags,
            0,
        )
    }

    pub fn new_with_stride(
        vertex_buffers: &[cu::DevicePtr],
        num_vertices: u32,
        vertex_format: VertexFormat,
        flags: &GeometryFlags,
        vertex_stride_in_bytes: u32,
    ) -> Self {
        BuildInputTriangleArray {
            vertex_buffers: vertex_buffers.as_ptr(),
            num_vertices,
            vertex_format,
            vertex_stride_in_bytes,
            index_buffer: cu::DevicePtr::null(),
            num_index_triplets: 0,
            index_format: IndicesFormat::None,
            index_stride_in_bytes: 0,
            pre_transform: cu::DevicePtr::null(),
            flags: flags as *const _,
            num_sbt_records: 1,
            sbt_index_offset_buffer: cu::DevicePtr::null(),
            sbt_index_offset_size_in_bytes: 0,
            sbt_index_offset_stride_in_bytes: 0,
            primitive_index_offset: 0,
            transform_format: TransformFormat::None,
        }
    }

    pub fn build(self) -> sys::OptixBuildInput {
        let triangle_array = unsafe {
            std::mem::transmute::<
                BuildInputTriangleArray,
                sys::OptixBuildInputTriangleArray,
            >(self)
        };
        sys::OptixBuildInput {
            type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
            input: sys::OptixBuildInputUnion { triangle_array },
        }
    }

    pub fn pre_transform(mut self, pre_transform: cu::DevicePtr) -> Self {
        self.pre_transform = pre_transform;
        self.transform_format = TransformFormat::MatrixFloat12;
        self
    }

    pub fn index_buffer(
        mut self,
        index_buffer: cu::DevicePtr,
        num_index_triplets: u32,
        index_format: IndicesFormat,
    ) -> Self {
        self.index_buffer = index_buffer;
        self.num_index_triplets = num_index_triplets;
        self.index_format = index_format;
        self.index_stride_in_bytes = 0;
        self
    }

    pub fn index_buffer_with_stride(
        mut self,
        index_buffer: cu::DevicePtr,
        num_index_triplets: u32,
        index_format: IndicesFormat,
        index_stride_in_bytes: u32,
    ) -> Self {
        self.index_buffer = index_buffer;
        self.num_index_triplets = num_index_triplets;
        self.index_format = index_format;
        self.index_stride_in_bytes = index_stride_in_bytes;
        self
    }
}