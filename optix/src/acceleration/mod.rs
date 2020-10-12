use crate::{
    sys, DeviceContext, DeviceCopy, DeviceStorage, Error, TypedBuffer,
};
use cu::DeviceAllocRef;
type Result<T, E = Error> = std::result::Result<T, E>;
use smallvec::SmallVec;

pub mod triangle_array;
pub use triangle_array::*;

pub mod instance_array;
pub use instance_array::*;

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

unsafe impl DeviceCopy for TraversableHandle {}

impl DeviceContext {
    /// Computes the device memory required for temporary and output buffers
    /// when building the acceleration structure. Use the returned sizes to
    /// allocate enough memory to pass to `accel_build()`
    pub fn accel_compute_memory_usage<T: BuildInputTriangleArray, I: BuildInputInstanceArray>(
        &self,
        accel_options: &[AccelBuildOptions],
        build_inputs: &[BuildInput<T, I>],
    ) -> Result<AccelBufferSizes> {
        let mut buffer_sizes = AccelBufferSizes::default();
        let build_sys: SmallVec<[_; 4]> = build_inputs.iter().map(|b| {
            match b {
                BuildInput::TriangleArray(bita) => {
                    sys::OptixBuildInput {
                        type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                        input: sys::OptixBuildInputUnion{
                            triangle_array: bita.to_sys(),
                        },
                    }
                }
                BuildInput::InstanceArray(biia) => {
                    sys::OptixBuildInput {
                        type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                        input: sys::OptixBuildInputUnion {
                            instance_array: biia.to_sys(),
                        }
                    }
                }
                _ => unimplemented!(),
            }
        }).collect();

        unsafe {
            sys::optixAccelComputeMemoryUsage(
                self.inner,
                accel_options.as_ptr() as *const _,
                build_sys.as_ptr(),
                build_sys.len() as u32,
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
    pub fn accel_build<T: BuildInputTriangleArray, I: BuildInputInstanceArray, S1: DeviceStorage, S2: DeviceStorage>(
        &self,
        stream: &cu::Stream,
        accel_options: &[AccelBuildOptions],
        build_inputs: &[BuildInput<T, I>],
        temp_buffer: &S1,
        output_buffer: &S2,
        emitted_properties: &mut [AccelEmitDesc],
    ) -> Result<TraversableHandle> {
        let mut traversable_handle = TraversableHandle { inner: 0 };
        let properties: Vec<sys::OptixAccelEmitDesc> =
            emitted_properties.iter().map(|p| p.into()).collect();

        let build_sys: SmallVec<[_; 4]> = build_inputs.iter().map(|b| {
            match b {
                BuildInput::TriangleArray(bita) => {
                    sys::OptixBuildInput {
                        type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                        input: sys::OptixBuildInputUnion{
                            triangle_array: bita.to_sys(),
                        },
                    }
                }
                BuildInput::InstanceArray(biia) => {
                    sys::OptixBuildInput {
                        type_: sys::OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_INSTANCES,
                        input: sys::OptixBuildInputUnion{
                            instance_array: biia.to_sys(),
                        },
                    }
                }
                _ => unimplemented!(),
            }
        }).collect();

        unsafe {
            sys::optixAccelBuild(
                self.inner,
                stream.inner(),
                accel_options.as_ptr() as *const _,
                build_sys.as_ptr(),
                build_sys.len() as u32,
                temp_buffer.device_ptr().0,
                temp_buffer.byte_size(),
                output_buffer.device_ptr().0,
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
                output_buffer.device_ptr().0,
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

pub struct TriangleArrayDefault;
impl BuildInputTriangleArray for TriangleArrayDefault {
    fn to_sys(&self) -> sys::OptixBuildInputTriangleArray {
        unreachable!()
    }
}
pub struct InstanceArrayDefault;
impl BuildInputInstanceArray for InstanceArrayDefault {
    fn to_sys(&self) -> sys::OptixBuildInputInstanceArray {
        unreachable!()
    }
}

pub enum BuildInput<T: BuildInputTriangleArray = TriangleArrayDefault, I: BuildInputInstanceArray = InstanceArrayDefault> {
    TriangleArray(T),
    CurveArray,
    CustomPrimitiveArray,
    InstanceArray(I),
}

// impl<'i, A: DeviceAllocRef> BuildInput<(),InstanceArray<'i, A>> {
//     pub fn instance_array<'i, A: DeviceAllocRef>(instances: &'i TypedBuffer<Instance, A>) -> BuildInput<(), InstanceArray<'i, A>> {
//         BuildInput::<(), InstanceArray<'i, A>>::InstanceArray(InstanceArray::new(instances))
//     }
// }

pub enum AccelEmitDesc {
    CompactedSize(cu::DevicePtr),
    Aabbs(cu::DevicePtr),
}

impl From<&AccelEmitDesc> for sys::OptixAccelEmitDesc {
    fn from(aed: &AccelEmitDesc) -> Self {
        match aed {
            AccelEmitDesc::CompactedSize(p) => Self {
                result: p.0,
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_COMPACTED_SIZE,
            },
            AccelEmitDesc::Aabbs(p) => Self {
                result: p.0,
                type_: sys::OptixAccelPropertyType_OPTIX_PROPERTY_TYPE_AABBS,
            }
        }
    }
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum GeometryFlags {
    None = sys::OptixGeometryFlags::None as u32,
    DisableAnyHit = sys::OptixGeometryFlags::DisableAnyHit as u32,
    RequireSingleAnyHitCall =
        sys::OptixGeometryFlags::RequireSingleAnyHitCall as u32,
}