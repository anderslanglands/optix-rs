#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![feature(untagged_unions)]
pub mod sys;

pub mod error;
pub use error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

pub mod device_context;
pub use device_context::DeviceContext;

pub mod module;
pub use module::{
    CompileDebugLevel, CompileOptimizationLevel, ExceptionFlags, Module,
    ModuleCompileOptions, PipelineCompileOptions, TraversableGraphFlags,
};

pub mod program_group;
pub use program_group::{
    ProgramGroup, ProgramGroupDesc, ProgramGroupModule, StackSizes,
};

pub mod pipeline;
pub use pipeline::{Pipeline, PipelineLinkOptions};

pub mod shader_binding_table;
pub use shader_binding_table::{SbtRecord, ShaderBindingTable};

pub mod acceleration;
pub use acceleration::{
    AccelBuildOptions, AccelEmitDesc, BuildFlags, BuildInput, BuildOperation,
    GeometryFlags, IndexTriple, IndicesFormat, MotionFlags, TraversableHandle,
    TriangleArray, Vertex, VertexFormat,
};

pub mod buffer;
pub use buffer::{Buffer, DeviceStorage, DeviceVariable, TypedBuffer};

pub mod image;
pub use self::image::{Image2D, Pixel, PixelFormat};

pub mod denoiser;
pub use denoiser::{
    Denoiser, DenoiserInputKind, DenoiserOptions, DenoiserParams, DenoiserSizes,
};

pub mod math;
pub use math::*;

/// Initialize the OptiX library function table. This function *MUST* be called
/// before any other optix functions.
pub fn init() -> Result<()> {
    unsafe {
        sys::optixInit()
            .to_result()
            .map_err(|source| Error::Initialization { source })?;

        Ok(())
    }
}

pub fn launch<S: DeviceStorage>(
    pipeline: &Pipeline,
    stream: &cu::Stream,
    buf_launch_params: &S,
    sbt: &sys::OptixShaderBindingTable,
    width: u32,
    height: u32,
    depth: u32,
) -> Result<()> {
    unsafe {
        sys::optixLaunch(
            pipeline.inner,
            stream.inner(),
            buf_launch_params.device_ptr().ptr(),
            buf_launch_params.byte_size(),
            sbt,
            width,
            height,
            depth,
        )
        .to_result()
        .map_err(|source| Error::Launch { source })
    }
}

/// Specifies the size of the opaque SBT record header data
pub const SBT_RECORD_HEADER_SIZE: usize =
    sys::OptixSbtRecordHeaderSize as usize;
/// Specifies the alignment requirement for SBT records
pub const SBT_RECORD_ALIGNMENT: usize = sys::OptixSbtRecordAlignment as usize;
/// Specifies the required alignment for temporary and output buffers used for
/// acceleration structures
pub const ACCEL_BUFFER_BYTE_ALIGNMENT: usize =
    sys::OptixAccelBufferByteAlignment as usize;
pub const INSTANCE_BYTE_ALIGNMENT: usize =
    sys::OptixInstanceByteAlignment as usize;
pub const AABB_BUFFER_BYTE_ALIGNMENT: usize =
    sys::OptixAabbBufferByteAlignment as usize;
pub const GEOMETRY_TRANSFORM_BYTE_ALIGNMENT: usize =
    sys::OptixGeometryTransformByteAlignment as usize;
pub const TRANSFORM_BYTE_ALIGNMENT: usize =
    sys::OptixTransformByteAlignment as usize;

#[cfg(test)]
mod tests {
    use crate as optix;
    #[test]
    fn it_works() -> Result<(), Box<dyn std::error::Error>> {
        cu::init()?;
        optix::init()?;
        Ok(())
    }
}

/// This trait specifies that the type in question can safely be bitwise-copied
/// to the device.
pub unsafe trait DeviceCopy: Sized {
    /// Use this to specify any alignment requirements for the type. For
    /// example, anything that translates to float4 on the CUDA side wants
    /// to be aligned to 16 bytes, transformation matrices want to be
    /// aligned to 64 bytes, etc.
    fn device_align() -> usize {
        std::mem::align_of::<Self>()
    }
}

#[macro_export]
macro_rules! impl_device_copy {
    ($($t:ty)*) => {
        $(
            unsafe impl DeviceCopy for $t {}
        )*
    }
}

#[macro_export]
macro_rules! impl_device_copy_align {
    ($($t:ty : $a:expr)*) => {
        $(
            unsafe impl DeviceCopy for $t {
                fn device_align() -> usize {
                    $a
                }
            }
        )*
    }
}

impl_device_copy!(
    usize u8 u16 u32 u64 u128
    isize i8 i16 i32 i64 i128
    f32 f64
    bool char

    std::num::NonZeroU8 std::num::NonZeroU16 std::num::NonZeroU32 std::num::NonZeroU64 std::num::NonZeroU128
);
unsafe impl<T: DeviceCopy> DeviceCopy for Option<T> {}
unsafe impl<L: DeviceCopy, R: DeviceCopy> DeviceCopy for Result<L, R> {}
unsafe impl<T: ?Sized + DeviceCopy> DeviceCopy for std::marker::PhantomData<T> {}
unsafe impl<T: DeviceCopy> DeviceCopy for std::num::Wrapping<T> {}

macro_rules! impl_device_copy_array {
    ($($n:expr)*) => {
        $(
            unsafe impl<T: DeviceCopy> DeviceCopy for [T;$ n] {}
        )*
    }
}

impl_device_copy_array! {
    1 2 3 4 5 6 7 8 9 10
    11 12 13 14 15 16 17 18 19 20
    21 22 23 24 25 26 27 28 29 30
    31 32
}
unsafe impl DeviceCopy for () {}
unsafe impl<A: DeviceCopy, B: DeviceCopy> DeviceCopy for (A, B) {}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy> DeviceCopy
    for (A, B, C)
{
}
unsafe impl<A: DeviceCopy, B: DeviceCopy, C: DeviceCopy, D: DeviceCopy>
    DeviceCopy for (A, B, C, D)
{
}
unsafe impl<
        A: DeviceCopy,
        B: DeviceCopy,
        C: DeviceCopy,
        D: DeviceCopy,
        E: DeviceCopy,
    > DeviceCopy for (A, B, C, D, E)
{
}
unsafe impl<
        A: DeviceCopy,
        B: DeviceCopy,
        C: DeviceCopy,
        D: DeviceCopy,
        E: DeviceCopy,
        F: DeviceCopy,
    > DeviceCopy for (A, B, C, D, E, F)
{
}
unsafe impl<
        A: DeviceCopy,
        B: DeviceCopy,
        C: DeviceCopy,
        D: DeviceCopy,
        E: DeviceCopy,
        F: DeviceCopy,
        G: DeviceCopy,
    > DeviceCopy for (A, B, C, D, E, F, G)
{
}
unsafe impl<
        A: DeviceCopy,
        B: DeviceCopy,
        C: DeviceCopy,
        D: DeviceCopy,
        E: DeviceCopy,
        F: DeviceCopy,
        G: DeviceCopy,
        H: DeviceCopy,
    > DeviceCopy for (A, B, C, D, E, F, G, H)
{
}
