#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
#![feature(untagged_unions)]
//! # Introduction
//! Rust bindings for [NVidia's Optix 7.1 raytracing API](https://raytracing-docs.nvidia.com)
//!
//! These bindings are *not safe*. The OptiX API has you construct a scene graph
//! of objects referencing each other by pointers. It's certainly possible to
//! wrap these up in safe objects using lifetimes or Rc's to track ownership
//! (and indeed an earlier version of this crate did just that) but doing so
//! means imposing constrictive design decisions on the user, which may or may
//! not be acceptable for their use case.
//!
//! Instead, what this crate provides is a thin, ergonomic wrapper around the C
//! API and leaves it up to the user to build their preferred ownership
//! abstractions around it.
//!
//! This means that:
//! - Related functionality is grouped onto struct methods, e.g. [ProgramGroup],
//!   [Pipeline] etc.
//! - Configuration is simplified by the addition of builders for complex config
//!   objects, or by providing higher-level functions for common functionality
//! - Rather than mutable out pointers, all functions return `Result<T,
//!   optix::Error>`, where [optix::Error](Error) implements [std::error::Error]
//! - Device memory management is eased with utility types such as [TypedBuffer]
//!   and [DeviceVariable], and functions that expect references to device
//!   memory are generic over the storage type.
//! - All functions that allocate are generic over a [cu::DeviceAllocRef], which
//!   allows the user to provide their own allocators. The design for this is
//!   very similar to wg-allocator. The default allocator simply calls through
//!   to `cuMemAlloc` and the underlying [cuda crate](crate::cu) also provides a
//!   simple bump allocator as an example, that is also used by the example
//!   programs.
//!
//! # Examples
//! The examples are a direct translation of Ingo Wald's OptiX 7 Siggraph course
//! and it is highly recommended you read them to see how the crate works
//!
//! # Building
//! In order to build the crate you must have the Optix 7.1 SDK and a recent
//! version of the CUDA SDK installed. The build script expects to be able to
//! find these using the environment variables `OPTIX_ROOT` and `CUDA_ROOT`,
//! respectively. You'll also need at least driver version 450 installed.
//!
//! ```
//! env OPTIX_ROOT=/path/to/optix CUDA_ROOT=/usr/local/cuda-11.0 cargo build
//! ```

//! This crate has been tested on Linux only. It *should* work on Windows, and I
//! will gratefully accept PRs for fixing any issues there.
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
    AccelBuildOptions, AccelEmitDesc, BuildFlags, BuildInput,
    BuildInputInstanceArray, BuildOperation, GeometryFlags, IndexTriple,
    IndicesFormat, InstanceArray, InstanceFlags, MotionFlags,
    TraversableHandle, TriangleArray, Vertex, VertexFormat, Instance
};

pub mod buffer;
pub use buffer::{Buffer, DeviceStorage, DeviceVariable, TypedBuffer, BufferSlice};

pub mod image;
pub use self::image::{Image2D, Pixel, PixelFormat};

pub mod denoiser;
pub use denoiser::{
    Denoiser, DenoiserInputKind, DenoiserOptions, DenoiserParams, DenoiserSizes,
};

pub mod math;
pub use math::*;

pub use cu;

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

/// Launch the given [Pipeline] on the given [Stream](cu::Stream).
///
/// # Safety
/// You must ensure that:
/// - Any [ProgramGroup]s reference by the [Pipeline] are still alive
/// - Any [DevicePtr]s contained in `buf_launch_params` point to valid,
///   correctly aligned memory
/// - Any [SbtRecord]s and associated data referenced by the
///   [OptixShaderBindingTable] are alive and valid
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
            buf_launch_params.device_ptr().0,
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
