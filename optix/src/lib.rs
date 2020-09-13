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
pub use program_group::{ProgramGroup, ProgramGroupDesc, ProgramGroupModule};

pub mod pipeline;
pub use pipeline::{Pipeline, PipelineLinkOptions};

pub mod shader_binding_table;
pub use shader_binding_table::{SbtRecord, ShaderBindingTable};

/// Initialize the OptiX library function table. This function *MUST* be called
/// before any other optix functions.
pub fn init() -> Result<()> {
    unsafe {
        sys::optixInit()
            .to_result()
            .map_err(|source| Error::InitializationFailed { source })?;

        Ok(())
    }
}

pub fn launch<T: cu::DeviceCopy>(
    pipeline: &Pipeline,
    stream: &cu::Stream,
    buf_launch_params: &cu::TypedBuffer<T>,
    sbt: &sys::OptixShaderBindingTable,
    width: u32,
    height: u32,
    depth: u32,
) -> Result<()> {

    unsafe {
        sys::optixLaunch(
            pipeline.inner,
            stream.inner(),
            buf_launch_params.device_ptr().device_ptr(),
            buf_launch_params.byte_size(),
            sbt,
            width, height, depth
        ).to_result().map_err(|source| Error::LaunchFailed{source})
    }
}

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
