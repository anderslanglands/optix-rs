#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate bitflags;

use optix_sys as sys;

pub mod error;
pub use error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

type Ref<T> = std::sync::Arc<T>;

pub mod device_context;
pub use device_context::DeviceContext;

pub mod module;
pub use module::{
    CompileDebugLevel, CompileOptimizationLevel, Module, ModuleCompileOptions,
    ModuleRef, PipelineCompileOptions,
};

pub mod program_group;
pub use program_group::{
    ProgramGroupDesc, ProgramGroupModule, ProgramGroupRef,
};

pub mod pipeline;
pub use pipeline::{PipelineLinkOptions, PipelineRef};

pub mod shader_binding_table;
pub use shader_binding_table::{ShaderBindingTable, ShaderBindingTableBuilder};

pub mod acceleration;
pub use acceleration::*;

pub fn init() -> Result<()> {
    unsafe {
        let res = sys::optixInit();
        if res != sys::OptixResult::OPTIX_SUCCESS {
            return Err(Error::InitializationFailed { cerr: res.into() });
        }

        Ok(())
    }
}

pub trait SbtRecord: Sized {
    fn pack(&mut self, pg: &ProgramGroupRef);
}

pub trait DeviceShareable {
    type DeviceType: Copy;
    fn to_device(&self) -> Self::DeviceType;
}

impl DeviceShareable for cuda::Buffer {
    type DeviceType = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::DeviceType {
        self.as_device_ptr()
    }
}

impl DeviceShareable for i32 {
    type DeviceType = i32;
    fn to_device(&self) -> i32 {
        *self
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
