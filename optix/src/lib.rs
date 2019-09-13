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
pub use shader_binding_table::{
    SbtRecord, ShaderBindingTable, ShaderBindingTableBuilder,
};

pub mod acceleration;
pub use acceleration::*;

pub mod buffer;
pub use buffer::*;

pub mod math;

/// Initialize the OptiX library function table. This function *MUST* be called
/// before any other optix functions.
pub fn init() -> Result<()> {
    unsafe {
        let res = sys::optixInit();
        if res != sys::OptixResult::OPTIX_SUCCESS {
            return Err(Error::InitializationFailed { cerr: res.into() });
        }

        Ok(())
    }
}

/// Trait to represent a type that can convert itself to a CUDA-compatible
/// target type.
pub trait DeviceShareable {
    type Target: Copy;
    fn to_device(&self) -> Self::Target;
    fn cuda_decl(nested: bool) -> String;
}

impl DeviceShareable for cuda::Buffer {
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.as_device_ptr()
    }
    fn cuda_decl(_: bool) -> String {
        "void*".into()
    }
}

impl DeviceShareable for cuda::TextureObject {
    type Target = cuda::cudaTextureObject_t;
    fn to_device(&self) -> Self::Target {
        self.as_device_ptr()
    }
    fn cuda_decl(_: bool) -> String {
        "cudaTextureObject_t".into()
    }
}

impl DeviceShareable for Option<cuda::TextureObject> {
    type Target = cuda::cudaTextureObject_t;
    fn to_device(&self) -> Self::Target {
        match self {
            Some(t) => t.as_device_ptr(),
            None => 0,
        }
    }
    fn cuda_decl(_: bool) -> String {
        "cudaTextureObject_t".into()
    }
}

impl DeviceShareable for Option<std::rc::Rc<cuda::TextureObject>> {
    type Target = cuda::cudaTextureObject_t;
    fn to_device(&self) -> Self::Target {
        match self {
            Some(t) => t.as_device_ptr(),
            None => 0,
        }
    }
    fn cuda_decl(_: bool) -> String {
        "cudaTextureObject_t".into()
    }
}

impl DeviceShareable for i32 {
    type Target = i32;
    fn to_device(&self) -> i32 {
        *self
    }
    fn cuda_decl(_: bool) -> String {
        "int".into()
    }
}

impl DeviceShareable for bool {
    type Target = bool;
    fn to_device(&self) -> bool {
        *self
    }
    fn cuda_decl(_: bool) -> String {
        "bool".into()
    }
}

/// Wrapper type to represent a variable that is shared between Rust and CUDA.
pub struct SharedVariable<T>
where
    T: DeviceShareable,
{
    var: T,
    buffer: cuda::Buffer,
}

impl<T> SharedVariable<T>
where
    T: DeviceShareable,
{
    /// Create a new SharedVariable, taking ownership of the variable `var`.
    ///
    /// Once the `SharedVariable` has been created, it can be uploaded to the
    /// device using the `SharedVariable::upload()` method. `SharedVariable`
    /// handles management of the underlying CUDA buffer used for device-side
    /// storage. `SharedVariable` implements Deref targeting the wrapped
    /// variable type for easy access.
    pub fn new(var: T) -> Result<SharedVariable<T>> {
        let cvar = var.to_device();
        let buffer = cuda::Buffer::with_data(std::slice::from_ref(&cvar))?;
        Ok(SharedVariable { var, buffer })
    }

    /// Upload the wrapped variable to the device. Any changes to the variable
    /// on the Rust side will not be reflected on the device until this is
    /// called.
    pub fn upload(&mut self) -> Result<()> {
        let cvar = self.var.to_device();
        self.buffer.upload(std::slice::from_ref(&cvar))?;
        Ok(())
    }

    /// Get a reference to the `cuda::Buffer` representing the device-side
    /// storage for this variable.
    pub fn variable_buffer(&self) -> &cuda::Buffer {
        &self.buffer
    }
}

impl<T> std::ops::Deref for SharedVariable<T>
where
    T: DeviceShareable,
{
    type Target = T;
    fn deref(&self) -> &T {
        &self.var
    }
}

impl<T> std::ops::DerefMut for SharedVariable<T>
where
    T: DeviceShareable,
{
    fn deref_mut(&mut self) -> &mut T {
        &mut self.var
    }
}

/// Macro to generate a newtype wrapper with DeviceShareable and Deref
/// implemented
#[macro_export]
macro_rules! wrap_copyable_for_device {
    ($ty:ty, $newtype:ident, $fmt:expr, $cmp:literal) => {
        #[derive(Copy, Clone)]
        pub struct $newtype($ty);

        impl DeviceShareable for $newtype {
            type Target = $newtype;
            fn to_device(&self) -> Self::Target {
                *self
            }
            fn cuda_decl(_: bool) -> String {
                stringify!($ty).into()
            }
        }

        impl BufferElement for $newtype {
            const FORMAT: BufferFormat = $fmt;
            const COMPONENTS: usize = $cmp;
        }

        impl std::ops::Deref for $newtype {
            type Target = $ty;
            fn deref(&self) -> &$ty {
                &self.0
            }
        }

        impl std::ops::DerefMut for $newtype {
            fn deref_mut(&mut self) -> &mut $ty {
                &mut self.0
            }
        }

        impl From<$ty> for $newtype {
            fn from(v: $ty) -> $newtype {
                $newtype(v)
            }
        }
    };
}

/// Macro to generate a newtype wrapper with DeviceShareable and Deref
/// implemented
#[macro_export]
macro_rules! math_type {
    ($ty:ty, $fmt:expr, $cmp:literal) => {
        impl DeviceShareable for $ty {
            type Target = $ty;
            fn to_device(&self) -> Self::Target {
                *self
            }
            fn cuda_decl(_: bool) -> String {
                stringify!($ty).into()
            }
        }

        impl BufferElement for $ty {
            const FORMAT: BufferFormat = $fmt;
            const COMPONENTS: usize = $cmp;
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
