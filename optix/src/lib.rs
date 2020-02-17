#[macro_use]
extern crate bitflags;

use optix_sys as sys;

pub mod cuda;
use cuda::Allocator;

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
    SbtData, SbtRecord, ShaderBindingTable, ShaderBindingTableBuilder,
};

pub mod acceleration;
pub use acceleration::*;

pub mod buffer;
pub use buffer::*;

pub mod texture;
pub use texture::*;

pub mod instance;
pub use instance::{make_instance, Instance, InstanceFlags};

pub mod math;

/// Initialize the OptiX library function table. This function *MUST* be called
/// before any other optix functions.
pub fn init() -> Result<()> {
    unsafe {
        let res = sys::optixInit();
        if res != sys::OptixResult::OPTIX_SUCCESS {
            return Err(Error::InitializationFailed { source: res.into() });
        }

        Ok(())
    }
}

/// Trait to represent a type that can convert itself to a CUDA-compatible
/// target type.
pub trait DeviceShareable {
    type Target;
    fn to_device(&self) -> Self::Target;
    fn cuda_decl() -> String {
        Self::cuda_type()
    }
    fn cuda_type() -> String;
    fn zero() -> Self::Target;
}

impl<'a, AllocT> DeviceShareable for cuda::Buffer<'a, AllocT>
where
    AllocT: Allocator,
{
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.as_device_ptr()
    }
    fn cuda_type() -> String {
        "void*".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

impl DeviceShareable for cuda::TextureObject {
    type Target = cuda::cudaTextureObject_t;
    fn to_device(&self) -> Self::Target {
        self.as_device_ptr()
    }
    fn cuda_type() -> String {
        "cudaTextureObject_t".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

// impl DeviceShareable for Option<cuda::TextureObject> {
//     type Target = cuda::cudaTextureObject_t;
//     fn to_device(&self) -> Self::Target {
//         match self {
//             Some(t) => t.to_device(),
//             None => 0,
//         }
//     }
//     fn cuda_type() -> String {
//         "cudaTextureObject_t".into()
//     }
//     fn zero() -> Self::Target {
//         0
//     }
// }

// impl DeviceShareable for Option<std::rc::Rc<cuda::TextureObject>> {
//     type Target = cuda::cudaTextureObject_t;
//     fn to_device(&self) -> Self::Target {
//         match self {
//             Some(t) => t.to_device(),
//             None => 0,
//         }
//     }
//     fn cuda_type() -> String {
//         "cudaTextureObject_t".into()
//     }
//     fn zero() -> Self::Target {
//         0
//     }
// }

impl DeviceShareable for i8 {
    type Target = i8;
    fn to_device(&self) -> i8 {
        *self
    }
    fn cuda_type() -> String {
        "int".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

impl DeviceShareable for i16 {
    type Target = i16;
    fn to_device(&self) -> i16 {
        *self
    }
    fn cuda_type() -> String {
        "int".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

impl DeviceShareable for i32 {
    type Target = i32;
    fn to_device(&self) -> i32 {
        *self
    }
    fn cuda_type() -> String {
        "int".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

impl DeviceShareable for i64 {
    type Target = i64;
    fn to_device(&self) -> i64 {
        *self
    }
    fn cuda_type() -> String {
        "int".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

impl DeviceShareable for f32 {
    type Target = f32;
    fn to_device(&self) -> f32 {
        *self
    }
    fn cuda_type() -> String {
        "float".into()
    }
    fn zero() -> Self::Target {
        0.0
    }
}

impl DeviceShareable for f64 {
    type Target = f64;
    fn to_device(&self) -> f64 {
        *self
    }
    fn cuda_type() -> String {
        "float".into()
    }
    fn zero() -> Self::Target {
        0.0
    }
}

impl DeviceShareable for u8 {
    type Target = u8;
    fn to_device(&self) -> u8 {
        *self
    }
    fn cuda_type() -> String {
        "unsigned int".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

impl DeviceShareable for u16 {
    type Target = u16;
    fn to_device(&self) -> u16 {
        *self
    }
    fn cuda_type() -> String {
        "unsigned int".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

impl DeviceShareable for u32 {
    type Target = u32;
    fn to_device(&self) -> u32 {
        *self
    }
    fn cuda_type() -> String {
        "unsigned int".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

impl DeviceShareable for u64 {
    type Target = u64;
    fn to_device(&self) -> u64 {
        *self
    }
    fn cuda_type() -> String {
        "unsigned int".into()
    }
    fn zero() -> Self::Target {
        0
    }
}

impl DeviceShareable for bool {
    type Target = bool;
    fn to_device(&self) -> bool {
        *self
    }
    fn cuda_type() -> String {
        "bool".into()
    }
    fn zero() -> Self::Target {
        false
    }
}

impl<T> DeviceShareable for &T
where
    T: DeviceShareable,
{
    type Target = T::Target;
    fn to_device(&self) -> T::Target {
        (**self).to_device()
    }

    fn cuda_type() -> String {
        T::cuda_type()
    }

    fn cuda_decl() -> String {
        T::cuda_decl()
    }

    fn zero() -> Self::Target {
        T::zero()
    }
}

impl<T> DeviceShareable for std::rc::Rc<T>
where
    T: DeviceShareable,
{
    type Target = T::Target;
    fn to_device(&self) -> T::Target {
        (**self).to_device()
    }

    fn cuda_type() -> String {
        T::cuda_type()
    }

    fn cuda_decl() -> String {
        T::cuda_decl()
    }

    fn zero() -> Self::Target {
        T::zero()
    }
}

impl<T> DeviceShareable for std::rc::Rc<std::cell::RefCell<T>>
where
    T: DeviceShareable,
{
    type Target = T::Target;
    fn to_device(&self) -> T::Target {
        self.borrow().to_device()
    }

    fn cuda_type() -> String {
        T::cuda_type()
    }

    fn cuda_decl() -> String {
        T::cuda_decl()
    }

    fn zero() -> Self::Target {
        T::zero()
    }
}

impl<T> DeviceShareable for Option<T>
where
    T: DeviceShareable,
{
    type Target = T::Target;
    fn to_device(&self) -> T::Target {
        match self {
            Some(t) => t.to_device(),
            None => T::zero(),
        }
    }

    fn cuda_type() -> String {
        T::cuda_type()
    }

    fn cuda_decl() -> String {
        T::cuda_decl()
    }

    fn zero() -> Self::Target {
        T::zero()
    }
}

/// Wrapper type to represent a variable that is shared between Rust and CUDA.
pub struct SharedVariable<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    var: T,
    buffer: cuda::Buffer<'a, AllocT>,
}

impl<'a, AllocT, T> SharedVariable<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    /// Create a new SharedVariable, taking ownership of the variable `var`.
    ///
    /// Once the `SharedVariable` has been created, it can be uploaded to the
    /// device using the `SharedVariable::upload()` method. `SharedVariable`
    /// handles management of the underlying CUDA buffer used for device-side
    /// storage. `SharedVariable` implements Deref targeting the wrapped
    /// variable type for easy access.
    pub fn new(
        var: T,
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<SharedVariable<'a, AllocT, T>> {
        let cvar = var.to_device();
        let buffer = cuda::Buffer::with_data(
            std::slice::from_ref(&cvar),
            // FIXME: let trait implementors declare their preferred alignment
            std::mem::align_of::<T>(),
            tag,
            allocator,
        )?;
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
    pub fn variable_buffer(&self) -> &cuda::Buffer<'a, AllocT> {
        &self.buffer
    }
}

impl<'a, AllocT, T> DeviceShareable for SharedVariable<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.buffer.as_device_ptr()
    }

    fn cuda_type() -> String {
        format!("{}*", T::cuda_type())
    }

    fn cuda_decl() -> String {
        format!("{}*", T::cuda_decl())
    }

    fn zero() -> Self::Target {
        0
    }
}

impl<'a, AllocT, T> std::ops::Deref for SharedVariable<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    type Target = T;
    fn deref(&self) -> &T {
        &self.var
    }
}

impl<'a, AllocT, T> std::ops::DerefMut for SharedVariable<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    fn deref_mut(&mut self) -> &mut T {
        &mut self.var
    }
}

/// Wrapper type to share a Vec with CUDA as a buffer.
pub struct SharedVec<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    vec: Vec<T>,
    buffer: cuda::Buffer<'a, AllocT>,
}

impl<'a, AllocT, T> SharedVec<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    /// Create a new SharedVec, taking ownership of the Vec `vec`.
    ///
    /// Once the `SharedVec` has been created, it can be uploaded to the
    /// device using the `SharedVec::upload()` method. `SharedVec`
    /// handles management of the underlying CUDA buffer used for device-side
    /// storage. `SharedVec` implements Deref targeting the wrapped
    /// vec for easy access.
    pub fn new(
        vec: Vec<T>,
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<SharedVec<'a, AllocT, T>> {
        let cvec: Vec<T::Target> = vec.iter().map(|t| t.to_device()).collect();
        let buffer = cuda::Buffer::with_data(
            &cvec,
            // FIXME: let trait implementors declare their preferred alignemnt
            std::mem::align_of::<T>(),
            tag,
            allocator,
        )?;
        Ok(SharedVec { vec, buffer })
    }

    /// Upload the wrapped vec to the device. Any changes to the variable
    /// on the Rust side will not be reflected on the device until this is
    /// called.
    pub fn upload(&mut self) -> Result<()> {
        let cvec: Vec<T::Target> =
            self.vec.iter().map(|t| t.to_device()).collect();
        self.buffer.upload(&cvec)?;
        Ok(())
    }

    /// Get a reference to the `cuda::Buffer` representing the device-side
    /// storage for this variable.
    pub fn variable_buffer(&self) -> &cuda::Buffer<'a, AllocT> {
        &self.buffer
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct SharedVecD {
    pub ptr: cuda::CUdeviceptr,
    pub len: usize,
}

impl<'a, AllocT, T> DeviceShareable for SharedVec<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    type Target = SharedVecD;
    fn to_device(&self) -> SharedVecD {
        SharedVecD {
            ptr: self.buffer.as_device_ptr(),
            len: self.vec.len(),
        }
    }

    fn cuda_type() -> String {
        format!("SharedVec<{}>", T::cuda_type())
    }

    fn cuda_decl() -> String {
        r#"
template <typename T> 
struct SharedVec {
    T* ptr; 
    size_t len; 

    const T& operator[](size_t i) const {
        return ptr[i];
    } 

    T& operator[](size_t i) {
        return ptr[i];
    } 

    bool is_null() const {
        return ptr == nullptr;
    }

    bool is_empty() const {
        return len == 0;
    }
};
        "#
        .into()
    }

    fn zero() -> Self::Target {
        SharedVecD { ptr: 0, len: 0 }
    }
}

impl<'a, AllocT, T> std::ops::Deref for SharedVec<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    type Target = Vec<T>;
    fn deref(&self) -> &Vec<T> {
        &self.vec
    }
}

impl<'a, AllocT, T> std::ops::DerefMut for SharedVec<'a, AllocT, T>
where
    AllocT: Allocator,
    T: DeviceShareable,
{
    fn deref_mut(&mut self) -> &mut Vec<T> {
        &mut self.vec
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
            fn cuda_type() -> String {
                stringify!($ty).into()
            }
            fn zero() -> Self::Target {
                0
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
    ($ty:ty, $cuty:ty, $fmt:expr, $cmp:literal, $cmpty:ty, $align:literal) => {
        impl DeviceShareable for $ty {
            type Target = $ty;
            fn to_device(&self) -> Self::Target {
                *self
            }
            fn cuda_type() -> String {
                stringify!($cuty).into()
            }
            fn zero() -> Self::Target {
                zero::<$ty>()
            }
        }

        impl BufferElement for $ty {
            const FORMAT: BufferFormat = $fmt;
            const COMPONENTS: usize = $cmp;
            const ALIGNMENT: usize = $align;
            type ComponentType = $cmpty;
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        println!("{}", TestEnum::cuda_decl());
        println!("{}", Life::cuda_decl());
        assert_eq!(2 + 2, 4);
    }

    use super::DeviceShareable;

    #[optix_derive::device_shared]
    struct Life<'a> {
        x: &'a i32,
    }

    #[optix_derive::device_shared]
    enum TestEnum {
        A,
        B,
        C,
        D,
    }
}
