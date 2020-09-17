#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]
pub mod sys;

pub mod error;
pub use error::Error;

pub mod device;
pub use device::{Device, DeviceAttribute};

pub mod context;
pub use context::{Context, ContextFlags};

pub mod module;
pub use module::{Function, Module};

pub mod stream;
pub use stream::{Stream, StreamFlags};

pub mod execution;
pub use execution::Dim;

pub mod memory;
pub use memory::DevicePtr;

pub mod allocator;
pub use allocator::{DefaultDeviceAlloc, DeviceAllocRef};

pub mod texture;
pub use texture::{AddressMode, ArrayFormat, FilterMode, TexObject, TextureReadFlags};

type Result<T, E = Error> = std::result::Result<T, E>;

pub fn init() -> Result<()> {
    unsafe { sys::cuInit(0).to_result() }
}

pub fn driver_get_version() -> Result<i32> {
    unsafe {
        let mut v = 0i32;
        sys::cuDriverGetVersion(&mut v).to_result().map(|_| v)
    }
}

/*
#[cfg(test)]
mod tests {

    use crate as cu;
    #[test]
    fn add() -> Result<(), Box<dyn std::error::Error>> {
        use cu::DeviceStorage;

        cu::init()?;
        let device = cu::Device::get(0)?;
        let _ctx = device.ctx_create(
            cu::ContextFlags::SCHED_AUTO | cu::ContextFlags::MAP_HOST,
        )?;

        let module = cu::Module::load_string(include_str!(concat!(
            env!("OUT_DIR"),
            "/add.ptx"
        )))?;

        let function = module.get_function("sum")?;
        let stream = cu::Stream::create(cu::StreamFlags::NON_BLOCKING)?;

        let buf_x = cu::Buffer::from_slice(&[2.0f32; 10])?;
        let buf_y = cu::Buffer::from_slice(&[4.0f32; 10])?;
        let mut out = vec![0.0f32; 10];
        let buf_out = cu::Buffer::from_slice(&out)?;
        let len = 10u32;

        cu::launch!(
            function,
            1,
            1,
            0,
            stream,
            buf_x.device_ptr(),
            buf_y.device_ptr(),
            buf_out.device_ptr(),
            len
        )?;

        cu::Context::synchronize()?;

        buf_out.copy_to_slice(&mut out)?;

        assert_eq!(out, [6.0f32; 10]);

        println!("{:?}", out);

        Ok(())
    }
}
*/
