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

pub mod buffer;
pub use buffer::{Buffer, TypedBuffer};

pub mod allocator;
pub use allocator::{DefaultDeviceAlloc, DeviceAllocRef};

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

#[cfg(test)]
mod tests {

    use crate as cu;

    #[test]
    fn add() -> Result<(), Box<dyn std::error::Error>> {
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

pub unsafe trait DeviceCopy {
    // Empty
}

macro_rules! impl_device_copy {
    ($($t:ty)*) => {
        $(
            unsafe impl DeviceCopy for $t {}
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

unsafe impl DeviceCopy for DevicePtr {}
