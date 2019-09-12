use optix_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

use std::rc::Rc;

use super::DeviceShareable;

/// Runtime-typed buffer
pub struct RtBuffer {
    buffer: cuda::Buffer,
    count: usize,
    format: BufferFormat,
}

impl RtBuffer {
    pub fn new<T>(data: &[T], format: BufferFormat) -> Result<RtBuffer> {
        let buffer = cuda::Buffer::with_data(data)?;

        Ok(RtBuffer {
            buffer,
            count: data.len(),
            format,
        })
    }

    pub fn as_ptr(&self) -> *const std::os::raw::c_void {
        self.buffer.as_ptr()
    }

    pub fn as_device_ptr(&self) -> cuda::CUdeviceptr {
        self.buffer.as_device_ptr()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn format(&self) -> BufferFormat {
        self.format
    }

    pub fn byte_size(&self) -> usize {
        self.buffer.byte_size()
    }
}

impl DeviceShareable for RtBuffer {
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.buffer.as_device_ptr()
    }
    fn cuda_decl(_: bool) -> String {
        "void*".into()
    }
}

impl DeviceShareable for Rc<RtBuffer> {
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.buffer.as_device_ptr()
    }
    fn cuda_decl(_: bool) -> String {
        "void*".into()
    }
}

pub struct CtBuffer<T>
where
    T: BufferElement,
{
    buffer: cuda::Buffer,
    count: usize,
    _t: std::marker::PhantomData<T>,
}

impl<T> CtBuffer<T>
where
    T: BufferElement,
{
    pub fn new(data: &[T]) -> Result<CtBuffer<T>> {
        let buffer = cuda::Buffer::with_data(data)?;

        Ok(CtBuffer {
            buffer,
            count: data.len(),
            _t: std::marker::PhantomData::<T> {},
        })
    }

    pub fn uninitialized(count: usize) -> Result<CtBuffer<T>> {
        let buffer = cuda::Buffer::new(count * std::mem::size_of::<T>())?;

        Ok(CtBuffer {
            buffer,
            count,
            _t: std::marker::PhantomData::<T> {},
        })
    }

    pub fn as_ptr(&self) -> *const std::os::raw::c_void {
        self.buffer.as_ptr()
    }

    pub fn as_device_ptr(&self) -> cuda::CUdeviceptr {
        self.buffer.as_device_ptr()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn format(&self) -> BufferFormat {
        T::FORMAT
    }

    pub fn byte_size(&self) -> usize {
        self.buffer.byte_size()
    }

    pub fn download(&self, dst: &mut [T]) -> Result<()> {
        Ok(self.buffer.download(dst)?)
    }
}

impl<T> DeviceShareable for CtBuffer<T>
where
    T: BufferElement,
{
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.buffer.as_device_ptr()
    }
    fn cuda_decl(_: bool) -> String {
        format!("{}*", T::FORMAT.device_name())
    }
}

impl<T> DeviceShareable for Rc<CtBuffer<T>>
where
    T: BufferElement,
{
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.buffer.as_device_ptr()
    }
    fn cuda_decl(_: bool) -> String {
        format!("{}*", T::FORMAT.device_name())
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BufferFormat {
    U8,
    U8x2,
    U8x3,
    U8x4,
    U16,
    U16x2,
    U16x3,
    U16x4,
    F16,
    F16x2,
    F16x3,
    F16x4,
    F32,
    F32x2,
    F32x3,
    F32x4,
    I32,
    I32x2,
    I32x3,
    I32x4,
}

impl BufferFormat {
    pub fn byte_size(&self) -> usize {
        match self {
            BufferFormat::U8 => 1,
            BufferFormat::U8x2 => 2,
            BufferFormat::U8x3 => 3,
            BufferFormat::U8x4 => 4,
            BufferFormat::U16 => 2,
            BufferFormat::U16x2 => 4,
            BufferFormat::U16x3 => 6,
            BufferFormat::U16x4 => 8,
            BufferFormat::F16 => 2,
            BufferFormat::F16x2 => 4,
            BufferFormat::F16x3 => 6,
            BufferFormat::F16x4 => 8,
            BufferFormat::F32 => 4,
            BufferFormat::F32x2 => 8,
            BufferFormat::F32x3 => 12,
            BufferFormat::F32x4 => 16,
            BufferFormat::I32 => 4,
            BufferFormat::I32x2 => 8,
            BufferFormat::I32x3 => 12,
            BufferFormat::I32x4 => 16,
        }
    }

    pub fn device_name(&self) -> &'static str {
        match self {
            BufferFormat::U8 => "u8",
            BufferFormat::U8x2 => "V2u8",
            BufferFormat::U8x3 => "V3u8",
            BufferFormat::U8x4 => "V4u8",
            BufferFormat::U16 => "u16",
            BufferFormat::U16x2 => "V2u16",
            BufferFormat::U16x3 => "V3u16",
            BufferFormat::U16x4 => "V4u16",
            BufferFormat::F16 => "f16",
            BufferFormat::F16x2 => "V2f16",
            BufferFormat::F16x3 => "V3f16",
            BufferFormat::F16x4 => "V4f16",
            BufferFormat::F32 => "f32",
            BufferFormat::F32x2 => "V2f32",
            BufferFormat::F32x3 => "V3f32",
            BufferFormat::F32x4 => "V4f32",
            BufferFormat::I32 => "i32",
            BufferFormat::I32x2 => "V2i32",
            BufferFormat::I32x3 => "V3i32",
            BufferFormat::I32x4 => "V4i32",
        }
    }
}

pub trait BufferElement {
    const FORMAT: BufferFormat;
    const COMPONENTS: usize;
}

impl BufferElement for u8 {
    const FORMAT: BufferFormat = BufferFormat::U8;
    const COMPONENTS: usize = 1;
}

impl BufferElement for [u8; 2] {
    const FORMAT: BufferFormat = BufferFormat::U8x2;
    const COMPONENTS: usize = 2;
}

impl BufferElement for [u8; 3] {
    const FORMAT: BufferFormat = BufferFormat::U8x3;
    const COMPONENTS: usize = 3;
}

impl BufferElement for [u8; 4] {
    const FORMAT: BufferFormat = BufferFormat::U8x4;
    const COMPONENTS: usize = 4;
}

impl BufferElement for u16 {
    const FORMAT: BufferFormat = BufferFormat::U16;
    const COMPONENTS: usize = 1;
}

impl BufferElement for [u16; 2] {
    const FORMAT: BufferFormat = BufferFormat::U16x2;
    const COMPONENTS: usize = 2;
}

impl BufferElement for [u16; 3] {
    const FORMAT: BufferFormat = BufferFormat::U16x3;
    const COMPONENTS: usize = 3;
}

impl BufferElement for [u16; 4] {
    const FORMAT: BufferFormat = BufferFormat::U16x4;
    const COMPONENTS: usize = 4;
}

impl BufferElement for i32 {
    const FORMAT: BufferFormat = BufferFormat::I32;
    const COMPONENTS: usize = 1;
}

impl BufferElement for [i32; 2] {
    const FORMAT: BufferFormat = BufferFormat::I32x2;
    const COMPONENTS: usize = 2;
}

impl BufferElement for [i32; 3] {
    const FORMAT: BufferFormat = BufferFormat::I32x3;
    const COMPONENTS: usize = 3;
}

impl BufferElement for [i32; 4] {
    const FORMAT: BufferFormat = BufferFormat::I32x4;
    const COMPONENTS: usize = 4;
}

impl BufferElement for f32 {
    const FORMAT: BufferFormat = BufferFormat::F32;
    const COMPONENTS: usize = 1;
}

impl BufferElement for [f32; 2] {
    const FORMAT: BufferFormat = BufferFormat::F32x2;
    const COMPONENTS: usize = 2;
}

impl BufferElement for [f32; 3] {
    const FORMAT: BufferFormat = BufferFormat::F32x3;
    const COMPONENTS: usize = 3;
}

impl BufferElement for [f32; 4] {
    const FORMAT: BufferFormat = BufferFormat::F32x4;
    const COMPONENTS: usize = 4;
}
