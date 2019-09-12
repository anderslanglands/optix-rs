use optix_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

use std::rc::Rc;

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

use super::DeviceShareable;
impl DeviceShareable for RtBuffer {
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.buffer.as_device_ptr()
    }
}

impl DeviceShareable for Rc<RtBuffer> {
    type Target = cuda::CUdeviceptr;
    fn to_device(&self) -> Self::Target {
        self.buffer.as_device_ptr()
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
}
