use crate::{sys, DeviceStorage, TypedBuffer, DeviceCopy};

#[repr(u32)]
pub enum PixelFormat {
    Half3 = sys::OptixPixelFormat_OPTIX_PIXEL_FORMAT_HALF3,
    Half4 = sys::OptixPixelFormat_OPTIX_PIXEL_FORMAT_HALF4,
    Float3 = sys::OptixPixelFormat_OPTIX_PIXEL_FORMAT_FLOAT3,
    Float4 = sys::OptixPixelFormat_OPTIX_PIXEL_FORMAT_FLOAT4,
    Uchar3 = sys::OptixPixelFormat_OPTIX_PIXEL_FORMAT_UCHAR3,
    Uchar4 = sys::OptixPixelFormat_OPTIX_PIXEL_FORMAT_UCHAR4,
}

#[repr(C)]
pub struct Image2D {
    data: cu::DevicePtr,
    width: u32,
    height: u32,
    row_stride_in_bytes: u32,
    pixel_stride_in_bytes: u32,
    format: PixelFormat,
}

impl Image2D {
    pub fn new(
        data: cu::DevicePtr,
        width: u32,
        height: u32,
        row_stride_in_bytes: u32,
        pixel_stride_in_bytes: u32,
        format: PixelFormat,
    ) -> Image2D {
        Image2D {
            data,
            width,
            height,
            row_stride_in_bytes,
            pixel_stride_in_bytes,
            format,
        }
    }

    pub fn from_buffer<P: Pixel, A: cu::DeviceAllocRef>(
        buffer: TypedBuffer<P, A>,
        width: u32,
        height: u32,
    ) -> Image2D {
        Image2D {
            data: buffer.device_ptr(),
            width, 
            height,
            row_stride_in_bytes: P::BYTE_SIZE as u32 * width,
            pixel_stride_in_bytes: P::BYTE_SIZE as u32,
            format: P::FORMAT,
        }
    }
}

pub trait Pixel: DeviceCopy + Sized {
    const FORMAT: PixelFormat;
    const BYTE_SIZE: usize;
}