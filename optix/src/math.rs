use super::{
    buffer::{BufferElement, BufferFormat},
    math_type, DeviceShareable,
};

#[cfg(feature = "math-imath")]
pub use imath::*;

math_type!(V2u8, BufferFormat::U8x2, 2);
math_type!(V3u8, BufferFormat::U8x3, 3);
math_type!(V4u8, BufferFormat::U8x4, 4);

math_type!(V2u16, BufferFormat::U16x2, 2);
math_type!(V3u16, BufferFormat::U16x3, 3);
math_type!(V4u16, BufferFormat::U16x4, 4);

math_type!(V2i32, BufferFormat::I32x2, 2);
math_type!(V3i32, BufferFormat::I32x3, 3);
math_type!(V4i32, BufferFormat::I32x4, 4);

math_type!(V2f32, BufferFormat::F32x2, 2);
math_type!(V3f32, BufferFormat::F32x3, 3);
math_type!(V4f32, BufferFormat::F32x4, 4);

impl DeviceShareable for M4f32 {
    type Target = M4f32;

    fn to_device(&self) -> M4f32 {
        *self
    }

    fn cuda_type() -> String {
        "M4f32".into()
    }
}
