use super::{
    buffer::{BufferElement, BufferFormat},
    math_type, DeviceShareable,
};

#[cfg(feature = "math-imath")]
pub use imath::*;

#[cfg(feature = "math-imath")]
math_type!(V2u8, BufferFormat::U8x2, 2);
#[cfg(feature = "math-imath")]
math_type!(V3u8, BufferFormat::U8x3, 3);
#[cfg(feature = "math-imath")]
math_type!(V4u8, BufferFormat::U8x4, 4);

#[cfg(feature = "math-imath")]
math_type!(V2u16, BufferFormat::U16x2, 2);
#[cfg(feature = "math-imath")]
math_type!(V3u16, BufferFormat::U16x3, 3);
#[cfg(feature = "math-imath")]
math_type!(V4u16, BufferFormat::U16x4, 4);

#[cfg(feature = "math-imath")]
math_type!(V2i32, BufferFormat::I32x2, 2);
#[cfg(feature = "math-imath")]
math_type!(V3i32, BufferFormat::I32x3, 3);
#[cfg(feature = "math-imath")]
math_type!(V4i32, BufferFormat::I32x4, 4);

#[cfg(feature = "math-imath")]
math_type!(V2f32, BufferFormat::F32x2, 2);
#[cfg(feature = "math-imath")]
math_type!(V3f32, BufferFormat::F32x3, 3);
#[cfg(feature = "math-imath")]
math_type!(V4f32, BufferFormat::F32x4, 4);
