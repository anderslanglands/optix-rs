use super::{
    buffer::{BufferElement, BufferFormat},
    math_type, DeviceShareable,
};

#[cfg(feature = "math-imath")]
pub use imath::*;

cfg_if::cfg_if! {
    if #[cfg(feature="math-nalgebra")] {

        pub use nalgebra_glm::I8Vec2 as V2u8;
        pub use nalgebra_glm::I8Vec3 as V3u8;
        pub use nalgebra_glm::I8Vec4 as V4u8;

        pub use nalgebra_glm::I16Vec2 as V2u16;
        pub use nalgebra_glm::I16Vec3 as V3u16;
        pub use nalgebra_glm::I16Vec4 as V4u16;

        pub use nalgebra_glm::IVec2 as V2i32;
        pub use nalgebra_glm::IVec3 as V3i32;
        pub use nalgebra_glm::IVec4 as V4i32;

        pub use nalgebra_glm::Vec2 as V2f32;
        pub use nalgebra_glm::Vec3 as V3f32;
        pub use nalgebra_glm::Vec4 as V4f32;

        pub use nalgebra_glm::DVec2 as V2f64;
        pub use nalgebra_glm::DVec3 as V3f64;
        pub use nalgebra_glm::DVec4 as V4f64;

        pub use nalgebra_glm::Mat4x4 as M4f32;
        pub use nalgebra_glm::DMat4x4 as M4f64;

        #[inline(always)]
        pub fn v2f32(x: f32, y: f32) -> V2f32 {
            V2f32::new(x, y)
        }

        #[inline(always)]
        pub fn v3f32(x: f32, y: f32, z: f32) -> V3f32 {
            V3f32::new(x, y, z)
        }

        #[inline(always)]
        pub fn v4f32(x: f32, y: f32, z: f32, w: f32) -> V4f32 {
            V4f32::new(x, y, z, w)
        }

        #[inline(always)]
        pub fn v2f64(x: f64, y: f64) -> V2f64 {
            V2f64::new(x, y)
        }

        #[inline(always)]
        pub fn v3f64(x: f64, y: f64, z: f64) -> V3f64 {
            V3f64::new(x, y, z)
        }

        #[inline(always)]
        pub fn v4f64(x: f64, y: f64, z: f64, w: f64) -> V4f64 {
            V4f64::new(x, y, z, w)
        }

        #[inline(always)]
        pub fn v2i32(x: i32, y: i32) -> V2i32 {
            V2i32::new(x, y)
        }

        #[inline(always)]
        pub fn v3i32(x: i32, y: i32, z: i32) -> V3i32 {
            V3i32::new(x, y, z)
        }

        #[inline(always)]
        pub fn v4i32(x: i32, y: i32, z: i32, w: i32) -> V4i32 {
            V4i32::new(x, y, z, w)
        }

        pub use nalgebra_glm::{
            normalize,
            cross,
            translate,
            translation,
            scale,
            scaling,
            rotate,
            rotation,
            perspective_fov_rh,
            perspective_fov_rh_zo,
            perspective_fov_lh,
            inverse,
            inverse_transpose,
            transpose,
        };

        pub use nalgebra_glm::{Dimension, Scalar, Number, RealField};

        pub fn cast_slice_v3i32(s: &[i32]) -> &[V3i32] {
            if s.len() % 3 != 0 {
                panic!("Tried to cast slice of length {} to V3i32", s.len());
            }

            unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const V3i32, s.len() / 3)
            }
        }

        pub fn cast_slice_v3f32(s: &[f32]) -> &[V3f32] {
            if s.len() % 3 != 0 {
                panic!("Tried to cast slice of length {} to V3f32", s.len());
            }

            unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const V3f32, s.len() / 3)
            }
        }

        pub fn cast_slice_v2f32(s: &[f32]) -> &[V2f32] {
            if s.len() % 2 != 0 {
                panic!("Tried to cast slice of length {} to V2f32", s.len());
            }

            unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const V2f32, s.len() / 2)
            }
        }

        pub fn cast_slice_m4f32(s: &[f32]) -> &[M4f32] {
            if s.len() % 16 != 0 {
                panic!("Tried to cast slice of length {} to M4f32", s.len());
            }

            unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const M4f32, s.len() / 16)
            }
        }

        pub fn cast_slice_m4f64(s: &[f64]) -> &[M4f64] {
            if s.len() % 16 != 0 {
                panic!("Tried to cast slice of length {} to M4f64", s.len());
            }

            unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const M4f64, s.len() / 16)
            }
        }

        use nalgebra_glm::{TVec3, vec3, min2, max2};

        pub struct Box3<T> where T: RealField {
            pub min: TVec3<T>,
            pub max: TVec3<T>,
        }

        impl<T> Box3<T> where T: RealField {
            pub fn new(min: TVec3<T>, max: TVec3<T>) -> Box3<T> {
                Box3{min, max}
            }

            pub fn make_empty() -> Box3<T> {
                let max = T::min_value();
                let min = T::max_value();
                Box3 {
                    min: vec3(min, min, min),
                    max: vec3(max, max, max),
                }
            }

            pub fn center(&self) -> TVec3<T> {
                let d = self.max - self.min;
                let half = T::from_f32(0.5).unwrap();
                self.min + d.component_mul(&vec3(half, half, half))
            }

            pub fn extend_by_pnt(&mut self, pnt: TVec3<T>) {
                self.min = min2(&self.min, &pnt);
                self.max = max2(&self.max, &pnt);
            }
        }

        pub type Box3f32 = Box3<f32>;
        pub type Box3f64 = Box3<f64>;
    }
}

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
