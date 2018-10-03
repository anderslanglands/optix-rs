use nalgebra::{Vector2, Vector3, Vector4};

pub type V2f32 = Vector2<f32>;
pub type V3f32 = Vector3<f32>;
pub type V4f32 = Vector4<f32>;

// short constructors
pub fn v2f(x: f32, y: f32) -> V2f32 {
    V2f32::new(x, y)
}

pub fn v3f(x: f32, y: f32, z: f32) -> V3f32 {
    V3f32::new(x, y, z)
}

pub fn v4f(x: f32, y: f32, z: f32, w: f32) -> V4f32 {
    V4f32::new(x, y, z, w)
}
