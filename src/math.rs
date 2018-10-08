use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

pub type V2f32 = Vector2<f32>;
pub type V3f32 = Vector3<f32>;
pub type V4f32 = Vector4<f32>;
pub type M4f32 = Matrix4<f32>;

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

pub type V2i32 = Vector2<i32>;
pub type V3i32 = Vector3<i32>;
pub type V4i32 = Vector4<i32>;

// short constructors
pub fn v2i(x: i32, y: i32) -> V2i32 {
    V2i32::new(x, y)
}

pub fn v3i(x: i32, y: i32, z: i32) -> V3i32 {
    V3i32::new(x, y, z)
}

pub fn v4i(x: i32, y: i32, z: i32, w: i32) -> V4i32 {
    V4i32::new(x, y, z, w)
}

// matrix constructors
pub fn m4f_translation(x: f32, y: f32, z: f32) -> M4f32 {
    M4f32::new_translation(&v3f(x, y, z))
}

pub fn m4f_rotation(axis: V3f32, angle: f32) -> M4f32 {
    nalgebra::Rotation3::from_axis_angle(
        &nalgebra::Unit::new_normalize(axis),
        angle,
    ).to_homogeneous()
}
