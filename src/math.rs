use nalgebra_glm as glm;
use nalgebra_glm::{
    DMat4x4, DVec2, DVec3, DVec4, IVec2, IVec3, IVec4, Mat4x4, U32Vec2,
    U32Vec3, U32Vec4, Vec2, Vec3, Vec4,
};

pub type V2f32 = Vec2;
pub type V3f32 = Vec3;
pub type V4f32 = Vec4;
pub type V2f64 = DVec2;
pub type V3f64 = DVec3;
pub type V4f64 = DVec4;
pub type M4f32 = Mat4x4;
pub type M4f64 = DMat4x4;

// short constructors
pub fn v2f32(x: f32, y: f32) -> V2f32 {
    V2f32::new(x, y)
}

pub fn v3f32(x: f32, y: f32, z: f32) -> V3f32 {
    V3f32::new(x, y, z)
}

pub fn v4f32(x: f32, y: f32, z: f32, w: f32) -> V4f32 {
    V4f32::new(x, y, z, w)
}

pub fn v2f64(x: f64, y: f64) -> V2f64 {
    V2f64::new(x, y)
}

pub fn v3f64(x: f64, y: f64, z: f64) -> V3f64 {
    V3f64::new(x, y, z)
}

pub fn v4f64(x: f64, y: f64, z: f64, w: f64) -> V4f64 {
    V4f64::new(x, y, z, w)
}

pub type V2i32 = IVec2;
pub type V3i32 = IVec3;
pub type V4i32 = IVec4;
pub type V2u32 = U32Vec2;
pub type V3u32 = U32Vec3;
pub type V4u32 = U32Vec4;

// short constructors
pub fn v2i32(x: i32, y: i32) -> V2i32 {
    V2i32::new(x, y)
}

pub fn v3i32(x: i32, y: i32, z: i32) -> V3i32 {
    V3i32::new(x, y, z)
}

pub fn v4i32(x: i32, y: i32, z: i32, w: i32) -> V4i32 {
    V4i32::new(x, y, z, w)
}

// short constructors
pub fn v2u32(x: u32, y: u32) -> V2u32 {
    V2u32::new(x, y)
}

pub fn v3u32(x: u32, y: u32, z: u32) -> V3u32 {
    V3u32::new(x, y, z)
}

pub fn v4u32(x: u32, y: u32, z: u32, w: u32) -> V4u32 {
    V4u32::new(x, y, z, w)
}

// matrix constructors
pub fn m4f32_translation(x: f32, y: f32, z: f32) -> M4f32 {
    M4f32::new_translation(&v3f32(x, y, z))
}

pub fn m4f32_rotation(axis: V3f32, angle: f32) -> M4f32 {
    let m = M4f32::identity();
    glm::rotate(&m, angle, &axis)
}

pub fn m4f32_scaling(x: f32, y: f32, z: f32) -> M4f32 {
    M4f32::new_nonuniform_scaling(&v3f32(x, y, z))
}

pub fn m4f64_translation(x: f64, y: f64, z: f64) -> M4f64 {
    M4f64::new_translation(&v3f64(x, y, z))
}

pub fn m4f64_rotation(axis: V3f64, angle: f64) -> M4f64 {
    let m = M4f64::identity();
    glm::rotate(&m, angle, &axis)
}

pub fn m4f64_scaling(x: f64, y: f64, z: f64) -> M4f64 {
    M4f64::new_nonuniform_scaling(&v3f64(x, y, z))
}

#[test]
fn test_matrix() {
    let xf_trans = m4f32_translation(2.0, 3.0, 4.0);
    println!("{}", xf_trans);
    println!("{:?}", xf_trans.as_slice());

    let xf_rot = m4f32_rotation(v3f32(0.0, 1.0, 0.0), std::f32::consts::PI);
    println!("{}", xf_rot);
    println!("{:?}", xf_rot.as_slice());

    let xf_pre = xf_trans * xf_rot;
    println!("{}", xf_pre);
    println!("{:?}", xf_pre.as_slice());

    let xf_trans = m4f32_translation(0.5, 0.5, 0.0);
    let xf_scale = m4f32_scaling(0.5, 0.5, 1.0);
    let xf = xf_trans * xf_scale;
    dbg!(xf.as_slice());
}
