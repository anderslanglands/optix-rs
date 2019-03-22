use nalgebra::{Matrix4, Vector2, Vector3, Vector4};

pub type V2f32 = Vector2<f32>;
pub type V3f32 = Vector3<f32>;
pub type V4f32 = Vector4<f32>;
pub type V2f64 = Vector2<f64>;
pub type V3f64 = Vector3<f64>;
pub type V4f64 = Vector4<f64>;
pub type M4f32 = Matrix4<f32>;
pub type M4f64 = Matrix4<f64>;

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

pub type V2i32 = Vector2<i32>;
pub type V3i32 = Vector3<i32>;
pub type V4i32 = Vector4<i32>;

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

// matrix constructors
pub fn m4f32_translation(x: f32, y: f32, z: f32) -> M4f32 {
    M4f32::new_translation(&v3f32(x, y, z))
}

pub fn m4f32_rotation(axis: V3f32, angle: f32) -> M4f32 {
    nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
        .to_homogeneous()
}

pub fn m4f32_scaling(x: f32, y: f32, z: f32) -> M4f32 {
    M4f32::new_nonuniform_scaling(&v3f32(x, y, z))
}

pub fn m4f64_translation(x: f64, y: f64, z: f64) -> M4f64 {
    M4f64::new_translation(&v3f64(x, y, z))
}

pub fn m4f64_rotation(axis: V3f64, angle: f64) -> M4f64 {
    nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
        .to_homogeneous()
}

pub fn m4f64_scaling(x: f64, y: f64, z: f64) -> M4f64 {
    M4f64::new_nonuniform_scaling(&v3f64(x, y, z))
}

pub fn m4f64_look_at_rh(eye: &V3f64, look: &V3f64, up: &V3f64) -> M4f64 {
    M4f64::look_at_rh(
        &nalgebra::geometry::Point3::new(eye.x, eye.y, eye.z),
        &nalgebra::geometry::Point3::new(look.x, look.y, look.z),
        up,
    )
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
