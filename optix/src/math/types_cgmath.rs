use crate::impl_device_copy_align;
use crate::DeviceCopy;

impl_device_copy_align!(
    cgmath::Vector2<i32>:8
    cgmath::Vector3<i32>:4
    cgmath::Vector4<i32>:16

    cgmath::Vector2<f32>:8
    cgmath::Vector3<f32>:4
    cgmath::Vector4<f32>:16

    cgmath::Point2<f32>:8
    cgmath::Point3<f32>:4
);