use crate::impl_device_copy_align;
use crate::DeviceCopy;

impl_device_copy_align!(
    nalgebra_glm::I32Vec2:8
    nalgebra_glm::I32Vec3:4
    nalgebra_glm::I32Vec4:16

    nalgebra_glm::Vec2:8
    nalgebra_glm::Vec3:4
    nalgebra_glm::Vec4:16
);
