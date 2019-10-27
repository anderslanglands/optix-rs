use super::acceleration::TraversableHandle;
use super::math::M4f32;
use bitflags::bitflags;
use optix_sys as sys;
use std::mem::MaybeUninit;
use std::rc::Rc;

pub use sys::OptixInstance as Instance;

bitflags! {
    pub struct InstanceFlags: u32 {
        const NONE = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_NONE;
        const DISABLE_TRIANGLE_FACE_CULLING = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        const FLIP_TRIANGLE_FACING = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING;
        const DISABLE_ANYHIT = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        const ENFORCE_ANYHIT = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT;
        const DISABLE_TRANSFORM = sys::OptixInstanceFlags_OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM;
    }
}

pub fn make_instance(
    transform: &M4f32,
    instance_id: u32,
    sbt_offset: u32,
    visibility_mask: u32,
    flags: InstanceFlags,
    traversable: &TraversableHandle,
) -> Instance {
    let mut inst = MaybeUninit::<Instance>::uninit();
    let mut inst = unsafe {
        std::ptr::write_bytes(inst.as_mut_ptr(), 0, 1);
        inst.assume_init()
    };

    unsafe {
        std::ptr::copy_nonoverlapping(
            transform.x.as_ptr(),
            inst.transform.as_mut_ptr(),
            12,
        );
    }

    inst.instanceId = instance_id;
    inst.sbtOffset = sbt_offset;
    inst.visibilityMask = visibility_mask;
    inst.flags = flags.bits();
    inst.traversableHandle = traversable.hnd;

    inst
}
