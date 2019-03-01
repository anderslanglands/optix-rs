use crate::context::*;
use std::ffi::CString;

new_key_type! { pub struct AccelerationHandle; }

/// Selects the BVH building algorithm to be used by an `Accelerator`
#[derive(Copy, Clone)]
pub enum Builder {
    NoAccel,
    Bvh,
    Sbvh,
    Trbvh,
}

impl Context {
    /// Creates a new `Acceleration` on this `Context` with the specified
    /// `Builder` returning a handle that can be used to access it later.
    pub fn acceleration_create(&mut self, builder: Builder) -> Result<AccelerationHandle> {
        let (rt_acc, result) = unsafe {
            let mut rt_acc: RTacceleration = std::mem::zeroed();
            let result = rtAccelerationCreate(self.rt_ctx, &mut rt_acc);
            (rt_acc, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtAccelerationCreate", result));
        };

        let str = match builder {
            Builder::NoAccel => CString::new("NoAccel"),
            Builder::Bvh => CString::new("Bvh"),
            Builder::Sbvh => CString::new("Sbvh"),
            Builder::Trbvh => CString::new("Trbvh"),
        }
        .unwrap();
        let result = unsafe { rtAccelerationSetBuilder(rt_acc, str.as_ptr()) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtAccelerationCreate", result))
        } else {
            let acc = self.ga_acceleration_obj.insert(rt_acc);
            Ok(acc)
        }
    }

    /// Destroys this Acceleration an all objects attached to it. Note that the
    /// Acceleration will not actually be destroyed until all references to it from
    /// other scene graph objects are released.
    /// # Panics
    /// If mat is not a valid AccelerationHandle
    pub fn acceleration_destroy(&mut self, acc: AccelerationHandle) {
        let rt_acc = self.ga_acceleration_obj.remove(acc).unwrap();
        if unsafe { rtAccelerationDestroy(rt_acc) } != RtResult::SUCCESS {
            panic!("Error destroying acceleration: {:?}", acc);
        }
    }

    /// Check that the Acceleration and all objects attached to it are correctly
    /// set up.
    /// # Panics
    /// If mat is not a valid AccelerationHandle
    pub fn acceleration_validate(&self, acc: AccelerationHandle) -> Result<()> {
        let rt_acc = *self.ga_acceleration_obj.get(acc).unwrap();
        let result = unsafe { rtAccelerationValidate(rt_acc) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtAccelerationValidate", result))
        } else {
            Ok(())
        }
    }
}
