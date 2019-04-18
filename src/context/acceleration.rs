use crate::context::*;
use std::cell::RefCell;
use std::ffi::CString;
use std::rc::Rc;

pub struct Acceleration {
    pub(crate) rt_acc: RTacceleration,
}

pub type AccelerationHandle = Rc<RefCell<Acceleration>>;

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
    pub fn acceleration_create(
        &mut self,
        builder: Builder,
    ) -> Result<AccelerationHandle> {
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
            let acc = Rc::new(RefCell::new(Acceleration { rt_acc }));
            self.accelerations.push(Rc::clone(&acc));
            Ok(acc)
        }
    }

    pub fn acceleration_mark_dirty(
        &mut self,
        acc: &AccelerationHandle,
    ) -> Result<()> {
        let result = unsafe { rtAccelerationMarkDirty(acc.borrow().rt_acc) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtAccelerationMarkDirty", result))
        } else {
            Ok(())
        }
    }

    /// Check that the Acceleration and all objects attached to it are correctly
    /// set up.
    /// # Panics
    /// If mat is not a valid AccelerationHandle
    pub fn acceleration_validate(
        &self,
        acc: &AccelerationHandle,
    ) -> Result<()> {
        let result = unsafe { rtAccelerationValidate(acc.borrow().rt_acc) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtAccelerationValidate", result))
        } else {
            Ok(())
        }
    }
}
