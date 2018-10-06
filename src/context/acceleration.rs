use crate::context::*;
use crate::ginallocator::*;
use std::ffi::CString;

#[derive(Default, Debug, Copy, Clone)]
pub struct AccelerationMarker;
impl Marker for AccelerationMarker {
    const ID: &'static str = "Acceleration";
}
pub type AccelerationHandle = Handle<AccelerationMarker>;

#[derive(Copy, Clone)]
pub enum Builder {
    NoAccel,
    Bvh,
    Sbvh,
    Trbvh,
}

impl Context {
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
        }.unwrap();
        let result = unsafe { rtAccelerationSetBuilder(rt_acc, str.as_ptr()) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtAccelerationCreate", result))
        } else {
            let acc = self.ga_acceleration_obj.insert(rt_acc);
            Ok(acc)
        }
    }

    pub fn acceleration_destroy(&mut self, acc: AccelerationHandle) {
        let rt_acc = *self.ga_acceleration_obj.get(acc).unwrap();
        match self.ga_acceleration_obj.destroy(acc) {
            DestroyResult::StillAlive => (),
            DestroyResult::ShouldDrop => {
                if unsafe {
                    rtAccelerationDestroy(rt_acc)
                } != RtResult::SUCCESS {
                    panic!("Error destroying acceleration: {}", acc);       
                }
            }
        }
    }

    pub fn acceleration_validate(&self, acc: AccelerationHandle) -> Result<()> {
        let rt_acc = *self.ga_acceleration_obj.get(acc).unwrap();
        let result = unsafe {
            rtAccelerationValidate(rt_acc)
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtAccelerationValidate", result))
        } else {
            Ok(())
        }
    }
}
