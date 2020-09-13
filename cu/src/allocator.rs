use crate::{sys, Error, DevicePtr, memory::{mem_alloc, mem_free}};
type Result<T, E = Error> = std::result::Result<T, E>;

pub use std::alloc::Layout;
pub use std::ptr::NonNull;

// This is true for >= Kepler...
const MAX_ALIGNMENT: usize = 512;

pub unsafe trait DeviceAllocRef {
    fn alloc(&self, layout: Layout) -> Result<DevicePtr>;
    fn dealloc(&self, ptr: DevicePtr) -> Result<()>;
}

#[derive(Copy, Clone)]
pub struct DefaultDeviceAlloc;

unsafe impl DeviceAllocRef for DefaultDeviceAlloc {
    fn alloc(&self, layout: Layout) -> Result<DevicePtr> {
        if !layout.align().is_power_of_two() || layout.align() > MAX_ALIGNMENT {
            let msg = format!("Cannot satisfy alignment of {}", layout.align());
            panic!("{}", msg);
        }

        mem_alloc(layout.size())
    }

    fn dealloc(&self, ptr: DevicePtr) -> Result<()> {
        mem_free(ptr)
    }
}