use crate::{sys, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

pub type DevicePtr = sys::CUdeviceptr;

pub fn mem_alloc(size: usize) -> Result<DevicePtr> {
    let mut inner = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut inner, size as u64)
            .to_result()
            .map(|_| {
                inner
            })
    }
}

pub fn mem_free(ptr: DevicePtr) -> Result<()> {
    unsafe { sys::cuMemFree_v2(ptr).to_result() }
}
