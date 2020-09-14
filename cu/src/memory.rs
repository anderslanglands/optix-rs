use crate::{sys, Error, DeviceCopy};
type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct DevicePtr {
    pub(crate) inner: sys::CUdeviceptr,
}

impl DevicePtr {
    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.inner
    }

    pub fn null() -> DevicePtr {
        DevicePtr {
            inner: 0,
        }
    }
}

pub fn mem_alloc(size: usize) -> Result<DevicePtr> {
    let mut inner = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut inner, size as u64)
            .to_result()
            .map(|_| {
                DevicePtr { inner }
            })
    }
}

pub fn mem_free(ptr: DevicePtr) -> Result<()> {
    unsafe { sys::cuMemFree_v2(ptr.inner).to_result() }
}
