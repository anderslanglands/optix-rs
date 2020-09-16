use crate::{sys, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Hash)]
pub struct DevicePtr {
    ptr: sys::CUdeviceptr,
}

impl std::fmt::Display for DevicePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:p}", self.ptr as *const u64)
    }
}

impl DevicePtr {
    const TAG_SHIFT:u64 = 48;
    const TAG_MASK: u64 = ((1 << 16) - 1) << DevicePtr::TAG_SHIFT;
    const PTR_MASK: u64 = !DevicePtr::TAG_MASK;

    pub fn new(ptr: sys::CUdeviceptr) -> DevicePtr {
        DevicePtr { ptr }
    }

    pub fn null() -> DevicePtr {
        DevicePtr { ptr: 0 }
    }

    pub fn with_tag(ptr: sys::CUdeviceptr, tag: u16) -> DevicePtr {
        assert_eq!(ptr & DevicePtr::PTR_MASK, ptr);
        let ptr = ptr | ((tag as u64) << DevicePtr::TAG_SHIFT);
        DevicePtr { ptr }
    }

    pub fn ptr(&self) -> sys::CUdeviceptr {
        self.ptr & DevicePtr::PTR_MASK
    }

    pub fn tag(&self) -> u16 {
        ((self.ptr & DevicePtr::TAG_MASK) >> DevicePtr::TAG_SHIFT) as u16
    }
}

pub fn mem_alloc(size: usize) -> Result<DevicePtr> {
    let mut ptr = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut ptr, size as u64)
            .to_result()
            .map(|_| {
                DevicePtr{ptr}
            })
    }
}

pub fn mem_alloc_with_tag(size: usize, tag: u16) -> Result<DevicePtr> {
    let mut ptr = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut ptr, size as u64)
            .to_result()
            .map(|_| {
                DevicePtr::with_tag(ptr, tag)
            })
    }
}

pub fn mem_free(ptr: DevicePtr) -> Result<()> {
    unsafe { sys::cuMemFree_v2(ptr.ptr()).to_result() }
}
