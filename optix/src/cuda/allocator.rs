use super::{CUdeviceptr, Error};
use bitfield::*;
use optix_sys::cuda_sys as sys;
use std::cell::{Cell, Ref, RefCell};
use std::collections::HashMap;

type Result<T, E = Error> = std::result::Result<T, E>;

pub trait Allocator {
    unsafe fn alloc(
        &self,
        size: usize,
        alignment: usize,
        tag: u64,
    ) -> Result<Allocation>;
    unsafe fn dealloc(&self, allocation: Allocation) -> Result<()>;
}

pub struct Mallocator {}

impl Mallocator {
    pub fn new() -> Mallocator {
        Mallocator {}
    }
}

pub struct TaggedMallocator {
    total_allocated: Cell<usize>,
    allocs_by_tag: RefCell<HashMap<u64, usize>>,
}

impl TaggedMallocator {
    pub fn new() -> TaggedMallocator {
        TaggedMallocator {
            total_allocated: Cell::new(0),
            allocs_by_tag: RefCell::new(HashMap::new()),
        }
    }

    pub fn total_allocated(&self) -> usize {
        self.total_allocated.get()
    }

    pub fn tag_allocations(&self) -> Ref<HashMap<u64, usize>> {
        self.allocs_by_tag.borrow()
    }
}

pub struct Allocation {
    ptr: CUdeviceptr,
    tagged_size: TaggedSize,
}

impl Allocation {
    pub fn new(ptr: CUdeviceptr, size: usize, tag: u64) -> Allocation {
        let mut tagged_size = TaggedSize(0);
        tagged_size.set_size(size as u64);
        tagged_size.set_tag(tag);
        Allocation { ptr, tagged_size }
    }

    pub fn size(&self) -> usize {
        self.tagged_size.get_size() as usize
    }

    pub fn tag(&self) -> u64 {
        self.tagged_size.get_tag()
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

bitfield! {
    pub struct TaggedSize(u64);
    get_size, set_size: 39, 0;
    get_tag, set_tag: 63, 40;
}

impl Allocator for Mallocator {
    unsafe fn alloc(
        &self,
        size: usize,
        alignment: usize,
        tag: u64,
    ) -> Result<Allocation> {
        // cuda mallocs are always 256 or 512-byte aligned on recent GPUs so we
        // can satisfy anything up to that
        // FIXME: check this don't assume it's 512
        if alignment > 512 || !alignment.is_power_of_two() {
            return Err(Error::AllocationAlignment {
                size,
                align: alignment,
            });
        }

        let mut ptr = std::ptr::null_mut();
        let res = sys::cudaMalloc(&mut ptr, size);
        if res != sys::cudaError::cudaSuccess || ptr.is_null() {
            Err(Error::AllocationFailed {
                source: res.into(),
                size: size,
            })
        } else {
            Ok(Allocation::new(ptr as CUdeviceptr, size, tag))
        }
    }

    unsafe fn dealloc(&self, allocation: Allocation) -> Result<()> {
        sys::cudaFree(allocation.ptr as *mut std::os::raw::c_void);
        Ok(())
    }
}

impl Allocator for TaggedMallocator {
    unsafe fn alloc(
        &self,
        size: usize,
        alignment: usize,
        tag: u64,
    ) -> Result<Allocation> {
        // cuda mallocs are always 512-byte aligned on recent GPUs so we can
        // satisfy anything up to that
        // FIXME: check this don't assume it's 512
        if alignment > 512 || !alignment.is_power_of_two() {
            return Err(Error::AllocationAlignment {
                size,
                align: alignment,
            });
        }

        let mut ptr = std::ptr::null_mut();
        let res = sys::cudaMalloc(&mut ptr, size);
        if res != sys::cudaError::cudaSuccess || ptr.is_null() {
            Err(Error::AllocationFailed {
                source: res.into(),
                size: size,
            })
        } else {
            self.total_allocated.set(self.total_allocated.get() + size);
            *self.allocs_by_tag.borrow_mut().entry(tag).or_insert(0) += size;
            Ok(Allocation::new(ptr as CUdeviceptr, size, tag))
        }
    }

    unsafe fn dealloc(&self, allocation: Allocation) -> Result<()> {
        sys::cudaFree(allocation.ptr as *mut std::os::raw::c_void);
        self.total_allocated
            .set(self.total_allocated.get() - allocation.size());
        *self
            .allocs_by_tag
            .borrow_mut()
            .get_mut(&allocation.tag())
            .unwrap() -= allocation.size();
        Ok(())
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_bitfield() {
        use super::TaggedSize;
        let mut sz = TaggedSize(0);
        sz.set_size(578);
        sz.set_tag(1017);

        assert_eq!(sz.get_size(), 578);
        assert_eq!(sz.get_tag(), 1017);
    }
}
