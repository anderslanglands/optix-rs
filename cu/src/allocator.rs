use crate::{
    memory::{mem_alloc, mem_free},
    sys, DevicePtr, Error,
};
type Result<T, E = Error> = std::result::Result<T, E>;

use parking_lot::Mutex;
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

/// A block of memory allocated from the default device allocator that a custom
/// allocator will portion up and give out
struct DeviceBlock {
    ptr: DevicePtr,
    size: usize,
}

fn align_offset(ptr: u64, align: usize) -> u64 {
    let align = align as u64;
    ((!ptr) + 1) & (align - 1)
}

fn align_up(ptr: u64, align: usize) -> u64 {
    ptr + align_offset(ptr, align)
}

pub struct DeviceFrameAllocator {
    old_blocks: Vec<DevicePtr>,
    block: DevicePtr,
    block_size: usize,
    current_ptr: u64,
    current_end: u64,
}

impl DeviceFrameAllocator {
    pub fn new(block_size: usize) -> Result<Self> {
        // make sure the block size matches our alignment
        let block_size = align_up(block_size as u64, MAX_ALIGNMENT) as usize;
        let block = mem_alloc(block_size)?;
        let current_ptr = block;
        let current_end = current_ptr + block_size as u64;
        Ok(Self {
            old_blocks: Vec::new(),
            block,
            block_size,
            current_ptr,
            current_end,
        })
    }

    pub fn alloc(&mut self, layout: Layout) -> Result<DevicePtr> {
        if u64::MAX - self.current_ptr < (layout.size() + layout.align()) as u64
        {
            panic!("allocation too big for u64!");
        }

        let new_ptr = align_up(self.current_ptr, layout.align());

        if (new_ptr + layout.size() as u64) > self.current_end {
            // allocate a new block
            self.old_blocks.push(self.block);
            self.block = mem_alloc(self.block_size)?;
            self.current_end = self.block + self.block_size as u64;

            let new_ptr = align_up(self.block, layout.align());
            self.current_ptr = new_ptr + layout.size() as u64;
            Ok(new_ptr)
        } else {
            self.current_ptr = new_ptr + layout.size() as u64;
            Ok(new_ptr)
        }
    }

    pub fn dealloc(&self, _ptr: DevicePtr) -> Result<()> {
        Ok(())
    }
}

impl Drop for DeviceFrameAllocator {
    fn drop(&mut self) {
        // just let it leak
    }
}

unsafe impl DeviceAllocRef for &Mutex<DeviceFrameAllocator> {
    fn alloc(&self, layout: Layout) -> Result<DevicePtr> {
        self.lock().alloc(layout)
    }

    fn dealloc(&self, ptr: DevicePtr) -> Result<()> {
        self.lock().dealloc(ptr)
    }
}
