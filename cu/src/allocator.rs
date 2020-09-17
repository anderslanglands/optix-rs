use crate::{
    memory::{
        mem_alloc, mem_alloc_pitch, mem_alloc_pitch_with_tag,
        mem_alloc_with_tag, mem_free,
    },
    sys::CUdeviceptr,
    DevicePtr, Error,
};
type Result<T, E = Error> = std::result::Result<T, E>;

use parking_lot::Mutex;
pub use std::alloc::Layout;
pub use std::ptr::NonNull;

// This is true for >= Kepler...
const MAX_ALIGNMENT: usize = 512;

pub unsafe trait DeviceAllocRef {
    fn alloc(&self, layout: Layout) -> Result<DevicePtr>;
    fn alloc_with_tag(&self, layout: Layout, tag: u16) -> Result<DevicePtr>;
    fn alloc_pitch(
        &self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
    ) -> Result<(DevicePtr, usize)>;
    fn alloc_pitch_with_tag(
        &self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
        tag: u16,
    ) -> Result<(DevicePtr, usize)>;
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

    fn alloc_with_tag(&self, layout: Layout, tag: u16) -> Result<DevicePtr> {
        if !layout.align().is_power_of_two() || layout.align() > MAX_ALIGNMENT {
            let msg = format!("Cannot satisfy alignment of {}", layout.align());
            panic!("{}", msg);
        }

        mem_alloc_with_tag(layout.size(), tag)
    }

    fn alloc_pitch(
        &self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
    ) -> Result<(DevicePtr, usize)> {
        mem_alloc_pitch(width_in_bytes, height_in_rows, element_byte_size)
    }

    fn alloc_pitch_with_tag(
        &self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
        tag: u16,
    ) -> Result<(DevicePtr, usize)> {
        mem_alloc_pitch_with_tag(
            width_in_bytes,
            height_in_rows,
            element_byte_size,
            tag,
        )
    }

    fn dealloc(&self, ptr: DevicePtr) -> Result<()> {
        mem_free(ptr)
    }
}

fn align_offset(ptr: u64, align: usize) -> u64 {
    let align = align as u64;
    ((!ptr) + 1) & (align - 1)
}

fn align_up(ptr: u64, align: usize) -> u64 {
    ptr + align_offset(ptr, align)
}

pub struct DeviceFrameAllocator {
    old_blocks: Vec<CUdeviceptr>,
    block: CUdeviceptr,
    block_size: usize,
    current_ptr: CUdeviceptr,
    current_end: CUdeviceptr,
    num_allocs: usize,
    total_allocated: usize,
}

impl DeviceFrameAllocator {
    pub fn new(block_size: usize) -> Result<Self> {
        // make sure the block size matches our alignment
        let block_size = align_up(block_size as u64, MAX_ALIGNMENT) as usize;
        let block = mem_alloc(block_size)?.ptr();
        let current_ptr = block;
        let current_end = current_ptr + block_size as u64;
        Ok(Self {
            old_blocks: Vec::new(),
            block,
            block_size,
            current_ptr,
            current_end,
            num_allocs: 0,
            total_allocated: 0,
        })
    }

    fn alloc_impl(&mut self, layout: Layout) -> Result<CUdeviceptr> {
        if u64::MAX - self.current_ptr < (layout.size() + layout.align()) as u64
        {
            panic!("allocation too big for u64!");
        }

        self.num_allocs += 1;
        self.total_allocated += layout.size();

        let new_ptr = align_up(self.current_ptr, layout.align());

        if (new_ptr + layout.size() as u64) > self.current_end {
            // allocate a new block
            self.old_blocks.push(self.block);
            self.block = mem_alloc(self.block_size)?.ptr();
            self.current_end = self.block + self.block_size as u64;

            let new_ptr = align_up(self.block, layout.align());
            self.current_ptr = new_ptr + layout.size() as u64;
            Ok(new_ptr)
        } else {
            self.current_ptr = new_ptr + layout.size() as u64;
            Ok(new_ptr)
        }
    }

    pub fn alloc(&mut self, layout: Layout) -> Result<DevicePtr> {
        Ok(DevicePtr::new(self.alloc_impl(layout)?))
    }

    pub fn alloc_with_tag(
        &mut self,
        layout: Layout,
        tag: u16,
    ) -> Result<DevicePtr> {
        Ok(DevicePtr::with_tag(self.alloc_impl(layout)?, tag))
    }

    pub fn alloc_pitch(
        &self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
    ) -> Result<(DevicePtr, usize)> {
        mem_alloc_pitch(width_in_bytes, height_in_rows, element_byte_size)
    }

    pub fn alloc_pitch_with_tag(
        &self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
        tag: u16,
    ) -> Result<(DevicePtr, usize)> {
        mem_alloc_pitch_with_tag(
            width_in_bytes,
            height_in_rows,
            element_byte_size,
            tag,
        )
    }

    pub fn dealloc(&self, _ptr: DevicePtr) -> Result<()> {
        Ok(())
    }

    pub fn report(&self) -> (usize, usize) {
        (self.num_allocs, self.total_allocated)
    }
}

impl Drop for DeviceFrameAllocator {
    fn drop(&mut self) {
        // just let it leak
    }
}
