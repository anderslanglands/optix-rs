use crate::{
    memory::{
        mem_alloc, mem_alloc_pitch, 
        mem_free, Allocation
    },
    sys::CUdeviceptr,
    DevicePtr, Error,
};
type Result<T, E = Error> = std::result::Result<T, E>;

pub use std::alloc::Layout;
pub use std::ptr::NonNull;

// This is true for >= Kepler...
const MAX_ALIGNMENT: usize = 512;
// FIXME: we need to modify the allocators to grab this info from the device
// upon creation
const PITCH_ALIGNMENT: usize = 32;

const KB: usize = 1024;
const MB: usize = KB * KB;
const GB: usize = MB * 1024;

pub unsafe trait DeviceAllocImpl {
    fn alloc(&mut self, layout: Layout) -> Result<Allocation>;
    fn alloc_with_tag<T: Into<u16>>(&mut self, layout: Layout, _tag: T) -> Result<Allocation> {
        self.alloc(layout)
    }
    fn alloc_pitch(
        &mut self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
    ) -> Result<(Allocation, usize)>;
    fn alloc_pitch_with_tag<T: Into<u16>>(
        &mut self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
        _tag: T,
    ) -> Result<(Allocation, usize)> {
        self.alloc_pitch(width_in_bytes, height_in_rows, element_byte_size)
    }
    fn dealloc(&mut self, ptr: Allocation) -> Result<()>;
}

pub unsafe trait DeviceAllocRef {
    fn alloc(&self, layout: Layout) -> Result<Allocation>;
    fn alloc_with_tag<T: Into<u16>>(&self, layout: Layout, _tag: T) -> Result<Allocation> {
        self.alloc(layout)
    }
    fn alloc_pitch(
        &self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
    ) -> Result<(Allocation, usize)>;
    fn alloc_pitch_with_tag<T: Into<u16>>(
        &self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
        _tag: T,
    ) -> Result<(Allocation, usize)> {
        self.alloc_pitch(width_in_bytes, height_in_rows, element_byte_size)
    }
    fn dealloc(&self, ptr: Allocation) -> Result<()>;
}

#[derive(Copy, Clone)]
pub struct DefaultDeviceAlloc;

unsafe impl DeviceAllocRef for DefaultDeviceAlloc {
    fn alloc(&self, layout: Layout) -> Result<Allocation> {
        if !layout.align().is_power_of_two() || layout.align() > MAX_ALIGNMENT {
            let msg = format!("Cannot satisfy alignment of {}", layout.align());
            panic!("{}", msg);
        }

        mem_alloc(layout.size())
    }

    fn alloc_pitch(
        &self,
        width_in_bytes: usize,
        height_in_rows: usize,
        element_byte_size: usize,
    ) -> Result<(Allocation, usize)> {
        mem_alloc_pitch(width_in_bytes, height_in_rows, element_byte_size)
    }

    fn dealloc(&self, ptr: Allocation) -> Result<()> {
        mem_free(ptr.ptr)
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
    max_size: usize,
    current_ptr: CUdeviceptr,
    current_end: CUdeviceptr,
    num_allocs: usize,
    total_allocated: usize,
}

impl DeviceFrameAllocator {
    const DEFAULT_ALLOC_BIT: u16 = 1u16 << 15;
    const INTERNAL_TAG_SHIFT: u64 = 12;
    const INTERNAL_TAG_MASK: u16 =
        ((1 << 4) - 1) << DeviceFrameAllocator::INTERNAL_TAG_SHIFT;
    pub const EXTERNAL_TAG_MASK: u16 = !DeviceFrameAllocator::INTERNAL_TAG_MASK;

    pub fn new(block_size: usize, max_size: usize) -> Result<Self> {
        // make sure the block size matches our alignment
        let block_size = align_up(block_size as u64, MAX_ALIGNMENT) as usize;
        let block = mem_alloc(block_size)?.ptr.0;
        let current_ptr = block;
        let current_end = current_ptr + block_size as u64;
        Ok(Self {
            old_blocks: Vec::new(),
            block,
            block_size,
            max_size,
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

            // first check if the block size is big enough for the alloc
            while self.block_size < layout.size() {
                self.block_size *= 2;
            }

            self.block = mem_alloc(self.block_size)?.ptr.0;
            self.current_end = self.block + self.block_size as u64;

            let new_ptr = align_up(self.block, layout.align());
            self.current_ptr = new_ptr + layout.size() as u64;
            Ok(new_ptr)
        } else {
            self.current_ptr = new_ptr + layout.size() as u64;
            Ok(new_ptr)
        }
    }

    pub fn report(&self) -> (usize, usize) {
        (self.num_allocs, self.total_allocated)
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

unsafe impl DeviceAllocImpl for DeviceFrameAllocator {
    fn alloc(&mut self, layout: Layout) -> Result<Allocation> {
        if layout.size() < self.max_size {
            Ok(Allocation{ptr: DevicePtr::new(self.alloc_impl(layout)?), size: layout.size()})
        } else {
            let dptr = mem_alloc(layout.size())?;
            let ptr = dptr.ptr.0;
            Ok(Allocation{ptr: DevicePtr::new(
                ptr,
            ), size: layout.size()})
            // Ok(Allocation{ptr: DevicePtr::with_tag(
            //     ptr,
            //     0u16 | DeviceFrameAllocator::DEFAULT_ALLOC_BIT,
            // ), size: layout.size()})
        }
    }

    fn alloc_pitch(
        &mut self,
        width_in_bytes: usize,
        height_in_rows: usize,
        _element_byte_size: usize,
    ) -> Result<(Allocation, usize)> {
        // each row must be aligned to PITCH_ALIGNMENT bytes
        let pitch = align_up(width_in_bytes as u64, PITCH_ALIGNMENT);
        let layout = Layout::from_size_align(
            pitch as usize * height_in_rows,
            MAX_ALIGNMENT,
        )
        .expect("bad layout");

        if layout.size() < self.max_size {
            let ptr = self.alloc_impl(layout)?;
            Ok((Allocation{ptr: DevicePtr::new(ptr), size: layout.size()}, pitch as usize))
        } else {
            let dptr = mem_alloc(layout.size())?;
            let ptr = dptr.ptr.0;
            Ok((
                // Allocation{ptr: DevicePtr::with_tag(
                //     ptr,
                //     0u16 | DeviceFrameAllocator::DEFAULT_ALLOC_BIT,
                // ), size: layout.size()},
                Allocation{ptr: DevicePtr::new(
                    ptr,
                ), size: layout.size()},
                pitch as usize,
            ))
        }
    }

    fn dealloc(&mut self, _ptr: Allocation) -> Result<()> {
        Ok(())
    }
}

impl Drop for DeviceFrameAllocator {
    fn drop(&mut self) {
        // just let it leak
    }
}
