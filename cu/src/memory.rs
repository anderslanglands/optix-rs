use crate::{sys, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Hash)]
pub struct DevicePtr(pub sys::CUdeviceptr);

impl std::fmt::Display for DevicePtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:p}", self.0 as *const u64)
    }
}

impl DevicePtr {
    pub const TAG_SHIFT: u64 = 48;
    pub const TAG_MASK: u64 = ((1 << 16) - 1) << DevicePtr::TAG_SHIFT;
    pub const PTR_MASK: u64 = !DevicePtr::TAG_MASK;

    pub fn new(ptr: sys::CUdeviceptr) -> DevicePtr {
        DevicePtr(ptr)
    }

    pub fn null() -> DevicePtr {
        DevicePtr(0)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub struct Allocation {
    pub ptr: DevicePtr,
    pub size: usize,
}


pub fn mem_alloc(size: usize) -> Result<Allocation> {
    let mut ptr = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut ptr, size as u64)
            .to_result()
            .map(|_| Allocation{ptr: DevicePtr(ptr), size})
    }
}

pub fn mem_alloc_pitch(
    width_in_bytes: usize,
    height_in_rows: usize,
    element_size_in_bytes: usize,
) -> Result<(Allocation, usize)> {
    let mut ptr = 0;
    let mut pitch: usize = 0;
    unsafe {
        sys::cuMemAllocPitch_v2(
            &mut ptr,
            &mut pitch as *mut _ as *mut _,
            width_in_bytes as u64,
            height_in_rows as u64,
            element_size_in_bytes as u32,
        )
        .to_result()
        .map(|_| (Allocation{ptr: DevicePtr(ptr), size: pitch * height_in_rows}, pitch))
    }
}

pub fn mem_free(ptr: DevicePtr) -> Result<()> {
    unsafe { sys::cuMemFree_v2(ptr.0).to_result() }
}

pub unsafe fn memcpy2d_htod(
    dst_ptr: DevicePtr,
    dst_width_in_bytes: usize,
    dst_height_in_rows: usize,
    dst_pitch_in_bytes: usize,
    src_ptr: *const std::os::raw::c_void,
    src_width_in_bytes: usize,
    src_height_in_rows: usize,
    src_pitch_in_bytes: usize,
) -> Result<()> {
    let cpy = sys::CUDA_MEMCPY2D_st {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_HOST,
        srcHost: src_ptr,
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: src_pitch_in_bytes as u64,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst_ptr.0,
        dstArray: std::ptr::null_mut(),
        dstPitch: dst_pitch_in_bytes as u64,
        WidthInBytes: src_width_in_bytes as u64,
        Height: src_height_in_rows as u64,
    };

    sys::cuMemcpy2D_v2(&cpy as *const _).to_result()
}
