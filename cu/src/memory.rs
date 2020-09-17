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
    const TAG_SHIFT: u64 = 48;
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
            .map(|_| DevicePtr { ptr })
    }
}

pub fn mem_alloc_with_tag(size: usize, tag: u16) -> Result<DevicePtr> {
    let mut ptr = 0;
    unsafe {
        sys::cuMemAlloc_v2(&mut ptr, size as u64)
            .to_result()
            .map(|_| DevicePtr::with_tag(ptr, tag))
    }
}

pub fn mem_alloc_pitch(
    width_in_bytes: usize,
    height_in_rows: usize,
    element_size_in_bytes: usize,
) -> Result<(DevicePtr, usize)> {
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
        .map(|_| (DevicePtr { ptr }, pitch))
    }
}

pub fn mem_alloc_pitch_with_tag(
    width_in_bytes: usize,
    height_in_rows: usize,
    element_size_in_bytes: usize,
    tag: u16,
) -> Result<(DevicePtr, usize)> {
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
        .map(|_| (DevicePtr::with_tag(ptr, tag), pitch))
    }
}

pub fn mem_free(ptr: DevicePtr) -> Result<()> {
    unsafe { sys::cuMemFree_v2(ptr.ptr()).to_result() }
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
        dstDevice: dst_ptr.ptr(),
        dstArray: std::ptr::null_mut(),
        dstPitch: dst_pitch_in_bytes as u64,
        WidthInBytes: src_width_in_bytes as u64,
        Height: src_height_in_rows as u64,
    };

    sys::cuMemcpy2D_v2(&cpy as *const _).to_result()
}
