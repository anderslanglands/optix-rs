use optix_sys::cuda_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub struct Array {
    ptr: sys::cudaArray_t,
}

impl Array {
    pub fn new<T>(
        data: &[T],
        desc: ChannelFormatDesc,
        width: usize,
        height: usize,
        num_components: usize,
        flags: ArrayFlags,
    ) -> Result<Array> {
        let mut ptr = std::ptr::null_mut();
        unsafe {
            let res = sys::cudaMallocArray(
                &mut ptr,
                &desc as *const ChannelFormatDesc
                    as *const sys::cudaChannelFormatDesc,
                width,
                height,
                flags.bits(),
            );
            if res != sys::cudaError::cudaSuccess {
                return Err(Error::ArrayAllocationFailed {
                    source: res.into(),
                    desc,
                    width,
                    height,
                    num_components,
                    flags,
                });
            }

            let pitch = width * num_components * std::mem::size_of::<T>();
            let res = sys::cudaMemcpy2DToArray(
                ptr,
                0,
                0,
                data.as_ptr() as *const std::os::raw::c_void,
                pitch,
                pitch,
                height,
                super::MemcpyKind::HostToDevice as u32,
            );
            if res != sys::cudaError::cudaSuccess {
                return Err(Error::ArrayMemcpy2DFailed { source: res.into() });
            }

            Ok(Array { ptr })
        }
    }

    #[allow(non_snake_case)]
    pub fn new_3d<T>(
        data: &[T],
        desc: ChannelFormatDesc,
        width: usize,
        height: usize,
        depth: usize,
        num_components: usize,
        flags: ArrayFlags,
    ) -> Result<Array> {
        let mut ptr = std::ptr::null_mut();
        let extent = sys::cudaExtent {
            width,
            height,
            depth,
        };
        unsafe {
            let res = sys::cudaMalloc3DArray(
                &mut ptr,
                &desc as *const ChannelFormatDesc
                    as *const sys::cudaChannelFormatDesc,
                extent,
                flags.bits(),
            );
            if res != sys::cudaError::cudaSuccess {
                return Err(Error::Array3DAllocationFailed {
                    source: res.into(),
                    desc,
                    width,
                    height,
                    depth,
                    num_components,
                    flags,
                });
            }

            let pitch = width * num_components * std::mem::size_of::<T>();
            let srcPtr = sys::cudaPitchedPtr {
                ptr: data.as_ptr() as *const std::os::raw::c_void,
                pitch,
                xsize: width,
                ysize: height,
            };

            let params = sys::cudaMemcpy3DParms {
                srcArray: std::ptr::null_mut(),
                srcPos: sys::cudaPos { x: 0, y: 0, z: 0 },
                srcPtr,
                dstArray: ptr,
                dstPos: sys::cudaPos { x: 0, y: 0, z: 0 },
                dstPtr: sys::cudaPitchedPtr {
                    ptr: std::ptr::null_mut(),
                    pitch: 0,
                    xsize: 0,
                    ysize: 0,
                },
                extent,
                kind: sys::cudaMemcpyKind::cudaMemcpyHostToDevice,
            };

            let res = sys::cudaMemcpy3D(&params);
            if res != sys::cudaError::cudaSuccess {
                return Err(Error::ArrayMemcpy3DFailed { source: res.into() });
            }

            Ok(Array { ptr })
        }
    }

    pub fn as_device_ptr(&self) -> sys::cudaArray_t {
        self.ptr
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        unsafe {
            let res = sys::cudaFreeArray(self.ptr);
            if res != sys::cudaError::cudaSuccess {
                panic!("cudaFreeArray failed: {:?}", res);
            }
        }
    }
}

bitflags::bitflags! {
pub struct ArrayFlags: u32 {
    const DEFAULT = 0x00;
    const LAYERED = 0x01;
    const SURFACE_LOAD_STORE = 0x02;
    const CUBEMAP = 0x04;
    const TEXTURE_GATHER = 0x08;
    const COLOR_ATTACHMENT = 0x20;
}
}

#[repr(C)]
#[derive(Debug)]
pub struct ChannelFormatDesc {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub w: i32,
    pub f: ChannelFormatKind,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ChannelFormatKind {
    Signed = sys::cudaChannelFormatKind_cudaChannelFormatKindSigned,
    Unsigned = sys::cudaChannelFormatKind_cudaChannelFormatKindUnsigned,
    Float = sys::cudaChannelFormatKind_cudaChannelFormatKindFloat,
    None = sys::cudaChannelFormatKind_cudaChannelFormatKindNone,
}
