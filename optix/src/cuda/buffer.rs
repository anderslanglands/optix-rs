use optix_sys::cuda_sys::{
    cudaError, cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind, CUdeviceptr,
};

use std::os::raw::c_void;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

#[repr(u32)]
#[derive(Debug, Copy, Clone, Display)]
pub enum MemcpyKind {
    #[display(fmt = "Host to Host")]
    HostToHost = cudaMemcpyKind::cudaMemcpyHostToHost,
    #[display(fmt = "Host to Device")]
    HostToDevice = cudaMemcpyKind::cudaMemcpyHostToDevice,
    #[display(fmt = "Device to Host")]
    DeviceToHost = cudaMemcpyKind::cudaMemcpyDeviceToHost,
    #[display(fmt = "Device to Device")]
    DeviceToDevice = cudaMemcpyKind::cudaMemcpyDeviceToDevice,
    #[display(fmt = "Default")]
    Default = cudaMemcpyKind::cudaMemcpyDefault,
}

impl From<cudaMemcpyKind::Type> for MemcpyKind {
    fn from(k: cudaMemcpyKind::Type) -> MemcpyKind {
        match k {
            cudaMemcpyKind::cudaMemcpyHostToHost => MemcpyKind::HostToHost,
            cudaMemcpyKind::cudaMemcpyHostToDevice => MemcpyKind::HostToDevice,
            cudaMemcpyKind::cudaMemcpyDeviceToHost => MemcpyKind::DeviceToHost,
            cudaMemcpyKind::cudaMemcpyDeviceToDevice => {
                MemcpyKind::DeviceToDevice
            }
            cudaMemcpyKind::cudaMemcpyDefault => MemcpyKind::Default,
            _ => unreachable!(),
        }
    }
}

impl From<MemcpyKind> for cudaMemcpyKind::Type {
    fn from(k: MemcpyKind) -> cudaMemcpyKind::Type {
        match k {
            MemcpyKind::HostToHost => cudaMemcpyKind::cudaMemcpyHostToHost,
            MemcpyKind::HostToDevice => cudaMemcpyKind::cudaMemcpyHostToDevice,
            MemcpyKind::DeviceToHost => cudaMemcpyKind::cudaMemcpyDeviceToHost,
            MemcpyKind::DeviceToDevice => {
                cudaMemcpyKind::cudaMemcpyDeviceToDevice
            }
            MemcpyKind::Default => cudaMemcpyKind::cudaMemcpyDefault,
        }
    }
}

pub struct Buffer {
    size_in_bytes: usize,
    d_ptr: *mut c_void,
}

impl Buffer {
    pub fn new(size_in_bytes: usize) -> Result<Buffer> {
        unsafe {
            let mut d_ptr = std::ptr::null_mut();
            let res = cudaMalloc(&mut d_ptr, size_in_bytes);
            if res != cudaError::cudaSuccess {
                Err(Error::BufferAllocationFailed {
                    cerr: res.into(),
                    size: size_in_bytes,
                })
            } else {
                Ok(Buffer {
                    size_in_bytes,
                    d_ptr,
                })
            }
        }
    }

    pub fn with_data<T>(data: &[T]) -> Result<Buffer>
    where
        T: Sized,
    {
        unsafe {
            let size_in_bytes = std::mem::size_of::<T>() * data.len();
            let mut d_ptr = std::ptr::null_mut();

            if size_in_bytes != 0 {
                let res = cudaMalloc(&mut d_ptr, size_in_bytes);
                if res != cudaError::cudaSuccess {
                    return Err(Error::BufferAllocationFailed {
                        cerr: res.into(),
                        size: size_in_bytes,
                    });
                }

                let res = cudaMemcpy(
                    d_ptr,
                    data.as_ptr() as *const c_void,
                    size_in_bytes,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                );
                if res != cudaError::cudaSuccess {
                    return Err(Error::BufferUploadFailed { cerr: res.into() });
                }
            }

            Ok(Buffer {
                size_in_bytes,
                d_ptr,
            })
        }
    }

    pub fn upload<T>(&mut self, data: &[T]) -> Result<()> {
        let sz = data.len() * std::mem::size_of::<T>();
        if sz != self.size_in_bytes {
            return Err(Error::BufferUploadWrongSize {
                upload_size: sz,
                buffer_size: self.size_in_bytes,
            });
        }
        unsafe {
            let res = cudaMemcpy(
                self.d_ptr,
                data.as_ptr() as *const c_void,
                self.size_in_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            if res != cudaError::cudaSuccess {
                return Err(Error::BufferUploadFailed { cerr: res.into() });
            }
        }

        Ok(())
    }

    pub unsafe fn upload_ptr(
        &mut self,
        data: *const c_void,
        size: usize,
    ) -> Result<()> {
        let res = cudaMemcpy(
            self.d_ptr,
            data as *const c_void,
            size,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        if res != cudaError::cudaSuccess {
            return Err(Error::BufferUploadFailed { cerr: res.into() });
        }

        Ok(())
    }

    pub fn download<T>(&self, data: &mut [T]) -> Result<()> {
        let sz = data.len() * std::mem::size_of::<T>();
        if sz != self.size_in_bytes {
            return Err(Error::BufferDownloadWrongSize {
                download_size: sz,
                buffer_size: self.size_in_bytes,
            });
        }
        unsafe {
            let res = cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.d_ptr,
                self.size_in_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            if res != cudaError::cudaSuccess {
                return Err(Error::BufferDownloadFailed { cerr: res.into() });
            }
        }

        Ok(())
    }

    pub fn download_primitive<T>(&self) -> Result<T>
    where
        T: Default,
    {
        let sz = std::mem::size_of::<T>();
        if sz != self.size_in_bytes {
            return Err(Error::BufferDownloadWrongSize {
                download_size: sz,
                buffer_size: self.size_in_bytes,
            });
        }

        let mut data = T::default();
        unsafe {
            let res = cudaMemcpy(
                &mut data as *mut T as *mut c_void,
                self.d_ptr,
                self.size_in_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            if res != cudaError::cudaSuccess {
                return Err(Error::BufferDownloadFailed { cerr: res.into() });
            }
        }

        Ok(data)
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.d_ptr
    }

    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.d_ptr as CUdeviceptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.d_ptr
    }

    pub fn byte_size(&self) -> usize {
        self.size_in_bytes
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.d_ptr);
        }
    }
}

pub struct BufferArray {
    d_ptrs: Vec<*mut c_void>,
    size_in_bytes: usize,
}

impl BufferArray {
    pub fn new(bufs: Vec<Buffer>) -> BufferArray {
        let size_in_bytes = bufs[0].size_in_bytes;
        let d_ptrs: Vec<*mut c_void> =
            bufs.into_iter().map(|b| b.d_ptr).collect();

        BufferArray {
            d_ptrs,
            size_in_bytes,
        }
    }

    pub fn as_ptr(&self) -> *const *const c_void {
        self.d_ptrs.as_ptr() as *const *const c_void
    }

    pub fn as_device_ptr(&self) -> *const CUdeviceptr {
        self.d_ptrs.as_ptr() as *const CUdeviceptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut *mut c_void {
        self.d_ptrs.as_mut_ptr()
    }

    pub fn byte_size(&self) -> usize {
        self.size_in_bytes
    }
}
