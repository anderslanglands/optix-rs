use super::allocator::{Allocation, Allocator, Mallocator};
use optix_sys::cuda_sys::{
    cudaError, cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind, CUdeviceptr,
};

use std::os::raw::c_void;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

#[repr(u32)]
#[derive(Debug, Copy, Clone, thiserror::Error)]
pub enum MemcpyKind {
    #[error("Host to Host")]
    HostToHost = cudaMemcpyKind::cudaMemcpyHostToHost,
    #[error("Host to Device")]
    HostToDevice = cudaMemcpyKind::cudaMemcpyHostToDevice,
    #[error("Device to Host")]
    DeviceToHost = cudaMemcpyKind::cudaMemcpyDeviceToHost,
    #[error("Device to Device")]
    DeviceToDevice = cudaMemcpyKind::cudaMemcpyDeviceToDevice,
    #[error("Default")]
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

pub struct Buffer<'a, AllocT = Mallocator>
where
    AllocT: Allocator,
{
    allocation: Allocation,
    _alloc: &'a AllocT,
}

impl<'a, AllocT> Buffer<'a, AllocT>
where
    AllocT: Allocator,
{
    pub fn new(
        size_in_bytes: usize,
        alignment: usize,
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<Buffer<'a, AllocT>> {
        let allocation =
            unsafe { allocator.alloc(size_in_bytes, alignment, tag)? };
        Ok(Buffer {
            allocation,
            _alloc: allocator,
        })
    }

    pub fn with_data<T>(
        data: &[T],
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<Buffer<'a, AllocT>>
    where
        T: Sized,
    {
        let size_in_bytes = std::mem::size_of::<T>() * data.len();

        if size_in_bytes != 0 {
            let allocation = unsafe {
                allocator.alloc(
                    size_in_bytes,
                    std::mem::align_of::<T>(),
                    tag,
                )?
            };

            let res = unsafe {
                cudaMemcpy(
                    allocation.ptr() as *mut c_void,
                    data.as_ptr() as *const c_void,
                    size_in_bytes,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            if res != cudaError::cudaSuccess {
                return Err(Error::BufferUploadFailed { source: res.into() });
            }
            Ok(Buffer {
                allocation,
                _alloc: allocator,
            })
        } else {
            Err(Error::ZeroAllocation)
        }
    }

    pub fn upload<T>(&mut self, data: &[T]) -> Result<()> {
        let sz = data.len() * std::mem::size_of::<T>();
        if sz != self.allocation.size() {
            return Err(Error::BufferUploadWrongSize {
                upload_size: sz,
                buffer_size: self.allocation.size(),
            });
        }
        unsafe {
            let res = cudaMemcpy(
                self.allocation.ptr() as *mut c_void,
                data.as_ptr() as *const c_void,
                self.allocation.size(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            );
            if res != cudaError::cudaSuccess {
                return Err(Error::BufferUploadFailed { source: res.into() });
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
            self.allocation.ptr() as *mut c_void,
            data as *const c_void,
            size,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
        );
        if res != cudaError::cudaSuccess {
            return Err(Error::BufferUploadFailed { source: res.into() });
        }

        Ok(())
    }

    pub fn download<T>(&self, data: &mut [T]) -> Result<()> {
        let sz = data.len() * std::mem::size_of::<T>();
        if sz != self.allocation.size() {
            return Err(Error::BufferDownloadWrongSize {
                download_size: sz,
                buffer_size: self.allocation.size(),
            });
        }
        unsafe {
            let res = cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.allocation.ptr() as *mut c_void,
                self.allocation.size(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            if res != cudaError::cudaSuccess {
                return Err(Error::BufferDownloadFailed { source: res.into() });
            }
        }

        Ok(())
    }

    pub fn download_primitive<T>(&self) -> Result<T>
    where
        T: Default,
    {
        let sz = std::mem::size_of::<T>();
        if sz != self.allocation.size() {
            return Err(Error::BufferDownloadWrongSize {
                download_size: sz,
                buffer_size: self.allocation.size(),
            });
        }

        let mut data = T::default();
        unsafe {
            let res = cudaMemcpy(
                &mut data as *mut T as *mut c_void,
                self.allocation.ptr() as *mut c_void,
                self.allocation.size(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            );
            if res != cudaError::cudaSuccess {
                return Err(Error::BufferDownloadFailed { source: res.into() });
            }
        }

        Ok(data)
    }

    pub fn as_ptr(&self) -> *const c_void {
        self.allocation.ptr() as *const c_void
    }

    pub fn as_device_ptr(&self) -> CUdeviceptr {
        self.allocation.ptr() as CUdeviceptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.allocation.ptr() as *mut c_void
    }

    pub fn byte_size(&self) -> usize {
        self.allocation.size()
    }
}

impl<'a, AllocT> Drop for Buffer<'a, AllocT>
where
    AllocT: Allocator,
{
    fn drop(&mut self) {
        unsafe {
            let mut a = Allocation::new(0, 0, 0);
            std::mem::swap(&mut self.allocation, &mut a);
            self._alloc.dealloc(a).expect("dealloc failed")
        }
    }
}
