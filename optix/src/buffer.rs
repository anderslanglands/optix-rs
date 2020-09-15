use crate::{DeviceCopy, Error};
use cu::{sys, DefaultDeviceAlloc, DeviceAllocRef, DevicePtr};
type Result<T, E = Error> = std::result::Result<T, E>;

use std::alloc::Layout;

/// Trait used to specify types that represent some chunk of storage on the 
/// device. Could be a buffer or a shared variable.
pub trait DeviceStorage {
    /// Returns the address of this storage on the device
    fn device_ptr(&self) -> DevicePtr;
    /// Returns the size in bytes of the stored data
    fn byte_size(&self) -> usize;
}

/// An untyped, opaque chunk of device memory
pub struct Buffer<A: DeviceAllocRef = DefaultDeviceAlloc> {
    ptr: DevicePtr,
    byte_size: usize,
    alloc: A,
}

impl Buffer<DefaultDeviceAlloc> {
    pub fn from_slice<T>(slice: &[T]) -> Result<Buffer> {
        Buffer::from_slice_in(slice, DefaultDeviceAlloc)
    }
}

impl<A: DeviceAllocRef> Buffer<A> {
    pub fn from_slice_in<T>(slice: &[T], alloc: A) -> Result<Buffer<A>> {
        let byte_size = slice.len() * std::mem::size_of::<T>();
        let ptr = alloc
            .alloc(Layout::array::<T>(slice.len()).unwrap())
            .map_err(|source| Error::Allocation { source })?;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(Buffer {
            ptr,
            byte_size,
            alloc,
        })
    }

    pub fn uninitialized_with_align_in(
        byte_size: usize,
        align: usize,
        alloc: A,
    ) -> Result<Buffer<A>> {
        let ptr = alloc
            .alloc(Layout::from_size_align(byte_size, align).unwrap())
            .map_err(|source| Error::Allocation { source })?;
        Ok(Buffer {
            ptr,
            byte_size,
            alloc,
        })
    }

    pub fn copy_to_slice<T>(&self, slice: &mut [T]) -> Result<()> {
        let byte_size = slice.len() * std::mem::size_of::<T>();
        if byte_size != self.byte_size {
            panic!(
                "Tried to copy {} bytes to Buffer of {} bytes",
                byte_size, self.byte_size
            );
        }

        unsafe {
            sys::cuMemcpyDtoH_v2(
                slice.as_mut_ptr() as *mut _,
                self.ptr.ptr(),
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })
        }
    }
}

impl<A: DeviceAllocRef> DeviceStorage for Buffer<A> {
    fn device_ptr(&self) -> DevicePtr {
        self.ptr
    }

    fn byte_size(&self) -> usize {
        self.byte_size
    }
}

impl<A: DeviceAllocRef> Drop for Buffer<A> {
    fn drop(&mut self) {
        self.alloc.dealloc(self.ptr).expect("dealloc failed");
    }
}

/// Device storage for a buffer (array) of a particular type.
pub struct TypedBuffer<T: DeviceCopy, A: DeviceAllocRef = DefaultDeviceAlloc> {
    ptr: DevicePtr,
    len: usize,
    align: usize,
    alloc: A,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DeviceCopy> TypedBuffer<T, DefaultDeviceAlloc> {
    pub fn from_slice(slice: &[T]) -> Result<Self> {
        TypedBuffer::from_slice_in(slice, DefaultDeviceAlloc)
    }

    pub fn uninitialized(len: usize) -> Result<Self> {
        TypedBuffer::uninitialized_in(len, DefaultDeviceAlloc)
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> TypedBuffer<T, A> {
    pub fn from_slice_with_align_in(
        slice: &[T],
        align: usize,
        alloc: A,
    ) -> Result<TypedBuffer<T, A>> {
        let byte_size = slice.len() * std::mem::size_of::<T>();
        let ptr = alloc
            .alloc(Layout::from_size_align(byte_size, align).unwrap())
            .map_err(|source| Error::Allocation { source })?;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(TypedBuffer {
            ptr,
            len: slice.len(),
            align,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn from_slice_in(slice: &[T], alloc: A) -> Result<TypedBuffer<T, A>> {
        let byte_size = slice.len() * std::mem::size_of::<T>();
        let align = T::device_align();
        let ptr = alloc
            .alloc(Layout::from_size_align(byte_size, align).unwrap())
            .map_err(|source| Error::Allocation { source })?;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(TypedBuffer {
            ptr,
            len: slice.len(),
            align,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn uninitialized_in(len: usize, alloc: A) -> Result<TypedBuffer<T, A>> {
        let align = T::device_align();
        let size = len * std::mem::size_of::<T>();
        let ptr = alloc
            .alloc(Layout::from_size_align(size, align).unwrap())
            .map_err(|source| Error::Allocation { source })?;
        Ok(TypedBuffer {
            ptr,
            len,
            align,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn uninitialized_with_align_in(
        len: usize,
        align: usize,
        alloc: A,
    ) -> Result<TypedBuffer<T, A>> {
        let byte_size = len * std::mem::size_of::<T>();
        let ptr = alloc
            .alloc(Layout::from_size_align(byte_size, align).unwrap())
            .map_err(|source| Error::Allocation { source })?;
        Ok(TypedBuffer {
            ptr,
            len,
            align,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn resize(&mut self, len: usize) -> Result<()> {
        if len != self.len {
            self.alloc
                .dealloc(self.ptr)
                .map_err(|source| Error::Deallocation { source })?;
            self.ptr = self
                .alloc
                .alloc(Layout::from_size_align(
                    len * std::mem::size_of::<T>(),
                    self.align,
                ).unwrap())
                .map_err(|source| Error::Allocation { source })?;
            self.len = len;
        }
        Ok(())
    }

    pub fn download(&self, slice: &mut [T]) -> Result<()> {
        if slice.len() != self.len {
            panic!(
                "Tried to copy {} elements from TypedBuffer of length {}",
                slice.len(),
                self.len,
            );
        }

        let byte_size = slice.len() * std::mem::size_of::<T>();
        unsafe {
            sys::cuMemcpyDtoH_v2(
                slice.as_mut_ptr() as *mut _,
                self.ptr.ptr(),
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })
        }
    }

    pub fn upload(&mut self, slice: &[T]) -> Result<()> {
        if slice.len() != self.len {
            panic!(
                "Tried to copy {} elements to TypedBuffer of length {}",
                slice.len(),
                self.len,
            );
        }

        let byte_size = slice.len() * std::mem::size_of::<T>();
        unsafe {
            sys::cuMemcpyHtoD_v2(
                self.ptr.ptr(),
                slice.as_ptr() as *mut _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> Drop for TypedBuffer<T, A> {
    fn drop(&mut self) {
        self.alloc
            .dealloc(self.ptr)
            .expect("TypedBuffer dealloc failed");
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> DeviceStorage for TypedBuffer<T, A> {
    fn device_ptr(&self) -> DevicePtr {
        self.ptr
    }

    fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

/// `Deref`able wrapper around a POD variable whose value can be transferred to
/// and from the device.
pub struct DeviceVariable<T: DeviceCopy, A: DeviceAllocRef = DefaultDeviceAlloc> {
    ptr: cu::DevicePtr,
    alloc: A,
    inner: T,
}

impl<T:DeviceCopy, A: DeviceAllocRef> DeviceVariable<T, A> {
    pub fn new_in(inner: T, alloc: A) -> Result<DeviceVariable<T, A>> {
        let byte_size = std::mem::size_of::<T>();
        let ptr = alloc
            .alloc(Layout::from_size_align(byte_size, T::device_align()).unwrap())
            .map_err(|source| Error::Allocation { source })?;

        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.ptr(),
                &inner as *const _ as  *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(
            DeviceVariable::<T, A> {
                ptr, alloc, inner
            }
        )
    }

    pub fn download(&mut self) -> Result<()> {
        unsafe {
            sys::cuMemcpyDtoH_v2(
                &mut self.inner as *mut _ as *mut _,
                self.ptr.ptr(),
                std::mem::size_of::<T>() as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })
        }
    }

    pub fn upload(&mut self) -> Result<()> {
        unsafe {
            sys::cuMemcpyHtoD_v2(
                self.ptr.ptr(),
                &self.inner as *const _ as  *const _,
                std::mem::size_of::<T>() as u64
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })
        }
    }
}

impl<T:DeviceCopy> DeviceVariable<T, DefaultDeviceAlloc> {
    pub fn new(inner: T) -> Result<Self> {
        DeviceVariable::new_in(inner, DefaultDeviceAlloc)
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> DeviceStorage for DeviceVariable<T, A> {
    fn device_ptr(&self) -> DevicePtr {
        self.ptr
    }

    fn byte_size(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> std::ops::Deref for DeviceVariable<T, A> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> std::ops::DerefMut for DeviceVariable<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
