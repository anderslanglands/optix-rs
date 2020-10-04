use crate::{DeviceCopy, Error};
use cu::{sys, Allocation, DefaultDeviceAlloc, DeviceAllocRef, DevicePtr};
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

/// An untyped, opaque chunk of device memory that can be used to copy data to
/// and from the device.
pub struct Buffer<A: DeviceAllocRef = DefaultDeviceAlloc> {
    hnd: Allocation,
    byte_size: usize,
    alloc: A,
}

impl Buffer<DefaultDeviceAlloc> {
    /// Get a block of memory on the device from the default allocator
    /// sufficient and correctly aligned to hold `slice`, then copy the
    /// contents of `slice` to it.
    pub fn from_slice<T>(slice: &[T]) -> Result<Buffer> {
        Buffer::from_slice_in(slice, DefaultDeviceAlloc)
    }
}

impl<A: DeviceAllocRef> Buffer<A> {
    /// Get a block of memory on the device from allocator `alloc` sufficient
    /// and correctly aligned to hold `slice`, then copy the contents of
    /// `slice` to it.
    pub fn from_slice_in<T>(slice: &[T], alloc: A) -> Result<Buffer<A>> {
        let byte_size = slice.len() * std::mem::size_of::<T>();
        let ptr = alloc
            .alloc(Layout::array::<T>(slice.len()).unwrap())
            .map_err(|source| Error::Allocation { source })?;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.ptr.ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(Buffer {
            hnd: ptr,
            byte_size,
            alloc,
        })
    }

    /// Get a block of memory on the device from allocator `alloc` sufficient
    /// and correctly aligned to hold `slice`, then copy the contents of
    /// `slice` to it.
    pub fn from_slice_in_with_tag<T>(
        slice: &[T],
        alloc: A,
        tag: u16,
    ) -> Result<Buffer<A>> {
        let byte_size = slice.len() * std::mem::size_of::<T>();
        let ptr = alloc
            .alloc_with_tag(Layout::array::<T>(slice.len()).unwrap(), tag)
            .map_err(|source| Error::Allocation { source })?;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.ptr.ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(Buffer {
            hnd: ptr,
            byte_size,
            alloc,
        })
    }

    /// Get a block of uninitialized memory on the device of size `byte_size`
    /// bytes with alignment `align` from allocator `alloc`.
    pub fn uninitialized_with_align_in(
        byte_size: usize,
        align: usize,
        alloc: A,
    ) -> Result<Buffer<A>> {
        let ptr = alloc
            .alloc(Layout::from_size_align(byte_size, align).unwrap())
            .map_err(|source| Error::Allocation { source })?;
        Ok(Buffer {
            hnd: ptr,
            byte_size,
            alloc,
        })
    }

    /// Get a block of uninitialized memory on the device of size `byte_size`
    /// bytes with alignment `align` from allocator `alloc`.
    pub fn uninitialized_with_align_in_with_tag(
        byte_size: usize,
        align: usize,
        alloc: A,
        tag: u16,
    ) -> Result<Buffer<A>> {
        let ptr = alloc
            .alloc_with_tag(
                Layout::from_size_align(byte_size, align).unwrap(),
                tag,
            )
            .map_err(|source| Error::Allocation { source })?;
        Ok(Buffer {
            hnd: ptr,
            byte_size,
            alloc,
        })
    }

    /// Copy the contents of the buffer from the device into `slice`
    /// # Panics
    /// If the size of `slice` and the device buffer do not match
    pub fn download<T>(&self, slice: &mut [T]) -> Result<()> {
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
                self.hnd.ptr.ptr(),
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })
        }
    }

    pub fn resize_with_align(
        &mut self,
        new_size: usize,
        align: usize,
    ) -> Result<()> {
        if new_size == self.byte_size() {
            return Ok(());
        }

        self.alloc
            .dealloc(self.hnd)
            .map_err(|source| Error::Deallocation { source })?;
        let hnd = self
            .alloc
            .alloc(Layout::from_size_align(new_size, align).unwrap())
            .map_err(|source| Error::Allocation { source })?;

        self.hnd = hnd;
        Ok(())
    }

    pub fn resize_with_align_and_tag(
        &mut self,
        new_size: usize,
        align: usize,
        tag: u16,
    ) -> Result<()> {
        if new_size == self.byte_size() {
            return Ok(());
        }

        self.alloc
            .dealloc(self.hnd)
            .map_err(|source| Error::Deallocation { source })?;
        let hnd = self
            .alloc
            .alloc_with_tag(
                Layout::from_size_align(new_size, align).unwrap(),
                tag,
            )
            .map_err(|source| Error::Allocation { source })?;

        self.hnd = hnd;
        Ok(())
    }
}

impl<A: DeviceAllocRef> DeviceStorage for Buffer<A> {
    fn device_ptr(&self) -> DevicePtr {
        self.hnd.ptr
    }

    fn byte_size(&self) -> usize {
        self.byte_size
    }
}

impl<A: DeviceAllocRef> Drop for Buffer<A> {
    fn drop(&mut self) {
        self.alloc.dealloc(self.hnd).expect("dealloc failed");
    }
}

/// Device storage for a buffer (array) of a particular type.
pub struct TypedBuffer<T: DeviceCopy, A: DeviceAllocRef = DefaultDeviceAlloc> {
    hnd: Allocation,
    len: usize,
    align: usize,
    alloc: A,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: DeviceCopy> TypedBuffer<T, DefaultDeviceAlloc> {
    /// Get a block of memory on the device from the default allocator
    /// sufficient and correctly aligned to hold `slice`, then copy the
    /// contents of `slice` to it.
    pub fn from_slice(slice: &[T]) -> Result<Self> {
        TypedBuffer::from_slice_in(slice, DefaultDeviceAlloc)
    }

    /// Get a block of uninitialized memory on the device from the default
    /// allocator sufficient and correctly aligned to hold a slice of `len`
    /// elements of type `T`
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
                ptr.ptr.ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(TypedBuffer {
            hnd: ptr,
            len: slice.len(),
            align,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn from_slice_with_align_in_with_tag(
        slice: &[T],
        align: usize,
        alloc: A,
        tag: u16,
    ) -> Result<TypedBuffer<T, A>> {
        let byte_size = slice.len() * std::mem::size_of::<T>();
        let ptr = alloc
            .alloc_with_tag(
                Layout::from_size_align(byte_size, align).unwrap(),
                tag,
            )
            .map_err(|source| Error::Allocation { source })?;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.ptr.ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(TypedBuffer {
            hnd: ptr,
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
                ptr.ptr.ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(TypedBuffer {
            hnd: ptr,
            len: slice.len(),
            align,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn from_slice_in_with_tag(
        slice: &[T],
        alloc: A,
        tag: u16,
    ) -> Result<TypedBuffer<T, A>> {
        let byte_size = slice.len() * std::mem::size_of::<T>();
        let align = T::device_align();
        let ptr = alloc
            .alloc_with_tag(
                Layout::from_size_align(byte_size, align).unwrap(),
                tag,
            )
            .map_err(|source| Error::Allocation { source })?;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.ptr.ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(TypedBuffer {
            hnd: ptr,
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
            hnd: ptr,
            len,
            align,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn uninitialized_in_with_tag(
        len: usize,
        alloc: A,
        tag: u16,
    ) -> Result<TypedBuffer<T, A>> {
        let align = T::device_align();
        let size = len * std::mem::size_of::<T>();
        let ptr = alloc
            .alloc_with_tag(Layout::from_size_align(size, align).unwrap(), tag)
            .map_err(|source| Error::Allocation { source })?;
        Ok(TypedBuffer {
            hnd: ptr,
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
            hnd: ptr,
            len,
            align,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn uninitialized_with_align_in_with_tag(
        len: usize,
        align: usize,
        alloc: A,
        tag: u16,
    ) -> Result<TypedBuffer<T, A>> {
        let byte_size = len * std::mem::size_of::<T>();
        let ptr = alloc
            .alloc_with_tag(
                Layout::from_size_align(byte_size, align).unwrap(),
                tag,
            )
            .map_err(|source| Error::Allocation { source })?;
        Ok(TypedBuffer {
            hnd: ptr,
            len,
            align,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn resize(&mut self, len: usize) -> Result<()> {
        if len != self.len {
            self.alloc
                .dealloc(self.hnd)
                .map_err(|source| Error::Deallocation { source })?;
            self.hnd = self
                .alloc
                .alloc(
                    Layout::from_size_align(
                        len * std::mem::size_of::<T>(),
                        self.align,
                    )
                    .unwrap(),
                )
                .map_err(|source| Error::Allocation { source })?;
            self.len = len;
        }
        Ok(())
    }

    pub fn resize_with_tag(&mut self, len: usize, tag: u16) -> Result<()> {
        if len != self.len {
            self.alloc
                .dealloc(self.hnd)
                .map_err(|source| Error::Deallocation { source })?;
            self.hnd = self
                .alloc
                .alloc_with_tag(
                    Layout::from_size_align(
                        len * std::mem::size_of::<T>(),
                        self.align,
                    )
                    .unwrap(),
                    tag,
                )
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
                self.hnd.ptr.ptr(),
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
                self.hnd.ptr.ptr(),
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
            .dealloc(self.hnd)
            .expect("TypedBuffer dealloc failed");
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> DeviceStorage for TypedBuffer<T, A> {
    fn device_ptr(&self) -> DevicePtr {
        self.hnd.ptr
    }

    fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

/// `Deref`able wrapper around a POD variable whose value can be transferred to
/// and from the device.
pub struct DeviceVariable<T: DeviceCopy, A: DeviceAllocRef = DefaultDeviceAlloc>
{
    hnd: Allocation,
    alloc: A,
    inner: T,
}

impl<T: DeviceCopy, A: DeviceAllocRef> DeviceVariable<T, A> {
    pub fn new_in(inner: T, alloc: A) -> Result<DeviceVariable<T, A>> {
        let byte_size = std::mem::size_of::<T>();
        let ptr = alloc
            .alloc(
                Layout::from_size_align(byte_size, T::device_align()).unwrap(),
            )
            .map_err(|source| Error::Allocation { source })?;

        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.ptr.ptr(),
                &inner as *const _ as *const _,
                byte_size as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })?;
        }
        Ok(DeviceVariable::<T, A> {
            hnd: ptr,
            alloc,
            inner,
        })
    }

    pub fn download(&mut self) -> Result<()> {
        unsafe {
            sys::cuMemcpyDtoH_v2(
                &mut self.inner as *mut _ as *mut _,
                self.hnd.ptr.ptr(),
                std::mem::size_of::<T>() as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })
        }
    }

    pub fn upload(&mut self) -> Result<()> {
        unsafe {
            sys::cuMemcpyHtoD_v2(
                self.hnd.ptr.ptr(),
                &self.inner as *const _ as *const _,
                std::mem::size_of::<T>() as u64,
            )
            .to_result()
            .map_err(|source| Error::Memcpy { source })
        }
    }
}

impl<T: DeviceCopy> DeviceVariable<T, DefaultDeviceAlloc> {
    pub fn new(inner: T) -> Result<Self> {
        DeviceVariable::new_in(inner, DefaultDeviceAlloc)
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> DeviceStorage for DeviceVariable<T, A> {
    fn device_ptr(&self) -> DevicePtr {
        self.hnd.ptr
    }

    fn byte_size(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> std::ops::Deref
    for DeviceVariable<T, A>
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: DeviceCopy, A: DeviceAllocRef> std::ops::DerefMut
    for DeviceVariable<T, A>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
