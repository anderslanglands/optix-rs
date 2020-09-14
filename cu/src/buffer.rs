use crate::{
    allocator::Layout, sys, DefaultDeviceAlloc, DeviceAllocRef, 
    DevicePtr, Error,
};
type Result<T, E = Error> = std::result::Result<T, E>;

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
        let ptr = alloc.alloc(Layout::array::<T>(slice.len()).unwrap())?;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.device_ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()?;
        }
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
                self.ptr.device_ptr(),
                byte_size as u64,
            )
            .to_result()
        }
    }

    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.ptr.device_ptr()
    }
}

impl<A: DeviceAllocRef> Drop for Buffer<A> {
    fn drop(&mut self) {
        self.alloc.dealloc(self.ptr).expect("dealloc failed");
    }
}

pub struct TypedBuffer<T, A: DeviceAllocRef = DefaultDeviceAlloc>
{
    ptr: DevicePtr,
    len: usize,
    alloc: A,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TypedBuffer<T, DefaultDeviceAlloc>
{
    pub fn from_slice(slice: &[T]) -> Result<Self> {
        TypedBuffer::from_slice_in(slice, DefaultDeviceAlloc)
    }

    pub fn uninitialized(len: usize) -> Result<Self> {
        TypedBuffer::uninitialized_in(len, DefaultDeviceAlloc)
    }
}

impl<T, A: DeviceAllocRef> TypedBuffer<T, A>
{
    pub fn from_slice_in(slice: &[T], alloc: A) -> Result<TypedBuffer<T, A>> {
        let byte_size = slice.len() * std::mem::size_of::<T>();
        let ptr = alloc.alloc(Layout::array::<T>(slice.len()).unwrap())?;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                ptr.device_ptr(),
                slice.as_ptr() as *const _,
                byte_size as u64,
            )
            .to_result()?;
        }
        Ok(TypedBuffer {
            ptr,
            len: slice.len(),
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn uninitialized_in(len: usize, alloc: A) -> Result<TypedBuffer<T, A>> {
        let byte_size = len * std::mem::size_of::<T>();
        let ptr = alloc.alloc(Layout::array::<T>(len).unwrap())?;
        Ok(TypedBuffer {
            ptr,
            len,
            alloc,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn resize(&mut self, len: usize) -> Result<()> {
        if len != self.len {
            self.alloc.dealloc(self.ptr)?;
            let byte_size = len * std::mem::size_of::<T>();
            self.ptr = self.alloc.alloc(Layout::array::<T>(len).unwrap())?;
        }
        Ok(())
    }

    pub fn copy_to_slice(&self, slice: &mut [T]) -> Result<()> {
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
                self.ptr.device_ptr(),
                byte_size as u64,
            )
            .to_result()
        }
    }

    pub fn copy_from_slice(&mut self, slice: &[T]) -> Result<()> {
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
                self.ptr.device_ptr(),
                slice.as_ptr() as *mut _,
                byte_size as u64,
            )
            .to_result()
        }
    }

    pub fn device_ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T, A: DeviceAllocRef> Drop for TypedBuffer<T, A>
{
    fn drop(&mut self) {
        self.alloc
            .dealloc(self.ptr)
            .expect("TypedBuffer dealloc failed");
    }
}
