use crate::context::*;
use crate::format_get_size;
use std::ops::{Index, IndexMut};

use std::cell::RefCell;
use std::rc::Rc;

#[cfg(feature = "colorspace")]
use colorspace::rgb::RGBf32;

pub enum BufferHandle {
    Buffer1d(Buffer1dHandle),
    Buffer2d(Buffer2dHandle),
}

impl Clone for BufferHandle {
    fn clone(&self) -> BufferHandle {
        match &self {
            BufferHandle::Buffer1d(buf) => {
                BufferHandle::Buffer1d(Rc::clone(buf))
            }
            BufferHandle::Buffer2d(buf) => {
                BufferHandle::Buffer2d(Rc::clone(buf))
            }
        }
    }
}

pub struct BufferID {
    pub(crate) buf: BufferHandle,
    pub id: i32,
}

impl Clone for BufferID {
    fn clone(&self) -> BufferID {
        BufferID {
            buf: self.buf.clone(),
            id: self.id,
        }
    }
}

pub struct BufferIDBuffer {
    pub buf: Buffer1dHandle,
    pub(crate) buffers: Vec<BufferID>,
}

pub struct BufferIDBufferID {
    pub buffer: BufferIDBuffer,
    pub id: BufferID,
}

impl BufferIDBufferID {
    pub fn new(ctx: &mut Context, buffer: BufferIDBuffer) -> BufferIDBufferID {
        let id = ctx.buffer_get_id_1d(&buffer.buf).unwrap();
        BufferIDBufferID { buffer, id }
    }
}

#[repr(C)]
pub struct Buffer1d {
    pub(crate) rt_buf: RTbuffer,
}

#[repr(C)]
pub struct Buffer2d {
    pub(crate) rt_buf: RTbuffer,
}

#[repr(C)]
pub struct Buffer3d {
    pub(crate) rt_buf: RTbuffer,
}

pub type Buffer1dHandle = Rc<RefCell<Buffer1d>>;
pub type Buffer2dHandle = Rc<RefCell<Buffer2d>>;
pub type Buffer3dHandle = Rc<RefCell<Buffer3d>>;

pub trait BufferElement: Clone {
    const FORMAT: Format;
}

impl BufferElement for f32 {
    const FORMAT: Format = Format::FLOAT;
}

#[cfg(feature = "colorspace")]
impl BufferElement for RGBf32 {
    const FORMAT: Format = Format::FLOAT3;
}

impl BufferElement for V2f32 {
    const FORMAT: Format = Format::FLOAT2;
}

impl BufferElement for V3f32 {
    const FORMAT: Format = Format::FLOAT3;
}

impl BufferElement for V4f32 {
    const FORMAT: Format = Format::FLOAT4;
}

impl BufferElement for V2i32 {
    const FORMAT: Format = Format::INT2;
}

impl BufferElement for V3i32 {
    const FORMAT: Format = Format::INT3;
}

impl BufferElement for V4i32 {
    const FORMAT: Format = Format::INT4;
}

impl BufferElement for V2u32 {
    const FORMAT: Format = Format::UNSIGNED_INT2;
}

impl BufferElement for V3u32 {
    const FORMAT: Format = Format::UNSIGNED_INT3;
}

impl BufferElement for V4u32 {
    const FORMAT: Format = Format::UNSIGNED_INT4;
}

impl BufferElement for V4u8 {
    const FORMAT: Format = Format::UNSIGNED_BYTE4;
}

impl BufferElement for i16 {
    const FORMAT: Format = Format::SHORT;
}

impl BufferElement for u16 {
    const FORMAT: Format = Format::UNSIGNED_SHORT;
}

impl BufferElement for i32 {
    const FORMAT: Format = Format::INT;
}

impl BufferElement for u32 {
    const FORMAT: Format = Format::UNSIGNED_INT;
}

#[repr(C)]
#[derive(Clone)]
pub struct BufferIdProxy(i32);

impl BufferElement for BufferIdProxy {
    const FORMAT: Format = Format::BUFFER_ID;
}

/// A wrapper for a read-only mapping of a `Buffer1d` to host memory. The buffer
/// is automatically unmapped when this struct is dropped.
pub struct ScopedBufMap1d<'a, T: 'a + BufferElement> {
    data: &'a [T],
    buf: &'a Buffer1dHandle,
}

impl<'a, T: BufferElement> ScopedBufMap1d<'a, T> {
    /// Returns the size of this buffer. This is synonymous with `len()`
    pub fn width(&self) -> usize {
        self.data.len()
    }

    /// Returns the size of this buffer. This is synonymous with `width()`
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the underlying data slice
    pub fn as_slice(&self) -> &[T] {
        self.data
    }
}

impl<'a, T: BufferElement> Index<usize> for ScopedBufMap1d<'a, T> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        &self.data[idx]
    }
}

impl<'a, T: BufferElement> Drop for ScopedBufMap1d<'a, T> {
    fn drop(&mut self) {
        unsafe {
            if rtBufferUnmap(self.buf.borrow().rt_buf) != RtResult::SUCCESS {
                panic!("rtBufferUnmap failed!");
            }
        }
    }
}

/// A wrapper for a read-write mapping of a `Buffer1d` to host memory. The
/// buffer is automatically unmapped when this struct is dropped.
pub struct ScopedBufMap1dMut<'a, T: 'a + BufferElement> {
    data: &'a mut [T],
    buf: &'a Buffer1dHandle,
}

impl<'a, T: BufferElement> ScopedBufMap1dMut<'a, T> {
    /// Returns the size of this buffer. This is synonymous with `len()`
    pub fn width(&self) -> usize {
        self.data.len()
    }

    /// Returns the size of this buffer. This is synonymous with `width()`
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the underlying data slice
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Get a mutable reference to the underlying data slice
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.data
    }
}

impl<'a, T: BufferElement> Index<usize> for ScopedBufMap1dMut<'a, T> {
    type Output = T;

    fn index(&self, idx: usize) -> &T {
        &self.data[idx]
    }
}

impl<'a, T: BufferElement> IndexMut<usize> for ScopedBufMap1dMut<'a, T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.data[idx]
    }
}

impl<'a, T: BufferElement> Drop for ScopedBufMap1dMut<'a, T> {
    fn drop(&mut self) {
        unsafe {
            if rtBufferUnmap(self.buf.borrow().rt_buf) != RtResult::SUCCESS {
                panic!("rtBufferUnmap failed!");
            }
        }
    }
}

/// A wrapper for a read-only mapping of a `Buffer1d` to host memory. The buffer
/// is automatically unmapped when this struct is dropped.
pub struct ScopedBufMap2d<'a, T: 'a + BufferElement> {
    data: &'a [T],
    buf: &'a Buffer2dHandle,
    width: usize,
    height: usize,
}

impl<'a, T: BufferElement> ScopedBufMap2d<'a, T> {
    /// Gets the width of this buffer
    pub fn width(&self) -> usize {
        self.width
    }

    /// Gets the height of this buffer
    pub fn height(&self) -> usize {
        self.height
    }

    /// Gets the size of this buffer (i.e. width * height)
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the underlying data slice
    pub fn as_slice(&self) -> &[T] {
        self.data
    }
}

impl<'a, T: BufferElement> Index<(usize, usize)> for ScopedBufMap2d<'a, T> {
    type Output = T;

    fn index(&self, idx: (usize, usize)) -> &T {
        &self.data[idx.1 * self.width + idx.0]
    }
}

impl<'a, T: BufferElement> Drop for ScopedBufMap2d<'a, T> {
    fn drop(&mut self) {
        unsafe {
            if rtBufferUnmap(self.buf.borrow().rt_buf) != RtResult::SUCCESS {
                panic!("rtBufferUnmap failed!");
            }
        }
    }
}

/// A wrapper for a read-write mapping of a `Buffer1d` to host memory. The
/// buffer is automatically unmapped when this struct is dropped.
pub struct ScopedBufMap2dMut<'a, T: 'a + BufferElement> {
    data: &'a mut [T],
    buf: &'a Buffer2dHandle,
    width: usize,
    height: usize,
}

impl<'a, T: BufferElement> ScopedBufMap2dMut<'a, T> {
    /// Gets the width of this buffer
    pub fn width(&self) -> usize {
        self.width
    }

    /// Gets the height of this buffer
    pub fn height(&self) -> usize {
        self.height
    }

    /// Gets the size of this buffer (i.e. width * height)
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the underlying data slice
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Get a mutable reference to the underlying data slice
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.data
    }
}

impl<'a, T: BufferElement> Index<(usize, usize)> for ScopedBufMap2dMut<'a, T> {
    type Output = T;

    fn index(&self, idx: (usize, usize)) -> &T {
        &self.data[idx.1 * self.width + idx.0]
    }
}

impl<'a, T: BufferElement> IndexMut<(usize, usize)>
    for ScopedBufMap2dMut<'a, T>
{
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut T {
        &mut self.data[idx.1 * self.width + idx.0]
    }
}

impl<'a, T: BufferElement> Drop for ScopedBufMap2dMut<'a, T> {
    fn drop(&mut self) {
        unsafe {
            if rtBufferUnmap(self.buf.borrow().rt_buf) != RtResult::SUCCESS {
                panic!("rtBufferUnmap failed!");
            }
        }
    }
}

impl Context {
    /// Creates a new `Buffer1d` on this Context, returning a `Buffer1dHandle`
    /// that can be used to access it later.
    pub fn buffer_create_1d(
        &mut self,
        width: usize,
        format: Format,
        buffer_type: BufferType,
        flags: BufferFlag,
    ) -> Result<Buffer1dHandle> {
        let (rt_buf, result) = unsafe {
            let mut rt_buf: RTbuffer = std::mem::uninitialized();
            let result = rtBufferCreate(
                self.rt_ctx,
                buffer_type as u32 | flags as u32,
                &mut rt_buf,
            );
            (rt_buf, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferCreate", result));
        }

        let result = unsafe { rtBufferSetFormat(rt_buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let result = unsafe { rtBufferSetSize1D(rt_buf, width as u64) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize1D", result));
        }

        let buf = Rc::new(RefCell::new(Buffer1d { rt_buf }));
        self.buffer1ds.push(Rc::clone(&buf));

        self.buffer_mem
            .insert(rt_buf, format_get_size(format) * width);

        Ok(buf)
    }

    pub fn buffer_create_from_glbo_1d(
        &mut self,
        width: usize,
        format: Format,
        buffer_type: BufferType,
        gl_id: u32,
    ) -> Result<Buffer1dHandle> {
        let (rt_buf, result) = unsafe {
            let mut rt_buf: RTbuffer = std::mem::uninitialized();
            let result = rtBufferCreateFromGLBO(
                self.rt_ctx,
                buffer_type as u32,
                gl_id,
                &mut rt_buf,
            );
            (rt_buf, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferCreateFromGLBO", result));
        }

        let result = unsafe { rtBufferSetFormat(rt_buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let result = unsafe { rtBufferSetSize1D(rt_buf, width as u64) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize1D", result));
        }

        let buf = Rc::new(RefCell::new(Buffer1d { rt_buf }));
        self.buffer1ds.push(Rc::clone(&buf));

        self.buffer_mem
            .insert(rt_buf, format_get_size(format) * width);

        Ok(buf)
    }

    pub fn buffer_create_1d_named<S: Into<String>>(
        &mut self,
        width: usize,
        format: Format,
        buffer_type: BufferType,
        flags: BufferFlag,
        name: S,
    ) -> Result<Buffer1dHandle> {
        self.buffer_create_1d(width, format, buffer_type, flags)
            .map(|h| {
                self.buffer_names.insert(h.borrow().rt_buf, name.into());
                h
            })
    }

    /// Get the format of this buffer
    pub fn buffer_get_format_1d(&self, buf: &Buffer1dHandle) -> Result<Format> {
        let (format, result) = unsafe {
            let mut format = Format::UNKNOWN;
            let result = rtBufferGetFormat(
                buf.borrow().rt_buf,
                &mut format as *mut Format,
            );
            (format, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferGetFormat", result));
        }
        Ok(format)
    }

    /// Set the format of this buffer
    pub fn buffer_set_format_1d(
        &mut self,
        buf: &Buffer1dHandle,
        format: Format,
    ) -> Result<()> {
        let result = unsafe { rtBufferSetFormat(buf.borrow().rt_buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let width = self.buffer_get_size_1d(buf)?;
        self.buffer_mem
            .insert(buf.borrow().rt_buf, width * format_get_size(format));

        Ok(())
    }

    /// Get the size of this buffer.
    pub fn buffer_get_size_1d(&self, buf: &Buffer1dHandle) -> Result<usize> {
        let (size, result) = unsafe {
            let mut width = 0u64;
            let result =
                rtBufferGetSize1D(buf.borrow().rt_buf, &mut width as *mut u64);
            (width, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferGetSize1D", result));
        }

        Ok(size as usize)
    }

    /// Set the size of this buffer.
    pub fn buffer_set_size_1d(
        &mut self,
        buf: &Buffer1dHandle,
        width: usize,
    ) -> Result<()> {
        let result =
            unsafe { rtBufferSetSize1D(buf.borrow().rt_buf, width as u64) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize1D", result));
        }

        let f = self.buffer_get_format_1d(buf)?;
        self.buffer_mem
            .insert(buf.borrow().rt_buf, width * format_get_size(f));

        Ok(())
    }

    /// Create a new `Buffer1d` and fill it with the given data.
    pub fn buffer_create_from_slice_1d<T: BufferElement>(
        &mut self,
        data: &[T],
        buffer_type: BufferType,
        flags: BufferFlag,
    ) -> Result<Buffer1dHandle> {
        let hnd =
            self.buffer_create_1d(data.len(), T::FORMAT, buffer_type, flags)?;
        {
            let mut map = self.buffer_map_1d_mut::<T>(&hnd).unwrap();
            map.as_slice_mut().clone_from_slice(data);
        }
        Ok(hnd)
    }

    /// Create a new buffer of buffer ids and fill it with the given data.
    pub fn buffer_create_id_buffer(
        &mut self,
        data: Vec<BufferID>,
        buffer_type: BufferType,
        flags: BufferFlag,
    ) -> Result<BufferIDBuffer> {
        let hnd = self.buffer_create_1d(
            data.len(),
            Format::BUFFER_ID,
            buffer_type,
            flags,
        )?;
        {
            let mut map =
                self.buffer_map_1d_mut::<BufferIdProxy>(&hnd).unwrap();
            for (m, b) in map.as_slice_mut().iter_mut().zip(data.iter()) {
                m.0 = b.id;
            }
        }
        Ok(BufferIDBuffer {
            buf: hnd,
            buffers: data,
        })
    }

    /// Create a new named `Buffer1d` and fill it with the given data.
    pub fn buffer_create_from_slice_1d_named<
        T: BufferElement,
        S: Into<String>,
    >(
        &mut self,
        data: &[T],
        buffer_type: BufferType,
        flags: BufferFlag,
        name: S,
    ) -> Result<Buffer1dHandle> {
        let hnd = self.buffer_create_1d_named(
            data.len(),
            T::FORMAT,
            buffer_type,
            flags,
            name,
        )?;
        {
            let mut map = self.buffer_map_1d_mut::<T>(&hnd).unwrap();
            map.as_slice_mut().clone_from_slice(data);
        }
        Ok(hnd)
    }

    /// Creates a new `Buffer2d` on this Context, returning a `Buffer2dHandle`
    /// that can be used to access it later.
    pub fn buffer_create_2d(
        &mut self,
        width: usize,
        height: usize,
        format: Format,
        buffer_type: BufferType,
        flags: BufferFlag,
    ) -> Result<Buffer2dHandle> {
        let (rt_buf, result) = unsafe {
            let mut rt_buf: RTbuffer = std::mem::uninitialized();
            let result = rtBufferCreate(
                self.rt_ctx,
                buffer_type as u32 | flags as u32,
                &mut rt_buf,
            );
            (rt_buf, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferCreate", result));
        }

        let result = unsafe { rtBufferSetFormat(rt_buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let result =
            unsafe { rtBufferSetSize2D(rt_buf, width as u64, height as u64) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize2D", result));
        }

        let buf = Rc::new(RefCell::new(Buffer2d { rt_buf }));
        self.buffer2ds.push(Rc::clone(&buf));

        self.buffer_mem
            .insert(rt_buf, format_get_size(format) * width * height);

        Ok(buf)
    }

    pub fn buffer_create_from_glbo_2d(
        &mut self,
        width: usize,
        height: usize,
        format: Format,
        buffer_type: BufferType,
        gl_id: u32,
    ) -> Result<Buffer2dHandle> {
        let (rt_buf, result) = unsafe {
            let mut rt_buf: RTbuffer = std::mem::uninitialized();
            let result = rtBufferCreateFromGLBO(
                self.rt_ctx,
                buffer_type as u32,
                gl_id,
                &mut rt_buf,
            );
            (rt_buf, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferCreate", result));
        }

        let result = unsafe { rtBufferSetFormat(rt_buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let result =
            unsafe { rtBufferSetSize2D(rt_buf, width as u64, height as u64) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize2D", result));
        }

        let buf = Rc::new(RefCell::new(Buffer2d { rt_buf }));
        self.buffer2ds.push(Rc::clone(&buf));

        self.buffer_mem
            .insert(rt_buf, format_get_size(format) * width * height);

        Ok(buf)
    }

    pub fn buffer_create_2d_named<S: Into<String>>(
        &mut self,
        width: usize,
        height: usize,
        format: Format,
        buffer_type: BufferType,
        flags: BufferFlag,
        name: S,
    ) -> Result<Buffer2dHandle> {
        self.buffer_create_2d(width, height, format, buffer_type, flags)
            .map(|h| {
                self.buffer_names.insert(h.borrow().rt_buf, name.into());
                h
            })
    }

    /// Create a new `Buffer2d` and fill it with the given data.
    pub fn buffer_create_from_slice_2d<T: BufferElement>(
        &mut self,
        data: &[T],
        width: usize,
        height: usize,
        buffer_type: BufferType,
        flags: BufferFlag,
    ) -> Result<Buffer2dHandle> {
        let hnd = self.buffer_create_2d(
            width,
            height,
            T::FORMAT,
            buffer_type,
            flags,
        )?;
        {
            let mut map = self.buffer_map_2d_mut::<T>(&hnd).unwrap();
            map.as_slice_mut().clone_from_slice(data);
        }
        Ok(hnd)
    }

    /// Creates an unsized `Buffer2d` on this Context, returning a `Buffer2dHandle`
    /// that can be used to access it later. The size of the buffer must be set
    /// before it is used
    pub fn buffer_create_unsized_2d(
        &mut self,
        format: Format,
        buffer_type: BufferType,
        flags: BufferFlag,
    ) -> Result<Buffer2dHandle> {
        let (rt_buf, result) = unsafe {
            let mut rt_buf: RTbuffer = std::mem::uninitialized();
            let result = rtBufferCreate(
                self.rt_ctx,
                buffer_type as u32 | flags as u32,
                &mut rt_buf,
            );
            (rt_buf, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferCreate", result));
        }

        let result = unsafe { rtBufferSetFormat(rt_buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let buf = Rc::new(RefCell::new(Buffer2d { rt_buf }));
        self.buffer2ds.push(Rc::clone(&buf));

        self.buffer_mem.insert(rt_buf, 0);

        Ok(buf)
    }

    /// Get the size of this buffer.
    pub fn buffer_get_size_2d(
        &self,
        buf: &Buffer2dHandle,
    ) -> Result<(usize, usize)> {
        let (size, result) = unsafe {
            let mut width = 0u64;
            let mut height = 0u64;
            let result = rtBufferGetSize2D(
                buf.borrow().rt_buf,
                &mut width as *mut u64,
                &mut height as *mut u64,
            );
            ((width, height), result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferGetSize2D", result));
        }
        Ok((size.0 as usize, size.1 as usize))
    }

    /// Set the size of this buffer.
    pub fn buffer_set_size_2d(
        &mut self,
        buf: &Buffer2dHandle,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let result = unsafe {
            rtBufferSetSize2D(buf.borrow().rt_buf, width as u64, height as u64)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize2D", result));
        }

        let format = self.buffer_get_format_2d(&buf)?;
        self.buffer_mem.insert(
            buf.borrow().rt_buf,
            format_get_size(format) * width * height,
        );

        Ok(())
    }

    /// Get the format of this buffer
    pub fn buffer_get_format_2d(&self, buf: &Buffer2dHandle) -> Result<Format> {
        let (format, result) = unsafe {
            let mut format = Format::UNKNOWN;
            let result = rtBufferGetFormat(
                buf.borrow().rt_buf,
                &mut format as *mut Format,
            );
            (format, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferGetFormat", result));
        }
        Ok(format)
    }

    /// Set the format of this buffer
    pub fn buffer_set_format_2d(
        &mut self,
        buf: &Buffer2dHandle,
        format: Format,
    ) -> Result<()> {
        let result = unsafe { rtBufferSetFormat(buf.borrow().rt_buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let (width, height) = self.buffer_get_size_2d(&buf)?;
        self.buffer_mem.insert(
            buf.borrow().rt_buf,
            format_get_size(format) * width * height,
        );

        Ok(())
    }

    /// Map the buffer so that i can be read by host memory. The returned struct
    /// will automatically unmap the buffer when it drops.
    /// # Panics
    /// If buf is not a valid Buffer1dHandle
    pub fn buffer_map_1d<'a, T: BufferElement>(
        &self,
        buffer_handle: &'a Buffer1dHandle,
    ) -> Result<ScopedBufMap1d<'a, T>> {
        // first check that the formats align
        let format = self.buffer_get_format_1d(buffer_handle)?;
        if format != T::FORMAT {
            return Err(Error::IncompatibleBufferFormat {
                given: T::FORMAT,
                expected: format,
            });
        }

        let mut p: *const T = std::ptr::null();
        let result = unsafe {
            rtBufferMap(
                buffer_handle.borrow().rt_buf,
                (&mut p) as *mut _ as *mut *mut ::std::os::raw::c_void,
            )
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtBufferMap", result))
        } else {
            let (width, result) = unsafe {
                let mut width: RTsize = 0;
                let result = rtBufferGetSize1D(
                    buffer_handle.borrow().rt_buf,
                    &mut width,
                );
                (width, result)
            };

            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtBufferGetSize1D", result));
            }

            Ok(ScopedBufMap1d {
                data: unsafe { std::slice::from_raw_parts(p, width as usize) },
                buf: &buffer_handle,
            })
        }
    }

    /// Map the buffer so that i can be read and written by host memory.
    /// The returned struct will automatically unmap the buffer when it drops.
    /// # Panics
    /// If buf is not a valid Buffer1dHandle
    pub fn buffer_map_1d_mut<'a, T: BufferElement>(
        &mut self,
        buffer_handle: &'a Buffer1dHandle,
    ) -> Result<ScopedBufMap1dMut<'a, T>> {
        // first check that the formats align
        let format = self.buffer_get_format_1d(buffer_handle)?;
        if format != T::FORMAT {
            return Err(Error::IncompatibleBufferFormat {
                given: T::FORMAT,
                expected: format,
            });
        }

        let mut p: *mut T = std::ptr::null_mut();
        let result = unsafe {
            rtBufferMap(
                buffer_handle.borrow().rt_buf,
                (&mut p) as *mut _ as *mut *mut ::std::os::raw::c_void,
            )
        };
        if result != RtResult::SUCCESS {
            Err(Error::Optix((result, "rtBufferMap".to_owned())))
        } else {
            let (width, result) = unsafe {
                let mut width: RTsize = 0;
                let result = rtBufferGetSize1D(
                    buffer_handle.borrow().rt_buf,
                    &mut width,
                );
                (width, result)
            };

            if result != RtResult::SUCCESS {
                return Err(Error::Optix((
                    result,
                    "rtBufferGetSize1D".to_owned(),
                )));
            }

            Ok(ScopedBufMap1dMut {
                data: unsafe {
                    std::slice::from_raw_parts_mut(p, width as usize)
                },
                buf: &buffer_handle,
            })
        }
    }

    /// Map the buffer so that it can be read by host memory. The returned struct
    /// will automatically unmap the buffer when it drops.
    /// # Panics
    /// If buf is not a valid Buffer2dHandle
    pub fn buffer_map_2d<'a, T: BufferElement>(
        &self,
        buffer_handle: &'a Buffer2dHandle,
    ) -> Result<ScopedBufMap2d<'a, T>> {
        // first check that the formats align
        let format = self.buffer_get_format_2d(buffer_handle)?;
        let num_channels = if format != T::FORMAT {
            if T::FORMAT == Format::FLOAT {
                match format {
                    Format::FLOAT2 => Ok(2),
                    Format::FLOAT3 => Ok(3),
                    Format::FLOAT4 => Ok(4),
                    _ => Err(Error::IncompatibleBufferFormat {
                        given: T::FORMAT,
                        expected: format,
                    }),
                }
            } else {
                Err(Error::IncompatibleBufferFormat {
                    given: T::FORMAT,
                    expected: format,
                })
            }
        } else {
            Ok(1)
        }?;

        let mut p: *const T = std::ptr::null();
        let result = unsafe {
            rtBufferMap(
                buffer_handle.borrow().rt_buf,
                (&mut p) as *mut _ as *mut *mut ::std::os::raw::c_void,
            )
        };
        if result != RtResult::SUCCESS {
            Err(Error::Optix((result, "rtBufferMap".to_owned())))
        } else {
            let (width, height, result) = unsafe {
                let mut width: RTsize = 0;
                let mut height: RTsize = 0;
                let result = rtBufferGetSize2D(
                    buffer_handle.borrow().rt_buf,
                    &mut width,
                    &mut height,
                );
                (width, height, result)
            };

            if result != RtResult::SUCCESS {
                return Err(Error::Optix((
                    result,
                    "rtBufferGetSize2D".to_owned(),
                )));
            }

            Ok(ScopedBufMap2d {
                data: unsafe {
                    std::slice::from_raw_parts(
                        p,
                        (width * height) as usize * num_channels,
                    )
                },
                buf: &buffer_handle,
                width: width as usize,
                height: height as usize,
            })
        }
    }

    /// Map the buffer so that i can be read and written by host memory.
    /// The returned struct will automatically unmap the buffer when it drops.
    /// # Panics
    /// If buf is not a valid Buffer2dHandle
    pub fn buffer_map_2d_mut<'a, T: BufferElement>(
        &mut self,
        buffer_handle: &'a Buffer2dHandle,
    ) -> Result<ScopedBufMap2dMut<'a, T>> {
        // first check that the formats align
        // we support 'downcasting' a float tuple type to a float array
        let format = self.buffer_get_format_2d(buffer_handle)?;
        let num_channels = if format != T::FORMAT {
            if T::FORMAT == Format::FLOAT {
                match format {
                    Format::FLOAT2 => Ok(2),
                    Format::FLOAT3 => Ok(3),
                    Format::FLOAT4 => Ok(4),
                    _ => Err(Error::IncompatibleBufferFormat {
                        given: T::FORMAT,
                        expected: format,
                    }),
                }
            } else {
                Err(Error::IncompatibleBufferFormat {
                    given: T::FORMAT,
                    expected: format,
                })
            }
        } else {
            Ok(1)
        }?;

        let mut p: *mut T = std::ptr::null_mut();
        let result = unsafe {
            rtBufferMap(
                buffer_handle.borrow().rt_buf,
                (&mut p) as *mut _ as *mut *mut ::std::os::raw::c_void,
            )
        };
        if result != RtResult::SUCCESS {
            Err(Error::Optix((result, "rtBufferMap".to_owned())))
        } else {
            let (width, height, result) = unsafe {
                let mut width: RTsize = 0;
                let mut height: RTsize = 0;
                let result = rtBufferGetSize2D(
                    buffer_handle.borrow().rt_buf,
                    &mut width,
                    &mut height,
                );
                (width, height, result)
            };

            if result != RtResult::SUCCESS {
                return Err(Error::Optix((
                    result,
                    "rtBufferGetSize2D".to_owned(),
                )));
            }

            Ok(ScopedBufMap2dMut {
                data: unsafe {
                    std::slice::from_raw_parts_mut(
                        p,
                        (width * height) as usize * num_channels,
                    )
                },
                buf: &buffer_handle,
                width: width as usize,
                height: height as usize,
            })
        }
    }

    pub fn buffer_get_id_1d(
        &mut self,
        buffer_handle: &Buffer1dHandle,
    ) -> Result<BufferID> {
        let mut id = 0i32;
        let result =
            unsafe { rtBufferGetId(buffer_handle.borrow().rt_buf, &mut id) };
        if result != RtResult::SUCCESS {
            return Err(Error::Optix((result, "rtBufferGetId".to_owned())));
        } else {
            Ok(BufferID {
                buf: BufferHandle::Buffer1d(Rc::clone(buffer_handle)),
                id,
            })
        }
    }

    pub fn buffer_get_id_2d(
        &mut self,
        buffer_handle: &Buffer2dHandle,
    ) -> Result<BufferID> {
        let mut id = 0i32;
        let result =
            unsafe { rtBufferGetId(buffer_handle.borrow().rt_buf, &mut id) };
        if result != RtResult::SUCCESS {
            return Err(Error::Optix((result, "rtBufferGetId".to_owned())));
        } else {
            Ok(BufferID {
                buf: BufferHandle::Buffer2d(Rc::clone(buffer_handle)),
                id,
            })
        }
    }

    pub fn buffer_validate_1d(&self, buf: &Buffer1dHandle) -> Result<()> {
        let result = unsafe { rtBufferValidate(buf.borrow().rt_buf) };
        if result == RtResult::SUCCESS {
            Ok(())
        } else {
            Err(self.optix_error("rtBufferValidate 1d", result))
        }
    }

    pub fn buffer_validate_2d(&self, buf: &Buffer2dHandle) -> Result<()> {
        let result = unsafe { rtBufferValidate(buf.borrow().rt_buf) };
        if result == RtResult::SUCCESS {
            Ok(())
        } else {
            Err(self.optix_error("rtBufferValidate 2d", result))
        }
    }
}
