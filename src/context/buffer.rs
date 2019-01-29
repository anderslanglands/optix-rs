use crate::context::*;
use std::ops::{Index, IndexMut};

use slotmap::*;

new_key_type! { pub struct Buffer1dHandle; }
new_key_type! { pub struct Buffer2dHandle; }
new_key_type! { pub struct Buffer3dHandle; }

pub trait BufferElement: Clone {
    const FORMAT: Format;
}

impl BufferElement for f32 {
    const FORMAT: Format = Format::FLOAT;
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

impl BufferElement for i32 {
    const FORMAT: Format = Format::INT;
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

/// A wrapper for a read-only mapping of a `Buffer1d` to host memory. The buffer
/// is automatically unmapped when this struct is dropped.
pub struct ScopedBufMap1d<'a, T: 'a + BufferElement> {
    data: &'a [T],
    buf: &'a RTbuffer,
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
            if rtBufferUnmap(*self.buf) != RtResult::SUCCESS {
                panic!("rtBufferUnmap failed!");
            }
        }
    }
}

/// A wrapper for a read-write mapping of a `Buffer1d` to host memory. The
/// buffer is automatically unmapped when this struct is dropped.
pub struct ScopedBufMap1dMut<'a, T: 'a + BufferElement> {
    data: &'a mut [T],
    buf: &'a mut RTbuffer,
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
            if rtBufferUnmap(*self.buf) != RtResult::SUCCESS {
                panic!("rtBufferUnmap failed!");
            }
        }
    }
}

/// A wrapper for a read-only mapping of a `Buffer1d` to host memory. The buffer
/// is automatically unmapped when this struct is dropped.
pub struct ScopedBufMap2d<'a, T: 'a + BufferElement> {
    data: &'a [T],
    buf: &'a RTbuffer,
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
            if rtBufferUnmap(*self.buf) != RtResult::SUCCESS {
                panic!("rtBufferUnmap failed!");
            }
        }
    }
}

/// A wrapper for a read-write mapping of a `Buffer1d` to host memory. The
/// buffer is automatically unmapped when this struct is dropped.
pub struct ScopedBufMap2dMut<'a, T: 'a + BufferElement> {
    data: &'a mut [T],
    buf: &'a mut RTbuffer,
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

impl<'a, T: BufferElement> IndexMut<(usize, usize)> for ScopedBufMap2dMut<'a, T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut T {
        &mut self.data[idx.1 * self.width + idx.0]
    }
}

impl<'a, T: BufferElement> Drop for ScopedBufMap2dMut<'a, T> {
    fn drop(&mut self) {
        unsafe {
            if rtBufferUnmap(*self.buf) != RtResult::SUCCESS {
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
        let (buf, result) = unsafe {
            let mut buf: RTbuffer = std::mem::uninitialized();
            let result = rtBufferCreate(self.rt_ctx, buffer_type as u32 | flags as u32, &mut buf);
            (buf, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferCreate", result));
        }

        let result = unsafe { rtBufferSetFormat(buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let result = unsafe { rtBufferSetSize1D(buf, width as u64) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize1D", result));
        }

        let hnd = self.ga_buffer1d_obj.insert(buf);

        Ok(hnd)
    }

    /// Get the format of this buffer
    pub fn buffer_get_format_1d(&self, buf: Buffer1dHandle) -> Result<Format> {
        let buf = self.ga_buffer1d_obj.get(buf).unwrap();
        let (format, result) = unsafe {
            let mut format = Format::UNKNOWN;
            let result = rtBufferGetFormat(*buf, &mut format as *mut Format);
            (format, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferGetFormat", result));
        }
        Ok(format)
    }

    /// Set the format of this buffer
    pub fn buffer_set_format_1d(&mut self, buf: Buffer1dHandle, format: Format) -> Result<()> {
        let buf = self.ga_buffer1d_obj.get(buf).unwrap();
        let result = unsafe { rtBufferSetFormat(*buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }
        Ok(())
    }

    /// Create a new `Buffer1d` and fill it with the given data.
    pub fn buffer_create_from_slice_1d<T: BufferElement>(
        &mut self,
        data: &[T],
        buffer_type: BufferType,
        flags: BufferFlag,
    ) -> Result<Buffer1dHandle> {
        let hnd = self.buffer_create_1d(data.len(), T::FORMAT, buffer_type, flags)?;
        {
            let mut map = self.buffer_map_1d_mut::<T>(hnd).unwrap();
            map.as_slice_mut().clone_from_slice(data);
        }
        Ok(hnd)
    }

    /// Destroys this buffer. Not that the buffer will not actually be destroyed
    /// until all references to it from other scene graph objects are released.
    /// # Panics
    /// If buf is not a valid Buffer1dHandle
    pub fn buffer_destroy_1d(&mut self, buf: Buffer1dHandle) {
        let rt_buf = self.ga_buffer1d_obj.remove(buf).unwrap();
        if unsafe { rtBufferDestroy(rt_buf) } != RtResult::SUCCESS {
            panic!("Error destroying buffer {:?}", buf);
        }
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
        let (buf, result) = unsafe {
            let mut buf: RTbuffer = std::mem::uninitialized();
            let result = rtBufferCreate(self.rt_ctx, buffer_type as u32 | flags as u32, &mut buf);
            (buf, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferCreate", result));
        }

        let result = unsafe { rtBufferSetFormat(buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let result = unsafe { rtBufferSetSize2D(buf, width as u64, height as u64) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize2D", result));
        }

        let hnd = self.ga_buffer2d_obj.insert(buf);

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
        let (buf, result) = unsafe {
            let mut buf: RTbuffer = std::mem::uninitialized();
            let result = rtBufferCreate(self.rt_ctx, buffer_type as u32 | flags as u32, &mut buf);
            (buf, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferCreate", result));
        }

        let result = unsafe { rtBufferSetFormat(buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let hnd = self.ga_buffer2d_obj.insert(buf);

        Ok(hnd)
    }

    /// Get the size of this buffer.
    pub fn buffer_get_size_2d(&self, buf: Buffer2dHandle) -> Result<(usize, usize)> {
        let buf = self.ga_buffer2d_obj.get(buf).unwrap();
        let (size, result) = unsafe {
            let mut width = 0u64;
            let mut height = 0u64;
            let result = rtBufferGetSize2D(*buf, &mut width as *mut u64, &mut height as *mut u64);
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
        buf: Buffer2dHandle,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let buf = self.ga_buffer2d_obj.get(buf).unwrap();
        let result = unsafe { rtBufferSetSize2D(*buf, width as u64, height as u64) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize2D", result));
        }
        Ok(())
    }

    /// Get the format of this buffer
    pub fn buffer_get_format_2d(&self, buf: Buffer2dHandle) -> Result<Format> {
        let buf = self.ga_buffer2d_obj.get(buf).unwrap();
        let (format, result) = unsafe {
            let mut format = Format::UNKNOWN;
            let result = rtBufferGetFormat(*buf, &mut format as *mut Format);
            (format, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferGetFormat", result));
        }
        Ok(format)
    }

    /// Set the format of this buffer
    pub fn buffer_set_format_2d(&mut self, buf: Buffer2dHandle, format: Format) -> Result<()> {
        let buf = self.ga_buffer2d_obj.get(buf).unwrap();
        let result = unsafe { rtBufferSetFormat(*buf, format) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }
        Ok(())
    }

    /// Destroys this buffer. Not that the buffer will not actually be destroyed
    /// until all references to it from other scene graph objects are released.
    /// # Panics
    /// If buf is not a valid Buffer2dHandle
    pub fn buffer_destroy_2d(&mut self, buf: Buffer2dHandle) {
        let rt_buf = self.ga_buffer2d_obj.remove(buf).unwrap();
        if unsafe { rtBufferDestroy(rt_buf) } != RtResult::SUCCESS {
            panic!("Error destroying program {:?}", buf);
        }
    }

    /// Map the buffer so that i can be read by host memory. The returned struct
    /// will automatically unmap the buffer when it drops.
    /// # Panics
    /// If buf is not a valid Buffer1dHandle
    pub fn buffer_map_1d<'a, T: BufferElement>(
        &'a self,
        buffer_handle: Buffer1dHandle,
    ) -> Result<ScopedBufMap1d<'a, T>> {
        // first check that the formats align
        let format = self.buffer_get_format_1d(buffer_handle)?;
        if format != T::FORMAT {
            return Err(Error::IncompatibleBufferFormat {
                given: T::FORMAT,
                expected: format,
            });
        }

        match self.ga_buffer1d_obj.get(buffer_handle) {
            Some(buf) => {
                let mut p: *const T = std::ptr::null();
                let result = unsafe {
                    rtBufferMap(*buf, (&mut p) as *mut _ as *mut *mut ::std::os::raw::c_void)
                };
                if result != RtResult::SUCCESS {
                    Err(self.optix_error("rtBufferMap", result))
                } else {
                    let (width, result) = unsafe {
                        let mut width: RTsize = 0;
                        let result = rtBufferGetSize1D(*buf, &mut width);
                        (width, result)
                    };

                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtBufferGetSize1D", result));
                    }

                    Ok(ScopedBufMap1d {
                        data: unsafe { std::slice::from_raw_parts(p, width as usize) },
                        buf,
                    })
                }
            }
            None => Err(Error::HandleNotFoundError),
        }
    }

    /// Map the buffer so that i can be read and written by host memory.
    /// The returned struct will automatically unmap the buffer when it drops.
    /// # Panics
    /// If buf is not a valid Buffer1dHandle
    pub fn buffer_map_1d_mut<'a, T: BufferElement>(
        &'a mut self,
        buffer_handle: Buffer1dHandle,
    ) -> Result<ScopedBufMap1dMut<'a, T>> {
        // first check that the formats align
        let format = self.buffer_get_format_1d(buffer_handle)?;
        if format != T::FORMAT {
            return Err(Error::IncompatibleBufferFormat {
                given: T::FORMAT,
                expected: format,
            });
        }

        match self.ga_buffer1d_obj.get_mut(buffer_handle) {
            Some(buf) => {
                let mut p: *mut T = std::ptr::null_mut();
                let result = unsafe {
                    rtBufferMap(*buf, (&mut p) as *mut _ as *mut *mut ::std::os::raw::c_void)
                };
                if result != RtResult::SUCCESS {
                    Err(Error::Optix((result, "rtBufferMap".to_owned())))
                } else {
                    let (width, result) = unsafe {
                        let mut width: RTsize = 0;
                        let result = rtBufferGetSize1D(*buf, &mut width);
                        (width, result)
                    };

                    if result != RtResult::SUCCESS {
                        return Err(Error::Optix((result, "rtBufferGetSize1D".to_owned())));
                    }

                    Ok(ScopedBufMap1dMut {
                        data: unsafe { std::slice::from_raw_parts_mut(p, width as usize) },
                        buf,
                    })
                }
            }
            None => Err(Error::HandleNotFoundError),
        }
    }

    /// Map the buffer so that it can be read by host memory. The returned struct
    /// will automatically unmap the buffer when it drops.
    /// # Panics
    /// If buf is not a valid Buffer2dHandle
    pub fn buffer_map_2d<'a, T: BufferElement>(
        &'a self,
        buffer_handle: Buffer2dHandle,
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

        match self.ga_buffer2d_obj.get(buffer_handle) {
            Some(buf) => {
                let mut p: *const T = std::ptr::null();
                let result = unsafe {
                    rtBufferMap(*buf, (&mut p) as *mut _ as *mut *mut ::std::os::raw::c_void)
                };
                if result != RtResult::SUCCESS {
                    Err(Error::Optix((result, "rtBufferMap".to_owned())))
                } else {
                    let (width, height, result) = unsafe {
                        let mut width: RTsize = 0;
                        let mut height: RTsize = 0;
                        let result = rtBufferGetSize2D(*buf, &mut width, &mut height);
                        (width, height, result)
                    };

                    if result != RtResult::SUCCESS {
                        return Err(Error::Optix((result, "rtBufferGetSize2D".to_owned())));
                    }

                    Ok(ScopedBufMap2d {
                        data: unsafe {
                            std::slice::from_raw_parts(p, (width * height) as usize * num_channels)
                        },
                        buf,
                        width: width as usize,
                        height: height as usize,
                    })
                }
            }
            None => Err(Error::HandleNotFoundError),
        }
    }

    /// Map the buffer so that i can be read and written by host memory.
    /// The returned struct will automatically unmap the buffer when it drops.
    /// # Panics
    /// If buf is not a valid Buffer2dHandle
    pub fn buffer_map_2d_mut<'a, T: BufferElement>(
        &'a mut self,
        buffer_handle: Buffer2dHandle,
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

        match self.ga_buffer2d_obj.get_mut(buffer_handle) {
            Some(buf) => {
                let mut p: *mut T = std::ptr::null_mut();
                let result = unsafe {
                    rtBufferMap(*buf, (&mut p) as *mut _ as *mut *mut ::std::os::raw::c_void)
                };
                if result != RtResult::SUCCESS {
                    Err(Error::Optix((result, "rtBufferMap".to_owned())))
                } else {
                    let (width, height, result) = unsafe {
                        let mut width: RTsize = 0;
                        let mut height: RTsize = 0;
                        let result = rtBufferGetSize2D(*buf, &mut width, &mut height);
                        (width, height, result)
                    };

                    if result != RtResult::SUCCESS {
                        return Err(Error::Optix((result, "rtBufferGetSize2D".to_owned())));
                    }

                    Ok(ScopedBufMap2dMut {
                        data: unsafe {
                            std::slice::from_raw_parts_mut(
                                p,
                                (width * height) as usize * num_channels,
                            )
                        },
                        buf,
                        width: width as usize,
                        height: height as usize,
                    })
                }
            }
            None => Err(Error::HandleNotFoundError),
        }
    }
}
