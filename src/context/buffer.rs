use crate::context::*;
use crate::ginallocator::*;
use crate::math::*;
use std::ops::{Index, IndexMut};

#[derive(Default, Debug, Copy, Clone)]
pub struct Buffer1dMarker;
impl Marker for Buffer1dMarker {
    const ID: &'static str = "Buffer1d";
}
pub type Buffer1dHandle = Handle<Buffer1dMarker>;

#[derive(Default, Debug, Copy, Clone)]
pub struct Buffer2dMarker;
impl Marker for Buffer2dMarker {
    const ID: &'static str = "Buffer2d";
}
pub type Buffer2dHandle = Handle<Buffer2dMarker>;

#[derive(Default, Debug, Copy, Clone)]
pub struct Buffer3dMarker;
impl Marker for Buffer3dMarker {
    const ID: &'static str = "Buffer3d";
}
pub type Buffer3dHandle = Handle<Buffer3dMarker>;

pub trait BufferElement: Clone {
    const FORMAT: Format;
}

impl BufferElement for V4f32 {
    const FORMAT: Format = Format::FLOAT4;
}

pub struct ScopedBufMap2d<'a, T: 'a + BufferElement> {
    data: &'a [T],
    buf: &'a RTbuffer,
    width: usize,
    height: usize,
}

impl<'a, T: BufferElement> ScopedBufMap2d<'a, T> {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
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

pub struct ScopedBufMap2dMut<'a, T: 'a + BufferElement> {
    data: &'a mut [T],
    buf: &'a mut RTbuffer,
    width: usize,
    height: usize,
}

impl<'a, T: BufferElement> ScopedBufMap2dMut<'a, T> {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn len(&self) -> usize {
        self.data.len()
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
            if rtBufferUnmap(*self.buf) != RtResult::SUCCESS {
                panic!("rtBufferUnmap failed!");
            }
        }
    }
}

impl Context {
    pub fn buffer_create_2d<T: BufferElement>(
        &mut self,
        width: usize,
        height: usize,
        buffer_type: BufferType,
        flags: BufferFlag,
    ) -> Result<Buffer2dHandle> {
        let (buf, result) = unsafe {
            let mut buf: RTbuffer = std::mem::uninitialized();
            let result = rtBufferCreate(
                self.rt_ctx,
                buffer_type as u32 | flags as u32,
                &mut buf,
            );
            (buf, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferCreate", result));
        }

        let result = unsafe { rtBufferSetFormat(buf, T::FORMAT) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetFormat", result));
        }

        let result =
            unsafe { rtBufferSetSize2D(buf, width as u64, height as u64) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtBufferSetSize2D", result));
        }

        let hnd = self.ga_buffer2d_obj.insert(buf);

        Ok(hnd)
    }

    pub fn buffer_destroy_2d(&mut self, buf: Buffer2dHandle) {
        let rt_buf = *self.ga_buffer2d_obj.get(buf).unwrap();
        match self.ga_buffer2d_obj.destroy(buf) {
            DestroyResult::StillAlive => (),
            DestroyResult::ShouldDrop => {
                if unsafe {
                    rtBufferDestroy(rt_buf)
                } != RtResult::SUCCESS {
                    panic!("Error destroying program {}", buf);
                }
            }
        }
    }

    pub fn buffer_map_2d<'a, T: BufferElement>(
        &'a self,
        buffer_handle: Buffer2dHandle,
    ) -> Result<ScopedBufMap2d<'a, T>> {
        match self.ga_buffer2d_obj.get(buffer_handle) {
            Some(buf) => {
                let mut p: *const T = std::ptr::null();
                let result = unsafe {
                    rtBufferMap(
                        *buf,
                        (&mut p) as *mut _ as *mut *mut ::std::os::raw::c_void,
                    )
                };
                if result != RtResult::SUCCESS {
                    Err(self.optix_error("rtBufferMap", result))
                } else {
                    let (width, height, result) = unsafe {
                        let mut width: RTsize = 0;
                        let mut height: RTsize = 0;
                        let result = rtBufferGetSize2D(
                            *buf,
                            &mut width,
                            &mut height,
                        );
                        (width, height, result)
                    };

                    if result != RtResult::SUCCESS {
                        return Err(
                            self.optix_error("rtBufferGetSize2D", result)
                        );
                    }

                    Ok(ScopedBufMap2d {
                        data: unsafe {
                            std::slice::from_raw_parts(p, (width * height) as usize)
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
