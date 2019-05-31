use crate::context::*;

use std::cell::RefCell;
use std::rc::Rc;

pub struct TextureSampler {
    pub(crate) rt_ts: RTtexturesampler,
    pub(crate) buffer: BufferHandle,
}

impl Drop for TextureSampler {
    fn drop(&mut self) {
        let mut id = 0i32;
        unsafe {
            rtTextureSamplerGetId(self.rt_ts, &mut id);
        }
        println!("Dropping texture sampler {}", id);
    }
}

pub type TextureSamplerHandle = Rc<RefCell<TextureSampler>>;

pub struct TextureID {
    pub(crate) ts: TextureSamplerHandle,
    pub id: i32,
}

impl Drop for TextureID {
    fn drop(&mut self) {
        println!("Dropping texture ID {}", self.id);
    }
}

impl Clone for TextureID {
    fn clone(&self) -> TextureID {
        TextureID {
            ts: Rc::clone(&self.ts),
            id: self.id,
        }
    }
}

pub enum BufferHandle {
    Buffer1d(Buffer1dHandle),
    Buffer2d(Buffer2dHandle),
}

impl Context {
    pub fn texture_sampler_create_1d(
        &mut self,
        buf: Buffer1dHandle,
    ) -> Result<TextureSamplerHandle> {
        self.buffer_validate_1d(&buf)?;

        let (rt_ts, result) = unsafe {
            let mut rt_ts: RTtexturesampler = std::mem::zeroed();
            let result = rtTextureSamplerCreate(self.rt_ctx, &mut rt_ts);
            (rt_ts, result)
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerCreate", result));
        }

        let result = unsafe {
            rtTextureSamplerSetBuffer(rt_ts, 0, 0, buf.borrow().rt_buf)
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetBuffer 1d", result)
            );
        }

        let ts = Rc::new(RefCell::new(TextureSampler {
            rt_ts,
            buffer: BufferHandle::Buffer1d(buf),
        }));

        self.texture_samplers.push(Rc::clone(&ts));

        Ok(ts)
    }

    pub fn texture_sampler_create_2d(
        &mut self,
        buf: Buffer2dHandle,
    ) -> Result<TextureSamplerHandle> {
        self.buffer_validate_2d(&buf)?;

        let (rt_ts, result) = unsafe {
            let mut rt_ts: RTtexturesampler = std::mem::zeroed();
            let result = rtTextureSamplerCreate(self.rt_ctx, &mut rt_ts);
            (rt_ts, result)
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerCreate", result));
        }

        let result = unsafe {
            rtTextureSamplerSetBuffer(rt_ts, 0, 0, buf.borrow().rt_buf)
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetBuffer 2d", result)
            );
        }

        let ts = Rc::new(RefCell::new(TextureSampler {
            rt_ts,
            buffer: BufferHandle::Buffer2d(buf),
        }));

        self.texture_samplers.push(Rc::clone(&ts));

        Ok(ts)
    }

    pub fn texture_sampler_create_from_slice_2d<T: BufferElement>(
        &mut self,
        data: &[T],
        width: usize,
        height: usize,
    ) -> Result<TextureSamplerHandle> {
        let buf = self.buffer_create_from_slice_2d(
            data,
            width,
            height,
            BufferType::INPUT,
            BufferFlag::NONE,
        )?;

        self.buffer_validate_2d(&buf)?;

        let (rt_ts, result) = unsafe {
            let mut rt_ts: RTtexturesampler = std::mem::zeroed();
            let result = rtTextureSamplerCreate(self.rt_ctx, &mut rt_ts);
            (rt_ts, result)
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerCreate", result));
        }

        let result = unsafe {
            rtTextureSamplerSetBuffer(rt_ts, 0, 0, buf.borrow().rt_buf)
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetBuffer 2d", result)
            );
        }

        let ts = Rc::new(RefCell::new(TextureSampler {
            rt_ts,
            buffer: BufferHandle::Buffer2d(buf),
        }));

        self.texture_samplers.push(Rc::clone(&ts));

        Ok(ts)
    }

    pub fn texture_sampler_validate(
        &mut self,
        ts: &TextureSamplerHandle,
    ) -> Result<()> {
        let result = unsafe { rtTextureSamplerValidate(ts.borrow().rt_ts) };

        if result == RtResult::SUCCESS {
            Ok(())
        } else {
            Err(self.optix_error("rtTextureSamplerValidate", result))
        }
    }

    pub fn texture_sampler_set_buffer_1d(
        &mut self,
        ts: &TextureSamplerHandle,
        buf: Buffer1dHandle,
    ) -> Result<()> {
        self.buffer_validate_1d(&buf)?;

        let result = unsafe {
            rtTextureSamplerSetBuffer(
                ts.borrow().rt_ts,
                0,
                0,
                buf.borrow().rt_buf,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetBuffer 1d", result)
            );
        }

        ts.borrow_mut().buffer = BufferHandle::Buffer1d(buf);

        Ok(())
    }

    pub fn texture_sampler_set_buffer_2d(
        &mut self,
        ts: &TextureSamplerHandle,
        buf: Buffer2dHandle,
    ) -> Result<()> {
        self.buffer_validate_2d(&buf)?;

        let result = unsafe {
            rtTextureSamplerSetBuffer(
                ts.borrow().rt_ts,
                0,
                0,
                buf.borrow().rt_buf,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetBuffer 1d", result)
            );
        }

        ts.borrow_mut().buffer = BufferHandle::Buffer2d(buf);

        Ok(())
    }

    pub fn texture_sampler_set_filtering_modes(
        &mut self,
        ts: &TextureSamplerHandle,
        minification: FilterMode,
        magnification: FilterMode,
        mipmapping: FilterMode,
    ) -> Result<()> {
        let result = unsafe {
            rtTextureSamplerSetFilteringModes(
                ts.borrow().rt_ts,
                minification,
                magnification,
                mipmapping,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetFilteringModes", result)
            );
        } else {
            Ok(())
        }
    }

    pub fn texture_sampler_set_indexing_mode(
        &mut self,
        ts: &TextureSamplerHandle,
        indexmode: TextureIndexMode,
    ) -> Result<()> {
        let result = unsafe {
            rtTextureSamplerSetIndexingMode(ts.borrow().rt_ts, indexmode)
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetIndexingMode", result)
            );
        } else {
            Ok(())
        }
    }

    pub fn texture_sampler_set_max_anisotropy(
        &mut self,
        ts: &TextureSamplerHandle,
        value: f32,
    ) -> Result<()> {
        let result = unsafe {
            rtTextureSamplerSetMaxAnisotropy(ts.borrow().rt_ts, value)
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetMaxAnisotropy", result)
            );
        } else {
            Ok(())
        }
    }

    pub fn texture_sampler_set_mip_level_bias(
        &mut self,
        ts: &TextureSamplerHandle,
        value: f32,
    ) -> Result<()> {
        let result = unsafe {
            rtTextureSamplerSetMipLevelBias(ts.borrow().rt_ts, value)
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetMipLevelBias", result)
            );
        } else {
            Ok(())
        }
    }

    pub fn texture_sampler_set_mip_level_clamp(
        &mut self,
        ts: &TextureSamplerHandle,
        lower: f32,
        upper: f32,
    ) -> Result<()> {
        let result = unsafe {
            rtTextureSamplerSetMipLevelClamp(ts.borrow().rt_ts, lower, upper)
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtTextureSamplerSetMipLevelClamp", result)
            );
        } else {
            Ok(())
        }
    }

    pub fn texture_sampler_set_read_mode(
        &mut self,
        ts: &TextureSamplerHandle,
        mode: TextureReadMode,
    ) -> Result<()> {
        let result =
            unsafe { rtTextureSamplerSetReadMode(ts.borrow().rt_ts, mode) };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerSetReadMode", result));
        } else {
            Ok(())
        }
    }

    pub fn texture_sampler_set_wrap_mode(
        &mut self,
        ts: &TextureSamplerHandle,
        dimension: u32,
        mode: WrapMode,
    ) -> Result<()> {
        let result = unsafe {
            rtTextureSamplerSetWrapMode(ts.borrow().rt_ts, dimension, mode)
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerSetWrapMode", result));
        } else {
            Ok(())
        }
    }

    pub fn texture_sampler_get_id(
        &mut self,
        ts: &TextureSamplerHandle,
    ) -> Result<TextureID> {
        let mut id = 0i32;
        let result =
            unsafe { rtTextureSamplerGetId(ts.borrow().rt_ts, &mut id) };
        if result != RtResult::SUCCESS {
            return Err(Error::Optix((
                result,
                "rtTextureSamplerGetId".to_owned(),
            )));
        } else {
            Ok(TextureID {
                ts: Rc::clone(&ts),
                id,
            })
        }
    }
}
