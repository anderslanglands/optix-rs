use crate::context::*;

new_key_type! { pub struct TextureSamplerHandle; }

pub enum BufferHandle {
    Buffer1d(Buffer1dHandle),
    Buffer2d(Buffer2dHandle),
}

impl Context {
    pub fn texture_sampler_create_1d(
        &mut self,
        buf: Buffer1dHandle,
    ) -> Result<TextureSamplerHandle> {
        self.buffer_validate_1d(buf)?;
        let rt_buf = self.ga_buffer1d_obj.get(buf).unwrap();

        let (rt_ts, result) = unsafe {
            let mut rt_ts: RTtexturesampler = std::mem::zeroed();
            let result = rtTextureSamplerCreate(self.rt_ctx, &mut rt_ts);
            (rt_ts, result)
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerCreate", result));
        }

        let result = unsafe { rtTextureSamplerSetBuffer(rt_ts, 0, 0, *rt_buf) };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerSetBuffer 1d", result));
        }

        let ts = self.ga_texture_sampler_obj.insert(rt_ts);
        self.gd_texture_sampler_buffer
            .insert(ts, BufferHandle::Buffer1d(buf));

        Ok(ts)
    }

    pub fn texture_sampler_create_2d(
        &mut self,
        buf: Buffer2dHandle,
    ) -> Result<TextureSamplerHandle> {
        self.buffer_validate_2d(buf)?;
        let rt_buf = self.ga_buffer2d_obj.get(buf).unwrap();

        let (rt_ts, result) = unsafe {
            let mut rt_ts: RTtexturesampler = std::mem::zeroed();
            let result = rtTextureSamplerCreate(self.rt_ctx, &mut rt_ts);
            (rt_ts, result)
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerCreate", result));
        }

        let result = unsafe { rtTextureSamplerSetBuffer(rt_ts, 0, 0, *rt_buf) };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerSetBuffer 2d", result));
        }

        let ts = self.ga_texture_sampler_obj.insert(rt_ts);

        Ok(ts)
    }

    pub fn texture_sampler_destroy(&mut self, ts: TextureSamplerHandle) {
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();

        let result = unsafe { rtTextureSamplerDestroy(*rt_ts) };

        if result != RtResult::SUCCESS {
            panic!("rtTextureSamplerDestroy");
        }
    }

    pub fn texture_sampler_validate(
        &mut self,
        ts: TextureSamplerHandle,
    ) -> Result<()> {
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();

        let result = unsafe { rtTextureSamplerValidate(*rt_ts) };

        if result == RtResult::SUCCESS {
            Ok(())
        } else {
            Err(self.optix_error("rtTextureSamplerValidate", result))
        }
    }

    pub fn texture_sampler_set_buffer_1d(
        &mut self,
        ts: TextureSamplerHandle,
        buf: Buffer1dHandle,
    ) -> Result<()> {
        self.buffer_validate_1d(buf)?;
        let rt_buf = self.ga_buffer1d_obj.get(buf).unwrap();
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();

        let result =
            unsafe { rtTextureSamplerSetBuffer(*rt_ts, 0, 0, *rt_buf) };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerSetBuffer 1d", result));
        }

        self.gd_texture_sampler_buffer
            .insert(ts, BufferHandle::Buffer1d(buf));

        Ok(())
    }

    pub fn texture_sampler_set_buffer_2d(
        &mut self,
        ts: TextureSamplerHandle,
        buf: Buffer2dHandle,
    ) -> Result<()> {
        self.buffer_validate_2d(buf)?;
        let rt_buf = self.ga_buffer2d_obj.get(buf).unwrap();
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();

        let result =
            unsafe { rtTextureSamplerSetBuffer(*rt_ts, 0, 0, *rt_buf) };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerSetBuffer 1d", result));
        }

        self.gd_texture_sampler_buffer
            .insert(ts, BufferHandle::Buffer2d(buf));

        Ok(())
    }

    pub fn texture_sampler_set_filtering_modes(
        &mut self,
        ts: TextureSamplerHandle,
        minification: FilterMode,
        magnification: FilterMode,
        mipmapping: FilterMode,
    ) -> Result<()> {
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();
        let result = unsafe {
            rtTextureSamplerSetFilteringModes(
                *rt_ts,
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
        ts: TextureSamplerHandle,
        indexmode: TextureIndexMode,
    ) -> Result<()> {
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();
        let result =
            unsafe { rtTextureSamplerSetIndexingMode(*rt_ts, indexmode) };

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
        ts: TextureSamplerHandle,
        value: f32,
    ) -> Result<()> {
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();
        let result = unsafe { rtTextureSamplerSetMaxAnisotropy(*rt_ts, value) };

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
        ts: TextureSamplerHandle,
        value: f32,
    ) -> Result<()> {
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();
        let result = unsafe { rtTextureSamplerSetMipLevelBias(*rt_ts, value) };

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
        ts: TextureSamplerHandle,
        lower: f32,
        upper: f32,
    ) -> Result<()> {
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();
        let result =
            unsafe { rtTextureSamplerSetMipLevelClamp(*rt_ts, lower, upper) };

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
        ts: TextureSamplerHandle,
        mode: TextureReadMode,
    ) -> Result<()> {
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();
        let result = unsafe { rtTextureSamplerSetReadMode(*rt_ts, mode) };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerSetReadMode", result));
        } else {
            Ok(())
        }
    }

    pub fn texture_sampler_set_wrap_mode(
        &mut self,
        ts: TextureSamplerHandle,
        dimension: u32,
        mode: WrapMode,
    ) -> Result<()> {
        let rt_ts = self.ga_texture_sampler_obj.get(ts).unwrap();
        let result =
            unsafe { rtTextureSamplerSetWrapMode(*rt_ts, dimension, mode) };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTextureSamplerSetWrapMode", result));
        } else {
            Ok(())
        }
    }
}
