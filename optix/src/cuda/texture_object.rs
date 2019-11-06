use optix_sys::cuda_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

#[repr(C)]
pub struct TextureDesc {
    ///  Texture address mode for up to 3 dimensions
    pub address_mode: [TextureAddressMode; 3],
    /// Texture filter mode
    pub filter_mode: TextureFilterMode,
    /// Texture read mode
    pub read_mode: TextureReadMode,
    /// Perform sRGB->Linear conversion during texture read
    pub srgb: i32,
    /// Texture border color
    pub border_color: [f32; 4],
    /// Indicates whether texture reads are normalized or not
    pub normalized_coords: i32,
    /// Limit to the anisotropy ratio
    pub max_anisotropy: u32,
    /// Mipmap filter mode
    pub mipmap_filter_mode: TextureFilterMode,
    /// Offset applied to the supplied mipmap level
    pub mipmap_level_bias: f32,
    /// Lower end of the mipmap level range to clamp access to
    pub min_mipmap_level_clamp: f32,
    /// Upper end of the mipmap level range to clamp access to
    pub max_mipmap_level_clamp: f32,
}

impl TextureDesc {
    pub fn new() -> TextureDescBuilder {
        TextureDescBuilder {
            desc: TextureDesc::default(),
        }
    }
}

impl Default for TextureDesc {
    fn default() -> TextureDesc {
        TextureDesc {
            address_mode: [TextureAddressMode::Clamp; 3],
            filter_mode: TextureFilterMode::Point,
            read_mode: TextureReadMode::ElementType,
            srgb: 0,
            border_color: [0.0f32; 4],
            normalized_coords: 0,
            max_anisotropy: 1,
            mipmap_filter_mode: TextureFilterMode::Point,
            mipmap_level_bias: 0.0,
            min_mipmap_level_clamp: 0.0,
            max_mipmap_level_clamp: 99.0,
        }
    }
}

pub struct TextureDescBuilder {
    desc: TextureDesc,
}

impl TextureDescBuilder {
    pub fn address_mode(mut self, mode: [TextureAddressMode; 3]) -> Self {
        self.desc.address_mode = mode;
        self
    }

    pub fn filter_mode(mut self, filter_mode: TextureFilterMode) -> Self {
        self.desc.filter_mode = filter_mode;
        self
    }

    pub fn read_mode(mut self, read_mode: TextureReadMode) -> Self {
        self.desc.read_mode = read_mode;
        self
    }

    pub fn srgb(mut self, srgb: bool) -> Self {
        self.desc.srgb = if srgb { 1 } else { 0 };
        self
    }

    pub fn border_color(mut self, border_color: [f32; 4]) -> Self {
        self.desc.border_color = border_color;
        self
    }

    pub fn normalized_coords(mut self, normalized_coords: bool) -> Self {
        self.desc.normalized_coords = if normalized_coords { 1 } else { 0 };
        self
    }

    pub fn max_anisotropy(mut self, max_anisotropy: u32) -> Self {
        self.desc.max_anisotropy = max_anisotropy;
        self
    }

    pub fn mipmap_filter_mode(
        mut self,
        mipmap_filter_mode: TextureFilterMode,
    ) -> Self {
        self.desc.mipmap_filter_mode = mipmap_filter_mode;
        self
    }

    pub fn mipmap_level_bias(mut self, mipmap_level_bias: f32) -> Self {
        self.desc.mipmap_level_bias = mipmap_level_bias;
        self
    }

    pub fn min_mipmap_level_clamp(
        mut self,
        min_mipmap_level_clamp: f32,
    ) -> Self {
        self.desc.min_mipmap_level_clamp = min_mipmap_level_clamp;
        self
    }

    pub fn max_mipmap_level_clamp(
        mut self,
        max_mipmap_level_clamp: f32,
    ) -> Self {
        self.desc.max_mipmap_level_clamp = max_mipmap_level_clamp;
        self
    }

    pub fn build(self) -> TextureDesc {
        self.desc
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum TextureAddressMode {
    Wrap = sys::cudaTextureAddressMode_cudaAddressModeWrap,
    Clamp = sys::cudaTextureAddressMode_cudaAddressModeClamp,
    Mirror = sys::cudaTextureAddressMode_cudaAddressModeMirror,
    Border = sys::cudaTextureAddressMode_cudaAddressModeBorder,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum TextureFilterMode {
    Point = sys::cudaTextureFilterMode_cudaFilterModePoint,
    Linear = sys::cudaTextureFilterMode_cudaFilterModeLinear,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug)]
pub enum TextureReadMode {
    ElementType = sys::cudaTextureReadMode_cudaReadModeElementType,
    NormalizedFloat = sys::cudaTextureReadMode_cudaReadModeNormalizedFloat,
}

pub enum ResourceDesc {
    Array(super::array::Array),
}

pub struct TextureObject {
    ptr: sys::cudaTextureObject_t,
    _res_desc: ResourceDesc,
}

impl TextureObject {
    pub fn new(
        res_desc: ResourceDesc,
        tex_desc: &TextureDesc,
    ) -> Result<TextureObject> {
        unsafe {
            let d_res_desc = match &res_desc {
                ResourceDesc::Array(array) => sys::cudaResourceDesc {
                    resType: sys::cudaResourceType_cudaResourceTypeArray,
                    res: sys::cudaResourceDescUnion {
                        array: sys::cudaResourceDescUnionArray {
                            array: array.as_device_ptr(),
                        },
                    },
                },
            };
            let mut ptr = 0;
            let res = sys::cudaCreateTextureObject(
                &mut ptr,
                &d_res_desc,
                tex_desc as *const TextureDesc as *const sys::cudaTextureDesc,
                std::ptr::null(),
            );
            if res != sys::cudaError::cudaSuccess {
                return Err(Error::TextureObjectCreationFailed {
                    cerr: res.into(),
                });
            }

            Ok(TextureObject {
                ptr,
                _res_desc: res_desc,
            })
        }
    }

    pub fn as_device_ptr(&self) -> sys::cudaTextureObject_t {
        self.ptr
    }
}

impl Drop for TextureObject {
    fn drop(&mut self) {
        unsafe {
            let res = sys::cudaDestroyTextureObject(self.ptr);
            if res != sys::cudaError::cudaSuccess {
                panic!("cudaDestroyTextureObject failed: {:?}", res);
            }
        }
    }
}
