use crate::{sys, Error, DevicePtr};
type Result<T, E = Error> = std::result::Result<T, E>;

#[repr(transparent)]
pub struct TexObject {
    pub(crate) inner: sys::CUtexObject,
}

impl TexObject {
    pub fn null() -> TexObject {
        TexObject { inner: 0 }
    }

    pub fn inner(&self) -> sys::CUtexObject {
        self.inner
    }

    pub fn create_pitch2d(
        ptr: DevicePtr,
        format: ArrayFormat,
        num_channels: u32,
        width: usize,
        height: usize,
        pitch_in_bytes: usize,
    ) -> TexObjectBuilder {
        let res_desc = ResourceDesc {
            res_type: ResourceType::Pitch2d,
            res_union: ResourceDescUnion {
                pitch2d: ResourceDescPitch2D {
                    ptr,
                    format,
                    num_channels,
                    width,
                    height,
                    pitch_in_bytes,
                }
            },
            flags: 0,
        };

        TexObjectBuilder {
            res_desc,
            tex_desc: sys::CUDA_TEXTURE_DESC_st::default(),
        }
    }
}

pub struct TexObjectBuilder {
    res_desc: ResourceDesc,
    tex_desc: sys::CUDA_TEXTURE_DESC_st,
}

impl TexObjectBuilder {
    pub fn build(self) -> Result<TexObject> {
        let mut inner: sys::CUtexObject = 0;
        unsafe {
            sys::cuTexObjectCreate(
                &mut inner as *mut _,
                &self.res_desc as *const _ as *const _,
                &self.tex_desc as *const _,
                std::ptr::null(),
            ).to_result()
            .map(|_| TexObject{inner})
        }
    }

    pub fn address_mode(mut self, address_mode: AddressMode) -> Self {
        self.tex_desc.addressMode[0] = address_mode as u32;
        self.tex_desc.addressMode[1] = address_mode as u32;
        self.tex_desc.addressMode[2] = address_mode as u32;
        self
    }

    pub fn address_mode3(mut self, address_mode: [AddressMode; 3]) -> Self {
        self.tex_desc.addressMode[0] = address_mode[0] as u32;
        self.tex_desc.addressMode[1] = address_mode[1] as u32;
        self.tex_desc.addressMode[2] = address_mode[2] as u32;
        self
    }

    pub fn filter_mode(mut self, filter_mode: FilterMode) -> Self {
        self.tex_desc.filterMode = filter_mode as u32;
        self
    }

    pub fn flags(mut self, flags: TextureReadFlags) -> Self {
        self.tex_desc.flags = flags.bits();
        self
    }

    pub fn max_anisotropy(mut self, max_anisotropy: u32) -> Self {
        self.tex_desc.maxAnisotropy = max_anisotropy;
        self
    }

    pub fn mipmap_filter_mode(mut self, mipmap_filter_mode: FilterMode) -> Self {
        self.tex_desc.mipmapFilterMode = mipmap_filter_mode as u32;
        self
    }

    pub fn mipmap_level_bias(mut self, mipmap_level_bias: f32) -> Self {
        self.tex_desc.mipmapLevelBias = mipmap_level_bias;
        self
    }

    pub fn mipmap_level_clamp(mut self, min: f32, max: f32) -> Self {
        self.tex_desc.minMipmapLevelClamp = min;
        self.tex_desc.maxMipmapLevelClamp = max;
        self
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Array {
    pub(crate) inner: sys::CUarray,
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MipmappedArray {
    pub(crate) inner: sys::CUmipmappedArray,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ResourceDesc {
    res_type: ResourceType,
    res_union: ResourceDescUnion,
    flags: u32,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ResourceType {
    Array = sys::CUresourcetype_enum_CU_RESOURCE_TYPE_ARRAY,
    MipmappedArray = sys::CUresourcetype_enum_CU_RESOURCE_TYPE_MIPMAPPED_ARRAY,
    Linear = sys::CUresourcetype_enum_CU_RESOURCE_TYPE_LINEAR,
    Pitch2d = sys::CUresourcetype_enum_CU_RESOURCE_TYPE_PITCH2D,
}


#[repr(C)]
#[derive(Copy, Clone)]
pub union ResourceDescUnion {
    array: ResourceDescArray,
    mipmap: ResourceDescMipmap,
    linear: ResourceDescLinear,
    pitch2d: ResourceDescPitch2D,
    reserved: sys::CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ResourceDescArray {
    pub(crate) array: Array,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ResourceDescMipmap {
    pub(crate) mipmapped_array: MipmappedArray,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ResourceDescLinear {
    ptr: DevicePtr,
    format: ArrayFormat,
    num_channels: u32,
    size_in_bytes: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ResourceDescPitch2D {
    ptr: DevicePtr,
    format: ArrayFormat,
    num_channels: u32,
    width: usize,
    height: usize,
    pitch_in_bytes: usize,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ArrayFormat {
    Uint8 = sys::CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT8,
    Uint16 = sys::CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT16,
    UInt32 = sys::CUarray_format_enum::CU_AD_FORMAT_UNSIGNED_INT32,
    Int8 = sys::CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT8,
    Int16 = sys::CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT16,
    Int32 = sys::CUarray_format_enum::CU_AD_FORMAT_SIGNED_INT32,
    Half = sys::CUarray_format_enum::CU_AD_FORMAT_HALF,
    Float = sys::CUarray_format_enum::CU_AD_FORMAT_FLOAT,
}

pub trait Texel {
    const FORMAT: ArrayFormat;
    const NUM_CHANNELS: u32;
}

bitflags::bitflags! {
    pub struct TextureReadFlags: u32 {
        const READ_AS_INTEGER = sys::TextureReadFlags::ReadAsInteger as u32;
        const NORMALIZED_COORDINATES = sys::TextureReadFlags::NormalizedCoordinates as u32;
        const SRGB = sys::TextureReadFlags::Srgb as u32;
        const DISABLE_TRILINEAR_OPTIIMIZATION = sys::TextureReadFlags::DisableTrilinearOptimization as u32;
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TextureDesc {
    address_mode: [AddressMode; 3],
    filter_mode: FilterMode,
    flags: TextureReadFlags,
    max_anisotropy: u32,
    mipmap_filter_mode:FilterMode,
    mipmap_level_bias: f32,
    min_mipmap_level_clamp: f32,
    max_mipmap_level_clamp: f32,
}

impl Default for sys::CUDA_TEXTURE_DESC_st {
    fn default() -> Self {
        Self {
            addressMode: [
                sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_WRAP,
                sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_WRAP,
                sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_WRAP,
            ],
            filterMode: sys::CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
            flags: 0,
            maxAnisotropy: 0,
            mipmapFilterMode: sys::CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
            mipmapLevelBias: 0.0,
            minMipmapLevelClamp: 0.0,
            maxMipmapLevelClamp: 0.0,
            borderColor: [0.0f32; 4],
            reserved: [0; 12],
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AddressMode {
    Wrap = sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_WRAP,
    Clamp = sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_CLAMP,
    Mirror = sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_MIRROR,
    Border = sys::CUaddress_mode_enum::CU_TR_ADDRESS_MODE_BORDER,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum FilterMode {
    Point = sys::CUfilter_mode_enum::CU_TR_FILTER_MODE_POINT,
    Linear = sys::CUfilter_mode_enum::CU_TR_FILTER_MODE_LINEAR,
}