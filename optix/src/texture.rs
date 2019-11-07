use super::buffer::BufferElement;
use super::cuda;
use super::error::Error;
use super::math::*;
use super::DeviceShareable;
type Result<T, E = Error> = std::result::Result<T, E>;

pub struct Texture {
    texture_object: cuda::TextureObject,
}

pub trait TextureElement: BufferElement {
    const CHANNEL_DESC: cuda::ChannelFormatDesc;
}

#[repr(u32)]
pub enum WrapMode {
    Clamp = cuda::TextureAddressMode::Clamp as u32,
    Wrap = cuda::TextureAddressMode::Wrap as u32,
    Border = cuda::TextureAddressMode::Border as u32,
    Mirror = cuda::TextureAddressMode::Mirror as u32,
}

impl From<WrapMode> for cuda::TextureAddressMode {
    fn from(m: WrapMode) -> cuda::TextureAddressMode {
        match m {
            WrapMode::Clamp => cuda::TextureAddressMode::Clamp,
            WrapMode::Wrap => cuda::TextureAddressMode::Wrap,
            WrapMode::Border => cuda::TextureAddressMode::Border,
            WrapMode::Mirror => cuda::TextureAddressMode::Mirror,
        }
    }
}

impl Texture {
    pub fn new<T>(
        pixels: &[T],
        width: usize,
        height: usize,
        wrap_mode: WrapMode,
        srgb: bool,
    ) -> Result<Texture>
    where
        T: TextureElement,
    {
        let pixels = unsafe {
            std::slice::from_raw_parts(
                pixels.as_ptr() as *const T::ComponentType,
                pixels.len() * T::COMPONENTS,
            )
        };
        let array = cuda::Array::new(
            pixels,
            T::CHANNEL_DESC,
            width,
            height,
            T::COMPONENTS,
            cuda::ArrayFlags::DEFAULT,
        )
        .unwrap();

        let tex_desc = cuda::TextureDesc::new()
            .address_mode([wrap_mode.into(); 3])
            .filter_mode(cuda::TextureFilterMode::Linear)
            .read_mode(cuda::TextureReadMode::NormalizedFloat)
            .normalized_coords(true)
            .srgb(srgb)
            .build();

        let texture_object = cuda::TextureObject::new(
            cuda::ResourceDesc::Array(array),
            &tex_desc,
        )
        .unwrap();

        Ok(Texture { texture_object })
    }
}

impl TextureElement for u8 {
    const CHANNEL_DESC: cuda::ChannelFormatDesc = cuda::ChannelFormatDesc {
        x: 8,
        y: 0,
        z: 0,
        w: 0,
        f: cuda::ChannelFormatKind::Unsigned,
    };
}

impl TextureElement for f32 {
    const CHANNEL_DESC: cuda::ChannelFormatDesc = cuda::ChannelFormatDesc {
        x: 32,
        y: 0,
        z: 0,
        w: 0,
        f: cuda::ChannelFormatKind::Float,
    };
}

impl TextureElement for V4u8 {
    const CHANNEL_DESC: cuda::ChannelFormatDesc = cuda::ChannelFormatDesc {
        x: 8,
        y: 8,
        z: 8,
        w: 8,
        f: cuda::ChannelFormatKind::Unsigned,
    };
}

impl TextureElement for V4f32 {
    const CHANNEL_DESC: cuda::ChannelFormatDesc = cuda::ChannelFormatDesc {
        x: 32,
        y: 32,
        z: 32,
        w: 32,
        f: cuda::ChannelFormatKind::Float,
    };
}

impl DeviceShareable for Texture {
    type Target = cuda::cudaTextureObject_t;
    fn to_device(&self) -> Self::Target {
        self.texture_object.as_device_ptr()
    }
    fn cuda_type() -> String {
        "cudaTextureObject_t".into()
    }
}

impl DeviceShareable for Option<Texture> {
    type Target = cuda::cudaTextureObject_t;
    fn to_device(&self) -> Self::Target {
        match self {
            Some(t) => t.texture_object.as_device_ptr(),
            None => 0,
        }
    }
    fn cuda_type() -> String {
        "cudaTextureObject_t".into()
    }
}

impl DeviceShareable for Option<std::rc::Rc<Texture>> {
    type Target = cuda::cudaTextureObject_t;
    fn to_device(&self) -> Self::Target {
        match self {
            Some(t) => t.texture_object.as_device_ptr(),
            None => 0,
        }
    }
    fn cuda_type() -> String {
        "cudaTextureObject_t".into()
    }
}
