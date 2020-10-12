use crate::{
    sys, DeviceContext, DeviceCopy, DeviceStorage, Error, Image2D, TypedBuffer,
};
type Result<T, E = Error> = std::result::Result<T, E>;

pub struct Denoiser {
    inner: sys::OptixDenoiser,
}

impl DeviceContext {
    pub fn denoiser_create(
        &self,
        input_kind: DenoiserInputKind,
    ) -> Result<Denoiser> {
        let mut inner: sys::OptixDenoiser = std::ptr::null_mut();
        let options = DenoiserOptions { input_kind };
        unsafe {
            sys::optixDenoiserCreate(
                self.inner,
                &options as *const _ as *const _,
                &mut inner,
            )
            .to_result()
            .map(|_| Denoiser { inner })
            .map_err(|source| Error::CreateDenoiser { source })
        }
    }

    pub fn denoiser_destroy(&self, denoiser: Denoiser) -> Result<()> {
        unsafe {
            sys::optixDenoiserDestroy(denoiser.inner)
                .to_result()
                .map_err(|source| Error::DestroyDenoiser { source })
        }
    }
}

impl Denoiser {
    pub fn compute_intensity<S1: DeviceStorage, S2: DeviceStorage>(
        &self,
        stream: cu::Stream,
        input_image: &Image2D,
        output_intensity: &mut S2,
        scratch: &S2,
    ) -> Result<()> {
        if output_intensity.byte_size() < std::mem::size_of::<f32>() {
            panic!("compute_intensity() required an output intensity storage of at least a single float");
        }
        unsafe {
            sys::optixDenoiserComputeIntensity(
                self.inner,
                stream.inner(),
                input_image as *const _ as *const _,
                output_intensity.device_ptr().0,
                scratch.device_ptr().0,
                scratch.byte_size(),
            )
            .to_result()
            .map_err(|source| Error::DenoiserComputeIntensity { source })
        }
    }

    pub fn compute_memory_resources(
        &self,
        width: u32,
        height: u32,
    ) -> Result<DenoiserSizes> {
        let mut sizes = DenoiserSizes::default();
        unsafe {
            sys::optixDenoiserComputeMemoryResources(
                self.inner,
                width,
                height,
                &mut sizes as *mut _ as *mut _,
            )
            .to_result()
            .map(|_| sizes)
            .map_err(|source| Error::DenoiserComputeMemoryResources { source })
        }
    }

    pub fn invoke<S1: DeviceStorage, S2: DeviceStorage>(
        &self,
        stream: &cu::Stream,
        params: &DenoiserParams,
        denoiser_state: &S1,
        input_layers: &[Image2D],
        input_offset_x: u32,
        input_offset_y: u32,
        output_layer: &mut Image2D,
        scratch: &S2,
    ) -> Result<()> {
        unsafe {
            sys::optixDenoiserInvoke(
                self.inner,
                stream.inner(),
                params as *const _ as *const _,
                denoiser_state.device_ptr().0,
                denoiser_state.byte_size(),
                input_layers.as_ptr() as *const _,
                input_layers.len() as u32,
                input_offset_x,
                input_offset_y,
                output_layer as *mut _ as *mut _,
                scratch.device_ptr().0,
                scratch.byte_size(),
            )
            .to_result()
            .map_err(|source| Error::DenoiserInvoke { source })
        }
    }

    pub fn set_model_kind_ldr(&self) -> Result<()> {
        unsafe {
            sys::optixDenoiserSetModel(
                self.inner,
                sys::OptixDenoiserModelKind_OPTIX_DENOISER_MODEL_KIND_LDR,
                std::ptr::null_mut(),
                0,
            )
            .to_result()
            .map_err(|source| Error::DenoiserSetModelLdr { source })
        }
    }

    pub fn set_model_kind_hdr(&self) -> Result<()> {
        unsafe {
            sys::optixDenoiserSetModel(
                self.inner,
                sys::OptixDenoiserModelKind_OPTIX_DENOISER_MODEL_KIND_HDR,
                std::ptr::null_mut(),
                0,
            )
            .to_result()
            .map_err(|source| Error::DenoiserSetModelLdr { source })
        }
    }

    pub fn setup<S1: DeviceStorage, S2: DeviceStorage>(
        &self,
        stream: &cu::Stream,
        input_width: u32,
        input_height: u32,
        denoiser_state: &S1,
        scratch: &S2,
    ) -> Result<()> {
        unsafe {
            sys::optixDenoiserSetup(
                self.inner,
                stream.inner(),
                input_width,
                input_height,
                denoiser_state.device_ptr().0,
                denoiser_state.byte_size(),
                scratch.device_ptr().0,
                scratch.byte_size(),
            )
            .to_result()
            .map_err(|source| Error::DenoiserSetup { source })
        }
    }
}

#[repr(u32)]
pub enum DenoiserInputKind {
    Rgb = sys::OptixDenoiserInputKind_OPTIX_DENOISER_INPUT_RGB,
    RgbAledbo = sys::OptixDenoiserInputKind_OPTIX_DENOISER_INPUT_RGB_ALBEDO,
    RgbAlbedoNormal =
        sys::OptixDenoiserInputKind_OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL,
}

#[repr(u32)]
pub enum DenoiserModelKind {
    User = sys::OptixDenoiserModelKind_OPTIX_DENOISER_MODEL_KIND_USER,
    Ldr = sys::OptixDenoiserModelKind_OPTIX_DENOISER_MODEL_KIND_LDR,
    Hdr = sys::OptixDenoiserModelKind_OPTIX_DENOISER_MODEL_KIND_HDR,
}

#[repr(C)]
pub struct DenoiserOptions {
    pub input_kind: DenoiserInputKind,
}

#[repr(C)]
pub struct DenoiserParams {
    denoise_alpha: bool,
    hdr_intensity: cu::DevicePtr,
    blend_factor: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct DenoiserSizes {
    static_size_in_bytes: usize,
    with_overlap_scratch_size_in_bytes: usize,
    without_overlap_scratch_size_in_bytes: usize,
    overlap_window_size_in_pixels: u32,
}
