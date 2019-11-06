pub mod context;
pub use context::{Context, ContextRef};
pub mod buffer;
pub use buffer::{Buffer, BufferArray, MemcpyKind};
pub mod error;
pub mod nvrtc;
pub mod stream;
pub use stream::Stream;
pub mod texture_object;
pub use texture_object::{
    ResourceDesc, TextureAddressMode, TextureDesc, TextureDescBuilder,
    TextureFilterMode, TextureObject, TextureReadMode,
};
pub mod array;
pub use array::{Array, ArrayFlags, ChannelFormatDesc, ChannelFormatKind};

pub use error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

use optix_sys::cuda_sys as sys;

use std::ffi::CStr;

pub use sys::{cudaTextureObject_t, CUdeviceptr};

pub fn init() {
    unsafe {
        sys::cudaFree(std::ptr::null_mut());
    }
}

pub fn get_device_count() -> i32 {
    let mut count = 0i32;
    unsafe {
        sys::cudaGetDeviceCount(&mut count as *mut i32);
    }

    count
}

pub fn set_device(device: i32) -> Result<()> {
    unsafe {
        let res = sys::cudaSetDevice(device);
        if res != sys::cudaError_enum::CUDA_SUCCESS as u32 {
            return Err(Error::CouldNotSetDevice {
                cerr: res.into(),
                device,
            });
        }
    }

    Ok(())
}

pub fn get_device_properties(device: i32) -> Result<DeviceProp> {
    unsafe {
        let mut prop = std::mem::MaybeUninit::<sys::cudaDeviceProp>::uninit();
        let res = sys::cudaGetDeviceProperties(prop.as_mut_ptr(), device);
        if res != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(Error::CouldNotGetDeviceProperties {
                cerr: res.into(),
                device,
            });
        }
        let prop = prop.assume_init();
        Ok(DeviceProp { prop })
    }
}

pub fn device_synchronize() -> Result<()> {
    let res = unsafe {
        sys::cudaDeviceSynchronize();
        sys::cudaGetLastError()
    };
    if res != sys::cudaError_enum::CUDA_SUCCESS {
        Err(Error::DeviceSyncFailed { cerr: res.into() })
    } else {
        Ok(())
    }
}

pub struct DeviceProp {
    prop: sys::cudaDeviceProp,
}

impl DeviceProp {
    pub fn name(&self) -> String {
        unsafe {
            CStr::from_ptr(self.prop.name.as_ptr() as *const i8)
                .to_string_lossy()
                .into_owned()
        }
    }

    pub fn total_global_mem(&self) -> usize {
        self.prop.totalGlobalMem
    }

    pub fn shared_mem_per_block(&self) -> usize {
        self.prop.sharedMemPerBlock
    }

    pub fn registers_per_block(&self) -> i32 {
        self.prop.regsPerBlock
    }

    pub fn warp_size(&self) -> i32 {
        self.prop.warpSize
    }

    pub fn mem_pitch(&self) -> usize {
        self.prop.memPitch
    }

    pub fn max_threads_per_block(&self) -> i32 {
        self.prop.maxThreadsPerBlock
    }

    pub fn max_threads_dim(&self) -> &[i32; 3] {
        &self.prop.maxThreadsDim
    }

    pub fn max_grid_size(&self) -> &[i32; 3] {
        &self.prop.maxGridSize
    }

    pub fn clock_rate(&self) -> i32 {
        self.prop.clockRate
    }

    pub fn total_const_mem(&self) -> usize {
        self.prop.totalConstMem
    }

    pub fn major(&self) -> i32 {
        self.prop.major
    }

    pub fn minor(&self) -> i32 {
        self.prop.minor
    }

    pub fn texture_alignment(&self) -> usize {
        self.prop.textureAlignment
    }

    pub fn texture_pitch_alignment(&self) -> usize {
        self.prop.texturePitchAlignment
    }

    pub fn multi_processor_count(&self) -> i32 {
        self.prop.multiProcessorCount
    }

    pub fn kernel_exec_timeout_enabled(&self) -> bool {
        self.prop.kernelExecTimeoutEnabled != 0
    }

    pub fn integrated(&self) -> bool {
        self.prop.integrated != 0
    }

    pub fn can_map_host_memory(&self) -> bool {
        return self.prop.canMapHostMemory != 0;
    }

    pub fn compute_mode(&self) -> ComputeMode {
        match self.prop.computeMode as u32 {
            sys::cudaComputeMode_cudaComputeModeDefault => ComputeMode::Default,
            sys::cudaComputeMode_cudaComputeModeExclusive => {
                ComputeMode::Exclusive
            }
            sys::cudaComputeMode_cudaComputeModeProhibited => {
                ComputeMode::Prohibited
            }
            sys::cudaComputeMode_cudaComputeModeExclusiveProcess => {
                ComputeMode::ExclusiveProcess
            }
            _ => unreachable!(),
        }
    }

    pub fn max_texture_1d(&self) -> i32 {
        self.prop.maxTexture1D
    }

    pub fn max_texture_1d_mipmap(&self) -> i32 {
        self.prop.maxTexture1DMipmap
    }

    pub fn max_texture_1d_linear(&self) -> i32 {
        self.prop.maxTexture1DLinear
    }

    pub fn max_texture_2d(&self) -> &[i32; 2] {
        &self.prop.maxTexture2D
    }

    pub fn max_texture_2d_mipmap(&self) -> &[i32; 2] {
        &self.prop.maxTexture2DMipmap
    }

    pub fn max_texture_2d_linear(&self) -> &[i32; 3] {
        &self.prop.maxTexture2DLinear
    }

    pub fn max_texture_2d_gather(&self) -> &[i32; 2] {
        &self.prop.maxTexture2DGather
    }

    pub fn max_texture_3d(&self) -> &[i32; 3] {
        &self.prop.maxTexture3D
    }

    pub fn max_texture_3d_alt(&self) -> &[i32; 3] {
        &self.prop.maxTexture3DAlt
    }

    pub fn max_texture_cubemap(&self) -> i32 {
        self.prop.maxTextureCubemap
    }

    pub fn max_texture_1d_layered(&self) -> &[i32; 2] {
        &self.prop.maxTexture1DLayered
    }

    pub fn max_texture_2d_layered(&self) -> &[i32; 3] {
        &self.prop.maxTexture2DLayered
    }

    pub fn max_texture_cubemap_layered(&self) -> &[i32; 2] {
        &self.prop.maxTextureCubemapLayered
    }

    pub fn max_surface_1d(&self) -> i32 {
        self.prop.maxSurface1D
    }

    pub fn max_surface_2d(&self) -> &[i32; 2] {
        &self.prop.maxSurface2D
    }

    pub fn max_surface_3d(&self) -> &[i32; 3] {
        &self.prop.maxSurface3D
    }

    pub fn max_surface_1d_layered(&self) -> &[i32; 2] {
        &self.prop.maxSurface1DLayered
    }

    pub fn max_surface_2d_layered(&self) -> &[i32; 3] {
        &self.prop.maxSurface2DLayered
    }

    pub fn max_surface_cubemap(&self) -> i32 {
        self.prop.maxSurfaceCubemap
    }

    pub fn max_surface_cubemap_layered(&self) -> &[i32; 2] {
        &self.prop.maxSurfaceCubemapLayered
    }

    pub fn surface_alignment(&self) -> usize {
        self.prop.surfaceAlignment
    }
}

#[derive(Copy, Clone, Debug, Display)]
pub enum ComputeMode {
    #[display(
        fmt = "< Default compute mode (Multiple threads can use cuda::set_device() with this device)"
    )]
    Default,
    #[display(
        fmt = "< Compute-exclusive-thread mode (Only one thread in one process will be able to use cuda::set_device() with this device)"
    )]
    Exclusive,
    #[display(
        fmt = "< Compute-prohibited mode (No threads can use cuda::set_device() with this device)"
    )]
    Prohibited,
    #[display(
        fmt = "< Compute-exclusive-process mode (Many threads in one process will be able to use cuda::set_device() with this device)"
    )]
    ExclusiveProcess,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
