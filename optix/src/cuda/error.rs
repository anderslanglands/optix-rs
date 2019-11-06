use optix_sys::cuda_sys::Error as CudaError;

#[derive(Display, Debug)]
pub enum Error {
    #[display(fmt = "Buffer allocation of size {} bytes failed.", size)]
    BufferAllocationFailed { cerr: CudaError, size: usize },
    #[display(
        fmt = "Buffer allocation of {:?}, {}x{} flags: {:?}",
        desc,
        width,
        height,
        flags
    )]
    ArrayAllocationFailed {
        cerr: CudaError,
        desc: super::ChannelFormatDesc,
        width: usize,
        height: usize,
        flags: super::ArrayFlags,
    },
    #[display(fmt = "Array memcpy 2d failed.")]
    ArrayMemcpy2DFailed { cerr: CudaError },
    #[display(fmt = "Buffer upload failed.")]
    BufferUploadFailed { cerr: CudaError },
    #[display(fmt = "Buffer download failed.")]
    BufferDownloadFailed { cerr: CudaError },
    #[display(
        fmt = "Tried to upload {} bytes of data to a buffer of {} bytes",
        upload_size,
        buffer_size
    )]
    BufferUploadWrongSize {
        upload_size: usize,
        buffer_size: usize,
    },
    #[display(
        fmt = "Tried to download {} bytes of data from a buffer of {} bytes",
        download_size,
        buffer_size
    )]
    BufferDownloadWrongSize {
        download_size: usize,
        buffer_size: usize,
    },
    #[display(fmt = "Could not set device {}", device)]
    CouldNotSetDevice { cerr: CudaError, device: i32 },
    #[display(fmt = "Failed to create stream")]
    StreamCreationFailed { cerr: CudaError },
    #[display(fmt = "Could not get device {} properties", device)]
    CouldNotGetDeviceProperties { cerr: CudaError, device: i32 },
    #[display(fmt = "Could not get current context")]
    CouldNotGetCurrentContext { cerr: CudaError },
    #[display(fmt = "Device sync failed")]
    DeviceSyncFailed { cerr: CudaError },
    #[display(fmt = "Texture object creation failed")]
    TextureObjectCreationFailed { cerr: CudaError },
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::BufferAllocationFailed { cerr, .. } => Some(cerr),
            Error::BufferUploadFailed { cerr, .. } => Some(cerr),
            Error::CouldNotSetDevice { cerr, .. } => Some(cerr),
            Error::StreamCreationFailed { cerr, .. } => Some(cerr),
            _ => None,
        }
    }
}
