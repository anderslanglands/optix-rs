use optix_sys::cuda_sys::Error as CudaError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("allocation of size {size:} bytes failed.")]
    AllocationFailed { source: CudaError, size: usize },
    #[error("allocation of size {size:} bytes failed as could not satisfy alignment of {align:} bytes.")]
    AllocationAlignment { size: usize, align: usize },
    #[error("Tried to allocate zero bytes")]
    ZeroAllocation,
    #[error("Buffer allocation of size {size:} bytes failed.")]
    BufferAllocationFailed { source: CudaError, size: usize },
    #[error(
        "Array allocation of {desc:?}, {width:}x{height:} flags: {flags:?}"
    )]
    ArrayAllocationFailed {
        source: CudaError,
        desc: super::ChannelFormatDesc,
        width: usize,
        height: usize,
        num_components: usize,
        flags: super::ArrayFlags,
    },
    #[error(
        "3D Array allocation of {desc:?}, {width:}x{height:}x{depth:} flags: {flags:?}"
    )]
    Array3DAllocationFailed {
        source: CudaError,
        desc: super::ChannelFormatDesc,
        width: usize,
        height: usize,
        depth: usize,
        num_components: usize,
        flags: super::ArrayFlags,
    },
    #[error("Array memcpy 2d failed.")]
    ArrayMemcpy2DFailed { source: CudaError },
    #[error("Array memcpy 3d failed.")]
    ArrayMemcpy3DFailed { source: CudaError },
    #[error("Buffer upload failed.")]
    BufferUploadFailed { source: CudaError },
    #[error("Buffer download failed.")]
    BufferDownloadFailed { source: CudaError },
    #[error(
        "Tried to upload {upload_size:} bytes of data to a buffer of {buffer_size:} bytes"
    )]
    BufferUploadWrongSize {
        upload_size: usize,
        buffer_size: usize,
    },
    #[error(
        "Tried to download {download_size:} bytes of data from a buffer of {buffer_size:} bytes"
    )]
    BufferDownloadWrongSize {
        download_size: usize,
        buffer_size: usize,
    },
    #[error("Could not set device {device:}")]
    CouldNotSetDevice { source: CudaError, device: i32 },
    #[error("Failed to create stream")]
    StreamCreationFailed { source: CudaError },
    #[error("Could not get device {device:} properties")]
    CouldNotGetDeviceProperties { source: CudaError, device: i32 },
    #[error("Could not get current context")]
    CouldNotGetCurrentContext { source: CudaError },
    #[error("Device sync failed")]
    DeviceSyncFailed { source: CudaError },
    #[error("Texture object creation failed")]
    TextureObjectCreationFailed { source: CudaError },
    #[error("Could not get mem info")]
    GetMemInfoFailed { source: CudaError },
}
