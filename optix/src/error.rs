use optix_sys as sys;

use super::BufferFormat;

#[derive(Display, Debug)]
pub enum Error {
    #[display(fmt = "OptiX initialization failed")]
    InitializationFailed { cerr: sys::Error },
    #[display(fmt = "Failed to create OptiX device context")]
    DeviceContextCreateFailed { cerr: sys::Error },
    #[display(fmt = "A device context method failed")]
    DeviceContextMethodFailed { cerr: sys::Error },
    #[display(fmt = "Failed to set disk cache path '{}'", "path.display()")]
    SetCacheLocationFailed {
        cerr: sys::Error,
        path: std::path::PathBuf,
    },
    #[display(fmt = "Module creation failed:\n{}", log)]
    ModuleCreationFailed { cerr: sys::Error, log: String },
    #[display(fmt = "ProgramGroup creation failed:\n{}", log)]
    ProgramGroupCreationFailed { cerr: sys::Error, log: String },
    #[display(fmt = "Pipeline creation failed:\n{}", log)]
    PipelineCreationFailed { cerr: sys::Error, log: String },
    #[display(fmt = "OptiX launch failed")]
    LaunchFailed { cerr: sys::Error },
    #[display(fmt = "CUDA error")]
    CudaError { cerr: cuda::Error },
    #[display(fmt = "Incorrect vertex buffer format: {:?}", format)]
    IncorrectVertexBufferFormat { format: super::BufferFormat },
    #[display(fmt = "Incorrect index buffer format: {:?}", format)]
    IncorrectIndexBufferFormat { format: super::BufferFormat },
    #[display(fmt = "Failed to compute accel memory usage")]
    AccelComputeMemoryUsageFailed { cerr: sys::Error },
    #[display(fmt = "Failed to build accel")]
    AccelBuildFailed { cerr: sys::Error },
    #[display(fmt = "Failed to compact accel")]
    AccelCompactFailed { cerr: sys::Error },
    #[display(
        fmt = "Buffer shape mismatch. Expected {:?}x{}",
        e_format,
        e_count
    )]
    BufferShapeMismatch {
        e_format: BufferFormat,
        e_count: usize,
    },
}

impl From<cuda::Error> for Error {
    fn from(e: cuda::Error) -> Error {
        Error::CudaError { cerr: e }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::InitializationFailed { cerr, .. } => Some(cerr),
            Error::DeviceContextCreateFailed { cerr, .. } => Some(cerr),
            Error::DeviceContextMethodFailed { cerr, .. } => Some(cerr),
            Error::SetCacheLocationFailed { cerr, .. } => Some(cerr),
            Error::ModuleCreationFailed { cerr, .. } => Some(cerr),
            Error::ProgramGroupCreationFailed { cerr, .. } => Some(cerr),
            Error::PipelineCreationFailed { cerr, .. } => Some(cerr),
            Error::LaunchFailed { cerr, .. } => Some(cerr),
            Error::CudaError { cerr, .. } => Some(cerr),
            Error::AccelComputeMemoryUsageFailed { cerr, .. } => Some(cerr),
            Error::AccelBuildFailed { cerr, .. } => Some(cerr),
            Error::AccelCompactFailed { cerr, .. } => Some(cerr),
            _ => None,
        }
    }
}
