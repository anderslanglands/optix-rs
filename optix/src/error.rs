use super::cuda;
use optix_sys as sys;

use super::BufferFormat;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("OptiX initialization failed")]
    InitializationFailed { source: sys::Error },
    #[error("Failed to create OptiX device context")]
    DeviceContextCreateFailed { source: sys::Error },
    #[error("A device context method failed")]
    DeviceContextMethodFailed { source: sys::Error },
    #[error("Failed to set disk cache path '{}'", "path.display()")]
    SetCacheLocationFailed {
        source: sys::Error,
        path: std::path::PathBuf,
    },
    #[error("Module creation failed:\n{log:}")]
    ModuleCreationFailed { source: sys::Error, log: String },
    #[error("ProgramGroup creation failed:\n{log:}")]
    ProgramGroupCreationFailed { source: sys::Error, log: String },
    #[error("Pipeline creation failed:\n{log:}")]
    PipelineCreationFailed { source: sys::Error, log: String },
    #[error("OptiX launch failed")]
    LaunchFailed { source: sys::Error },
    #[error("CUDA error")]
    CudaError {
        #[from]
        source: cuda::Error,
    },
    #[error("Incorrect vertex buffer format: {format:?}")]
    IncorrectVertexBufferFormat { format: super::BufferFormat },
    #[error("Incorrect index buffer format: {format:?}")]
    IncorrectIndexBufferFormat { format: super::BufferFormat },
    #[error("Failed to compute accel memory usage")]
    AccelComputeMemoryUsageFailed { source: sys::Error },
    #[error("Failed to build accel")]
    AccelBuildFailed { source: sys::Error },
    #[error("Failed to compact accel")]
    AccelCompactFailed { source: sys::Error },
    #[error("Buffer shape mismatch. Expected {e_format:?}x{e_count:}")]
    BufferShapeMismatch {
        e_format: BufferFormat,
        e_count: usize,
    },
}
