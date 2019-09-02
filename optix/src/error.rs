use optix_sys as sys;

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
}

impl From<cuda::Error> for Error {
    fn from(e: cuda::Error) -> Error {
        Error::CudaError { cerr: e }
    }
}
