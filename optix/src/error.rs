use enum_primitive_derive::Primitive;
use num_traits::FromPrimitive;

use crate::sys;

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("OptiX initialization failed")]
    Initialization { source: OptixError },
    #[error("Failed to create device context")]
    DeviceContextCreation { source: OptixError },
    #[error("Failed to set cache database sizes")]
    DeviceContextSetCacheDatabaseSizes { source: OptixError },
    #[error("Failed to enable compilation cache")]
    DeviceContextSetCacheEnabled { source: OptixError },
    #[error("Failed to set cache location")]
    DeviceContextSetCacheLocation { source: OptixError },
    #[error("pipeline_launch_params_variable_name must be specified on PipelineCompileOptions")]
    PipelineLaunchParamsVariableNameNotSpecified,
    #[error("Failed to create module")]
    ModuleCreation { source: OptixError, log: String },
    #[error("Failed to get builtin IS module")]
    BuiltinIsModuleGet { source: OptixError },
    #[error("Failed to destroy module")]
    ModuleDestroy { source: OptixError },
    #[error("Failed to create program group")]
    ProgramGroupCreation { source: OptixError, log: String },
    #[error("Failed to destroy program group")]
    ProgramGroupDestroy { source: OptixError },
    #[error("Failed to get program group stack sizes")]
    ProgramGroupGetStackSizes { source: OptixError },
    #[error("Failed to create pipelin")]
    PipelineCreationFailed { source: OptixError, log: String },
    #[error("Failed to set pipeline stack size")]
    PipelineSetStackSize { source: OptixError },
    #[error("Failed to pack SBT record")]
    PackSbtRecord { source: OptixError },
    #[error(
        "Wrong number of raygen records supplied. Expected 1, got: {len:}"
    )]
    WrongRaygenRecordLen { len: usize },
    #[error("Launch failed")]
    Launch { source: OptixError },
    #[error("Failed to compute accel memory usage")]
    AccelComputeMemoryUsage { source: OptixError },
    #[error("Failed to build acceleration structure")]
    AccelBuild { source: OptixError },
    #[error("Failed to compact accel structure")]
    AccelCompact { source: OptixError },
    #[error("Allocation failed")]
    Allocation { source: cu::Error },
    #[error("Memcpy failed")]
    Memcpy { source: cu::Error },
    #[error("Deallocation failed")]
    Deallocation { source: cu::Error },
    #[error("Failed to destroy pipeline object")]
    PipelineDestroy { source: OptixError },
    #[error("Failed to create denoiser")]
    CreateDenoiser { source: OptixError },
    #[error("Failed to destroy denoiser")]
    DestroyDenoiser { source: OptixError },
    #[error("Failed to compute intensity")]
    DenoiserComputeIntensity { source: OptixError },
    #[error("Failed to compute denoiser memory resources")]
    DenoiserComputeMemoryResources { source: OptixError },
    #[error("Failed to invoke denoiser")]
    DenoiserInvoke { source: OptixError },
    #[error("Failed to set LDR denoiser model")]
    DenoiserSetModelLdr { source: OptixError },
    #[error("Failed to setup denoiser state")]
    DenoiserSetup { source: OptixError },
}

impl sys::OptixResult {
    pub fn to_result(&self) -> Result<(), OptixError> {
        if *self == sys::OptixResult::OPTIX_SUCCESS {
            return Ok(());
        }

        let v = self.0 as u32;

        if let Some(e) = OptixError::from_u32(v) {
            Err(e)
        } else {
            panic!("OptiX returned an unhandled error code: {}", v)
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Primitive)]
pub enum OptixError {
    InvalidValue = 7001,
    HostOutOfMemory = 7002,
    InvalidOperation = 7003,
    FileIoError = 7004,
    InvalidFileFormat = 7005,
    DiskCacheInvalidPath = 7010,
    DiskCachePermissionError = 7011,
    DiskCacheDatabaseError = 7012,
    DiskCacheInvalidData = 7013,
    LaunchFailure = 7050,
    InvalidDeviceContext = 7051,
    CudaNotInitialized = 7052,
    InvalidPtx = 7200,
    InvalidLaunchParameter = 7201,
    InvalidPayloadAccess = 7202,
    InvalidAttributeAccess = 7203,
    InvalidFunctionUse = 7204,
    InvalidFunctionArguments = 7205,
    PipelineOutOfConstantMemory = 7250,
    PipelineLinkError = 7251,
    InternalCompilerError = 7299,
    DenoiserModelNotSet = 7300,
    DenoiserNotInitialized = 7301,
    AccelNotCompatible = 7400,
    NotSupported = 7800,
    UnsupportedAbiVersion = 7801,
    FunctionTableSizeMismatch = 7802,
    InvalidEntryFunctionOptions = 7803,
    LibraryNotFound = 7804,
    EntrySymbolNotFound = 7805,
    CudaError = 7900,
    InternalError = 7990,
    Unknown = 7999,
}

use std::fmt;
impl fmt::Display for OptixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for OptixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
