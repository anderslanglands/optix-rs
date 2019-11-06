include!(concat!(env!("OUT_DIR"), "/cuda_wrapper.rs"));

#[repr(C)]
pub union cudaResourceDescUnion {
    pub array: cudaResourceDescUnionArray,
    pub mipmap: cudaResourceDescUnionMipmap,
    pub linear: cudaResourceDescUnionLinear,
    pub pitch2D: cudaResourceDescUnionPitch2D,
}

#[repr(C)]
pub struct cudaResourceDescUnionArray {
    pub array: cudaArray_t,
}

#[repr(C)]
pub struct cudaResourceDescUnionMipmap {
    pub mipmap: cudaMipmappedArray_t,
}

#[repr(C)]
pub struct cudaResourceDescUnionLinear {
    pub devPtr: *mut std::os::raw::c_void,
    pub desc: cudaChannelFormatDesc,
    pub sizeInBytes: usize,
}

#[repr(C)]
pub struct cudaResourceDescUnionPitch2D {
    pub devPtr: *mut std::os::raw::c_void,
    pub desc: cudaChannelFormatDesc,
    pub width: usize,
    pub height: usize,
    pub pitchInBytes: usize,
}

#[repr(C)]
pub struct cudaResourceDesc {
    pub resType: cudaResourceType,
    pub res: cudaResourceDescUnion,
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<cudaError::Type> for Error {
    fn from(e: cudaError::Type) -> Error {
        match e {
            cudaError::cudaErrorInvalidValue => Error::InvalidValue,
            cudaError::cudaErrorMemoryAllocation => Error::MemoryAllocation,
            cudaError::cudaErrorInitializationError => {
                Error::InitializationError
            }
            cudaError::cudaErrorCudartUnloading => Error::CudartUnloading,
            cudaError::cudaErrorProfilerDisabled => Error::ProfilerDisabled,
            cudaError::cudaErrorProfilerNotInitialized => {
                Error::ProfilerNotInitialized
            }
            cudaError::cudaErrorProfilerAlreadyStarted => {
                Error::ProfilerAlreadyStarted
            }
            cudaError::cudaErrorProfilerAlreadyStopped => {
                Error::ProfilerAlreadyStopped
            }
            cudaError::cudaErrorInvalidConfiguration => {
                Error::InvalidConfiguration
            }
            cudaError::cudaErrorInvalidPitchValue => Error::InvalidPitchValue,
            cudaError::cudaErrorInvalidSymbol => Error::InvalidSymbol,
            cudaError::cudaErrorInvalidHostPointer => Error::InvalidHostPointer,
            cudaError::cudaErrorInvalidDevicePointer => {
                Error::InvalidDevicePointer
            }
            cudaError::cudaErrorInvalidTexture => Error::InvalidTexture,
            cudaError::cudaErrorInvalidTextureBinding => {
                Error::InvalidTextureBinding
            }
            cudaError::cudaErrorInvalidChannelDescriptor => {
                Error::InvalidChannelDescriptor
            }
            cudaError::cudaErrorInvalidMemcpyDirection => {
                Error::InvalidMemcpyDirection
            }
            cudaError::cudaErrorAddressOfConstant => Error::AddressOfConstant,
            cudaError::cudaErrorTextureFetchFailed => Error::TextureFetchFailed,
            cudaError::cudaErrorTextureNotBound => Error::TextureNotBound,
            cudaError::cudaErrorSynchronizationError => {
                Error::SynchronizationError
            }
            cudaError::cudaErrorInvalidFilterSetting => {
                Error::InvalidFilterSetting
            }

            cudaError::cudaErrorInvalidNormSetting => Error::InvalidNormSetting,

            cudaError::cudaErrorMixedDeviceExecution => {
                Error::MixedDeviceExecution
            }

            cudaError::cudaErrorNotYetImplemented => Error::NotYetImplemented,

            cudaError::cudaErrorMemoryValueTooLarge => {
                Error::MemoryValueTooLarge
            }

            cudaError::cudaErrorInsufficientDriver => Error::InsufficientDriver,

            cudaError::cudaErrorInvalidSurface => Error::InvalidSurface,

            cudaError::cudaErrorDuplicateVariableName => {
                Error::DuplicateVariableName
            }

            cudaError::cudaErrorDuplicateTextureName => {
                Error::DuplicateTextureName
            }

            cudaError::cudaErrorDuplicateSurfaceName => {
                Error::DuplicateSurfaceName
            }

            cudaError::cudaErrorDevicesUnavailable => Error::DevicesUnavailable,

            cudaError::cudaErrorIncompatibleDriverContext => {
                Error::IncompatibleDriverContext
            }

            cudaError::cudaErrorMissingConfiguration => {
                Error::MissingConfiguration
            }

            cudaError::cudaErrorPriorLaunchFailure => Error::PriorLaunchFailure,

            cudaError::cudaErrorLaunchMaxDepthExceeded => {
                Error::LaunchMaxDepthExceeded
            }

            cudaError::cudaErrorLaunchFileScopedTex => {
                Error::LaunchFileScopedTex
            }

            cudaError::cudaErrorLaunchFileScopedSurf => {
                Error::LaunchFileScopedSurf
            }

            cudaError::cudaErrorSyncDepthExceeded => Error::SyncDepthExceeded,

            cudaError::cudaErrorLaunchPendingCountExceeded => {
                Error::LaunchPendingCountExceeded
            }

            cudaError::cudaErrorInvalidDeviceFunction => {
                Error::InvalidDeviceFunction
            }

            cudaError::cudaErrorNoDevice => Error::NoDevice,

            cudaError::cudaErrorInvalidDevice => Error::InvalidDevice,

            cudaError::cudaErrorStartupFailure => Error::StartupFailure,

            cudaError::cudaErrorInvalidKernelImage => Error::InvalidKernelImage,

            cudaError::cudaErrorDeviceUninitilialized => {
                Error::DeviceUninitilialized
            }

            cudaError::cudaErrorMapBufferObjectFailed => {
                Error::MapBufferObjectFailed
            }

            cudaError::cudaErrorUnmapBufferObjectFailed => {
                Error::UnmapBufferObjectFailed
            }

            cudaError::cudaErrorArrayIsMapped => Error::ArrayIsMapped,

            cudaError::cudaErrorAlreadyMapped => Error::AlreadyMapped,

            cudaError::cudaErrorNoKernelImageForDevice => {
                Error::NoKernelImageForDevice
            }

            cudaError::cudaErrorAlreadyAcquired => Error::AlreadyAcquired,

            cudaError::cudaErrorNotMapped => Error::NotMapped,

            cudaError::cudaErrorNotMappedAsArray => Error::NotMappedAsArray,

            cudaError::cudaErrorNotMappedAsPointer => Error::NotMappedAsPointer,

            cudaError::cudaErrorECCUncorrectable => Error::ECCUncorrectable,

            cudaError::cudaErrorUnsupportedLimit => Error::UnsupportedLimit,

            cudaError::cudaErrorDeviceAlreadyInUse => Error::DeviceAlreadyInUse,

            cudaError::cudaErrorPeerAccessUnsupported => {
                Error::PeerAccessUnsupported
            }

            cudaError::cudaErrorInvalidPtx => Error::InvalidPtx,

            cudaError::cudaErrorInvalidGraphicsContext => {
                Error::InvalidGraphicsContext
            }

            cudaError::cudaErrorNvlinkUncorrectable => {
                Error::NvlinkUncorrectable
            }

            cudaError::cudaErrorJitCompilerNotFound => {
                Error::JitCompilerNotFound
            }

            cudaError::cudaErrorInvalidSource => Error::InvalidSource,

            cudaError::cudaErrorFileNotFound => Error::FileNotFound,

            cudaError::cudaErrorSharedObjectSymbolNotFound => {
                Error::SharedObjectSymbolNotFound
            }

            cudaError::cudaErrorSharedObjectInitFailed => {
                Error::SharedObjectInitFailed
            }

            cudaError::cudaErrorOperatingSystem => Error::OperatingSystem,

            cudaError::cudaErrorInvalidResourceHandle => {
                Error::InvalidResourceHandle
            }

            cudaError::cudaErrorIllegalState => Error::IllegalState,

            cudaError::cudaErrorSymbolNotFound => Error::SymbolNotFound,

            cudaError::cudaErrorNotReady => Error::NotReady,

            cudaError::cudaErrorIllegalAddress => Error::IllegalAddress,

            cudaError::cudaErrorLaunchOutOfResources => {
                Error::LaunchOutOfResources
            }

            cudaError::cudaErrorLaunchTimeout => Error::LaunchTimeout,

            cudaError::cudaErrorLaunchIncompatibleTexturing => {
                Error::LaunchIncompatibleTexturing
            }

            cudaError::cudaErrorPeerAccessAlreadyEnabled => {
                Error::PeerAccessAlreadyEnabled
            }

            cudaError::cudaErrorPeerAccessNotEnabled => {
                Error::PeerAccessNotEnabled
            }

            cudaError::cudaErrorSetOnActiveProcess => Error::SetOnActiveProcess,

            cudaError::cudaErrorContextIsDestroyed => Error::ContextIsDestroyed,

            cudaError::cudaErrorAssert => Error::Assert,

            cudaError::cudaErrorTooManyPeers => Error::TooManyPeers,

            cudaError::cudaErrorHostMemoryAlreadyRegistered => {
                Error::HostMemoryAlreadyRegistered
            }

            cudaError::cudaErrorHostMemoryNotRegistered => {
                Error::HostMemoryNotRegistered
            }

            cudaError::cudaErrorHardwareStackError => Error::HardwareStackError,

            cudaError::cudaErrorIllegalInstruction => Error::IllegalInstruction,

            cudaError::cudaErrorMisalignedAddress => Error::MisalignedAddress,

            cudaError::cudaErrorInvalidAddressSpace => {
                Error::InvalidAddressSpace
            }

            cudaError::cudaErrorInvalidPc => Error::InvalidPc,

            cudaError::cudaErrorLaunchFailure => Error::LaunchFailure,

            cudaError::cudaErrorCooperativeLaunchTooLarge => {
                Error::CooperativeLaunchTooLarge
            }

            cudaError::cudaErrorNotPermitted => Error::NotPermitted,

            cudaError::cudaErrorNotSupported => Error::NotSupported,

            cudaError::cudaErrorSystemNotReady => Error::SystemNotReady,

            cudaError::cudaErrorSystemDriverMismatch => {
                Error::SystemDriverMismatch
            }

            cudaError::cudaErrorCompatNotSupportedOnDevice => {
                Error::CompatNotSupportedOnDevice
            }

            cudaError::cudaErrorStreamCaptureUnsupported => {
                Error::StreamCaptureUnsupported
            }

            cudaError::cudaErrorStreamCaptureInvalidated => {
                Error::StreamCaptureInvalidated
            }

            cudaError::cudaErrorStreamCaptureMerge => Error::StreamCaptureMerge,

            cudaError::cudaErrorStreamCaptureUnmatched => {
                Error::StreamCaptureUnmatched
            }

            cudaError::cudaErrorStreamCaptureUnjoined => {
                Error::StreamCaptureUnjoined
            }

            cudaError::cudaErrorStreamCaptureIsolation => {
                Error::StreamCaptureIsolation
            }

            cudaError::cudaErrorStreamCaptureImplicit => {
                Error::StreamCaptureImplicit
            }

            cudaError::cudaErrorCapturedEvent => Error::CapturedEvent,

            cudaError::cudaErrorStreamCaptureWrongThread => {
                Error::StreamCaptureWrongThread
            }

            cudaError::cudaErrorUnknown => Error::Unknown,

            cudaError::cudaErrorApiFailureBase => Error::ApiFailureBase,

            _ => unreachable!(),
        }
    }
}

#[derive(Display, Debug, Copy, Clone)]
pub enum Error {
    #[display(
        fmt = "This indicates that one or more of the parameters passed to the API call  is not within an acceptable range of values."
    )]
    InvalidValue,
    #[display(
        fmt = "The API call failed because it was unable to allocate enough memory to  perform the requested operation."
    )]
    MemoryAllocation,
    #[display(
        fmt = "The API call failed because the CUDA driver and runtime could not be  initialized."
    )]
    InitializationError,
    #[display(
        fmt = "This indicates that a CUDA Runtime API call cannot be executed because  it is being called during process shut down, at a point in time after  CUDA driver has been unloaded."
    )]
    CudartUnloading,
    #[display(
        fmt = "This indicates profiler is not initialized for this run. This can  happen when the application is running with external profiling tools  like visual profiler."
    )]
    ProfilerDisabled,
    #[display(
        fmt = "deprecated  This error return is deprecated as of CUDA 5.0. It is no longer an error  to attempt to enablegdisable the profiling via ::cudaProfilerStart or  ::cudaProfilerStop without initialization."
    )]
    ProfilerNotInitialized,
    #[display(
        fmt = "deprecated  This error return is deprecated as of CUDA 5.0. It is no longer an error  to call cudaProfilerStart() when profiling is already enabled."
    )]
    ProfilerAlreadyStarted,
    #[display(
        fmt = "deprecated  This error return is deprecated as of CUDA 5.0. It is no longer an error  to call cudaProfilerStop() when profiling is already disabled."
    )]
    ProfilerAlreadyStopped,
    #[display(
        fmt = "This indicates that a kernel launch is requesting resources that can  never be satisfied by the current device. Requesting more shared memory  per block than the device supports will trigger this error, as will  requesting too many threads or blocks. See ::cudaDeviceProp for more  device limitations."
    )]
    InvalidConfiguration,
    #[display(
        fmt = "This indicates that one or more of the pitch-related parameters passed  to the API call is not within the acceptable range for pitch."
    )]
    InvalidPitchValue,
    #[display(
        fmt = "This indicates that the symbol namegidentifier passed to the API call  is not a valid name or identifier."
    )]
    InvalidSymbol,
    #[display(
        fmt = "This indicates that at least one host pointer passed to the API call is  not a valid host pointer.  deprecated  This error return is deprecated as of CUDA 10.1."
    )]
    InvalidHostPointer,
    #[display(
        fmt = "This indicates that at least one device pointer passed to the API call is  not a valid device pointer.  deprecated  This error return is deprecated as of CUDA 10.1."
    )]
    InvalidDevicePointer,
    #[display(
        fmt = "This indicates that the texture passed to the API call is not a valid  texture."
    )]
    InvalidTexture,
    #[display(
        fmt = "This indicates that the texture binding is not valid. This occurs if you  call ::cudaGetTextureAlignmentOffset() with an unbound texture."
    )]
    InvalidTextureBinding,
    #[display(
        fmt = "This indicates that the channel descriptor passed to the API call is not  valid. This occurs if the format is not one of the formats specified by  ::cudaChannelFormatKind, or if one of the dimensions is invalid."
    )]
    InvalidChannelDescriptor,
    #[display(
        fmt = "This indicates that the direction of the memcpy passed to the API call is  not one of the types specified by ::cudaMemcpyKind."
    )]
    InvalidMemcpyDirection,
    #[display(
        fmt = "This indicated that the user has taken the address of a constant variable,  which was forbidden up until the CUDA 3.1 release.  deprecated  This error return is deprecated as of CUDA 3.1. Variables in constant  memory may now have their address taken by the runtime via  ::cudaGetSymbolAddress(). "
    )]
    AddressOfConstant,
    #[display(
        fmt = "This indicated that a texture fetch was not able to be performed.  This was previously used for device emulation of texture operations.  deprecated  This error return is deprecated as of CUDA 3.1. Device emulation mode was  removed with the CUDA 3.1 release."
    )]
    TextureFetchFailed,
    #[display(
        fmt = "This indicated that a texture was not bound for access.  This was previously used for device emulation of texture operations.  deprecated  This error return is deprecated as of CUDA 3.1. Device emulation mode was  removed with the CUDA 3.1 release."
    )]
    TextureNotBound,
    #[display(
        fmt = "This indicated that a synchronization operation had failed.  This was previously used for some device emulation functions.  deprecated  This error return is deprecated as of CUDA 3.1. Device emulation mode was  removed with the CUDA 3.1 release."
    )]
    SynchronizationError,
    #[display(
        fmt = "This indicates that a non-float texture was being accessed with linear  filtering. This is not supported by CUDA."
    )]
    InvalidFilterSetting,
    #[display(
        fmt = "This indicates that an attempt was made to read a non-float texture as a  normalized float. This is not supported by CUDA."
    )]
    InvalidNormSetting,
    #[display(
        fmt = "Mixing of device and device emulation code was not allowed.  deprecated  This error return is deprecated as of CUDA 3.1. Device emulation mode was  removed with the CUDA 3.1 release."
    )]
    MixedDeviceExecution,
    #[display(
        fmt = "This indicates that the API call is not yet implemented. Production  releases of CUDA will never return this error.  deprecated  This error return is deprecated as of CUDA 4.1."
    )]
    NotYetImplemented,
    #[display(
        fmt = "This indicated that an emulated device pointer exceeded the 32-bit address  range.  deprecated  This error return is deprecated as of CUDA 3.1. Device emulation mode was  removed with the CUDA 3.1 release."
    )]
    MemoryValueTooLarge,
    #[display(
        fmt = "This indicates that the installed NVIDIA CUDA driver is older than the  CUDA runtime library. This is not a supported configuration. Users should  install an updated NVIDIA display driver to allow the application to run."
    )]
    InsufficientDriver,
    #[display(
        fmt = "This indicates that the surface passed to the API call is not a valid  surface."
    )]
    InvalidSurface,
    #[display(
        fmt = "This indicates that multiple global or constant variables (across separate  CUDA source files in the application) share the same string name."
    )]
    DuplicateVariableName,
    #[display(
        fmt = "This indicates that multiple textures (across separate CUDA source  files in the application) share the same string name."
    )]
    DuplicateTextureName,
    #[display(
        fmt = "This indicates that multiple surfaces (across separate CUDA source  files in the application) share the same string name."
    )]
    DuplicateSurfaceName,
    #[display(
        fmt = "This indicates that all CUDA devices are busy or unavailable at the current  time. Devices are often busygunavailable due to use of  ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long  running CUDA kernels have filled up the GPU and are blocking new work  from starting. They can also be unavailable due to memory constraints  on a device that already has active CUDA work being performed."
    )]
    DevicesUnavailable,
    #[display(
        fmt = "This indicates that the current context is not compatible with this  the CUDA Runtime. This can only occur if you are using CUDA  RuntimegDriver interoperability and have created an existing Driver  context using the driver API. The Driver context may be incompatible  either because the Driver context was created using an older version  of the API, because the Runtime API call expects a primary driver  context and the Driver context is not primary, or because the Driver  context has been destroyed. Please see ref CUDART_DRIVER Interactions  with the CUDA Driver API for more information."
    )]
    IncompatibleDriverContext,
    #[display(
        fmt = "The device function being invoked (usually via ::cudaLaunchKernel()) was not  previously configured via the ::cudaConfigureCall() function."
    )]
    MissingConfiguration,
    #[display(
        fmt = "This indicated that a previous kernel launch failed. This was previously  used for device emulation of kernel launches.  deprecated  This error return is deprecated as of CUDA 3.1. Device emulation mode was  removed with the CUDA 3.1 release."
    )]
    PriorLaunchFailure,
    #[display(
        fmt = "This error indicates that a device runtime grid launch did not occur  because the depth of the child grid would exceed the maximum supported  number of nested grid launches."
    )]
    LaunchMaxDepthExceeded,
    #[display(
        fmt = "This error indicates that a grid launch did not occur because the kernel  uses file-scoped textures which are unsupported by the device runtime.  Kernels launched via the device runtime only support textures created with  the Texture Object API's."
    )]
    LaunchFileScopedTex,
    #[display(
        fmt = "This error indicates that a grid launch did not occur because the kernel  uses file-scoped surfaces which are unsupported by the device runtime.  Kernels launched via the device runtime only support surfaces created with  the Surface Object API's."
    )]
    LaunchFileScopedSurf,
    #[display(
        fmt = "This error indicates that a call to ::cudaDeviceSynchronize made from  the device runtime failed because the call was made at grid depth greater  than than either the default (2 levels of grids) or user specified device  limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on  launched grids at a greater depth successfully, the maximum nested  depth at which ::cudaDeviceSynchronize will be called must be specified  with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit  api before the host-side launch of a kernel using the device runtime.  Keep in mind that additional levels of sync depth require the runtime  to reserve large amounts of device memory that cannot be used for  user allocations."
    )]
    SyncDepthExceeded,
    #[display(
        fmt = "This error indicates that a device runtime grid launch failed because  the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.  For this launch to proceed successfully, ::cudaDeviceSetLimit must be  called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher  than the upper bound of outstanding launches that can be issued to the  device runtime. Keep in mind that raising the limit of pending device  runtime launches will require the runtime to reserve device memory that  cannot be used for user allocations."
    )]
    LaunchPendingCountExceeded,
    #[display(
        fmt = "The requested device function does not exist or is not compiled for the  proper device architecture."
    )]
    InvalidDeviceFunction,
    #[display(
        fmt = "This indicates that no CUDA-capable devices were detected by the installed  CUDA driver."
    )]
    NoDevice,
    #[display(
        fmt = "This indicates that the device ordinal supplied by the user does not  correspond to a valid CUDA device."
    )]
    InvalidDevice,
    #[display(
        fmt = "This indicates an internal startup failure in the CUDA runtime."
    )]
    StartupFailure,
    #[display(fmt = "This indicates that the device kernel image is invalid.")]
    InvalidKernelImage,
    #[display(
        fmt = "This most frequently indicates that there is no context bound to the  current thread. This can also be returned if the context passed to an  API call is not a valid handle (such as a context that has had  ::cuCtxDestroy() invoked on it). This can also be returned if a user  mixes different API versions (i.e. 3010 context with 3020 API calls).  See ::cuCtxGetApiVersion() for more details."
    )]
    DeviceUninitilialized,
    #[display(
        fmt = "This indicates that the buffer object could not be mapped."
    )]
    MapBufferObjectFailed,
    #[display(
        fmt = "This indicates that the buffer object could not be unmapped."
    )]
    UnmapBufferObjectFailed,
    #[display(
        fmt = "This indicates that the specified array is currently mapped and thus  cannot be destroyed."
    )]
    ArrayIsMapped,
    #[display(fmt = "This indicates that the resource is already mapped.")]
    AlreadyMapped,
    #[display(
        fmt = "This indicates that there is no kernel image available that is suitable  for the device. This can occur when a user specifies code generation  options for a particular CUDA source file that do not include the  corresponding device configuration."
    )]
    NoKernelImageForDevice,
    #[display(
        fmt = "This indicates that a resource has already been acquired."
    )]
    AlreadyAcquired,
    #[display(fmt = "This indicates that a resource is not mapped.")]
    NotMapped,
    #[display(
        fmt = "This indicates that a mapped resource is not available for access as an  array."
    )]
    NotMappedAsArray,
    #[display(
        fmt = "This indicates that a mapped resource is not available for access as a  pointer."
    )]
    NotMappedAsPointer,
    #[display(
        fmt = "This indicates that an uncorrectable ECC error was detected during  execution."
    )]
    ECCUncorrectable,
    #[display(
        fmt = "This indicates that the ::cudaLimit passed to the API call is not  supported by the active device."
    )]
    UnsupportedLimit,
    #[display(
        fmt = "This indicates that a call tried to access an exclusive-thread device that  is already in use by a different thread."
    )]
    DeviceAlreadyInUse,
    #[display(
        fmt = "This error indicates that P2P access is not supported across the given  devices."
    )]
    PeerAccessUnsupported,
    #[display(
        fmt = "A PTX compilation failed. The runtime may fall back to compiling PTX if  an application does not contain a suitable binary for the current device."
    )]
    InvalidPtx,
    #[display(
        fmt = "This indicates an error with the OpenGL or DirectX context."
    )]
    InvalidGraphicsContext,
    #[display(
        fmt = "This indicates that an uncorrectable NVLink error was detected during the  execution."
    )]
    NvlinkUncorrectable,
    #[display(
        fmt = "This indicates that the PTX JIT compiler library was not found. The JIT Compiler  library is used for PTX compilation. The runtime may fall back to compiling PTX  if an application does not contain a suitable binary for the current device."
    )]
    JitCompilerNotFound,
    #[display(
        fmt = "This indicates that the device kernel source is invalid."
    )]
    InvalidSource,
    #[display(fmt = "This indicates that the file specified was not found.")]
    FileNotFound,
    #[display(
        fmt = "This indicates that a link to a shared object failed to resolve."
    )]
    SharedObjectSymbolNotFound,
    #[display(
        fmt = "This indicates that initialization of a shared object failed."
    )]
    SharedObjectInitFailed,
    #[display(fmt = "This error indicates that an OS call failed.")]
    OperatingSystem,
    #[display(
        fmt = "This indicates that a resource handle passed to the API call was not  valid. Resource handles are opaque types like ::cudaStream_t and  ::cudaEvent_t."
    )]
    InvalidResourceHandle,
    #[display(
        fmt = "This indicates that a resource required by the API call is not in a  valid state to perform the requested operation."
    )]
    IllegalState,
    #[display(
        fmt = "This indicates that a named symbol was not found. Examples of symbols  are globalgconstant variable names, texture names, and surface names."
    )]
    SymbolNotFound,
    #[display(
        fmt = "This indicates that asynchronous operations issued previously have not  completed yet. This result is not actually an error, but must be indicated  differently than ::cudaSuccess (which indicates completion). Calls that  may return this value include ::cudaEventQuery() and ::cudaStreamQuery()."
    )]
    NotReady,
    #[display(
        fmt = "The device encountered a load or store instruction on an invalid memory address.  This leaves the process in an inconsistent state and any further CUDA work  will return the same error. To continue using CUDA, the process must be terminated  and relaunched."
    )]
    IllegalAddress,
    #[display(
        fmt = "This indicates that a launch did not occur because it did not have  appropriate resources. Although this error is similar to  ::cudaErrorInvalidConfiguration, this error usually indicates that the  user has attempted to pass too many arguments to the device kernel, or the  kernel launch specifies too many threads for the kernel's register count."
    )]
    LaunchOutOfResources,
    #[display(
        fmt = "This indicates that the device kernel took too long to execute. This can  only occur if timeouts are enabled - see the device property  ref ::cudaDeviceProp::kernelExecTimeoutEnabled kernelExecTimeoutEnabled  for more information.  This leaves the process in an inconsistent state and any further CUDA work  will return the same error. To continue using CUDA, the process must be terminated  and relaunched."
    )]
    LaunchTimeout,
    #[display(
        fmt = "This error indicates a kernel launch that uses an incompatible texturing  mode."
    )]
    LaunchIncompatibleTexturing,
    #[display(
        fmt = "This error indicates that a call to ::cudaDeviceEnablePeerAccess() is  trying to re-enable peer addressing on from a context which has already  had peer addressing enabled."
    )]
    PeerAccessAlreadyEnabled,
    #[display(
        fmt = "This error indicates that ::cudaDeviceDisablePeerAccess() is trying to  disable peer addressing which has not been enabled yet via  ::cudaDeviceEnablePeerAccess()."
    )]
    PeerAccessNotEnabled,
    #[display(
        fmt = "This indicates that the user has called ::cudaSetValidDevices(),  ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),  ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or  ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by  calling non-device management operations (allocating memory and  launching kernels are examples of non-device management operations).  This error can also be returned if using runtimegdriver  interoperability and there is an existing ::CUcontext active on the  host thread."
    )]
    SetOnActiveProcess,
    #[display(
        fmt = "This error indicates that the context current to the calling thread  has been destroyed using ::cuCtxDestroy, or is a primary context which  has not yet been initialized."
    )]
    ContextIsDestroyed,
    #[display(
        fmt = "An assert triggered in device code during kernel execution. The device  cannot be used again. All existing allocations are invalid. To continue  using CUDA, the process must be terminated and relaunched."
    )]
    Assert,
    #[display(
        fmt = "This error indicates that the hardware resources required to enable  peer access have been exhausted for one or more of the devices  passed to ::cudaEnablePeerAccess()."
    )]
    TooManyPeers,
    #[display(
        fmt = "This error indicates that the memory range passed to ::cudaHostRegister()  has already been registered."
    )]
    HostMemoryAlreadyRegistered,
    #[display(
        fmt = "This error indicates that the pointer passed to ::cudaHostUnregister()  does not correspond to any currently registered memory region."
    )]
    HostMemoryNotRegistered,
    #[display(
        fmt = "Device encountered an error in the call stack during kernel execution,  possibly due to stack corruption or exceeding the stack size limit.  This leaves the process in an inconsistent state and any further CUDA work  will return the same error. To continue using CUDA, the process must be terminated  and relaunched."
    )]
    HardwareStackError,
    #[display(
        fmt = "The device encountered an illegal instruction during kernel execution  This leaves the process in an inconsistent state and any further CUDA work  will return the same error. To continue using CUDA, the process must be terminated  and relaunched."
    )]
    IllegalInstruction,
    #[display(
        fmt = "The device encountered a load or store instruction  on a memory address which is not aligned.  This leaves the process in an inconsistent state and any further CUDA work  will return the same error. To continue using CUDA, the process must be terminated  and relaunched."
    )]
    MisalignedAddress,
    #[display(
        fmt = "While executing a kernel, the device encountered an instruction  which can only operate on memory locations in certain address spaces  (global, shared, or local), but was supplied a memory address not  belonging to an allowed address space.  This leaves the process in an inconsistent state and any further CUDA work  will return the same error. To continue using CUDA, the process must be terminated  and relaunched."
    )]
    InvalidAddressSpace,
    #[display(
        fmt = "The device encountered an invalid program counter.  This leaves the process in an inconsistent state and any further CUDA work  will return the same error. To continue using CUDA, the process must be terminated  and relaunched."
    )]
    InvalidPc,
    #[display(
        fmt = "An exception occurred on the device while executing a kernel. Common  causes include dereferencing an invalid device pointer and accessing  out of bounds shared memory. Less common cases can be system specific - more  information about these cases can be found in the system specific user guide.  This leaves the process in an inconsistent state and any further CUDA work  will return the same error. To continue using CUDA, the process must be terminated  and relaunched."
    )]
    LaunchFailure,
    #[display(
        fmt = "This error indicates that the number of blocks launched per grid for a kernel that was  launched via either ::cudaLaunchCooperativeKernel or ::cudaLaunchCooperativeKernelMultiDevice  exceeds the maximum number of blocks as allowed by ::cudaOccupancyMaxActiveBlocksPerMultiprocessor  or ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors  as specified by the device attribute ::cudaDevAttrMultiProcessorCount."
    )]
    CooperativeLaunchTooLarge,
    #[display(
        fmt = "This error indicates the attempted operation is not permitted."
    )]
    NotPermitted,
    #[display(
        fmt = "This error indicates the attempted operation is not supported  on the current system or device."
    )]
    NotSupported,
    #[display(
        fmt = "This error indicates that the system is not yet ready to start any CUDA  work.  To continue using CUDA, verify the system configuration is in a  valid state and all required driver daemons are actively running.  More information about this error can be found in the system specific  user guide."
    )]
    SystemNotReady,
    #[display(
        fmt = "This error indicates that there is a mismatch between the versions of  the display driver and the CUDA driver. Refer to the compatibility documentation  for supported versions."
    )]
    SystemDriverMismatch,
    #[display(
        fmt = "This error indicates that the system was upgraded to run with forward compatibility  but the visible hardware detected by CUDA does not support this configuration.  Refer to the compatibility documentation for the supported hardware matrix or ensure  that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES  environment variable."
    )]
    CompatNotSupportedOnDevice,
    #[display(
        fmt = "The operation is not permitted when the stream is capturing."
    )]
    StreamCaptureUnsupported,
    #[display(
        fmt = "The current capture sequence on the stream has been invalidated due to  a previous error."
    )]
    StreamCaptureInvalidated,
    #[display(
        fmt = "The operation would have resulted in a merge of two independent capture  sequences."
    )]
    StreamCaptureMerge,
    #[display(fmt = "The capture was not initiated in this stream.")]
    StreamCaptureUnmatched,
    #[display(
        fmt = "The capture sequence contains a fork that was not joined to the primary  stream."
    )]
    StreamCaptureUnjoined,
    #[display(
        fmt = "A dependency would have been created which crosses the capture sequence  boundary. Only implicit in-stream ordering dependencies are allowed to  cross the boundary."
    )]
    StreamCaptureIsolation,
    #[display(
        fmt = "The operation would have resulted in a disallowed implicit dependency on  a current capture sequence from cudaStreamLegacy."
    )]
    StreamCaptureImplicit,
    #[display(
        fmt = "The operation is not permitted on an event which was last recorded in a  capturing stream."
    )]
    CapturedEvent,
    #[display(
        fmt = "A stream capture sequence not initiated with the ::cudaStreamCaptureModeRelaxed  argument to ::cudaStreamBeginCapture was passed to ::cudaStreamEndCapture in a  different thread."
    )]
    StreamCaptureWrongThread,
    #[display(
        fmt = "This indicates that an unknown internal error has occurred."
    )]
    Unknown,
    #[display(
        fmt = "Any unhandled CUDA driver error is added to this value and returned via  the runtime. Production releases of CUDA should not return such errors.  deprecated  This error return is deprecated as of CUDA 4.1."
    )]
    ApiFailureBase,
}
