use enum_primitive_derive::Primitive;
use num_traits::FromPrimitive;
use crate::sys;

impl sys::cudaError_enum {
    pub fn to_result(&self) -> Result<(), Error> {
        if self.0 == 0 {
            return Ok(());
        }

        let v = self.0 as u32;

        if let Some(e) = Error::from_u32(v) {
            Err(e)
        } else {
            panic!("CUDA returned an unhandled error code: {}", v)
        }
    }
}

#[derive(Debug, Copy, Clone, thiserror::Error, Primitive)]
pub enum Error {
    #[error("This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.")]
    InvalidValue = 1,
    #[error("The API call failed because it was unable to allocate enough memory to perform the requested operation.")]
    OutOfMemory = 2,
    #[error("This indicates that the CUDA driver has not been initialized with ::cuInit() or that initialization has failed.")]
    NotInitialized = 3,
    #[error("This indicates that the CUDA driver is in the process of shutting down.")]
    Deinitialized = 4,
    #[error("This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.")]
    ProfilerDisabled = 5,
    #[error("\\deprecated This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable the profiling via ::cuProfilerStart or ::cuProfilerStop without initialization.")]
    ProfilerNotInitialized = 6,
    #[error("\\deprecated This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStart() when profiling is already enabled.")]
    ProfilerAlreadyStarted = 7,
    #[error("\\deprecated This error return is deprecated as of CUDA 5.0. It is no longer an error to call cuProfilerStop() when profiling is already disabled.")]
    ProfilerAlreadyStopped = 8,
    #[error("This indicates that no CUDA-capable devices were detected by the installed CUDA driver.")]
    NoDevice = 100,
    #[error("This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device.")]
    InvalidDevice = 101,
    #[error("This indicates that the device kernel image is invalid. This can also indicate an invalid CUDA module.")]
    InvalidImage = 200,
    #[error("This most frequently indicates that there is no context bound to the current thread. This can also be returned if the context passed to an API call is not a valid handle (such as a context that has had ::cuCtxDestroy() invoked on it). This can also be returned if a user mixes different API versions (i.e. 3010 context with 3020 API calls). See ::cuCtxGetApiVersion() for more details.")]
    InvalidContext = 201,
    #[error("This indicated that the context being supplied as a parameter to the API call was already the active context. \\deprecated This error return is deprecated as of CUDA 3.2. It is no longer an error to attempt to push the active context via ::cuCtxPushCurrent().")]
    ContextAlreadyCurrent = 202,
    #[error("This indicates that a map or register operation has failed.")]
    MapFailed = 205,
    #[error(
        "This indicates that an unmap or unregister operation has failed."
    )]
    UnmapFailed = 206,
    #[error("This indicates that the specified array is currently mapped and thus cannot be destroyed.")]
    ArrayIsMapped = 207,
    #[error("This indicates that the resource is already mapped.")]
    AlreadyMapped = 208,
    #[error("This indicates that there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration.")]
    NoBinaryForGpu = 209,
    #[error("This indicates that a resource has already been acquired.")]
    AlreadyAcquired = 210,
    #[error("This indicates that a resource is not mapped.")]
    NotMapped = 211,
    #[error("This indicates that a mapped resource is not available for access as an array.")]
    NotMappedAsArray = 212,
    #[error("This indicates that a mapped resource is not available for access as a pointer.")]
    NotMappedAsPointer = 213,
    #[error("This indicates that an uncorrectable ECC error was detected during execution.")]
    EccUncorrectable = 214,
    #[error("This indicates that the ::CUlimit passed to the API call is not supported by the active device.")]
    UnsupportedLimit = 215,
    #[error("This indicates that the ::CUcontext passed to the API call can only be bound to a single CPU thread at a time but is already bound to a CPU thread.")]
    ContextAlreadyInUse = 216,
    #[error("This indicates that peer access is not supported across the given devices.")]
    PeerAccessUnsupported = 217,
    #[error("This indicates that a PTX JIT compilation failed.")]
    InvalidPtx = 218,
    #[error("This indicates an error with OpenGL or DirectX context.")]
    InvalidGraphicsContext = 219,
    #[error("This indicates that an uncorrectable NVLink error was detected during the execution.")]
    NvlinkUncorrectable = 220,
    #[error("This indicates that the PTX JIT compiler library was not found.")]
    JitCompilerNotFound = 221,
    #[error("This indicates that the device kernel source is invalid.")]
    InvalidSource = 300,
    #[error("This indicates that the file specified was not found.")]
    FileNotFound = 301,
    #[error(
        "This indicates that a link to a shared object failed to resolve."
    )]
    SharedObjectSymbolNotFound = 302,
    #[error("This indicates that initialization of a shared object failed.")]
    SharedObjectInitFailed = 303,
    #[error("This indicates that an OS call failed.")]
    OperatingSystem = 304,
    #[error("This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like ::CUstream and ::CUevent.")]
    InvalidHandle = 400,
    #[error("This indicates that a resource required by the API call is not in a valid state to perform the requested operation.")]
    IllegalState = 401,
    #[error("This indicates that a named symbol was not found. Examples of symbols are global/constant variable names, texture names, and surface names.")]
    NotFound = 500,
    #[error("This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than ::CUDA_SUCCESS (which indicates completion). Calls that may return this value include ::cuEventQuery() and ::cuStreamQuery().")]
    NotReady = 600,
    #[error("While executing a kernel, the device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.")]
    IllegalAddress = 700,
    #[error("This indicates that a launch did not occur because it did not have appropriate resources. This error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count. Passing arguments of the wrong size (i.e. a 64-bit pointer when a 32-bit int is expected) is equivalent to passing too many arguments and can also result in this error.")]
    LaunchOutOfResources = 701,
    #[error("This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device attribute ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.")]
    LaunchTimeout = 702,
    #[error("This error indicates a kernel launch that uses an incompatible texturing mode.")]
    LaunchIncompatibleTexturing = 703,
    #[error("This error indicates that a call to ::cuCtxEnablePeerAccess() is trying to re-enable peer access to a context which has already had peer access to it enabled.")]
    PeerAccessAlreadyEnabled = 704,
    #[error("This error indicates that ::cuCtxDisablePeerAccess() is trying to disable peer access which has not been enabled yet via ::cuCtxEnablePeerAccess().")]
    PeerAccessNotEnabled = 705,
    #[error("This error indicates that the primary context for the specified device has already been initialized.")]
    PrimaryContextActive = 708,
    #[error("This error indicates that the context current to the calling thread has been destroyed using ::cuCtxDestroy, or is a primary context which has not yet been initialized.")]
    ContextIsDestroyed = 709,
    #[error("A device-side assert triggered during kernel execution. The context cannot be used anymore, and must be destroyed. All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA.")]
    Assert = 710,
    #[error("This error indicates that the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to ::cuCtxEnablePeerAccess().")]
    TooManyPeers = 711,
    #[error("This error indicates that the memory range passed to ::cuMemHostRegister() has already been registered.")]
    HostMemoryAlreadyRegistered = 712,
    #[error("This error indicates that the pointer passed to ::cuMemHostUnregister() does not correspond to any currently registered memory region.")]
    HostMemoryNotRegistered = 713,
    #[error("While executing a kernel, the device encountered a stack error. This can be due to stack corruption or exceeding the stack size limit. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.")]
    HardwareStackError = 714,
    #[error("While executing a kernel, the device encountered an illegal instruction. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.")]
    IllegalInstruction = 715,
    #[error("While executing a kernel, the device encountered a load or store instruction on a memory address which is not aligned. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.")]
    MisalignedAddress = 716,
    #[error("While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.")]
    InvalidAddressSpace = 717,
    #[error("While executing a kernel, the device program counter wrapped its address space. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.")]
    InvalidPc = 718,
    #[error("An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases can be found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.")]
    LaunchFailed = 719,
    #[error("This error indicates that the number of blocks launched per grid for a kernel that was launched via either ::cuLaunchCooperativeKernel or ::cuLaunchCooperativeKernelMultiDevice exceeds the maximum number of blocks as allowed by ::cuOccupancyMaxActiveBlocksPerMultiprocessor or ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags times the number of multiprocessors as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.")]
    CooperativeLaunchTooLarge = 720,
    #[error(
        "This error indicates that the attempted operation is not permitted."
    )]
    NotPermitted = 800,
    #[error("This error indicates that the attempted operation is not supported on the current system or device.")]
    NotSupported = 801,
    #[error("This error indicates that the system is not yet ready to start any CUDA work.  To continue using CUDA, verify the system configuration is in a valid state and all required driver daemons are actively running. More information about this error can be found in the system specific user guide.")]
    SystemNotReady = 802,
    #[error("This error indicates that there is a mismatch between the versions of the display driver and the CUDA driver. Refer to the compatibility documentation for supported versions.")]
    SystemDriverMismatch = 803,
    #[error("This error indicates that the system was upgraded to run with forward compatibility but the visible hardware detected by CUDA does not support this configuration. Refer to the compatibility documentation for the supported hardware matrix or ensure that only supported hardware is visible during initialization via the CUDA_VISIBLE_DEVICES environment variable.")]
    CompatNotSupportedOnDevice = 804,
    #[error("This error indicates that the operation is not permitted when the stream is capturing.")]
    StreamCaptureUnsupported = 900,
    #[error("This error indicates that the current capture sequence on the stream has been invalidated due to a previous error.")]
    StreamCaptureInvalidated = 901,
    #[error("This error indicates that the operation would have resulted in a merge of two independent capture sequences.")]
    StreamCaptureMerge = 902,
    #[error("This error indicates that the capture was not initiated in this stream.")]
    StreamCaptureUnmatched = 903,
    #[error("This error indicates that the capture sequence contains a fork that was not joined to the primary stream.")]
    StreamCaptureUnjoined = 904,
    #[error("This error indicates that a dependency would have been created which crosses the capture sequence boundary. Only implicit in-stream ordering dependencies are allowed to cross the boundary.")]
    StreamCaptureIsolation = 905,
    #[error("This error indicates a disallowed implicit dependency on a current capture sequence from cudaStreamLegacy.")]
    StreamCaptureImplicit = 906,
    #[error("This error indicates that the operation is not permitted on an event which was last recorded in a capturing stream.")]
    CapturedEvent = 907,
    #[error("A stream capture sequence not initiated with the ::CU_STREAM_CAPTURE_MODE_RELAXED argument to ::cuStreamBeginCapture was passed to ::cuStreamEndCapture in a different thread.")]
    StreamCaptureWrongThread = 908,
    #[error("This error indicates that the timeout specified for the wait operation has lapsed.")]
    Timeout = 909,
    #[error("This error indicates that the graph update was not performed because it included changes which violated constraints specific to instantiated graph update.")]
    GraphExecUpdateFailure = 910,
    #[error("This indicates that an unknown internal error has occurred.")]
    Unknown = 999,
}
