#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use cuda_sys::{CUcontext, CUdeviceptr, CUstream};

include!(concat!(env!("OUT_DIR"), "/optix_wrapper.rs"));

extern "C" {
    pub fn optixInit() -> OptixResult;
}

#[repr(C)]
pub struct SbtRecordHeader {
    header: [u8; OptixSbtRecordHeaderSize],
}

impl SbtRecordHeader {
    pub fn as_mut_ptr(&mut self) -> *mut std::os::raw::c_void {
        self.header.as_mut_ptr() as *mut std::os::raw::c_void
    }
}

impl Default for SbtRecordHeader {
    fn default() -> SbtRecordHeader {
        SbtRecordHeader {
            header: [0u8; OptixSbtRecordHeaderSize],
        }
    }
}

#[repr(u32)]
#[derive(Debug)]
pub enum Error {
    InvalidValue = OptixResult::OPTIX_ERROR_INVALID_VALUE as u32,
    HostOutOfMemory = OptixResult::OPTIX_ERROR_HOST_OUT_OF_MEMORY as u32,
    InvalidOperation = OptixResult::OPTIX_ERROR_INVALID_OPERATION as u32,
    FileIoError = OptixResult::OPTIX_ERROR_FILE_IO_ERROR as u32,
    InvalidFileFormat = OptixResult::OPTIX_ERROR_INVALID_FILE_FORMAT as u32,
    DiskCacheInvalidPath =
        OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_PATH as u32,
    DiskCachePermissionError =
        OptixResult::OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR as u32,
    DiskCacheDatabaseError =
        OptixResult::OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR as u32,
    DiskCacheInvalidData =
        OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_DATA as u32,
    LaunchFailure = OptixResult::OPTIX_ERROR_LAUNCH_FAILURE as u32,
    InvalidDeviceContext =
        OptixResult::OPTIX_ERROR_INVALID_DEVICE_CONTEXT as u32,
    CudaNotInitialized = OptixResult::OPTIX_ERROR_CUDA_NOT_INITIALIZED as u32,
    InvalidPtx = OptixResult::OPTIX_ERROR_INVALID_PTX as u32,
    InvalidLaunchParameter =
        OptixResult::OPTIX_ERROR_INVALID_LAUNCH_PARAMETER as u32,
    InvalidPayloadAccess =
        OptixResult::OPTIX_ERROR_INVALID_PAYLOAD_ACCESS as u32,
    InvalidAttributeAccess =
        OptixResult::OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS as u32,
    InvalidFunctionUse = OptixResult::OPTIX_ERROR_INVALID_FUNCTION_USE as u32,
    InvalidFunctionArguments =
        OptixResult::OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS as u32,
    PipelineOutOfConstantMemory =
        OptixResult::OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY as u32,
    PipelineLinkError = OptixResult::OPTIX_ERROR_PIPELINE_LINK_ERROR as u32,
    InternalCompilerError =
        OptixResult::OPTIX_ERROR_INTERNAL_COMPILER_ERROR as u32,
    DenoiserModelNotSet =
        OptixResult::OPTIX_ERROR_DENOISER_MODEL_NOT_SET as u32,
    DenoiserNotInitialized =
        OptixResult::OPTIX_ERROR_DENOISER_NOT_INITIALIZED as u32,
    AccelNotCompatible = OptixResult::OPTIX_ERROR_ACCEL_NOT_COMPATIBLE as u32,
    NotSupported = OptixResult::OPTIX_ERROR_NOT_SUPPORTED as u32,
    UnsupportedAbiVersion =
        OptixResult::OPTIX_ERROR_UNSUPPORTED_ABI_VERSION as u32,
    FunctionTableSizeMismatch =
        OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH as u32,
    InvalidEntryFunctionOptions =
        OptixResult::OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS as u32,
    LibraryNotFound = OptixResult::OPTIX_ERROR_LIBRARY_NOT_FOUND as u32,
    EntrySymbolNotFound =
        OptixResult::OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND as u32,
    CudaError = OptixResult::OPTIX_ERROR_CUDA_ERROR as u32,
    InternalError = OptixResult::OPTIX_ERROR_INTERNAL_ERROR as u32,
    Unknown = OptixResult::OPTIX_ERROR_UNKNOWN as u32,
}

use std::fmt;
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<OptixResult> for Error {
    fn from(res: OptixResult) -> Error {
        match res {
            OptixResult::OPTIX_ERROR_INVALID_VALUE => Error::InvalidValue,
            OptixResult::OPTIX_ERROR_HOST_OUT_OF_MEMORY => {
                Error::HostOutOfMemory
            }
            OptixResult::OPTIX_ERROR_INVALID_OPERATION => {
                Error::InvalidOperation
            }
            OptixResult::OPTIX_ERROR_FILE_IO_ERROR => Error::FileIoError,
            OptixResult::OPTIX_ERROR_INVALID_FILE_FORMAT => {
                Error::InvalidFileFormat
            }
            OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_PATH => {
                Error::DiskCacheInvalidPath
            }
            OptixResult::OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR => {
                Error::DiskCachePermissionError
            }
            OptixResult::OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR => {
                Error::DiskCacheDatabaseError
            }
            OptixResult::OPTIX_ERROR_DISK_CACHE_INVALID_DATA => {
                Error::DiskCacheInvalidData
            }
            OptixResult::OPTIX_ERROR_LAUNCH_FAILURE => Error::LaunchFailure,
            OptixResult::OPTIX_ERROR_INVALID_DEVICE_CONTEXT => {
                Error::InvalidDeviceContext
            }
            OptixResult::OPTIX_ERROR_CUDA_NOT_INITIALIZED => {
                Error::CudaNotInitialized
            }
            OptixResult::OPTIX_ERROR_INVALID_PTX => Error::InvalidPtx,
            OptixResult::OPTIX_ERROR_INVALID_LAUNCH_PARAMETER => {
                Error::InvalidLaunchParameter
            }
            OptixResult::OPTIX_ERROR_INVALID_PAYLOAD_ACCESS => {
                Error::InvalidPayloadAccess
            }
            OptixResult::OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS => {
                Error::InvalidAttributeAccess
            }
            OptixResult::OPTIX_ERROR_INVALID_FUNCTION_USE => {
                Error::InvalidFunctionUse
            }
            OptixResult::OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS => {
                Error::InvalidFunctionArguments
            }
            OptixResult::OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY => {
                Error::PipelineOutOfConstantMemory
            }
            OptixResult::OPTIX_ERROR_PIPELINE_LINK_ERROR => {
                Error::PipelineLinkError
            }
            OptixResult::OPTIX_ERROR_INTERNAL_COMPILER_ERROR => {
                Error::InternalCompilerError
            }
            OptixResult::OPTIX_ERROR_DENOISER_MODEL_NOT_SET => {
                Error::DenoiserModelNotSet
            }
            OptixResult::OPTIX_ERROR_DENOISER_NOT_INITIALIZED => {
                Error::DenoiserNotInitialized
            }
            OptixResult::OPTIX_ERROR_ACCEL_NOT_COMPATIBLE => {
                Error::AccelNotCompatible
            }
            OptixResult::OPTIX_ERROR_NOT_SUPPORTED => Error::NotSupported,
            OptixResult::OPTIX_ERROR_UNSUPPORTED_ABI_VERSION => {
                Error::UnsupportedAbiVersion
            }
            OptixResult::OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH => {
                Error::FunctionTableSizeMismatch
            }
            OptixResult::OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS => {
                Error::InvalidEntryFunctionOptions
            }
            OptixResult::OPTIX_ERROR_LIBRARY_NOT_FOUND => {
                Error::LibraryNotFound
            }
            OptixResult::OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND => {
                Error::EntrySymbolNotFound
            }
            OptixResult::OPTIX_ERROR_CUDA_ERROR => Error::CudaError,
            OptixResult::OPTIX_ERROR_INTERNAL_ERROR => Error::InternalError,
            OptixResult::OPTIX_ERROR_UNKNOWN => Error::Unknown,
            _ => unreachable!(),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{optixInit, OptixResult_OPTIX_SUCCESS};
    use cuda_sys::{
        cudaFree, cudaGetDeviceCount, CUcontext, CUdeviceptr, CUstream,
    };

    #[test]
    fn it_works() {
        unsafe {
            cudaFree(std::ptr::null_mut());
            let mut num_devices = 0i32;
            cudaGetDeviceCount(&mut num_devices as *mut i32);
            if num_devices == 0 {
                panic!("No CUDA devices found");
            }
            println!("Found {} CUDA devices", num_devices);

            let result = optixInit();
            if result != OptixResult_OPTIX_SUCCESS {
                panic!("OptiX init failed!");
            }

            println!("OptiX initialized successfully! Yay!");
        }
    }

    #[test]
    fn alignment() {
        // we can't specify a constant expression for #[repr(align())]
        // so testing that the values still match is the best we can do
        assert_eq!(OptixSbtRecordHeaderSize, 32);
        assert_eq!(OptixSbtRecordAlignment, 16);
    }
}
