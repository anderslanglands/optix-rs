#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use cuda_sys::{CUcontext, CUdeviceptr, CUstream};

include!(concat!(env!("OUT_DIR"), "/optix_wrapper.rs"));

extern "C" {
    pub fn optixInit() -> OptixResult;
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
