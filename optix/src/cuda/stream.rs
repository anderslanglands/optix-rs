use optix_sys::cuda_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

pub struct Stream {
    s: sys::CUstream,
}

impl Stream {
    pub fn new() -> Result<Stream> {
        let mut s: sys::CUstream = std::ptr::null_mut();
        let res = unsafe { sys::cudaStreamCreate(&mut s) };

        if res != sys::cudaError::cudaSuccess {
            return Err(Error::StreamCreationFailed { source: res.into() });
        }

        Ok(Stream { s })
    }

    pub fn as_sys_ptr(&self) -> sys::CUstream {
        self.s
    }
}

impl Default for Stream {
    fn default() -> Stream {
        Stream {
            s: std::ptr::null_mut(),
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            if !self.s.is_null() {
                sys::cudaStreamDestroy(self.s);
            }
        }
    }
}
