use optix_sys::cuda_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Copy, Clone)]
pub struct ContextRef {
    ctx: sys::CUcontext,
}

impl ContextRef {
    pub fn ctx(&self) -> sys::CUcontext {
        self.ctx
    }
}

impl std::ops::Deref for ContextRef {
    type Target = sys::CUcontext;
    fn deref(&self) -> &sys::CUcontext {
        &self.ctx
    }
}

pub struct Context {}

impl Context {
    pub fn get_current() -> Result<ContextRef> {
        unsafe {
            let mut ctx = std::ptr::null_mut();
            let res = sys::cuCtxGetCurrent(&mut ctx);
            if res != sys::cudaError::cudaSuccess {
                return Err(Error::CouldNotGetCurrentContext {
                    source: res.into(),
                });
            }
            Ok(ContextRef { ctx })
        }
    }
}
