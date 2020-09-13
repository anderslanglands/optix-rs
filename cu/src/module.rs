use crate::{sys, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

use std::ffi::CString;

pub struct Module {
    inner: sys::CUmodule,
}

impl Module {
    pub fn load_string(data: &str) -> Result<Module> {
        let s = CString::new(data).expect("Invalid CString");
        let mut inner = std::ptr::null_mut();
        unsafe {
            sys::cuModuleLoadData(
                &mut inner,
                s.as_ptr() as *mut std::os::raw::c_void,
            )
            .to_result()
            .map(|_| Module { inner })
        }
    }

    pub fn get_function(&self, name: &str) -> Result<Function> {
        let mut inner = std::ptr::null_mut();
        let n = CString::new(name).expect("Invalid CString");
        unsafe {
            sys::cuModuleGetFunction(&mut inner, self.inner, n.as_ptr())
                .to_result()
                .map(|_| Function { inner })
        }
    }
}

pub struct Function {
    pub(crate) inner: sys::CUfunction,
}
