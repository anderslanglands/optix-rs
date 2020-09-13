use crate::{sys, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

use std::os::raw::{c_uint, c_void, c_char};
use std::ffi::CStr;

pub struct DeviceContext {
    pub(crate) inner: sys::OptixDeviceContext,
}

impl DeviceContext {
    pub fn create(cuda_context: &cu::Context) -> Result<DeviceContext> {
        unsafe {
            let mut inner = std::ptr::null_mut();
            sys::optixDeviceContextCreate(
                **cuda_context,
                std::ptr::null_mut(),
                &mut inner,
            )
            .to_result()
            .map(|_| DeviceContext { inner })
            .map_err(|source| Error::DeviceContextCreation { source })
        }
    }

    /// Sets the current log callback method.
    ///
    /// The following log levels are defined.
    /// * 0 - disable: Setting the callback level will disable all messages. The
    /// callback function will not be called in this case.
    /// * 1 - fatal: A non-recoverable error. The context and/or OptiX itself might
    /// no longer be in a usable state.
    /// * 2 - error: A recoverable error, e.g., when passing invalid call
    /// parameters.
    /// * 3 - warning: Hints that OptiX might not behave exactly as requested by
    /// the user or may perform slower than expected.
    /// * 4 - print: Status or progress messages.
    /// Higher levels might occur.
    ///
    /// # Safety
    /// Note that the callback must live longer than the DeviceContext which it's
    /// set for. This is because OptiX will fire messages when the underlying
    /// OptixDeviceContext is destroyed. In order to do ensure this we leak the
    /// closure `cb`. So don't go setting a new closure every frame.
    pub fn set_log_callback<F>(&mut self, cb: F, level: u32)
    where
        F: FnMut(u32, &str, &str) + 'static,
    {
        let (closure, trampoline) = unsafe { unpack_closure(cb) };
        let res = unsafe {
            sys::optixDeviceContextSetLogCallback(
                self.inner,
                Some(trampoline),
                closure,
                level,
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("Failed to set log callback");
        }
    }
}
type LogCallback =
    extern "C" fn(c_uint, *const c_char, *const c_char, *mut c_void);

/// Unpack a Rust closure, extracting a `void*` pointer to the data and a
/// trampoline function which can be used to invoke it.
///
/// # Safety
///
/// It is the user's responsibility to ensure the closure outlives the returned
/// `void*` pointer.
///
/// Calling the trampoline function with anything except the `void*` pointer
/// will result in *Undefined Behaviour*.
///
/// The closure should guarantee that it never panics, seeing as panicking
/// across the FFI barrier is *Undefined Behaviour*. You may find
/// `std::panic::catch_unwind()` useful.
unsafe fn unpack_closure<F>(closure: F) -> (*mut c_void, LogCallback)
where
    F: FnMut(u32, &str, &str),
{
    extern "C" fn trampoline<F>(
        level: c_uint,
        tag: *const c_char,
        msg: *const c_char,
        data: *mut c_void,
    ) where
        F: FnMut(u32, &str, &str),
    {
        let tag = unsafe { CStr::from_ptr(tag).to_string_lossy().into_owned() };
        let msg = unsafe { CStr::from_ptr(msg).to_string_lossy().into_owned() };
        let closure: &mut F = unsafe { &mut *(data as *mut F) };
        (*closure)(level, &tag, &msg);
    }

    let cb = Box::new(closure);
    let cb = Box::leak(cb);

    (cb as *mut F as *mut c_void, trampoline::<F>)
}
