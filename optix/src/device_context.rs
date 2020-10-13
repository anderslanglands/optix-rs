use crate::{sys, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

use std::ffi::CStr;
use std::os::raw::{c_char, c_uint, c_void};

use ustr::Ustr;

pub struct DeviceContext {
    pub(crate) inner: sys::OptixDeviceContext,
}

/// # Creating and initializing the `DeviceContext`
impl DeviceContext {
    /// Create a device context associated with the CUDA context specified with
    /// `cuda_context`.
    /// Unlike the rest of the API types that are created from the context,
    /// the context itself will automatically be destroyed when it is dropped.
    pub fn create(cuda_context: &cu::Context) -> Result<DeviceContext> {
        unsafe {
            let mut inner = std::ptr::null_mut();
            let opt = sys::OptixDeviceContextOptions {
                logCallbackFunction: None,
                logCallbackData: std::ptr::null_mut(),
                logCallbackLevel: 0,
                #[cfg(feature = "optix72")]
                validationMode: 0xFFFFFFFF,
            };
            sys::optixDeviceContextCreate(**cuda_context, &opt, &mut inner)
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
    /// * 1 - fatal: A non-recoverable error. The context and/or OptiX itself
    ///   might
    /// no longer be in a usable state.
    /// * 2 - error: A recoverable error, e.g., when passing invalid call
    /// parameters.
    /// * 3 - warning: Hints that OptiX might not behave exactly as requested by
    /// the user or may perform slower than expected.
    /// * 4 - print: Status or progress messages.
    /// Higher levels might occur.
    ///
    /// # Safety
    /// Note that the callback must live longer than the DeviceContext which
    /// it's set for. This is because OptiX will fire messages when the
    /// underlying OptixDeviceContext is destroyed. In order to do ensure
    /// this we leak the closure `cb`. So don't go setting a new closure
    /// every frame.
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

/// # Setting and inspecting the compilation cache settings
#[rustfmt::skip]
impl DeviceContext {
    /// Returns the low and high water marks, respectively, for disk cache
    /// garbage collection
    pub fn get_cache_database_sizes(&self) -> (usize, usize) {
        let mut low = 0usize;
        let mut high = 0usize;
        unsafe {
            sys::optixDeviceContextGetCacheDatabaseSizes(
                self.inner,
                &mut low as *mut _ as *mut _,
                &mut high as *mut _ as *mut _,
            )
            .to_result()
            .expect("optixDeviceContextGetCacheDatabaseSizes");
        }

        (low, high)
    }

    /// Indicates whether the disk cache is enabled or disabled.
    pub fn get_cache_enabled(&self) -> bool {
        let mut enabled = 0i32;
        unsafe {
            sys::optixDeviceContextGetCacheEnabled(
                self.inner,
                &mut enabled as *mut _,
            )
            .to_result()
            .expect("optixDeviceContextGetCacheEnabled");
        }

        enabled != 0
    }

    /// Sets the low and high water marks for disk cache garbage collection.
    /// Garbage collection is triggered when a new entry is written to the cache
    /// and the current cache data size plus the size of the cache entry that is
    /// about to be inserted exceeds the high water mark. Garbage collection
    /// proceeds until the size reaches the low water mark.
    ///
    /// Garbage collection will always free enough space to insert the new entry
    /// without exceeding the low water mark.
    ///
    /// Setting either limit to zero will disable garbage
    /// collection.
    ///
    /// An error will be returned if both limits are non-zero and
    /// the high water mark is smaller than the low water mark.
    ///
    /// Note that garbage collection is performed only on writes to the disk
    /// cache. No garbage collection is triggered on disk cache initialization
    /// or immediately when calling this function, but on subsequent inserting
    /// of data into the database. If the size of a compiled module exceeds
    /// the value configured for the high water mark and garbage collection is
    /// enabled, the module will not be added to the cache and a warning will be
    /// added to the log.
    pub fn set_cache_database_sizes(
        &mut self,
        low_watermark: usize,
        high_watermark: usize,
    ) -> Result<()> {
        unsafe {
            sys::optixDeviceContextSetCacheDatabaseSizes(
                self.inner,
                low_watermark,
                high_watermark,
            )
            .to_result()
            .map_err(|source| {
                Error::DeviceContextSetCacheDatabaseSizes { source }
            })
        }
    }

    /// Enables the cache if `enabled` is `true`, disables it otherwise.
    /// Note that no in-memory cache is used when caching is disabled.
    ///
    /// The cache database is initialized when the device context is created and
    /// when enabled through this function call. If the database cannot be
    /// initialized when the device context is created, caching will be
    /// disabled; a message is reported to the log callback if caching is
    /// enabled. In this case, the call to [DeviceContext::create()] does not
    /// return an error. To ensure that cache initialization succeeded on
    /// context creation, the status can be queried using
    /// [DeviceContext::get_cache_enabled()].
    ///
    /// If caching is disabled, the cache can be reconfigured and then enabled
    /// using [DeviceContext::set_cache_enabled()]. If the cache database
    /// cannot be initialized with [DeviceContext::set_cache_enabled()], an
    /// error is returned.
    ///
    /// Garbage collection is performed on the next write to the cache
    /// database, not when the cache is enabled.
    pub fn set_cache_enabled(&mut self, enabled: bool) -> Result<()> {
        unsafe {
            sys::optixDeviceContextSetCacheEnabled(
                self.inner,
                if enabled { 1 } else { 0 },
            )
            .to_result()
            .map_err(|source| Error::DeviceContextSetCacheEnabled { source })
        }
    }

    /// Sets the location of the disk cache.
    ///
    /// The location is specified by a directory. This directory should not be
    /// used for other purposes and will be created if it does not exist. An
    /// error will be returned if is not possible to create the disk cache at
    /// the specified location for any reason (e.g., the path is invalid or the
    /// directory is not writable). Caching will be disabled if the disk cache
    /// cannot be initialized in the new location. If caching is disabled, no
    /// error will be returned until caching is enabled. If the disk cache is
    /// located on a network file share, behavior is undefined.
    ///
    /// The location of the disk cache can be overridden with the environment
    /// variable `OPTIX_CACHE_PATH`. This environment variable takes precedence
    /// over the value passed to this function when the disk cache is enabled.
    ///
    /// The default location of the cache depends on the operating system:
    /// 
    /// | Operating System | Path |
    /// | ---------------- | ---- |
    /// | Windows | `%LOCALAPPDATA%\NVIDIA\OptixCache` |
    /// | Linux | `/var/tmp/OptixCache_username`, or `/tmp/OptixCache_username` if the first choice is not usable. The underscore and username suffix are omitted if the username cannot be obtained. |
    /// 
    pub fn set_cache_location(&mut self, location: Ustr) -> Result<()> {
        unsafe {
            sys::optixDeviceContextSetCacheLocation(
                self.inner,
                location.as_char_ptr(),
            )
            .to_result()
            .map_err(|source| Error::DeviceContextSetCacheLocation { source })
        }
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            sys::optixDeviceContextDestroy(self.inner)
                .to_result()
                .expect("Failed to destroy device context");
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
