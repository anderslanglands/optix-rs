use cuda::ContextRef;
use optix_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;

use super::module::ModuleRef;
use super::program_group::ProgramGroupRef;
use super::pipeline::PipelineRef;
use super::shader_binding_table::ShaderBindingTable;

pub struct DeviceContext {
    pub(crate) ctx: sys::OptixDeviceContext,

    // handles for all object types
    // pub(crate) modules: Vec<ModuleRef>,
    // pub(crate) program_groups: Vec<ProgramGroupRef>,
    pub(crate) pipelines: Vec<PipelineRef>,
}

pub struct Options {
    // TODO: logging stuff goes in here
}

impl DeviceContext {
    /// Create a device context associated with the `cuda::Context` referenced with `cuda_context`.
    pub fn create(
        cuda_context: ContextRef,
        options: Option<Options>,
    ) -> Result<DeviceContext> {
        unsafe {
            let mut ctx = std::ptr::null_mut();
            let res = sys::optixDeviceContextCreate(
                *cuda_context,
                std::ptr::null_mut(),
                &mut ctx,
            );
            if res != sys::OptixResult::OPTIX_SUCCESS {
                return Err(Error::DeviceContextCreateFailed {
                    cerr: res.into(),
                });
            }
            if ctx.is_null() {
                panic!("optixDeviceContextCreate returned NULL");
            }

            Ok(DeviceContext { ctx, 
            // modules: Vec::new(), 
            // program_groups: Vec::new(), 
            pipelines: Vec::new() })
        }
    }

    /// Returns the low and high water marks for disk cache garbage collection.
    pub fn get_cache_database_sizes(&self) -> (usize, usize) {
        let mut lo = 0usize;
        let mut hi = 0usize;
        let res = unsafe {
            sys::optixDeviceContextGetCacheDatabaseSizes(
                self.ctx, &mut lo, &mut hi,
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceContextGetCacheDatabaseSizes failed");
        }

        (lo, hi)
    }

    /// Indicates whether the disk cache is enabled or disabled.
    pub fn get_cache_enabled(&self) -> bool {
        let mut e = 0i32;
        let res =
            unsafe { sys::optixDeviceContextGetCacheEnabled(self.ctx, &mut e) };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceContextGetCacheEnabled failed");
        }

        e != 0
    }

    /// Returns the location of the disk cache.
    pub fn get_cache_location(&self) -> String {
        let mut bytes = [0i8; 4096];
        let res = unsafe {
            sys::optixDeviceContextGetCacheLocation(
                self.ctx,
                bytes.as_mut_ptr(),
                bytes.len(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetCacheLocation failed");
        }

        unsafe {
            CStr::from_ptr(bytes.as_ptr())
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Maximum value for OptixPipelineLinkOptions::maxTraceDepth
    pub fn max_trace_depth(&self) -> u32 {
        let mut value = 0u32;
        let res = unsafe {
            sys::optixDeviceContextGetProperty(self.ctx,
                sys::OptixDeviceProperty_OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH,
                &mut value as *mut u32 as *mut std::os::raw::c_void, 
                std::mem::size_of::<u32>(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetProperty failed");
        }

        value
    }

    /// Maximum value to pass into optixPipelineSetStackSize for parameter 
    /// maxTraversableGraphDepth
    pub fn max_traversable_graph_depth(&self) -> u32 {
        let mut value = 0u32;
        let res = unsafe {
            sys::optixDeviceContextGetProperty(self.ctx,
                sys::OptixDeviceProperty_OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,
                &mut value as *mut u32 as *mut std::os::raw::c_void, 
                std::mem::size_of::<u32>(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetProperty failed");
        }

        value
    }

    /// The maximum number of primitives (over all build inputs) as input to a 
    /// single Geometry Acceleration Structure (GAS)
    pub fn max_primtives_per_gas(&self) -> u32 {
        let mut value = 0u32;
        let res = unsafe {
            sys::optixDeviceContextGetProperty(self.ctx,
                sys::OptixDeviceProperty_OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
                &mut value as *mut u32 as *mut std::os::raw::c_void, 
                std::mem::size_of::<u32>(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetProperty failed");
        }

        value
    }

    /// The maximum number of instances that can be added to a single Instance 
    /// Acceleration Structure (IAS)
    pub fn max_instances_per_ias(&self) -> u32 {
        let mut value = 0u32;
        let res = unsafe {
            sys::optixDeviceContextGetProperty(self.ctx,
                sys::OptixDeviceProperty_OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
                &mut value as *mut u32 as *mut std::os::raw::c_void, 
                std::mem::size_of::<u32>(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetProperty failed");
        }

        value
    }

    /// The RT core version supported by the device (0 for no support, 10 for 
    /// version 1.0)
    pub fn rtcore_version(&self) -> u32 {
        let mut value = 0u32;
        let res = unsafe {
            sys::optixDeviceContextGetProperty(self.ctx,
                sys::OptixDeviceProperty_OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
                &mut value as *mut u32 as *mut std::os::raw::c_void, 
                std::mem::size_of::<u32>(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetProperty failed");
        }

        value
    }

    /// The maximum value for OptixInstance::instanceId
    pub fn max_instance_id(&self) -> u32 {
        let mut value = 0u32;
        let res = unsafe {
            sys::optixDeviceContextGetProperty(self.ctx,
                sys::OptixDeviceProperty_OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,
                &mut value as *mut u32 as *mut std::os::raw::c_void, 
                std::mem::size_of::<u32>(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetProperty failed");
        }

        value
    }

    /// The number of bits available for the OptixInstance::visibilityMask. 
    /// Higher bits must be set to zero
    pub fn num_bits_instance_visibility_mask(&self) -> u32 {
        let mut value = 0u32;
        let res = unsafe {
            sys::optixDeviceContextGetProperty(self.ctx,
                sys::OptixDeviceProperty_OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK,
                &mut value as *mut u32 as *mut std::os::raw::c_void, 
                std::mem::size_of::<u32>(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetProperty failed");
        }

        value
    }

    /// The maximum number for the sum of the number of SBT records of all build 
    /// inputs to a single Geometry Acceleration Structure (GAS)
    pub fn max_sbt_records_per_gas(&self) -> u32 {
        let mut value = 0u32;
        let res = unsafe {
            sys::optixDeviceContextGetProperty(self.ctx,
                sys::OptixDeviceProperty_OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS,
                &mut value as *mut u32 as *mut std::os::raw::c_void, 
                std::mem::size_of::<u32>(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetProperty failed");
        }

        value
    }

    /// The maximum value for OptixInstance::sbtOffset
    pub fn max_sbt_offset(&self) -> u32 {
        let mut value = 0u32;
        let res = unsafe {
            sys::optixDeviceContextGetProperty(self.ctx,
                sys::OptixDeviceProperty_OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET,
                &mut value as *mut u32 as *mut std::os::raw::c_void, 
                std::mem::size_of::<u32>(),
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceGetProperty failed");
        }

        value
    }

    /// Sets the low and high water marks for disk cache garbage collection.
    /// 
    /// Garbage collection is triggered when a new entry is written to the cache 
    /// and the current cache data size plus the size of the cache entry that is 
    /// about to be inserted exceeds the high water mark. Garbage collection 
    /// proceeds until the size reaches the low water mark. Garbage collection 
    /// will always free enough space to insert the new entry without exceeding 
    /// the low water mark. Setting either limit to zero will disable garbage 
    /// collection. An error will be returned if both limits are non-zero and 
    /// the high water mark is smaller than the low water mark.
    /// Note that garbage collection is performed only on writes to the disk 
    /// cache. No garbage collection is triggered on disk cache initialization or 
    /// immediately when calling this function, but on subsequent inserting of 
    /// data into the database.
    /// If the size of a compiled module exceeds the value configured for the 
    /// high water mark and garbage collection is enabled, the module will not 
    /// be added to the cache and a warning will be added to the log.
    pub fn set_database_cache_sizes(&mut self, low_water_mark: usize, high_water_mark: usize) {
        let res = unsafe {
            sys::optixDeviceContextSetCacheDatabaseSizes(self.ctx, low_water_mark, high_water_mark)
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceContextSetCacheDatabaseSizes failed");
        }
    }

    /// Enables or disables the disk cache.
    /// 
    /// If caching was previously disabled, enabling it will attempt to 
    /// initialize the disk cache database using the currently configured cache 
    /// location. An error will be returned if initialization fails.
    /// Note that no in-memory cache is used, so no caching behavior will be 
    /// observed if the disk cache is disabled.
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        let e = if enabled { 1 } else {0};
        let res = unsafe {
            sys::optixDeviceContextSetCacheEnabled(self.ctx, e)
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixDeviceContextSetCacheEnabled failed");
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
    /// The location of the disk cache can be overridden with the environment 
    /// variable OPTIX_CACHE_PATH. The environment variable takes precedence 
    /// over this setting.
    /// The default location depends on the operating system:
    /// * Windows: LOCALAPPDATA%\NVIDIA\OptixCache
    /// * Linux: /var/tmp/OptixCache_<username> (or /tmp/OptixCache_<username> 
    /// if the first choice is not usable), the underscore and username suffix 
    /// are omitted if the username cannot be obtained
    /// * MacOS X: /Library/Application Support/NVIDIA/OptixCache
    pub fn set_cache_location<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let cs = CString::new(path.as_ref().to_str().unwrap()).unwrap();
        let res = unsafe {
            sys::optixDeviceContextSetCacheLocation(self.ctx, cs.as_ptr())
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            return Err(Error::SetCacheLocationFailed{cerr: res.into(), path: path.as_ref().to_path_buf()});
        }

        Ok(())
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
    pub fn set_log_callback<F>(&mut self, cb: F, level: u32) where F: FnMut(u32, &str, &str) + 'static {
        let (closure, trampoline) = unsafe {unpack_closure(cb)};
        let res = unsafe {
            sys::optixDeviceContextSetLogCallback(self.ctx, Some(trampoline), closure, level)
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("Failed to set log callback");
        }
    }

    pub fn launch(&self, pipeline: &PipelineRef, stream: &cuda::Stream, launch_params: &cuda::Buffer, sbt: &ShaderBindingTable, width: u32, height: u32, depth: u32) -> Result<()> {
        let res = unsafe {
            sys::optixLaunch(
                pipeline.pipeline,
                stream.as_sys_ptr(),
                launch_params.as_device_ptr(),
                launch_params.byte_size(),
                &sbt.sbt,
                width, height, depth,
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            return Err(Error::LaunchFailed{cerr: res.into()});
        }

        Ok(())
    }
}

use std::os::raw::c_uint;

type LogCallback = extern "C" fn(c_uint, *const c_char, *const c_char, *mut c_void);

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
    extern "C" fn trampoline<F>(level: c_uint, tag: *const c_char, msg: *const c_char, data: *mut c_void)
    where
        F: FnMut(u32, &str, &str),
    {
        let tag = unsafe {CStr::from_ptr(tag).to_string_lossy().into_owned()};
        let msg = unsafe {CStr::from_ptr(msg).to_string_lossy().into_owned()};
        let closure: &mut F = unsafe { &mut *(data as *mut F) };
        (*closure)(level, &tag, &msg);
    }

    let cb = Box::new(closure);
    let cb = Box::leak(cb);

    (cb as *mut F as  *mut c_void, trampoline::<F>)
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            sys::optixDeviceContextDestroy(self.ctx);
        }
    }
}
