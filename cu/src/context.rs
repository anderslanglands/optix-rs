use enum_primitive_derive::Primitive;
use num_traits::FromPrimitive;

use crate::{sys, Device, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

pub struct Context {
    inner: sys::CUcontext,
}

impl std::ops::Deref for Context {
    type Target = sys::CUcontext;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Context {
    pub fn get_api_version(&self) -> Result<u32> {
        let mut v = 0u32;
        unsafe {
            sys::cuCtxGetApiVersion(self.inner, &mut v)
                .to_result()
                .map(|_| v)
        }
    }

    pub fn get_cache_config() -> Result<FuncCache> {
        let mut c = sys::CUfunc_cache_enum(0);
        unsafe {
            sys::cuCtxGetCacheConfig(&mut c).to_result().map(|_| {
                FuncCache::from_u32(c.0).expect("Unexpected FuncCache value")
            })
        }
    }

    pub fn get_current() -> Result<Context> {
        let mut inner: sys::CUcontext = std::ptr::null_mut();
        unsafe {
            sys::cuCtxGetCurrent(&mut inner)
                .to_result()
                .map(|_| Context { inner })
        }
    }

    pub fn get_device() -> Result<Device> {
        let mut inner: sys::CUdevice = 0;
        unsafe {
            sys::cuCtxGetDevice(&mut inner)
                .to_result()
                .map(|_| Device { inner })
        }
    }

    pub fn get_flags() -> Result<ContextFlags> {
        let mut flags = 0u32;
        unsafe {
            sys::cuCtxGetFlags(&mut flags).to_result().map(|_| {
                ContextFlags::from_bits(flags).expect("Unexpected flag bits")
            })
        }
    }

    pub fn get_limit(limit: Limit) -> Result<usize> {
        let mut sz = 0usize;
        unsafe {
            sys::cuCtxGetLimit(
                &mut sz as *mut usize as *mut u64,
                sys::CUlimit_enum(limit as u32),
            )
            .to_result()
            .map(|_| sz)
        }
    }

    pub fn get_shared_mem_config() -> Result<SharedMemConfig> {
        let mut c = 0u32;
        unsafe {
            sys::cuCtxGetSharedMemConfig(&mut c).to_result().map(|_| {
                SharedMemConfig::from_u32(c)
                    .expect("Invalid SharedMemConfig value")
            })
        }
    }

    pub fn pop_current() -> Result<Context> {
        let mut inner: sys::CUcontext = std::ptr::null_mut();
        unsafe {
            sys::cuCtxPopCurrent_v2(&mut inner)
                .to_result()
                .map(|_| Context { inner })
        }
    }

    pub fn push_current(ctx: Context) -> Result<()> {
        unsafe { sys::cuCtxPushCurrent_v2(ctx.inner).to_result() }
    }

    pub fn set_cache_config(cache: FuncCache) -> Result<()> {
        unsafe {
            sys::cuCtxSetCacheConfig(sys::CUfunc_cache(cache as u32))
                .to_result()
        }
    }

    pub fn set_current(ctx: Context) -> Result<()> {
        unsafe { sys::cuCtxSetCurrent(ctx.inner).to_result() }
    }

    pub fn set_limit(limit: Limit, sz: usize) -> Result<()> {
        unsafe {
            sys::cuCtxSetLimit(sys::CUlimit_enum(limit as u32), sz as u64)
                .to_result()
        }
    }

    pub fn set_shared_mem_config(config: SharedMemConfig) -> Result<()> {
        unsafe { sys::cuCtxSetSharedMemConfig(config as u32).to_result() }
    }

    pub fn synchronize() -> Result<()> {
        unsafe { sys::cuCtxSynchronize().to_result() }
    }
}

pub fn destroy(ctx: Context) -> Result<()> {
    unsafe { sys::cuCtxDestroy_v2(ctx.inner).to_result() }
}

impl Device {
    pub fn ctx_create(&self, flags: ContextFlags) -> Result<Context> {
        let mut inner = std::ptr::null_mut();
        unsafe {
            sys::cuCtxCreate_v2(&mut inner, flags.bits(), self.inner)
                .to_result()
                .map(|_| Context { inner })
        }
    }
}

bitflags::bitflags! {
    pub struct ContextFlags: u32 {
        const SCHED_AUTO = 0;
        const SCHED_SPIN = 1;
        const SCHED_YIELD = 2;
        const SCHED_BLOCKING_SYNC = 4;
        const BLOCKING_SYNC = 4;
        const SCHED_MASK = 7;
        const MAP_HOST = 8;
        const FLAGS_MASK = 31;
    }
}

#[derive(Debug, PartialEq, Copy, Clone, Primitive)]
pub enum FuncCache {
    PreferNone = 1,
    PreferShared = 2,
    PreferL1 = 3,
    PreferEqual = 4,
}

#[repr(u32)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Limit {
    StackSize = 0,
    PrintfFifoSize = 1,
    MallocHeapSize = 2,
    DevRuntimeSyncDepth = 3,
    DevRuntimePendingLaunchCount = 4,
    MaxL2FetchGranularity = 5,
    PersistingL2CacheSize = 6,
    Max = 7,
}

#[repr(u32)]
#[derive(Debug, PartialEq, Copy, Clone, Primitive)]
pub enum SharedMemConfig {
    DefaultBankSize =
        sys::CUsharedconfig_enum::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE,
    FourByteBankSize =
        sys::CUsharedconfig_enum::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE,
    EightByteBankSize =
        sys::CUsharedconfig_enum::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE,
}
