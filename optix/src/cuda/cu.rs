use optix_sys::cuda_sys as sys;

use sys::cu;

use std::ffi::CString;
use std::os::raw::c_void;

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("Failed to initialize CUDA driver API")]
    Init { source: cu::Error },
    #[error("Failed to get device {ordinal:}")]
    DeviceGet { source: cu::Error, ordinal: i32 },
    #[error("Failed to get device count")]
    DeviceGetCount { source: cu::Error },
    #[error("Failed to create context")]
    ContextCreate { source: cu::Error },
    #[error("Failed to synchronize context")]
    ContextSynchronize { source: cu::Error },
    #[error("Failed to load module data")]
    ModuleLoad { source: cu::Error, ptx: String },
    #[error("Could not get function \"{name:}\" from module")]
    ModuleGetFunction { source: cu::Error, name: String },
    #[error("Launch failed")]
    KernelLaunch { source: cu::Error },
}
type Result<T, E = Error> = std::result::Result<T, E>;

pub fn init() -> Result<()> {
    let result = unsafe { sys::cuInit(0) };
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        Err(Error::Init {
            source: cu::Error(result),
        })
    } else {
        Ok(())
    }
}

pub struct Device {
    pub(crate) ptr: sys::CUdevice,
}

impl Device {
    pub fn get(ordinal: i32) -> Result<Device> {
        let mut ptr: sys::CUdevice = 0;
        unsafe {
            let result = sys::cuDeviceGet(&mut ptr, ordinal);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                Err(Error::DeviceGet {
                    source: cu::Error(result),
                    ordinal,
                })
            } else {
                Ok(Device { ptr })
            }
        }
    }

    pub fn ctx_create(&self, flags: ContextFlags) -> Result<Context> {
        let mut ptr: sys::CUcontext = std::ptr::null_mut();
        unsafe {
            let result = sys::cuCtxCreate(&mut ptr, flags.bits(), self.ptr);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                Err(Error::ContextCreate {
                    source: cu::Error(result),
                })
            } else {
                Ok(Context { ptr })
            }
        }
    }

    pub fn get_count() -> Result<i32> {
        let mut count: i32 = 0;
        let result = unsafe { sys::cuDeviceGetCount(&mut count) };
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            Err(Error::DeviceGetCount {
                source: cu::Error(result),
            })
        } else {
            Ok(count)
        }
    }
}

bitflags::bitflags! {
    pub struct ContextFlags: u32 {
        const AUTO = sys::CUctx_flags_enum_CU_CTX_SCHED_AUTO;
        const SPIN = sys::CUctx_flags_enum_CU_CTX_SCHED_SPIN;
        const YIELD = sys::CUctx_flags_enum_CU_CTX_SCHED_YIELD;
        const BLOCKING_SYNC = sys::CUctx_flags_enum_CU_CTX_SCHED_BLOCKING_SYNC;
        const SCHED_MASK = sys::CUctx_flags_enum_CU_CTX_SCHED_MASK;
        const MAP_HOST = sys::CUctx_flags_enum_CU_CTX_MAP_HOST;
        const LMEM_RESIZE_TO_MAX = sys::CUctx_flags_enum_CU_CTX_LMEM_RESIZE_TO_MAX;
    }
}

pub struct Context {
    pub(crate) ptr: sys::CUcontext,
}

impl Context {
    /// Blocks waiting for the context to finish its work
    pub fn synchronize() -> Result<()> {
        let result = unsafe { sys::cuCtxSynchronize() };
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            Err(Error::ContextSynchronize {
                source: cu::Error(result),
            })
        } else {
            Ok(())
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let result = sys::cuCtxDestroy(self.ptr);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                panic!("Error destroying context {:?}: {}", self.ptr, result);
            }
        }
    }
}

pub struct Module {
    ptr: sys::CUmodule,
}

impl Module {
    pub fn load_data(ptx: &str) -> Result<Module> {
        let mut ptr: sys::CUmodule = std::ptr::null_mut();
        let result = unsafe {
            sys::cuModuleLoadData(&mut ptr, ptx.as_ptr() as *const c_void)
        };
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            Err(Error::ModuleLoad {
                source: cu::Error(result),
                ptx: ptx.into(),
            })
        } else {
            Ok(Module { ptr })
        }
    }

    pub fn get_function(&self, name: &str) -> Result<Function> {
        let mut ptr: sys::CUfunction = std::ptr::null_mut();
        let result = unsafe {
            let cname = CString::new(name).unwrap();
            sys::cuModuleGetFunction(&mut ptr, self.ptr, cname.as_ptr())
        };
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            Err(Error::ModuleGetFunction {
                source: cu::Error(result),
                name: name.into(),
            })
        } else {
            Ok(Function { ptr })
        }
    }
}

// impl Drop for Module {
//     fn drop(&mut self) {
//         unsafe {
//             let result = sys::cuModuleUnload(self.ptr);
//             if result != sys::cudaError_enum::CUDA_SUCCESS {
//                 panic!("Error unloading module {:p}: {}", self.ptr, result);
//             }
//         }
//     }
// }

pub struct Function {
    pub(crate) ptr: sys::CUfunction,
}

impl Function {
    pub unsafe fn launch(
        &self,
        grid_dims: Dims,
        block_dims: Dims,
        shared_mem_bytes: usize,
        stream: &Stream,
        kernel_params: &[*const c_void],
    ) -> Result<()> {
        let result = sys::cuLaunchKernel(
            self.ptr,
            grid_dims.x,
            grid_dims.y,
            grid_dims.z,
            block_dims.x,
            block_dims.y,
            block_dims.z,
            shared_mem_bytes as u32,
            stream.ptr,
            kernel_params.as_ptr(),
            std::ptr::null(),
        );
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            Err(Error::KernelLaunch {
                source: cu::Error(result),
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Dims {
    x: u32,
    y: u32,
    z: u32,
}

impl Dims {
    pub fn x(x: u32) -> Dims {
        Dims { x, y: 1, z: 1 }
    }

    pub fn xy(x: u32, y: u32) -> Dims {
        Dims { x, y, z: 1 }
    }

    pub fn xyz(x: u32, y: u32, z: u32) -> Dims {
        Dims { x, y, z }
    }
}

pub struct Stream {
    pub(crate) ptr: sys::CUstream,
}

impl Stream {
    pub fn default() -> Stream {
        Stream {
            ptr: std::ptr::null_mut(),
        }
    }
}

pub unsafe fn launch_kernel(
    f: &Function,
    grid_dims: Dims,
    block_dims: Dims,
    shared_mem_bytes: usize,
    stream: &Stream,
    kernel_params: &[*const c_void],
    extra_options: &[*const c_void],
) -> Result<()> {
    let result = sys::cuLaunchKernel(
        f.ptr,
        grid_dims.x,
        grid_dims.y,
        grid_dims.z,
        block_dims.x,
        block_dims.y,
        block_dims.z,
        shared_mem_bytes as u32,
        stream.ptr,
        kernel_params.as_ptr(),
        extra_options.as_ptr(),
    );
    if result != sys::cudaError_enum::CUDA_SUCCESS {
        Err(Error::KernelLaunch {
            source: cu::Error(result),
        })
    } else {
        Ok(())
    }
}
