use optix_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

pub use super::device_context::DeviceContext;

use std::ffi::{CStr, CString};

#[repr(u32)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub enum CompileOptimizationLevel {
    Level0 =
        sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
    Level1 =
        sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
    Level2 =
        sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_2,
    Level3 =
        sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
}

#[repr(u32)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub enum CompileDebugLevel {
    None = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
    LineInfo = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO,
    FULL = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL,
}

#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub struct ModuleCompileOptions {
    pub max_register_count: i32,
    pub opt_level: CompileOptimizationLevel,
    pub debug_level: CompileDebugLevel,
}

impl From<ModuleCompileOptions> for sys::OptixModuleCompileOptions {
    fn from(o: ModuleCompileOptions) -> sys::OptixModuleCompileOptions {
        sys::OptixModuleCompileOptions {
            maxRegisterCount: o.max_register_count,
            optLevel: o.opt_level as u32,
            debugLevel: o.debug_level as u32,
        }
    }
}

bitflags! {
    pub struct TraversableGraphFlags: u32 {
        const ALLOW_ANY = sys::OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        const ALLOW_SINGLE_GAS = sys::OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        const ALLOW_SINGLE_LEVEL_INSTANCING = sys::OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    }
}

bitflags! {
    pub struct ExceptionFlags: u32 {
        const NONE = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE;
        const STACK_OVERFLOW = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        const TRACE_DEPTH = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
        const USER = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_USER;
        const DEBUG = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_DEBUG;
    }
}

#[derive(Debug, Hash, PartialEq, Clone)]
pub struct PipelineCompileOptions {
    pub uses_motion_blur: bool,
    pub traversable_graph_flags: TraversableGraphFlags,
    pub num_payload_values: i32,
    pub num_attribute_values: i32,
    pub exception_flags: ExceptionFlags,
    pub pipeline_launch_params_variable_name: String,
}

pub struct Module {
    pub(crate) module: sys::OptixModule,
}

pub type ModuleRef = super::Ref<Module>;

impl DeviceContext {
    pub fn module_create_from_ptx(
        &mut self,
        module_compile_options: ModuleCompileOptions,
        pipeline_compile_options: &PipelineCompileOptions,
        ptx: &str,
    ) -> Result<(ModuleRef, String)> {
        let cptx = CString::new(ptx).unwrap();
        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let launch_param = CString::new(
            pipeline_compile_options
                .pipeline_launch_params_variable_name
                .as_str(),
        )
        .unwrap();

        let popt = sys::OptixPipelineCompileOptions {
            usesMotionBlur: if pipeline_compile_options.uses_motion_blur {
                1
            } else {
                0
            },
            traversableGraphFlags: pipeline_compile_options
                .traversable_graph_flags
                .bits(),
            numPayloadValues: pipeline_compile_options.num_payload_values,
            numAttributeValues: pipeline_compile_options.num_attribute_values,
            exceptionFlags: pipeline_compile_options.exception_flags.bits(),
            pipelineLaunchParamsVariableName: launch_param.as_ptr(),
        };

        let mopt = module_compile_options.into();
        let mut module = std::ptr::null_mut();
        let res = unsafe {
            sys::optixModuleCreateFromPTX(
                self.ctx,
                &mopt,
                &popt,
                cptx.as_ptr(),
                cptx.as_bytes().len(),
                log.as_mut_ptr() as *mut i8,
                &mut log_len,
                &mut module,
            )
        };

        let log = CStr::from_bytes_with_nul(&log[0..log_len])
            .unwrap()
            .to_string_lossy()
            .into_owned();

        if res != sys::OptixResult::OPTIX_SUCCESS {
            return Err(Error::ModuleCreationFailed {
                source: res.into(),
                log,
            });
        }

        let module = super::Ref::new(Module { module });
        // self.modules.push(super::Ref::clone(&module));
        Ok((module, log))
    }
}
