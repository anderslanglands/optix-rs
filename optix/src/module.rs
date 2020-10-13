use crate::{sys, DeviceContext, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

use std::ffi::{CStr, CString};

#[derive(Clone)]
#[repr(transparent)]
pub struct Module {
    pub(crate) inner: sys::OptixModule,
}

#[repr(u32)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub enum CompileOptimizationLevel {
    Default =
        sys::OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
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
    Full = sys::OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL,
}

cfg_if::cfg_if! {
    if #[cfg(feature="optix72")] {
        #[repr(C)]
        #[derive(Debug, Hash, PartialEq, Copy, Clone)]
        pub struct ModuleCompileOptions {
            pub max_register_count: i32,
            pub opt_level: CompileOptimizationLevel,
            pub debug_level: CompileDebugLevel,
        }

        impl From<&ModuleCompileOptions> for sys::OptixModuleCompileOptions {
            fn from(o: &ModuleCompileOptions) -> sys::OptixModuleCompileOptions {
                sys::OptixModuleCompileOptions {
                    maxRegisterCount: o.max_register_count,
                    optLevel: o.opt_level as u32,
                    debugLevel: o.debug_level as u32,
                    boundValues: std::ptr::null(),
                    numBoundValues: 0,
                }
            }
        }
    } else {
        #[repr(C)]
        #[derive(Debug, Hash, PartialEq, Copy, Clone)]
        pub struct ModuleCompileOptions {
            pub max_register_count: i32,
            pub opt_level: CompileOptimizationLevel,
            pub debug_level: CompileDebugLevel,
        }

        impl From<&ModuleCompileOptions> for sys::OptixModuleCompileOptions {
            fn from(o: &ModuleCompileOptions) -> sys::OptixModuleCompileOptions {
                sys::OptixModuleCompileOptions {
                    maxRegisterCount: o.max_register_count,
                    optLevel: o.opt_level as u32,
                    debugLevel: o.debug_level as u32,
                }
            }
        }
    }
}

bitflags::bitflags! {
    pub struct TraversableGraphFlags: u32 {
        const ALLOW_ANY = sys::OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        const ALLOW_SINGLE_GAS = sys::OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        const ALLOW_SINGLE_LEVEL_INSTANCING = sys::OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    }
}

bitflags::bitflags! {
    pub struct ExceptionFlags: u32 {
        const NONE = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_NONE;
        const STACK_OVERFLOW = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        const TRACE_DEPTH = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
        const USER = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_USER;
        const DEBUG = sys::OptixExceptionFlags::OPTIX_EXCEPTION_FLAG_DEBUG;
    }
}

bitflags::bitflags! {
    pub struct PrimitiveTypeFlags: i32 {
        const DEFAULT = 0;
        const CUSTOM =  sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        const ROUND_QUADRATIC_BSPLINE = sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE;
        const ROUND_CUBIC_BSPLINE =  sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
        const ROUND_LINEAR =  sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR;
        const TRIANGLE = sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    }
}

#[repr(u32)]
pub enum PrimitiveType {
    RoundQuadraticBspline = sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE as u32,
    RoundCubicBspline =  sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE as u32,
    RoundLinear =  sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR as u32,
    Triangle = sys::OptixPrimitiveTypeFlags_OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE as u32,
}

#[derive(Debug, Hash, PartialEq, Clone)]
pub struct PipelineCompileOptions {
    uses_motion_blur: bool,
    traversable_graph_flags: TraversableGraphFlags,
    num_payload_values: i32,
    num_attribute_values: i32,
    exception_flags: ExceptionFlags,
    pipeline_launch_params_variable_name: Option<ustr::Ustr>,
    primitive_type_flags: PrimitiveTypeFlags,
}

impl PipelineCompileOptions {
    pub fn new() -> PipelineCompileOptions {
        PipelineCompileOptions {
            uses_motion_blur: false,
            traversable_graph_flags: TraversableGraphFlags::ALLOW_ANY,
            num_payload_values: 0,
            num_attribute_values: 0,
            exception_flags: ExceptionFlags::NONE,
            pipeline_launch_params_variable_name: None,
            primitive_type_flags: PrimitiveTypeFlags::DEFAULT,
        }
    }

    pub fn build(&self) -> Result<sys::OptixPipelineCompileOptions> {
        if let Some(name) = self.pipeline_launch_params_variable_name {
            Ok(sys::OptixPipelineCompileOptions {
                usesMotionBlur: if self.uses_motion_blur { 1 } else { 0 },
                traversableGraphFlags: self.traversable_graph_flags.bits(),
                numPayloadValues: self.num_payload_values,
                numAttributeValues: self.num_attribute_values,
                exceptionFlags: self.exception_flags.bits(),
                pipelineLaunchParamsVariableName: unsafe { name.as_char_ptr() },
                usesPrimitiveTypeFlags: self.primitive_type_flags.bits() as u32,
            })
        } else {
            Err(Error::PipelineLaunchParamsVariableNameNotSpecified)
        }
    }

    pub fn uses_motion_blur(mut self, umb: bool) -> Self {
        self.uses_motion_blur = umb;
        self
    }

    pub fn traversable_graph_flags(
        mut self,
        tgf: TraversableGraphFlags,
    ) -> Self {
        self.traversable_graph_flags = tgf;
        self
    }

    pub fn num_payload_values(mut self, npv: i32) -> Self {
        self.num_payload_values = npv;
        self
    }

    pub fn num_attribute_values(mut self, nav: i32) -> Self {
        self.num_attribute_values = nav;
        self
    }

    pub fn exception_flags(mut self, ef: ExceptionFlags) -> Self {
        self.exception_flags = ef;
        self
    }

    pub fn pipeline_launch_params_variable_name(
        mut self,
        name: ustr::Ustr,
    ) -> Self {
        self.pipeline_launch_params_variable_name = Some(name);
        self
    }
}

/// # Creating and destroying `Module`s
impl DeviceContext {
    pub fn module_create_from_ptx(
        &mut self,
        module_compile_options: &ModuleCompileOptions,
        pipeline_compile_options: &PipelineCompileOptions,
        ptx: &str,
    ) -> Result<(Module, String)> {
        let cptx = CString::new(ptx).unwrap();
        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let mopt = module_compile_options.into();
        let popt = pipeline_compile_options.build()?;

        let mut inner = std::ptr::null_mut();
        let res = unsafe {
            sys::optixModuleCreateFromPTX(
                self.inner,
                &mopt as *const _,
                &popt,
                cptx.as_ptr(),
                cptx.as_bytes().len(),
                log.as_mut_ptr() as *mut i8,
                &mut log_len,
                &mut inner,
            )
            .to_result()
        };

        let log = CStr::from_bytes_with_nul(&log[0..log_len])
            .unwrap()
            .to_string_lossy()
            .into_owned();

        match res {
            Ok(()) => Ok((Module { inner }, log)),
            Err(source) => Err(Error::ModuleCreation { source, log }),
        }
    }

    pub fn builtin_is_module_get(
        &self,
        module_compile_options: &ModuleCompileOptions,
        pipeline_compile_options: &PipelineCompileOptions,
        builtin_is_module_type: PrimitiveType,
        uses_motion_blur: bool,
    ) -> Result<Module> {
        let is_options = sys::OptixBuiltinISOptions {
            builtinISModuleType: builtin_is_module_type as u32,
            usesMotionBlur: if uses_motion_blur { 1 } else { 0 },
        };

        let mut inner = std::ptr::null_mut();

        unsafe {
            sys::optixBuiltinISModuleGet(
                self.inner,
                module_compile_options as *const _ as *const _,
                pipeline_compile_options as *const _ as *const _,
                &is_options as *const _,
                &mut inner,
            )
            .to_result()
            .map(|_| Module { inner })
            .map_err(|source| Error::BuiltinIsModuleGet { source })
        }
    }

    /// Destroy a module created with [DeviceContext::module_create_from_ptx()]
    /// # Safety
    /// Modules must not be destroyed while they are still used by any program
    /// group.
    /// A Module must not be destroyed while it is
    /// still in use by concurrent API calls in other threads.
    pub fn module_destroy(&mut self, module: Module) -> Result<()> {
        unsafe {
            sys::optixModuleDestroy(module.inner)
                .to_result()
                .map_err(|source| Error::ModuleDestroy { source })
        }
    }
}
