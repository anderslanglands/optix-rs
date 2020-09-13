use crate::{sys, DeviceContext, Error};
type Result<T, E = Error> = std::result::Result<T, E>;

pub use sys::{CompileDebugLevel, CompileOptimizationLevel};
pub type ModuleCompileOptions = sys::OptixModuleCompileOptions;

use std::ffi::{CStr, CString};

#[derive(Clone)]
#[repr(transparent)]
pub struct Module {
    pub(crate) inner: sys::OptixModule,
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

        let popt = pipeline_compile_options.build()?;

        let mut inner = std::ptr::null_mut();
        let res = unsafe {
            sys::optixModuleCreateFromPTX(
                self.inner,
                module_compile_options,
                &popt,
                cptx.as_ptr(),
                cptx.as_bytes().len(),
                log.as_mut_ptr() as *mut i8,
                &mut log_len,
                &mut inner,
            ).to_result()
        };

        let log = CStr::from_bytes_with_nul(&log[0..log_len])
            .unwrap()
            .to_string_lossy()
            .into_owned();

        match res {
            Ok(()) => Ok((Module{inner}, log)),
            Err(source) => Err(Error::ModuleCreationFailed { source, log }),
        }
    }
}
