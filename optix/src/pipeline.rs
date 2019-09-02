use optix_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

use super::device_context::DeviceContext;
use super::module::{CompileDebugLevel, PipelineCompileOptions};
use super::program_group::ProgramGroupRef;

use std::ffi::{CStr, CString};

pub struct PipelineLinkOptions {
    pub max_trace_depth: u32,
    pub debug_level: CompileDebugLevel,
    pub override_uses_motion_blur: bool,
}

impl From<PipelineLinkOptions> for sys::OptixPipelineLinkOptions {
    fn from(o: PipelineLinkOptions) -> sys::OptixPipelineLinkOptions {
        sys::OptixPipelineLinkOptions {
            maxTraceDepth: o.max_trace_depth,
            debugLevel: o.debug_level as u32,
            overrideUsesMotionBlur: o.override_uses_motion_blur as i32,
        }
    }
}

pub struct Pipeline {
    pub(crate) pipeline: sys::OptixPipeline,
}

pub type PipelineRef = super::Ref<Pipeline>;

impl DeviceContext {
    pub fn pipeline_create(
        &mut self,
        pipeline_compile_options: &PipelineCompileOptions,
        link_options: PipelineLinkOptions,
        program_groups: &[ProgramGroupRef],
    ) -> Result<(PipelineRef, String)> {
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

        let link_options: sys::OptixPipelineLinkOptions = link_options.into();

        let pgs: Vec<sys::OptixProgramGroup> =
            program_groups.iter().map(|pg| pg.pg).collect();

        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let mut pipeline: sys::OptixPipeline = std::ptr::null_mut();

        let res = unsafe {
            sys::optixPipelineCreate(
                self.ctx,
                &popt,
                &link_options,
                pgs.as_ptr(),
                pgs.len() as u32,
                log.as_mut_ptr() as *mut i8,
                &mut log_len,
                &mut pipeline,
            )
        };

        let log = CStr::from_bytes_with_nul(&log[0..log_len])
            .unwrap()
            .to_string_lossy()
            .into_owned();

        if res != sys::OptixResult::OPTIX_SUCCESS {
            return Err(Error::PipelineCreationFailed {
                cerr: res.into(),
                log,
            });
        }
        let pipeline = super::Ref::new(Pipeline { pipeline });
        self.pipelines.push(super::Ref::clone(&pipeline));
        Ok((pipeline, log))
    }

    /// Sets the stack sizes for a pipeline.
    ///
    /// Users are encouraged to see the programming guide and the
    /// implementations of the helper functions to understand how to
    /// construct the stack sizes based on their particular needs.
    /// If this method is not used, an internal default implementation is used.
    /// The default implementation is correct (but not necessarily optimal) as
    /// long as the maximum depth of call trees of CC and DC programs is at most
    /// 2 and no motion transforms are used.
    /// The maxTraversableGraphDepth responds to the maximal number of
    /// traversables visited when calling trace. Every acceleration structure
    /// and motion transform count as one level of traversal. E.g., for a simple
    /// IAS (instance acceleration structure) -> GAS (geometry acceleration
    /// structure) traversal graph, the maxTraversableGraphDepth is two. For
    /// IAS -> MT (motion transform) -> GAS, the maxTraversableGraphDepth is
    /// three. Note that it does not matter whether a IAS or GAS has motion
    /// or not, it always counts as one. Launching optix with exceptions
    /// turned on (see OPTIX_EXCEPTION_FLAG_TRACE_DEPTH) will throw an
    /// exception if the specified maxTraversableGraphDepth is too small.
    ///
    /// #Arguments
    /// * `direct_callable_stack_size_from_traversable` - The direct stack size
    /// requirement for direct callables invoked from IS or AH
    /// * `direct_callable_stack_size_from_state` - The direct stack size
    /// requirement for direct callables invoked from RG, MS, or CH.
    /// * `continuation_stack_size` - The continuation stack requirement.
    /// * `max_traversable_graph_depth` - The maximum depth of a traversable
    ///   graph
    /// passed to trace
    ///
    /// # Panics
    /// If the FFI call to optixPipelineSetStackSize returns an error
    pub fn pipeline_set_stack_size(
        &self,
        pipeline: &mut PipelineRef,
        direct_callable_stack_size_from_traversable: u32,
        direct_callable_stack_size_from_state: u32,
        continuation_stack_size: u32,
        max_traversable_graph_depth: u32,
    ) {
        let res = unsafe {
            sys::optixPipelineSetStackSize(
                pipeline.pipeline,
                direct_callable_stack_size_from_traversable,
                direct_callable_stack_size_from_state,
                continuation_stack_size,
                max_traversable_graph_depth,
            )
        };
        if res != sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixPipelineSetStackSize failed");
        }
    }
}
