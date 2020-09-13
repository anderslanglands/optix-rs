use cu::sys::{CUcontext, CUdeviceptr, CUstream};

include!(concat!(env!("OUT_DIR"), "/optix_wrapper.rs"));

pub type size_t = usize;

extern "C" {
    pub fn optixInit() -> OptixResult;
}

#[repr(C)]
pub struct SbtRecordHeader {
    header: [u8; OptixSbtRecordHeaderSize as usize],
}

impl SbtRecordHeader {
    pub fn as_mut_ptr(&mut self) -> *mut std::os::raw::c_void {
        self.header.as_mut_ptr() as *mut std::os::raw::c_void
    }
}

impl Default for SbtRecordHeader {
    fn default() -> SbtRecordHeader {
        SbtRecordHeader {
            header: [0u8; OptixSbtRecordHeaderSize as usize],
        }
    }
}

#[repr(C)]
pub union OptixBuildInputUnion {
    pub triangle_array: OptixBuildInputTriangleArray,
    pub aabb_array: OptixBuildInputCustomPrimitiveArray,
    pub instance_array: OptixBuildInputInstanceArray,
    pad: [std::os::raw::c_char; 1024],
}

impl Default for OptixBuildInputUnion {
    fn default() -> OptixBuildInputUnion {
        OptixBuildInputUnion { pad: [0i8; 1024] }
    }
}

#[repr(C)]
pub struct OptixBuildInput {
    pub type_: OptixBuildInputType,
    pub input: OptixBuildInputUnion,
}

fn _size_check() {
    unsafe {
        std::mem::transmute::<OptixBuildInput, [u8; 1024 + 8]>(
            OptixBuildInput {
                type_: OptixBuildInputType_OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                input: { OptixBuildInputUnion { pad: [0; 1024] } },
            },
        );
    }
}

#[repr(C)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub struct OptixModuleCompileOptions {
    pub max_register_count: i32,
    pub opt_level: CompileOptimizationLevel,
    pub debug_level: CompileDebugLevel,
}

#[repr(C)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub struct OptixPipelineLinkOptions {
    pub max_trace_depth: u32,
    pub debug_level: CompileDebugLevel,
}

#[repr(u32)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub enum CompileOptimizationLevel {
    Default =
        OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
    Level0 = OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
    Level1 = OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_1,
    Level2 = OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_2,
    Level3 = OptixCompileOptimizationLevel::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3,
}

#[repr(u32)]
#[derive(Debug, Hash, PartialEq, Copy, Clone)]
pub enum CompileDebugLevel {
    None = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_NONE,
    LineInfo = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO,
    Full = OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_FULL,
}
