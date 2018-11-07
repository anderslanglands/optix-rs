#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::os::raw::c_char;

#[repr(u32)]
#[derive(PartialEq)]
pub enum NvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _nvrtcProgram {
    _unused: [u8; 0],
}
pub type nvrtcProgram = *mut _nvrtcProgram;

#[link(name = "nvrtc", kind = "dylib")]
extern "C" {
    pub fn nvrtcGetErrorString(result: NvrtcResult) -> *const c_char;
    pub fn nvrtcGetVersion(major: *const i32, minor: *const i32)
        -> NvrtcResult;

    pub fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: i32,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> NvrtcResult;

    pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> NvrtcResult;

    pub fn nvrtcCompileProgram(
        prog: nvrtcProgram,
        num_options: i32,
        options: *const *const c_char,
    ) -> NvrtcResult;

    pub fn nvrtcGetPTXSize(
        prog: nvrtcProgram,
        ptx_size: *mut usize,
    ) -> NvrtcResult;

    pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> NvrtcResult;

    pub fn nvrtcGetProgramLogSize(
        prog: nvrtcProgram,
        log_size: *mut usize,
    ) -> NvrtcResult;

    pub fn nvrtcGetProgramLog(
        prog: nvrtcProgram,
        log: *mut c_char,
    ) -> NvrtcResult;
}
