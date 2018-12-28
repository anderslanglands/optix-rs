use std::{ffi, fmt, io, ptr, result};
use crate::optix_bindings::*;
use crate::search_path;

#[derive(Debug)]
pub enum Error {
    Optix((RtResult, String)),
    Io(io::Error),
    Bounds,
    IncompatibleBuilderType,
    SearchPath(crate::search_path::Error),
    NulError(usize),
    HandleNotFoundError,
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::Io(err)
    }
}

impl From<search_path::Error> for Error {
    fn from(err: search_path::Error) -> Error {
        Error::SearchPath(err)
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(err: std::ffi::NulError) -> Error {
        Error::NulError(err.nul_position())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, output: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Optix(err) => {
                write!(output, "[ERROR OptiX {:?}] {}", err.0, err.1)
            }
            Error::Io(err) => write!(output, "[ERROR IO] {}", err),
            Error::Bounds => write!(output, "[ERROR out of bounds]"),
            Error::IncompatibleBuilderType => {
                write!(output, "[ERROR incompatible builder type]")
            },
            Error::SearchPath(err) => {
                write!(output, "[ERROR SearchPath {}", err)
            },
            Error::HandleNotFoundError => {
                write!(output, "[ERROR Hande Not Found]")
            }
            Error::NulError(pos) => {
                write!(output, "[ERROR Nul byte at position {}]", pos)
            }
        }
    }
}

impl Error {
    pub fn description(&self) -> String {
        format!("{}", self)
    }
}

pub(crate) type Result<T> = result::Result<T, Error>;

pub fn optix_error(msg: &str, ctx: RTcontext, result: RtResult) -> Error {
    Error::Optix((
        result,
        format!("{}: {}", msg, get_error_string(ctx, result)),
    ))
}

pub(crate) fn get_error_string(ctx: RTcontext, code: RtResult) -> String {
    let mut tmp: *const ::std::os::raw::c_char = ptr::null_mut();
    unsafe {
        rtContextGetErrorString(ctx, code, &mut tmp);
        ffi::CStr::from_ptr(tmp).to_string_lossy().into_owned()
    }
}

