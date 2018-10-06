//! # OptiX
//!
//! An oxidized wrapper for [NVidia's OptiX GPU raytracing library](https://developer.nvidia.com/optix)
//!

pub mod context;
pub mod error;
pub mod ginallocator;
pub mod math;
mod optix_bindings;
pub mod search_path;

use crate::error::{Error, Result};
use crate::optix_bindings::{rtGetVersion, RtResult};

/// Returns the version of the OptiX library in use.
pub fn get_version() -> Result<u32> {
    let mut version: ::std::os::raw::c_uint = 0;
    let result = unsafe { rtGetVersion(&mut version) };
    if result == RtResult::SUCCESS {
        Ok(version)
    } else {
        Err(Error::Optix((result, "rtGetVersion failed".to_owned())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tst_get_version() {
        println!("OptiX Version {}", get_version().unwrap());
    }
}
