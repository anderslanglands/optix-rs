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

pub use self::context::program::*;
pub use self::context::geometry::*;
pub use self::context::material::*;
pub use self::context::geometry_instance::*;
pub use self::context::buffer::*;
pub use self::context::variable::*;
pub use self::context::acceleration::*;
pub use self::context::geometry_group::*;
pub use self::context::transform::*;
pub use self::context::group::*;
pub use self::context::*;

use crate::error::Result;
pub use crate::error::Error;
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
