pub mod context;
pub mod error;
pub mod ginallocator;
mod optix_bindings;
pub mod search_path;
pub mod math;

use crate::error::{Error, Result};
use crate::optix_bindings::{rtGetVersion, RtResult};

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
