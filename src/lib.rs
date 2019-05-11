//! # OptiX
//!
//! An oxidized wrapper for [NVidia's OptiX GPU raytracing library](https://developer.nvidia.com/optix)
//!
use cfg_if::cfg_if;

cfg_if! {
if #[cfg(feature="optix5")] {
    mod optix5_bindings;
    use optix5_bindings as optix_bindings;
} else {
    mod optix6_bindings;
    use optix6_bindings as optix_bindings;
}

}

pub mod context;
pub mod cuda;
pub use cuda::mem_get_info;
mod cuda_bindings;
pub mod error;
pub mod math;
pub mod nvrtc;
mod nvrtc_bindings;

pub mod search_path;

pub use self::context::acceleration::*;
pub use self::context::buffer::*;
pub use self::context::geometry::*;
pub use self::context::geometry_group::*;
pub use self::context::geometry_instance::*;
#[cfg(not(feature = "optix5"))]
pub use self::context::geometry_triangles::*;
pub use self::context::group::*;
pub use self::context::material::*;
pub use self::context::program::*;
pub use self::context::texture_sampler::*;
pub use self::context::transform::*;
pub use self::context::variable::*;
pub use self::context::*;

pub use crate::error::Error;
use crate::error::Result;
pub use crate::optix_bindings::{
    rtGetVersion, rtVariableSetUserData, MotionKeyType, RTsize, RTvariable,
    RtResult, WrapMode,
};

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

cfg_if! {
if #[cfg(feature="optix5")] {
pub fn format_get_size(f: Format) -> usize {
    match f {
        Format::UNKNOWN => 0,
        Format::FLOAT => 4,
        Format::FLOAT2 => 8,
        Format::FLOAT3 => 12,
        Format::FLOAT4 => 16,
        Format::BYTE => 1,
        Format::BYTE2 => 2,
        Format::BYTE3 => 3,
        Format::BYTE4 => 4,
        Format::UNSIGNED_BYTE => 1,
        Format::UNSIGNED_BYTE2 => 2,
        Format::UNSIGNED_BYTE3 => 3,
        Format::UNSIGNED_BYTE4 => 4,
        Format::SHORT => 2,
        Format::SHORT2 => 4,
        Format::SHORT3 => 6,
        Format::SHORT4 => 8,
        Format::UNSIGNED_SHORT => 2,
        Format::UNSIGNED_SHORT2 => 4,
        Format::UNSIGNED_SHORT3 => 6,
        Format::UNSIGNED_SHORT4 => 8,
        Format::INT => 4,
        Format::INT2 => 8,
        Format::INT3 => 12,
        Format::INT4 => 16,
        Format::UNSIGNED_INT => 4,
        Format::UNSIGNED_INT2 => 8,
        Format::UNSIGNED_INT3 => 12,
        Format::UNSIGNED_INT4 => 16,
        Format::USER => 0,
        Format::BUFFER_ID => 4,
        Format::PROGRAM_ID => 4,
        Format::HALF => 2,
        Format::HALF2 => 4,
        Format::HALF3 => 6,
        Format::HALF4 => 8,
    }
}
} else {
pub fn format_get_size(f: Format) -> usize {
    match f {
        Format::UNKNOWN => 0,
        Format::FLOAT => 4,
        Format::FLOAT2 => 8,
        Format::FLOAT3 => 12,
        Format::FLOAT4 => 16,
        Format::BYTE => 1,
        Format::BYTE2 => 2,
        Format::BYTE3 => 3,
        Format::BYTE4 => 4,
        Format::UNSIGNED_BYTE => 1,
        Format::UNSIGNED_BYTE2 => 2,
        Format::UNSIGNED_BYTE3 => 3,
        Format::UNSIGNED_BYTE4 => 4,
        Format::SHORT => 2,
        Format::SHORT2 => 4,
        Format::SHORT3 => 6,
        Format::SHORT4 => 8,
        Format::UNSIGNED_SHORT => 2,
        Format::UNSIGNED_SHORT2 => 4,
        Format::UNSIGNED_SHORT3 => 6,
        Format::UNSIGNED_SHORT4 => 8,
        Format::INT => 4,
        Format::INT2 => 8,
        Format::INT3 => 12,
        Format::INT4 => 16,
        Format::UNSIGNED_INT => 4,
        Format::UNSIGNED_INT2 => 8,
        Format::UNSIGNED_INT3 => 12,
        Format::UNSIGNED_INT4 => 16,
        Format::USER => 0,
        Format::BUFFER_ID => 4,
        Format::PROGRAM_ID => 4,
        Format::HALF => 2,
        Format::HALF2 => 4,
        Format::HALF3 => 6,
        Format::HALF4 => 8,
        Format::LONG_LONG => 8,
        Format::LONG_LONG2 => 16,
        Format::LONG_LONG3 => 24,
        Format::LONG_LONG4 => 32,
        Format::UNSIGNED_LONG_LONG => 8,
        Format::UNSIGNED_LONG_LONG2 => 16,
        Format::UNSIGNED_LONG_LONG3 => 24,
        Format::UNSIGNED_LONG_LONG4 => 32,
        Format::UNSIGNED_BC1 => 8,
        Format::UNSIGNED_BC2 => 16,
        Format::UNSIGNED_BC3 => 16,
        Format::UNSIGNED_BC4 => 8,
        Format::BC4 => 8,
        Format::UNSIGNED_BC5 => 16,
        Format::BC5 => 16,
        Format::UNSIGNED_BC6H => 16,
        Format::BC6H => 16,
        Format::UNSIGNED_BC7 => 16,
    }
}
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
