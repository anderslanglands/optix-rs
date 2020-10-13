use cu::sys::{CUcontext, CUdeviceptr, CUstream};

include!(concat!(env!("OUT_DIR"), "/optix_wrapper.rs"));

pub type size_t = usize;

macro_rules! const_assert {
    ($x:expr $(,)?) => {
        #[allow(unknown_lints, eq_op)]
        const _: [(); 0 - !{
            const ASSERT: bool = $x;
            ASSERT
        } as usize] = [];
    };
}

// check that our configuration matches the optix version we're building against
cfg_if::cfg_if! {
if #[cfg(feature="optix72")] {
    const_assert!(OptixVersion == 70200);
} else {
    const_assert!(OptixVersion == 70100);
}

}

extern "C" {
    pub fn optixInit() -> OptixResult;
}

// The SBT record header is an opaque blob used by optix
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

// Manually define the build input union as the bindgen is pretty nasty
#[repr(C)]
pub union OptixBuildInputUnion {
    pub triangle_array: OptixBuildInputTriangleArray,
    pub curve_array: OptixBuildInputCurveArray,
    pub custom_primitive_array: OptixBuildInputCustomPrimitiveArray,
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

// Sanity check that the size of this union we're defining matches the one in
// optix header so we don't get any nasty surprises
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
