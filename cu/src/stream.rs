use crate::{sys, Error, Function, Dim};
type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Copy, Clone)]
pub struct Stream {
    pub(crate) inner: sys::CUstream,
}

impl Stream {
    pub fn create(flags: StreamFlags) -> Result<Stream> {
        let mut inner = std::ptr::null_mut();
        unsafe {
            sys::cuStreamCreate(&mut inner, flags.bits())
                .to_result()
                .map(|_| Stream { inner })
        }
    }

    pub fn inner(&self) -> sys::CUstream {
        self.inner
    }

    pub fn launch_kernel<D1: Into<Dim>, D2: Into<Dim>>(
        &self,
        f: Function,
        grid_dim: D1,
        block_dim: D2,
        shared_mem_bytes: u32,
        params: &[*mut std::ffi::c_void]
    ) -> Result<()> {
        let grid_dim: Dim = grid_dim.into();
        let block_dim: Dim = block_dim.into();
        unsafe {
            sys::cuLaunchKernel(
                f.inner,
                grid_dim.x,
                grid_dim.y,
                grid_dim.z,
                block_dim.x,
                block_dim.y,
                block_dim.z,
                shared_mem_bytes,
                self.inner,
                params.as_ptr() as *mut _,
                std::ptr::null_mut(),
            )
            .to_result()
        }
    }

}

bitflags::bitflags! {
    pub struct StreamFlags: u32 {
        const DEFAULT = sys::CUstream_flags_enum_CU_STREAM_DEFAULT;
        const NON_BLOCKING = sys::CUstream_flags_enum_CU_STREAM_NON_BLOCKING;
    }
}
