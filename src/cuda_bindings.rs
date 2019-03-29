#[link(name = "cudart", kind = "dylib")]
extern "C" {
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> u32;
}
