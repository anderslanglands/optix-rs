use crate::cuda_bindings::*;

pub fn mem_get_info() -> (usize, usize) {
    unsafe {
        let mut free = 0;
        let mut total = 0;

        let result =
            cudaMemGetInfo(&mut free as *mut usize, &mut total as *mut usize);

        (free, total)
    }
}

#[test]
fn test_mem_info() {
    let (free, total) = mem_get_info();
    println!("free: {}, total: {}", free, total);
}
