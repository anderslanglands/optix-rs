use optix::cuda;
use optix_sys::cuda_sys::{self, cudaFree, cudaGetDeviceCount};
use optix_sys::{optixInit, OptixResult};

static CU_SRC: &str = r#"
extern "C" __global__ void saxpy(float a, float* x, float* y, float* out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a * x[tid] + y[tid];
    }
}
"#;

static NUM_THREADS: usize = 128;
static NUM_BLOCKS: usize = 32;

fn main() {
    unsafe {
        cudaFree(std::ptr::null_mut());
        let mut num_devices = 0i32;
        cudaGetDeviceCount(&mut num_devices as *mut i32);
        if num_devices == 0 {
            panic!("No CUDA devices found");
        }
        println!("Found {} CUDA devices", num_devices);

        let header = cuda::nvrtc::Header {
            name: "launch_params.h".into(),
            contents: "".into(),
        };

        let ptx = compile_to_ptx(CU_SRC, header);
        let ptx = std::ffi::CString::new(ptx).unwrap();
        let cstr_saxpy = std::ffi::CString::new("saxpy").unwrap();

        let mut device: cuda_sys::CUdevice = 0;
        let mut context: cuda_sys::CUcontext = std::ptr::null_mut();
        let mut module: cuda_sys::CUmodule = std::ptr::null_mut();
        let mut kernel: cuda_sys::CUfunction = std::ptr::null_mut();

        let result = cuda_sys::cuDeviceGet(&mut device, 0);
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuDeviceGet");
        }
        let result = cuda_sys::cuCtxCreate(&mut context, 0, device);
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuCtxCreate");
        }
        let result = cuda_sys::cuModuleLoadDataEx(
            &mut module,
            ptx.as_ptr() as *const std::os::raw::c_void,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuModuleLoadDataEx");
        }
        let result = cuda_sys::cuModuleGetFunction(
            &mut kernel,
            module,
            cstr_saxpy.as_ptr(),
        );
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuModuleGetFunction");
        }

        let mut n = NUM_THREADS * NUM_BLOCKS;
        let buffer_size = n * std::mem::size_of::<f32>();
        let mut a = 5.1f32;

        let mut h_x = Vec::with_capacity(n);
        let mut h_y = Vec::with_capacity(n);
        let mut h_out = vec![0f32; n];
        for i in 0..n {
            h_x.push(i as f32);
            h_y.push(i as f32 * 2.0);
        }

        let mut d_x: cuda_sys::CUdeviceptr = 0;
        let mut d_y: cuda_sys::CUdeviceptr = 0;
        let mut d_out: cuda_sys::CUdeviceptr = 0;

        let result = cuda_sys::cuMemAlloc(&mut d_x, buffer_size);
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuMemAlloc d_x");
        }
        let result = cuda_sys::cuMemAlloc(&mut d_y, buffer_size);
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuMemAlloc d_y");
        }
        let result = cuda_sys::cuMemAlloc(&mut d_out, buffer_size);
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuMemAlloc d_out");
        }

        let result = cuda_sys::cuMemcpyHtoD(
            d_x,
            h_x.as_ptr() as *const std::os::raw::c_void,
            buffer_size,
        );
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuMemcpy d_x");
        }
        let result = cuda_sys::cuMemcpyHtoD(
            d_y,
            h_y.as_ptr() as *const std::os::raw::c_void,
            buffer_size,
        );
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuMemcpy d_y");
        }

        let mut args: [*mut std::os::raw::c_void; 5] = [
            &mut a as *mut f32 as *mut std::os::raw::c_void,
            &mut d_x as *mut u64 as *mut std::os::raw::c_void,
            &mut d_y as *mut u64 as *mut std::os::raw::c_void,
            &mut d_out as *mut u64 as *mut std::os::raw::c_void,
            &mut n as *mut usize as *mut std::os::raw::c_void,
        ];

        let result = cuda_sys::cuLaunchKernel(
            kernel,
            NUM_BLOCKS as u32,
            1,
            1,
            NUM_THREADS as u32,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cudaLaunchKernel");
        }

        let result = cuda_sys::cuCtxSynchronize();
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuCtxSynchronize");
        }

        let result = cuda_sys::cuMemcpyDtoH(
            h_out.as_mut_ptr() as *mut std::os::raw::c_void,
            d_out,
            buffer_size,
        );
        if result != cuda_sys::cudaError::cudaSuccess {
            panic!("cuMemcpyDtoH");
        }

        // println!("{:?}", h_out);
    }
}

fn compile_to_ptx(src: &str, header: cuda::nvrtc::Header) -> String {
    use cuda::nvrtc::Program;

    let optix_root = std::env::var("OPTIX_ROOT")
        .expect("OPTIX_ROOT not found. You must set OPTIX_ROOT either as an environment variable, or in build-settings.toml to point to the root of your OptiX installation.");

    let cuda_root = std::env::var("CUDA_ROOT")
        .expect("CUDA_ROOT not found. You must set CUDA_ROOT either as an environment variable, or in build-settings.toml to point to the root of your CUDA installation.");

    // Create a vector of options to pass to the compiler
    let optix_inc = format!("-I{}/include", optix_root);
    let cuda_inc = format!("-I{}/include", cuda_root);
    let common_inc = format!(
        "-I{}/examples/common",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    );

    let options = vec![
        optix_inc,
        cuda_inc,
        common_inc,
        "-I/usr/include/x86_64-linux-gnu".into(),
        "-I/usr/lib/gcc/x86_64-linux-gnu/7/include".into(),
        "-arch=compute_70".to_owned(),
        "-rdc=true".to_owned(),
        "-std=c++14".to_owned(),
        "-D__x86_64".to_owned(),
        "-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1".into(),
        "-default-device".into(),
    ];

    // The program object allows us to compile the cuda source and get ptx from
    // it if successful.
    let mut prg = Program::new(src, "devicePrograms", &vec![header]).unwrap();

    match prg.compile_program(&options) {
        Err(code) => {
            panic!("{}: {}", code, prg.get_program_log().unwrap());
        }
        Ok(_) => (),
    }

    let ptx = prg.get_ptx().unwrap();
    ptx
}
