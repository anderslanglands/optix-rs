use optix::cuda::{self, cu};
use optix_sys::cuda_sys::{self, cudaFree, cudaGetDeviceCount};
use optix_sys::{optixInit, OptixResult};

use std::error::Error;
pub fn log_error_chain(err: &impl std::error::Error) {
    eprintln!("{}", err);
    let mut err = err.source();
    while err.is_some() {
        let e = err.unwrap();
        eprintln!("    because: {}", e);
        err = e.source();
    }
}

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

fn main() -> Result<(), Box<dyn std::error::Error + 'static>> {
    cu::init()?;

    let alloc = cuda::TaggedMallocator::new();

    println!("Found {} CUDA devices", cu::Device::get_count()?);

    let header = cuda::nvrtc::Header {
        name: "launch_params.h".into(),
        contents: "".into(),
    };
    let ptx = compile_to_ptx(CU_SRC, header);

    let device = cu::Device::get(0)?;
    let context = device.ctx_create(cu::ContextFlags::AUTO)?;
    let module = cu::Module::load_data(&ptx)?;
    let kernel = module.get_function("saxpy")?;

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

    let d_x =
        cuda::Buffer::with_data(&h_x, std::mem::align_of::<f32>(), 0, &alloc)?;
    let d_y =
        cuda::Buffer::with_data(&h_y, std::mem::align_of::<f32>(), 0, &alloc)?;
    let d_out = cuda::Buffer::new(
        buffer_size as usize,
        std::mem::align_of::<f32>(),
        0,
        &alloc,
    )?;

    let args: [*const std::os::raw::c_void; 5] = [
        &a as *const f32 as *const std::os::raw::c_void,
        &d_x.as_device_ptr() as *const u64 as *const std::os::raw::c_void,
        &d_y.as_device_ptr() as *const u64 as *const std::os::raw::c_void,
        &d_out.as_device_ptr() as *const u64 as *const std::os::raw::c_void,
        &n as *const usize as *const std::os::raw::c_void,
    ];

    match unsafe {
        kernel.launch(
            cu::Dims::x(NUM_BLOCKS as u32),
            cu::Dims::x(NUM_THREADS as u32),
            0,
            &cu::Stream::default(),
            &args,
        )
    } {
        Err(e) => log_error_chain(&e),
        Ok(()) => (),
    };

    cu::Context::synchronize()?;

    d_out.download(&mut h_out)?;
    println!("{:?}", h_out);

    Ok(())
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
