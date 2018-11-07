use config::Config;
use crate::nvrtc_bindings::*;
use std::ffi::CString;
use std::os::raw::c_char;
use std::collections::HashMap;

pub fn get_error_string(result: NvrtcResult) -> String {
    unsafe {
        std::ffi::CStr::from_ptr(nvrtcGetErrorString(result))
            .to_string_lossy()
            .into_owned()
    }
}

pub struct Program {
    pub prog: nvrtcProgram,
}

pub struct Header {
    pub name: String,
    pub contents: String,
}

impl Program {
    pub fn new(
        src: &str,
        name: &str,
        headers: Vec<Header>,
    ) -> Result<Program, String> {
        let src = CString::new(src).unwrap();
        let name = CString::new(name).unwrap();
        let mut header_names = Vec::new();
        let mut header_contents = Vec::new();
        for h in headers {
            header_names.push(CString::new(h.name).unwrap());
            header_contents.push(CString::new(h.contents).unwrap());
        }
        let mut header_names_arr = Vec::new();
        let mut header_contents_arr = Vec::new();
        for i in 0..header_names.len() {
            header_names_arr.push(header_names[i].as_ptr() as *const c_char);
            header_contents_arr
                .push(header_contents[i].as_ptr() as *const c_char);
        }

        let (prog, result) = unsafe {
            let mut prog: nvrtcProgram = std::ptr::null_mut();
            let result = nvrtcCreateProgram(
                &mut prog,
                src.as_ptr() as *const c_char,
                name.as_ptr() as *const c_char,
                header_names.len() as i32,
                (&header_names_arr[..]).as_ptr() as *const *const c_char,
                (&header_contents_arr[..]).as_ptr() as *const *const c_char,
            );
            (prog, result)
        };

        if result != NvrtcResult::NVRTC_SUCCESS {
            Err(get_error_string(result))
        } else {
            Ok(Program { prog })
        }
    }

    pub fn compile_program(
        &mut self,
        options: Vec<String>,
    ) -> Result<(), String> {
        let mut coptions = Vec::new();
        for o in options {
            let c = CString::new(o).unwrap();
            coptions.push(c);
        }

        let mut options_arr = Vec::new();
        for o in &coptions {
            options_arr.push(o.as_ptr() as *const c_char);
        }

        let result = unsafe {
            nvrtcCompileProgram(
                self.prog,
                options_arr.len() as i32,
                options_arr.as_ptr() as *const *const c_char,
            )
        };
        if result != NvrtcResult::NVRTC_SUCCESS {
            Err(get_error_string(result))
        } else {
            Ok(())
        }
    }

    pub fn get_program_log(&self) -> Result<String, String> {
        let (log_size, result) = unsafe {
            let mut log_size: usize = 0;
            let result = nvrtcGetProgramLogSize(self.prog, &mut log_size);
            (log_size, result)
        };

        if result != NvrtcResult::NVRTC_SUCCESS {
            return Err(get_error_string(result));
        }

        let mut buffer = create_whitespace_cstring(log_size);

        let result = unsafe {
            nvrtcGetProgramLog(self.prog, buffer.as_ptr() as *mut c_char)
        };

        if result != NvrtcResult::NVRTC_SUCCESS {
            Err(get_error_string(result))
        } else {
            Ok(buffer.to_string_lossy().into_owned())
        }
    }

    pub fn get_ptx(&self) -> Result<String, String> {
        let (ptx_size, result) = unsafe {
            let mut ptx_size: usize = 0;
            let result = nvrtcGetPTXSize(self.prog, &mut ptx_size);
            (ptx_size, result)
        };

        if result != NvrtcResult::NVRTC_SUCCESS {
            return Err(get_error_string(result));
        }

        let mut buffer = create_whitespace_cstring(ptx_size);

        let result =
            unsafe { nvrtcGetPTX(self.prog, buffer.as_ptr() as *mut c_char) };

        if result != NvrtcResult::NVRTC_SUCCESS {
            Err(get_error_string(result))
        } else {
            Ok(buffer.to_string_lossy().into_owned())
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            nvrtcDestroyProgram(&mut self.prog);
        }
    }
}

fn create_whitespace_cstring(len: usize) -> CString {
    let mut buffer: Vec<u8> = Vec::with_capacity(len as usize + 1);
    buffer.extend([b' '].iter().cycle().take(len as usize));
    unsafe { CString::from_vec_unchecked(buffer) }
}

#[test]
fn test_compile() {
    // Grab OPTIX_ROOT and CUDA_ROOT from build-settings.toml
    // or from the environment
    let mut settings = Config::default();

    settings
        .merge(config::File::with_name("build-settings").required(false))
        .unwrap();
    settings.merge(config::Environment::new()).ok();

    let settings_map = settings
        .try_into::<HashMap<String, String>>()
        .unwrap_or(HashMap::new());

    let optix_root = settings_map
        .get("optix_root")
        .expect("OPTIX_ROOT not found. You must set OPTIX_ROOT either as an environment variable, or in build-settings.toml to point to the root of your OptiX installation.");

    let cuda_root = settings_map.get("cuda_root")
        .expect("CUDA_ROOT not found. You must set CUDA_ROOT either as an environment variable, or in build-settings.toml to point to the root of your CUDA installation.");

    // Create a vector of options to pass to the compiler
    let optix_inc = format!("-I{}/include", optix_root);
    let cuda_inc = format!("-I{}/include", cuda_root);

    let options = vec![
        optix_inc, 
        cuda_inc,
        "-arch=compute_30".to_owned(),
        "-rdc=true".to_owned(),
        "-std=c++14".to_owned(),
        "-D__x86_64".to_owned(),
        "--device-as-default-execution-space".to_owned(),
    ];

    // The program object allows us to compile the cuda source and get ptx from 
    // it if successful. 
    let mut prg = Program::new(
        "#include <optix.h>\n__device__ float myfun() { return 1.0f; }",
        "myfun",
        Vec::new(),
    ).unwrap();

    match prg.compile_program(options) {
        Err(code) => {
            panic!("{}: {}", code, prg.get_program_log().unwrap());
        }
        Ok(_) => {
            println!("Compilation successful");
            println!("{}", prg.get_ptx().unwrap());
        }
    }
}
