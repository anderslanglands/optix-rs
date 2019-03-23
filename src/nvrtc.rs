use crate::nvrtc_bindings::*;
use std::ffi::CString;
use std::os::raw::c_char;

use std::fmt;

#[derive(Debug)]
pub struct Error {
    error_string: String,
}

impl fmt::Display for Error {
    fn fmt(&self, output: &mut fmt::Formatter) -> fmt::Result {
        write!(output, "nvrtc compilation error: {}", self.error_string)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

pub fn get_error_string(result: NvrtcResult) -> Error {
    unsafe {
        Error {
            error_string: std::ffi::CStr::from_ptr(nvrtcGetErrorString(result))
                .to_string_lossy()
                .into_owned(),
        }
    }
}

/// A CUDA program object that can be compiled to generate PTX
pub struct Program {
    pub prog: nvrtcProgram,
}

/// Represents a header file that can be included by a program. The `name` is
/// the name by which the header will be referenced in the CUDA source. The
/// `contents` is just the CUDA contents of the header.
pub struct Header {
    pub name: String,
    pub contents: String,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Program {
    /// Create a new `Program` with the given `src`, using the entry point
    /// `name` and with a list of `headers` to include. If there are no headers
    /// to include, just pass an empty `Vec`.
    pub fn new(src: &str, name: &str, headers: Vec<Header>) -> Result<Program> {
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

    /// Compile this program with the given `options`
    pub fn compile_program(&mut self, options: &Vec<String>) -> Result<()> {
        let mut coptions = Vec::new();
        for o in options {
            let c = CString::new(o.clone()).unwrap();
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
            Err(Error {
                error_string: format!(
                    "{}\n{}",
                    get_error_string(result).error_string,
                    self.get_program_log().unwrap()
                ),
            })
        } else {
            Ok(())
        }
    }

    /// Get the program compilation log
    pub fn get_program_log(&self) -> Result<String> {
        let (log_size, result) = unsafe {
            let mut log_size: usize = 0;
            let result = nvrtcGetProgramLogSize(self.prog, &mut log_size);
            (log_size, result)
        };

        if result != NvrtcResult::NVRTC_SUCCESS {
            return Err(get_error_string(result));
        }

        let buffer = create_whitespace_cstring(log_size);

        let result = unsafe {
            nvrtcGetProgramLog(self.prog, buffer.as_ptr() as *mut c_char)
        };

        if result != NvrtcResult::NVRTC_SUCCESS {
            Err(get_error_string(result))
        } else {
            Ok(buffer.to_string_lossy().into_owned())
        }
    }

    /// Assuming a successful compilation, get the generated PTX as a `String`
    pub fn get_ptx(&self) -> Result<String> {
        let (ptx_size, result) = unsafe {
            let mut ptx_size: usize = 0;
            let result = nvrtcGetPTXSize(self.prog, &mut ptx_size);
            (ptx_size, result)
        };

        if result != NvrtcResult::NVRTC_SUCCESS {
            return Err(get_error_string(result));
        }

        let buffer = create_whitespace_cstring(ptx_size);

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
    let mut buffer: Vec<u8> = Vec::with_capacity(len as usize);
    buffer.extend([b' '].iter().cycle().take(len as usize - 1));
    unsafe { CString::from_vec_unchecked(buffer) }
}

#[test]
fn test_compile() {
    use config::Config;
    use std::collections::HashMap;
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
    )
    .unwrap();

    match prg.compile_program(&options) {
        Err(code) => {
            panic!("{}: {}", code, prg.get_program_log().unwrap());
        }
        Ok(_) => {
            println!("Compilation successful");
            println!("{}", prg.get_ptx().unwrap());
        }
    }
}
