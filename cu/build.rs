use std::path::Path;

fn main() {
    let cuda_root = std::env::var("CUDA_ROOT")
        .expect("Could not get CUDA_ROOT from environment");

    bindgen_cuda(&cuda_root);

    compile_to_ptx("src/cuda/add.cu");
}

fn compile_to_ptx(cu_path: &str) {
    println!("cargo:rerun-if-changed={}", cu_path);

    let full_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join(cu_path);

    let mut ptx_path =
        std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
            .join(full_path.file_name().unwrap());
    ptx_path.set_extension("ptx");

    std::process::Command::new("nvcc")
        .arg("-ptx")
        .arg(&full_path)
        .arg("-o")
        .arg(&ptx_path)
        .output()
        .expect("failed to run nvcc command");
}

fn bindgen_cuda(cuda_root: &str) {
    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .join("cuda_wrapper.rs");

    let header_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("src")
            .join("cuda_wrapper.h");

    let this_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("build.rs");

    if !out_path.is_file()
        || get_modified_time(&out_path) < get_modified_time(&header_path)
        || get_modified_time(&out_path) < get_modified_time(&this_path)
    {
        let bindings = bindgen::Builder::default()
            .header("src/cuda_wrapper.h")
            .clang_arg(format!("-I{}/include", cuda_root))
            .whitelist_recursively(false)
            .whitelist_type("CU.*")
            .whitelist_type("Nvrtc.*")
            .whitelist_type("_nvrtc.*")
            .whitelist_type("nvrtc.*")
            .whitelist_type("cuuint.*")
            .whitelist_type("size_t")
            .whitelist_type("textureReference")
            .whitelist_type("surfaceReference")
            .whitelist_type("dim3")
            .whitelist_type("cudaError_enum")
            .whitelist_function("cu.*")
            .whitelist_function("nvrtc.*")
            .layout_tests(false)
            .newtype_enum("cudaError_enum")
            .newtype_enum("CUdevice_attribute")
            .constified_enum_module("nvrtcResult")
            .constified_enum_module("CUctx_flags_enum")
            .constified_enum_module("CUarray_format_enum")
            .constified_enum_module("CUaddress_mode_enum")
            .constified_enum_module("CUfilter_mode_enum")
            .constified_enum_module("CUmemorytype_enum")
            .newtype_enum("CUfunc_cache_enum")
            .newtype_enum("CUlimit_enum")
            .whitelist_type("TextureReadFlags")
            .rustified_enum("TextureReadFlags")
            .constified_enum_module("CUsharedconfig_enum")
            .generate()
            .expect("Unable to generate cuda bindings");

        let dbg_path = std::path::PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR").unwrap(),
        );
        bindings
            .write_to_file(dbg_path.join("cuda_wrapper.rs"))
            .expect("Couldn't write bindings!");

        bindings
            .write_to_file(out_path)
            .expect("Couldn't write bindings!");
    }

    println!(
        "cargo:rustc-link-search=native={}",
        format!("{}/lib64", cuda_root)
    );
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
}

fn get_modified_time(path: &Path) -> std::time::SystemTime {
    std::fs::metadata(path).unwrap().modified().unwrap()
}
