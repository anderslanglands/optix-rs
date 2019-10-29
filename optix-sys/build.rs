use std::path::{Path, PathBuf};

fn main() {
    let optix_root = std::env::var("OPTIX_ROOT")
        .expect("Could not get OPTIX_ROOT from environment");
    let cuda_root = std::env::var("CUDA_ROOT")
        .expect("Could not get CUDA_ROOT from environment");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .join("optix_wrapper.rs");

    let header_path =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("src")
            .join("optix_wrapper.h");

    if !out_path.is_file()
        || get_modified_time(&out_path) < get_modified_time(&header_path)
    {
        let bindings = bindgen::Builder::default()
            .header("src/optix_wrapper.h")
            .clang_arg(format!("-I{}/include", optix_root))
            .clang_arg(format!("-I{}/include", cuda_root))
            .whitelist_recursively(false)
            .whitelist_type("Optix.*")
            .whitelist_type("RaygenRecord")
            .whitelist_type("MissRecord")
            .whitelist_type("HitgroupRecord")
            .blacklist_type("OptixBuildInput")
            .whitelist_function("optix.*")
            .whitelist_var("OptixSbtRecordHeaderSize")
            .whitelist_var("OptixSbtRecordAlignment")
            .layout_tests(false)
            .generate_comments(false)
            .rustified_enum("OptixResult")
            .constified_enum_module("OptixCompileOptimizationLevel")
            .constified_enum_module("OptixCompileDebugLevel")
            .constified_enum_module("OptixTraversableGraphFlags")
            .constified_enum_module("OptixExceptionFlags")
            .constified_enum_module("OptixProgramGroupKind")
            .rustified_enum("GeometryFlags")
            .rustified_enum("OptixGeometryFlags")
            .rustified_enum("OptixVertexFormat")
            .rustified_enum("OptixIndicesFormat")
            .rust_target(bindgen::RustTarget::Nightly)
            .rustfmt_bindings(true)
            .generate()
            .expect("Unable to generate optix bindings");

        let dbg_path = std::path::PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR").unwrap(),
        );
        bindings
            .write_to_file(dbg_path.join("optix_wrapper.rs"))
            .expect("Couldn't write bindings!");

        bindings
            .write_to_file(out_path)
            .expect("Couldn't write bindings!");
    }

    let dst_capi = cmake::Config::new("optix_stubs-capi")
        .define("INC_OPTIX", &format!("{}/include", optix_root))
        .define("INC_CUDA", &format!("{}/include", cuda_root))
        .always_configure(false)
        .build();

    println!("cargo:rustc-link-search=native={}", dst_capi.display());
    println!("cargo:rustc-link-lib=static=optix_stubs-capi");

    // println!(
    //     "cargo:rustc-link-search=native={}",
    //     format!("{}/lib64", cuda_root)
    // );
    // println!("cargo:rustc-link-lib=dylib=cudart");
    // println!("cargo:rustc-link-lib=dylib=cuda");
}

fn get_modified_time(path: &Path) -> std::time::SystemTime {
    std::fs::metadata(path).unwrap().modified().unwrap()
}
