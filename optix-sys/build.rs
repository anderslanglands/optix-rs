fn main() {
    let optix_root = std::env::var("OPTIX_ROOT")
        .expect("Could not get OPTIX_ROOT from environment");
    let cuda_root = std::env::var("CUDA_ROOT")
        .expect("Could not get CUDA_ROOT from environment");

    let bindings = bindgen::Builder::default()
        .header("src/optix_wrapper.h")
        .clang_arg(format!("-I{}/include", optix_root))
        .clang_arg(format!("-I{}/include", cuda_root))
        .whitelist_recursively(false)
        .whitelist_type("Optix.*")
        .whitelist_type("RaygenRecord")
        .whitelist_type("MissRecord")
        .whitelist_type("HitgroupRecord")
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
        .generate()
        .expect("Unable to generate optix bindings");

    let out_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("optix_wrapper.rs"))
        .expect("Couldn't write bindings!");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("optix_wrapper.rs"))
        .expect("Couldn't write bindings!");

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
