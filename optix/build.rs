use std::path::{Path, PathBuf};

fn main() {
    let optix_root = std::env::var("OPTIX_ROOT")
        .expect("Could not get OPTIX_ROOT from environment");
    let cuda_root = std::env::var("CUDA_ROOT")
        .expect("Could not get CUDA_ROOT from environment");

    bindgen_optix(&optix_root, &cuda_root);

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    let args = vec![
        format!("-I{}/include", optix_root),
        format!("-I{}/examples/common/gdt", manifest_dir),
        "-G".into(),
    ];

    compile_to_ptx("examples/02_pipeline/device_programs.cu", &args);
    compile_to_ptx("examples/03_window/device_programs.cu", &args);
    compile_to_ptx("examples/04_mesh/device_programs.cu", &args);
    compile_to_ptx("examples/05_sbtdata/device_programs.cu", &args);
    compile_to_ptx("examples/06_multiple/device_programs.cu", &args);
    compile_to_ptx("examples/07_obj/device_programs.cu", &args);
    compile_to_ptx("examples/08_texture/device_programs.cu", &args);
    compile_to_ptx("examples/09_shadow/device_programs.cu", &args);
    compile_to_ptx("examples/10_softshadow/device_programs.cu", &args);
}

fn compile_to_ptx(cu_path: &str, args: &[String]) {
    println!("cargo:rerun-if-changed={}", cu_path);

    let full_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join(cu_path);

    let mut ptx_path =
        std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
            .join(cu_path);
    ptx_path.set_extension("ptx");
    std::fs::create_dir_all(ptx_path.parent().unwrap()).unwrap();

    let output = std::process::Command::new("nvcc")
        .arg("-ptx")
        .arg(&full_path)
        .arg("-o")
        .arg(&ptx_path)
        .args(args)
        .output()
        .expect("failed to fun nvcc");

    if !output.status.success() {
        panic!("{}", unsafe { String::from_utf8_unchecked(output.stderr) });
    }
}

fn get_modified_time(path: &Path) -> std::time::SystemTime {
    std::fs::metadata(path).unwrap().modified().unwrap()
}

fn bindgen_optix(optix_root: &str, cuda_root: &str) {
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .join("optix_wrapper.rs");

    let header_path =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("src")
            .join("optix_wrapper.h");

    let this_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("build.rs");

    if !out_path.is_file()
        || get_modified_time(&out_path) < get_modified_time(&header_path)
        || get_modified_time(&out_path) < get_modified_time(&this_path)
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
            .blacklist_type("OptixModuleCompileOptions")
            .blacklist_type("OptixPipelineLinkOptions")
            .whitelist_function("optix.*")
            .whitelist_var("OptixSbtRecordHeaderSize")
            .whitelist_var("OptixSbtRecordAlignment")
            .whitelist_var("OptixAccelBufferByteAlignment")
            .whitelist_var("OptixInstanceByteAlignment")
            .whitelist_var("OptixAabbBufferByteAlignment")
            .whitelist_var("OptixGeometryTransformByteAlignment")
            .whitelist_var("OptixTransformByteAlignment")
            .layout_tests(false)
            .generate_comments(false)
            .newtype_enum("OptixResult")
            .constified_enum_module("OptixCompileOptimizationLevel")
            .constified_enum_module("OptixCompileDebugLevel")
            .constified_enum_module("OptixTraversableGraphFlags")
            .constified_enum_module("OptixExceptionFlags")
            .constified_enum_module("OptixProgramGroupKind")
            .rustified_enum("GeometryFlags")
            .rustified_enum("OptixGeometryFlags")
            .constified_enum("OptixVertexFormat")
            .constified_enum("OptixIndicesFormat")
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
}
