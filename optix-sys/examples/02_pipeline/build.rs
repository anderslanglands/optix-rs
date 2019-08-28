fn main() {
    let optix_root = std::env::var("OPTIX_ROOT")
        .expect("Could not get OPTIX_ROOT from environment");
    let cuda_root = std::env::var("CUDA_ROOT")
        .expect("Could not get CUDA_ROOT from environment");

    let out_path =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    cc::Build::new()
        .file("devicePrograms.cu")
        .include(format!("{}/include", optix_root))
        .include(format!("{}/include", cuda_root))
        .compile();

    panic!("BLAH");
}
