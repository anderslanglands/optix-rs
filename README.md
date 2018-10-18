# optix-rs
A Rust wrapper for NVidia's OptiX raytracing library.

In order to use this crate as a dependency you must have CMake (>= 3.5), OptiX (>= 5) and CUDA (>= 9) installed. Please take a look at the example project here: https://github.com/anderslanglands/optix-rs-pathtracer-example to see a minimal example of the setup required.

To build this crate from the repository you need to tell Cargo (and CMake) where to find OptiX and CUDA you can either create a build-settings.toml file alongside build.rs, with their path specified with `optix_root` and `cuda_root`, or set the `OPTIX_ROOT` and `CUDA_ROOT` environment variables to point to the installations.

```
env OPTIX_ROOT="/Developer/NVIDIA/Optix-5.0" CUDA_ROOT="/Developer/NVIDIA/CUDA-9.2" cargo run --example pathtracer
```

