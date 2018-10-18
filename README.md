# optix-rs
A Rust wrapper for NVidia's OptiX raytracing library.

To use the crate you must have OptiX and CUDA installed. To tell Cargo where to find them you can either create a build-settings.toml file alongside build.rs, with their path specified with `optix_root` and `cuda_root`, or set the `OPTIX_ROOT` and `CUDA_ROOT` environment variables to point to the installations.
