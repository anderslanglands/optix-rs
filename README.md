# optix-rs
Rust bindings to NVidia's OptiX ray tracing library. This crate is opinionated: lifetimes of all scene graph objects are managed by reference counting. This means that using the library is safe, but may carry some performance overhead compared to using the native C library directly.

To use the crate you must have OptiX (and CUDA) installed. To tell Cargo where to find them you can either create a build-settings.toml file alongside build.rs, with their path specified, or set the OPTIX_ROOT and CUDA_ROOT environment variables to point to the installations.
