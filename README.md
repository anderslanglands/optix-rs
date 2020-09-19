# Introduction
Rust bindings for [NVidia's Optix 7.1 raytracing API](https://raytracing-docs.nvidia.com)

These bindings are *not safe*. The OptiX API has you construct a scene graph
of objects referencing each other by pointers. It's certainly possible to
wrap these up in safe objects using lifetimes or Rc's to track ownership
(and indeed an earlier version of this crate did just that) but doing so
means imposing constrictive design decisions on the user, which may or may
not be acceptable for their use case.

Instead, what this crate provides is a thin, ergonomic wrapper around the C API and leaves it up to the user to build their preferred ownership abstractions around it. This means that:
- Related functionality is grouped onto struct methods, e.g. `ProgramGroup`,
`Pipeline` etc.
- Configuration is simplified by the addition of builders for complex config objects, or by providing higher-level functions for common functionality
- Rather than mutable out pointers, all functions return `Result<T, optix::Error>`, where `optix::Error` implements `std::error::Error` - Device memory management is eased with utility types such as `TypedBuffer` and `DeviceVariable`, and functions that expect references to device memory are generic over the storage type. 
- All functions that allocate are generic over a `cu::DeviceAllocRef`, which allows the user to provide their own allocators. The design for this is very similar to wg-allocator. The default allocator simply calls through to `cuMemAlloc` and the underlying cuda crate also provides a simple bump allocator as an example, that is also used by the example programs.

# Examples
The examples are a direct translation of Ingo Wald's OptiX 7 Siggraph course
and it is highly recommended you read them to see how the crate works

# Building
In order to build the crate you must have the Optix 7.1 SDK and a recent
version of the CUDA SDK installed. The build script expects to be able to
find these using the environment variables `OPTIX_ROOT` and `CUDA_ROOT`,
respectively. You'll also need at least driver version 450 installed.

This crate has been tested on Linux only. It *should* work on Windows, and I
will gratefully accept PRs for fixing any issues there.
