#ifndef RT_BSDFS_H
#define RT_BSDFS_H

#include <optix.h>
#include <optix_world.h>
#include <optixu/optixu_vector_types.h>
#include <optixu/optixu_math_namespace.h>
#include "shading_frame.cuh"

struct BsdfSample {
    optix::float3 omega_i;
    optix::float3 f;
    float pdf;
};

#endif
