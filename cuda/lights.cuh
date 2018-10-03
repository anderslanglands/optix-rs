#ifndef RT_LIGHTS_CUH
#define RT_LIGHTS_CUH

#include <optix.h>
#include <optix_world.h>
#include <optixu/optixu_vector_types.h>
#include <optixu/optixu_math_namespace.h>
#include "shading_frame.cuh"

struct LightSample {
    optix::float3 omega_i;
    optix::float3 L_i;
    float pdf;
};

#endif
