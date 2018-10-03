#include <optix.h>
#include "sampler.cuh"

struct PerRayData_radiance
{
    optix::float3 result;
    float  importance;
    int    depth;
    float z;
    Sampler sampler;
    uint2 pixel;
};

struct PerRayData_shadow
{
    optix::float3 attenuation;
};

