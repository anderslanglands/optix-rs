#pragma once
#include <optix.h>
#include "random.cuh"

struct LCGSampler {
    uint seed;
    uint dim;
    uint index;
    uint offset;

    __device__ __inline__ float next() {
        return rnd(seed);
    }
};

typedef LCGSampler Sampler;

// #define SAMPLER_BUFFER 1

#if(SAMPLER_BUFFER)

#define NEXT_SAMPLE(sampler, u) \
    if (sampler.dim >= 32) { \
        u = sampler.next(); \
    } \
else { \
    u = sample_buffer[make_uint2(sampler.index, sampler.dim)]; \
    sampler.dim +=1; \
}

#else

#define NEXT_SAMPLE(sampler, u) u = sampler.next();

#endif
