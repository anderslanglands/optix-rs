// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <cuda_runtime.h>
#include <optix_device.h>

#include "lcg.h"
#include "vec.h"

enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

namespace osc {
#include "launch_params.h"
}

using namespace osc;

#define NUM_LIGHT_SAMPLES 16
#define NUM_PIXEL_SAMPLES 4

namespace osc {

typedef LCG<16> Random;

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

/*! per-ray data now captures random numebr generator, so programs
    can access RNG state */
struct PRD {
    Random random;
    f32x3 pixelColor;
};

static __forceinline__ DEVICE void* unpackPointer(u32 i0, u32 i1) {
    const u64 uptr = static_cast<u64>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ DEVICE void packPointer(void* ptr, u32& i0, u32& i1) {
    const u64 uptr = reinterpret_cast<u64>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T> static __forceinline__ DEVICE T* getPRD() {
    const u32 u0 = optixGetPayload_0();
    const u32 u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__shadow() {
    /* not going to be used ... */
}

extern "C" __global__ void __closesthit__radiance() {
    const TriangleMeshSBTData& sbtData =
        *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    PRD& prd = *getPRD<PRD>();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const i32 primID = optixGetPrimitiveIndex();
    const i32x3 index = sbtData.index[primID];
    const f32 u = optixGetTriangleBarycentrics().x;
    const f32 v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const f32x3& A = sbtData.vertex[index.x];
    const f32x3& B = sbtData.vertex[index.y];
    const f32x3& C = sbtData.vertex[index.z];
    f32x3 Ng = cross(B - A, C - A);
    f32x3 Ns =
        (sbtData.normal.is_null())
            ? Ng
            : ((1.f - u - v) * sbtData.normal[index.x] +
               u * sbtData.normal[index.y] + v * sbtData.normal[index.z]);

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const f32x3 rayDir = optixGetWorldRayDirection();

    if (dot(rayDir, Ng) > 0.f)
        Ng = -Ng;
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f)
        Ns = Ns - 2.f * dot(Ng, Ns) * Ng;
    Ns = normalize(Ns);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    f32x3 diffuseColor = sbtData.color;
    if (sbtData.has_texture && !sbtData.texcoord.is_null()) {
        const f32x2 tc = (1.f - u - v) * sbtData.texcoord[index.x] +
                         u * sbtData.texcoord[index.y] +
                         v * sbtData.texcoord[index.z];

        f32x4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor = diffuseColor * make_f32x3(fromTexture);
    }

    // start with some ambient term
    f32x3 pixelColor = (0.01f + 0.1f * fabsf(dot(Ns, rayDir))) * diffuseColor;

    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const f32x3 surfPos = (1.f - u - v) * sbtData.vertex[index.x] +
                          u * sbtData.vertex[index.y] +
                          v * sbtData.vertex[index.z];

    const i32 numLightSamples = NUM_LIGHT_SAMPLES;
    for (i32 lightSampleID = 0; lightSampleID < numLightSamples;
         lightSampleID++) {
        // produce random light sample
        const f32x3 lightPos = optixLaunchParams.light.origin +
                               prd.random() * optixLaunchParams.light.du +
                               prd.random() * optixLaunchParams.light.dv;
        f32x3 lightDir = lightPos - surfPos;
        f32 lightDist = length(lightDir);
        lightDir = normalize(lightDir);

        // trace shadow ray:
        const f32 NdotL = dot(lightDir, Ns);
        if (NdotL >= 0.f) {
            f32x3 lightVisibility=make_float3(1.f,1.f,1.f);
            // the values we store the PRD poi32er in:
            u32 u0, u1;
            packPointer(&lightVisibility, u0, u1);
            optixTrace(optixLaunchParams.traversable, surfPos + 1e-3f * Ng,
                       lightDir,
                       1e-3f,                     // tmin
                       lightDist * (1.f - 1e-3f), // tmax
                       0.0f,                      // rayTime
                       OptixVisibilityMask(255),
                       // anyhit ON for shadow rays:
                       OPTIX_RAY_FLAG_NONE,
                       SHADOW_RAY_TYPE, // SBT offset
                       RAY_TYPE_COUNT,  // SBT stride
                       SHADOW_RAY_TYPE, // missSBTIndex
                       u0, u1);
            pixelColor =
                pixelColor +
                lightVisibility * optixLaunchParams.light.power * diffuseColor *
                    (NdotL / (lightDist * lightDist * numLightSamples));
        }
    }

    prd.pixelColor = pixelColor;
}

extern "C" __global__ void
__anyhit__radiance() { /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __anyhit__shadow() {
    // in this simple example, we terminate on ANY hit
    f32x3& prd = *getPRD<f32x3>();
    prd = make_float3(0.f,0.f,0.f);
    optixTerminateRay();
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid i32ersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
    PRD& prd = *getPRD<PRD>();
    // set to constant white as background color
    prd.pixelColor = make_float3(1.f,1.f,1.f);
}

extern "C" __global__ void __miss__shadow() {
    // misses shouldn't mess with shadow opacity - do nothing
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    // compute a test pattern based on pixel ID
    const i32 ix = optixGetLaunchIndex().x;
    const i32 iy = optixGetLaunchIndex().y;
    const i32 accum_id = optixLaunchParams.frame.accum_id;
    const auto& camera = optixLaunchParams.camera;

    PRD prd;
    prd.random.init(ix + accum_id * optixLaunchParams.frame.size.x,
                    iy + accum_id * optixLaunchParams.frame.size.y);
    prd.pixelColor = make_float3(0.f,0.f,0.f);

    // the values we store the PRD poi32er in:
    u32 u0, u1;
    packPointer(&prd, u0, u1);

    i32 numPixelSamples = NUM_PIXEL_SAMPLES;

    f32x3 pixelColor=make_float3(0.f,0.f,0.f);
    for (i32 sampleID = 0; sampleID < numPixelSamples; sampleID++) {
        // normalized screen plane position, in [0,1]^2
        const f32x2 screen(make_float2(ix + prd.random(), iy + prd.random()) /
                           make_float2(optixLaunchParams.frame.size.x,
                                 optixLaunchParams.frame.size.y));

        // generate ray direction
        f32x3 rayDir =
            normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal +
                      (screen.y - 0.5f) * camera.vertical);

        optixTrace(optixLaunchParams.traversable, camera.position, rayDir,
                   0.f,   // tmin
                   1e20f, // tmax
                   0.0f,  // rayTime
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                   RADIANCE_RAY_TYPE,             // SBT offset
                   RAY_TYPE_COUNT,                // SBT stride
                   RADIANCE_RAY_TYPE,             // missSBTIndex
                   u0, u1);
        pixelColor = pixelColor + prd.pixelColor;
    }

    const u32 fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.color_buffer[fbIndex] = make_float4(
        pixelColor.x / numPixelSamples, pixelColor.y / numPixelSamples,
        pixelColor.z / numPixelSamples, 1.0f);
}

} // namespace osc
