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

#include <optix_device.h>
#include <vec.h>

enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

namespace osc {
#include "launch_params.h"
}

using namespace osc;

namespace osc {

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

static __forceinline__ __device__ void* unpackPointer(u32 i0, u32 i1) {
    const u64 uptr = static_cast<u64>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, u32& i0,
                                                   u32& i1) {
    const u64 uptr = reinterpret_cast<u64>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T> static __forceinline__ __device__ T* getPRD() {
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

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int primID = optixGetPrimitiveIndex();
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
    f32x3 Ns = (!sbtData.normal.is_null())
                   ? ((1.f - u - v) * sbtData.normal[index.x] +
                      u * sbtData.normal[index.y] + v * sbtData.normal[index.z])
                   : Ng;

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
        //diffuseColor = diffuseColor * fromTexture.xyz();
        diffuseColor = diffuseColor * make_f32x3(fromTexture);
    }

    // ------------------------------------------------------------------
    // compute shadow
    // ------------------------------------------------------------------
    const f32x3 surfPos = (1.f - u - v) * sbtData.vertex[index.x] +
                          u * sbtData.vertex[index.y] +
                          v * sbtData.vertex[index.z];
    const f32x3 lightPos = make_f32x3(-907.108f, 2205.875f, -400.0267f);
    const f32x3 lightDir = lightPos - surfPos;

    // trace shadow ray:
    f32x3 lightVisibility=make_f32x3(1.f,1.f,1.f);
    // the values we store the PRD pointer in:
    u32 u0, u1;
    packPointer(&lightVisibility, u0, u1);
    optixTrace(optixLaunchParams.traversable, surfPos + 1e-3f * Ng, lightDir,
               1e-3f,       // tmin
               1.f - 1e-3f, // tmax
               0.0f,        // rayTime
               OptixVisibilityMask(255),
               // anyhit ON for shadow rays:
               OPTIX_RAY_FLAG_NONE,
               SHADOW_RAY_TYPE, // SBT offset
               RAY_TYPE_COUNT,  // SBT stride
               SHADOW_RAY_TYPE, // missSBTIndex
               u0, u1);

    // ------------------------------------------------------------------
    // final shading: a bit of ambient, a bit of directional ambient,
    // and directional component based on shadowing
    // ------------------------------------------------------------------
    const float cosDN = 0.1f + .8f * fabsf(dot(rayDir, Ns));

    f32x3& prd = *(f32x3*)getPRD<f32x3>();
    prd = (.1f + (.2f + .8f * lightVisibility) * cosDN) * diffuseColor;
}

extern "C" __global__ void
__anyhit__radiance() { /*! for this simple example, this will remain empty */
}

extern "C" __global__ void __anyhit__shadow() {
    // in this simple example, we terminate on ANY hit
    f32x3& prd = *(f32x3*)getPRD<f32x3>();
    prd = make_f32x3(0.f,0.f,0.f);
    optixTerminateRay();
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
    f32x3& prd = *(f32x3*)getPRD<f32x3>();
    // set to constant white as background color
    prd = make_f32x3(1.f,1.f,1.f);
}

extern "C" __global__ void __miss__shadow() {
    // misses shouldn't mess with shadow opacity - do nothing
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame() {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto& camera = optixLaunchParams.camera;

    // our per-ray data for this example. what we initialize it to
    // won't matter, since this value will be overwritten by either
    // the miss or hit program, anyway
    f32x3 pixelColorPRD = make_f32x3(0.f, 0.0f, 0.0f);

    // the values we store the PRD pointer in:
    u32 u0, u1;
    packPointer(&pixelColorPRD, u0, u1);

    // normalized screen plane position, in [0,1]^2
    const f32x2 screen =
        make_f32x2(f32(ix) + .5f, f32(iy) + .5f) /
        make_f32x2(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);

    // generate ray direction
    f32x3 rayDir =
        normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal +
                  (screen.y - 0.5f) * camera.vertical);

    optixTrace(optixLaunchParams.traversable, (float3)camera.position,
               (float3)rayDir,
               0.f,   // tmin
               1e20f, // tmax
               0.0f,  // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               RADIANCE_RAY_TYPE,             // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               RADIANCE_RAY_TYPE,             // missSBTIndex
               u0, u1);

    // and write to frame buffer ...
    const u32 fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.color_buffer[fbIndex] =
        make_float4(pixelColorPRD.x, pixelColorPRD.y, pixelColorPRD.z, 1.0f);
}

} // namespace osc
