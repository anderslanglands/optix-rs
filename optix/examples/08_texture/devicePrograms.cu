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

namespace osc {
#include "launch_params.h"
}

using namespace osc;

namespace osc {

/*! launch parameters in constant memory, filled in by optix upon
    optixLaunch (this gets filled in from the buffer we pass to
    optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

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

/*! helper function that creates a semi-random color from an ID */
inline __device__ V3f32 randomColor(int i) {
    int r = unsigned(i) * 13 * 17 + 0x234235;
    int g = unsigned(i) * 7 * 3 * 5 + 0x773477;
    int b = unsigned(i) * 11 * 19 + 0x223766;
    return V3f32((r & 255) / 255.f, (g & 255) / 255.f, (b & 255) / 255.f);
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
extern "C" __global__ void __closesthit__radiance() {
    const TriangleMeshSBTData& sbtData =
        *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int primID = optixGetPrimitiveIndex();
    const V3i32 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    V3f32 N(0.0f, 0.0f, 0.0f);
    if (!sbtData.normal.is_null()) {
        N = (1.f - u - v) * sbtData.normal[index.x] +
            u * sbtData.normal[index.y] + v * sbtData.normal[index.z];
    } else {
        const V3f32& A = sbtData.vertex[index.x];
        const V3f32& B = sbtData.vertex[index.y];
        const V3f32& C = sbtData.vertex[index.z];
        N = normalize(cross(B - A, C - A));
    }
    N = normalize(N);

    // ------------------------------------------------------------------
    // compute diffuse material color, including diffuse texture, if
    // available
    // ------------------------------------------------------------------
    V3f32 diffuseColor = sbtData.color;
    if (sbtData.has_texture && !sbtData.texcoord.is_null()) {
        const V2f32 tc = (1.f - u - v) * sbtData.texcoord[index.x] +
                         u * sbtData.texcoord[index.y] +
                         v * sbtData.texcoord[index.z];

        V4f32 fromTexture = tex2D<float4>(sbtData.texture, tc.x, 1.0f - tc.y);
        diffuseColor = diffuseColor * fromTexture.xyz();
    }

    // ------------------------------------------------------------------
    // perform some simple "NdotD" shading
    // ------------------------------------------------------------------
    const V3f32 rayDir = optixGetWorldRayDirection();
    const float cosDN = 0.2f + .8f * fabsf(dot(rayDir, N));

    V3f32& prd = *(V3f32*)getPRD<V3f32>();
    prd = cosDN * diffuseColor;
}

extern "C" __global__ void
__anyhit__radiance() { /*! for this simple example, this will remain empty */
}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance() {
    V3f32& prd = *(V3f32*)getPRD<V3f32>();
    // set to constant white as background color
    prd = V3f32(1.f, 1.0f, 1.0f);
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
    V3f32 pixelColorPRD = V3f32(0.f, 0.0f, 0.0f);

    // the values we store the PRD pointer in:
    u32 u0, u1;
    packPointer(&pixelColorPRD, u0, u1);

    // normalized screen plane position, in [0,1]^2
    const V2f32 screen =
        V2f32(f32(ix) + .5f, f32(iy) + .5f) /
        V2f32(optixLaunchParams.frame.size.x, optixLaunchParams.frame.size.y);

    // generate ray direction
    V3f32 rayDir =
        normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal +
                  (screen.y - 0.5f) * camera.vertical);

    optixTrace(optixLaunchParams.traversable, (float3)camera.position,
               (float3)rayDir,
               0.f,   // tmin
               1e20f, // tmax
               0.0f,  // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               SURFACE_RAY_TYPE,              // missSBTIndex
               u0, u1);

    // and write to frame buffer ...
    const u32 fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.color_buffer[fbIndex] =
        make_float4(pixelColorPRD.x, pixelColorPRD.y, pixelColorPRD.z, 1.0f);
}

} // namespace osc
