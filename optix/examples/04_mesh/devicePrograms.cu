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

#include "LaunchParams.h"

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
    const int primID = optixGetPrimitiveIndex();
    V3f32& prd = *(V3f32*)getPRD<V3f32>();
    prd = randomColor(primID);
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

    /*
if (ix == 960 / 2 && iy == 540 / 2) {
V3f32 p = camera.position;
printf("position: %f %f %f\n", p.x, p.y, p.z);
V3f32 d = camera.direction;
printf("direction: %f %f %f\n", d.x, d.y, d.z);
V3f32 h = camera.horizontal;
printf("horizontal: %f %f %f\n", h.x, h.y, h.z);
V3f32 v = camera.vertical;
printf("vertical: %f %f %f\n", v.x, v.y, v.z);
printf("raydir: %f %f %f\n", rayDir.x, rayDir.y, rayDir.z);
}
*/

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
    optixLaunchParams.frame.colorBuffer[fbIndex] =
        make_float4(pixelColorPRD.x, pixelColorPRD.y, pixelColorPRD.z, 1.0f);
}

} // namespace osc
