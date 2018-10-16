#include <optix.h>
#include <optix_world.h>
#include "random.cuh"
#include "raydata.cuh"

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(rtObject, scene_root, , );
rtBuffer<float4, 2> result_buffer;

rtDeclareVariable(Matrix4x4, camera_to_world, , );
rtDeclareVariable(Matrix4x4, raster_to_camera, , );

// a simple screen generator
RT_PROGRAM void generate_ray() {
    float x = float(launch_index.x);
    float y = float(launch_index.y);

    float camu = x / launch_dim.x;
    float camv = y / launch_dim.y;

    float3 result = make_float3(0.0f);

    PerRayData_radiance prd;
    prd.depth = 0;
    prd.z = 1000.0f;
    prd.pixel = launch_index;

    float4 Pras4 = make_float4(x, y, 0.0f, 1.0f);
    float4 Pcam4 = raster_to_camera * Pras4;
    float4 Dcam4 = make_float4(Pcam4.x, Pcam4.y, -1 * Pcam4.z, 0.0f);
    float4 P_origin_cam = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 P_origin_world = camera_to_world * P_origin_cam;
    float4 D_world = camera_to_world * Dcam4;
    float4 P_world = camera_to_world * Pcam4;
    auto origin = make_float3(P_origin_world);
    auto direction = normalize(make_float3(Pcam4));

    optix::Ray ray = optix::make_Ray(origin, direction, 0, 1e-5f, RT_DEFAULT_MAX);
    rtTrace(scene_root, ray, prd);

    result = prd.result;
    float alpha = prd.depth > 0 ? 1.0f : 0.0f;

    result_buffer[launch_index] = make_float4(result, alpha);
}

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

RT_PROGRAM void miss() {
    prd_radiance.result = make_float3(0.f);
}

RT_PROGRAM void miss_shadow() { prd_shadow.attenuation = make_float3(1); }
