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


rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

// Diffuse surface
RT_PROGRAM void mtl_ch_diffuse() {
    float3 world_shading_normal =
        normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal =
        normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

    float3 hit_point = ray.origin + ray.direction * t_hit;

    prd_radiance.result = make_float3(dot(world_shading_normal, -ray.direction));
    prd_radiance.z = t_hit;
    prd_radiance.depth = 1;
}

RT_PROGRAM void mtl_ah_shadow() {
    prd_shadow.attenuation = optix::make_float3(0.0f);
    rtTerminateRay();
}

// intersection
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3> index_buffer;
rtBuffer<int> material_buffer;

// rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );
// rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );

// rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void mesh_intersect(int primIdx) {
    const int3 v_idx = index_buffer[primIdx];

    const float3 p0 = vertex_buffer[v_idx.x];
    const float3 p1 = vertex_buffer[v_idx.y];
    const float3 p2 = vertex_buffer[v_idx.z];

    // Intersect ray with triangle
    float3 n;
    float t, beta, gamma;

    if (intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma)) {
        if (rtPotentialIntersection(t)) {

            shading_normal = geometric_normal = normalize(n);
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void bound(int primIdx, float result[6]) {
    const int3 v_idx = index_buffer[primIdx];

    const float3 v0 = vertex_buffer[v_idx.x];
    const float3 v1 = vertex_buffer[v_idx.y];
    const float3 v2 = vertex_buffer[v_idx.z];
    const float area = length(cross(v1 - v0, v2 - v0));

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (area > 0.0f && !isinf(area)) {
        aabb->m_min = fminf(fminf(v0, v1), v2);
        aabb->m_max = fmaxf(fmaxf(v0, v1), v2);
    } else {
        aabb->invalidate();
    }
}
