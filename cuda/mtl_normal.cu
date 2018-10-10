#include "raydata.cuh"
#include <optix.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

using namespace optix;

rtDeclareVariable(rtObject, scene_root, , );

rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

RT_PROGRAM void closest_hit() {
    float3 world_shading_normal =
        normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal =
        normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

    float3 hit_point = ray.origin + ray.direction * t_hit;

    prd.result = world_shading_normal * 0.5 + 0.5;
    // prd.result = make_float3(dot(world_shading_normal, -ray.direction));
    prd.z = t_hit;
    prd.depth = 1;
}

RT_PROGRAM void any_hit() {
    prd_shadow.attenuation = optix::make_float3(0.0f);
    rtTerminateRay();
}