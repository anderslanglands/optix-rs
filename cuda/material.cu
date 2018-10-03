#include "lights.cuh"
#include "random.cuh"
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

// clang-format off
rtBuffer<rtCallableProgramId<LightSample(ShadingFrame, float, float)> >
    lights_sample_fn;
rtBuffer<rtCallableProgramId<LightSample(float3, float3)> > lights_eval_fn;
rtBuffer<rtCallableProgramId<LightSample(float3, float3)> > lights_pdf_fn;
// clang-format on

RT_PROGRAM void closest_hit_radiance() {
    float3 world_shading_normal =
        normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal =
        normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

    float3 hit_point = ray.origin + ray.direction * t_hit;
    ShadingFrame frame(hit_point, world_shading_normal, -ray.direction);
    float3 result = make_float3(0.0f);

    for (int i = 0; i < lights_sample_fn.size(); ++i) {
        float u1 = prd.sampler.next();
        float u2 = prd.sampler.next();
        LightSample ls = lights_sample_fn[i](frame, u1, u2);

        if (ls.pdf > 0.0f) {
            optix::Ray ray =
                optix::make_Ray(frame.p, ls.omega_i, 1,
                                (2.0e-5f) * t_hit + 1.0e-5f, RT_DEFAULT_MAX);
            PerRayData_shadow shadow_prd;
            shadow_prd.attenuation = make_float3(1.0f);
            rtTrace(scene_root, ray, shadow_prd);

            result += (ls.L_i / ls.pdf) *
                      max(dot(world_shading_normal, ls.omega_i), 0.0f) * 0.18f /
                      M_PIf * shadow_prd.attenuation;
        }
    }
    prd.result = result;
    prd.z = t_hit;
    prd.depth = 1;
}

RT_PROGRAM void any_hit_shadow() {
    prd_shadow.attenuation = optix::make_float3(0.0f);
    rtTerminateRay();
}
