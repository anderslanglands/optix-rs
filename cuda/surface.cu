#include "bsdfs.cuh"
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
rtBuffer<rtCallableProgramId<BsdfSample(ShadingFrame, float, float)> >
    bsdf_sample_fn;
rtBuffer<rtCallableProgramId<BsdfSample(ShadingFrame, float3)> > bsdf_eval_fn;
rtBuffer<rtCallableProgramId<BsdfSample(ShadingFrame, float3)> > bsdf_pdf_fn;
// clang-format on
rtBuffer<float, 2> sample_buffer;
rtDeclareVariable(int, progression, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

__device__ __inline__ float power_heuristic(float fp, float gp) {
    return fp * fp / (fp * fp + gp * gp);
}

#define MIS

RT_PROGRAM void closest_hit() {
    float3 world_shading_normal =
        normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal =
        normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

    float3 hit_point = ray.origin + ray.direction * t_hit;
    float3 result = make_float3(0.0f);
    ShadingFrame frame(hit_point, world_shading_normal, -ray.direction);

    for (int i = 0; i < lights_sample_fn.size(); ++i) {
        float u1, u2;
        NEXT_SAMPLE(prd.sampler, u1);
        NEXT_SAMPLE(prd.sampler, u2);

        LightSample ls_from_light = lights_sample_fn[i](frame, u1, u2);
        if (ls_from_light.pdf == 0.0f) {
            continue;
        }

        for (int b = 0; b < bsdf_sample_fn.size(); ++b) {
            BsdfSample bs_from_light =
                bsdf_eval_fn[b](frame, ls_from_light.omega_i);
            optix::Ray ray =
                optix::make_Ray(hit_point, ls_from_light.omega_i, 1,
                                (2.0e-5f) * t_hit + 1.0e-5f, RT_DEFAULT_MAX);
            PerRayData_shadow shadow_prd;
            shadow_prd.attenuation = make_float3(1.0f);
            rtTrace(scene_root, ray, shadow_prd);

            const float3 result_light =
                (ls_from_light.L_i / ls_from_light.pdf) *
                max(dot(world_shading_normal, ls_from_light.omega_i), 0.0f) *
                bs_from_light.f * shadow_prd.attenuation;

#ifdef MIS
            const float weight_light =
                power_heuristic(ls_from_light.pdf, bs_from_light.pdf);

            result += result_light * weight_light;

            float b_u1, b_u2;
            NEXT_SAMPLE(prd.sampler, b_u1);
            NEXT_SAMPLE(prd.sampler, b_u2);

            BsdfSample bs_from_bsdf = bsdf_sample_fn[b](frame, b_u1, b_u2);
            LightSample ls_from_bsdf =
                lights_eval_fn[i](frame.p, bs_from_bsdf.omega_i);

            if (ls_from_bsdf.pdf == 0.0f) {
                // sample did not hit the light
                continue;
            }

            ray = optix::make_Ray(frame.p, bs_from_bsdf.omega_i, 1,
                                  2.0e-5f * t_hit + 1.0e-5f, RT_DEFAULT_MAX);
            shadow_prd.attenuation = make_float3(1.0f);
            rtTrace(scene_root, ray, shadow_prd);

            const float weight_bsdf =
                power_heuristic(bs_from_bsdf.pdf, ls_from_bsdf.pdf);

            const float3 result_bsdf =
                (ls_from_bsdf.L_i / bs_from_bsdf.pdf) *
                max(dot(world_shading_normal, bs_from_bsdf.omega_i), 0.0f) *
                bs_from_bsdf.f * shadow_prd.attenuation;

            result += result_bsdf * weight_bsdf;
#else
            result += result_light;
#endif
        }
    }
    prd.result = result;
    prd.z = t_hit;
    prd.depth = 1;
}

RT_PROGRAM void any_hit() {
    prd_shadow.attenuation = optix::make_float3(0.0f);
    rtTerminateRay();
}
