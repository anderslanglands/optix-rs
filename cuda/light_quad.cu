#include "lights.cuh"
#include "raydata.cuh"
#include <optixu/optixu_matrix_namespace.h>

using namespace optix;

rtDeclareVariable(Matrix4x4, light_to_world, , );
rtDeclareVariable(Matrix4x4, world_to_light, , );
rtDeclareVariable(float3, n_l, , );
rtDeclareVariable(float3, L_e, , );
rtDeclareVariable(float, area, , );

RT_CALLABLE_PROGRAM LightSample light_sample(ShadingFrame frame, const float u1,
                                             const float u2) {
    LightSample ls;

    float4 p_l = make_float4(u1 * 2 - 1, u2 * 2 - 1, 0.f, 1.f);
    float4 p_w = light_to_world * p_l;
    ls.omega_i = make_float3(p_w.x, p_w.y, p_w.z) - frame.p;
    if (dot(ls.omega_i, n_l) > 0) {
        ls.L_i = make_float3(0.0f, 0.0f, 0.0f);
        ls.pdf = 0.0f;
        return ls;
    }
    float d2 = dot(ls.omega_i, ls.omega_i);
    float d = sqrt(d2);
    ls.omega_i /= d;
    ls.L_i = L_e / area;
    ls.pdf = d2 / (abs(dot(ls.omega_i, n_l)) * area);
    return ls;
}

RT_CALLABLE_PROGRAM LightSample light_eval(const float3 p_s,
                                           const float3 omega_i) {
    LightSample ls;

    // check that ray is pointing to front size (-z) of light
    float cos_theta_l = dot(ls.omega_i, n_l);
    if (cos_theta_l > 0) {
        ls.L_i = make_float3(0.0f, 0.0f, 0.0f);
        ls.pdf = 0.0f;
        return ls;
    }

    // calculate intersection of omega_i with plane of light
    float4 ps_l = world_to_light * make_float4(p_s, 1.0f);
    float4 omega_l = world_to_light * make_float4(omega_i, 0.0f);
    float t = -ps_l.z / omega_l.z;

    // check that intersection point is inside the light extents
    float4 pl_l = ps_l + t * omega_l;
    if (abs(ps_l.x) > 1 || abs(ps_l.y) > 1) {
        ls.L_i = make_float3(0.0f, 0.0f, 0.0f);
        ls.pdf = 0.0f;
        return ls;
    }

    // transform the resulting point back to world space
    float4 pl_w = light_to_world * pl_l;
    float3 ld = make_float3(pl_w.x, pl_w.y, pl_w.z) - p_s;
    float d2 = dot(ld, ld);

    ls.L_i = L_e / area;
    ls.omega_i = omega_i;
    ls.pdf = d2 / (cos_theta_l * area);
    return ls;
}

RT_CALLABLE_PROGRAM float light_pdf(const float3 p_s, const float3 p_l) {
    return 1.0f;
}
