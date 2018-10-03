#include "bsdfs.cuh"
#include "raydata.cuh"
#include "shader_globals.cuh"

using namespace optix;

rtDeclareVariable(float3, color, , );
rtDeclareVariable(rtCallableProgramId<float3(ShaderGlobals)>, color_fun, , );
rtDeclareVariable(float, strength, , );
rtDeclareVariable(rtCallableProgramId<float(ShaderGlobals)>, strength_fun, , );

RT_CALLABLE_PROGRAM BsdfSample bsdf_sample(ShadingFrame frame, const float u1,
                                           const float u2) {
    BsdfSample bs;
    float3 omega_l;
    cosine_sample_hemisphere(u1, u2, omega_l);
    bs.omega_i = normalize(omega_l.z * frame.n + omega_l.x * frame.u +
                           omega_l.y * frame.v);
    bs.f = (color * strength) / M_PIf;
    bs.pdf = max(dot(bs.omega_i, frame.n), 0.0f) / M_PIf;
    return bs;
}

RT_CALLABLE_PROGRAM BsdfSample bsdf_eval(ShadingFrame frame,
                                         const float3 omega_i) {
    BsdfSample bs;
    bs.omega_i = omega_i;
    bs.f = (color * strength) / M_PIf;
    bs.pdf = max(dot(bs.omega_i, frame.n), 0.0f) / M_PIf;
    return bs;
}

RT_CALLABLE_PROGRAM float bsdf_pdf(ShadingFrame frame, const float3 omega_i) {
    return dot(omega_i, frame.n) / M_PIf;
}
