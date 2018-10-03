#include "bsdfs.cuh"
#include "raydata.cuh"
#include "shader_globals.cuh"

using namespace optix;

rtDeclareVariable(float3, color, , );
rtDeclareVariable(rtCallableProgramId<float3(ShaderGlobals)>, color_fun, , );
rtDeclareVariable(float, strength, , );
rtDeclareVariable(rtCallableProgramId<float(ShaderGlobals)>, strength_fun, , );
rtDeclareVariable(float, roughness, , );
rtDeclareVariable(rtCallableProgramId<float(ShaderGlobals)>, roughness_fun, , );
rtDeclareVariable(float, anisotropy, , );
rtDeclareVariable(rtCallableProgramId<float(ShaderGlobals)>, anisotropy_fun,
                  , );
rtDeclareVariable(float, rotation, , );
rtDeclareVariable(rtCallableProgramId<float(ShaderGlobals)>, rotation_fun, , );
rtDeclareVariable(float, ior, , );
rtDeclareVariable(rtCallableProgramId<float(ShaderGlobals)>, ior_fun, , );

static const float C_one_over_pi = 1.0f / M_PIf;
static const float C_two_pi = 2.0f * M_PIf;

inline __device__ float sqr(const float x) { return x * x; }

inline __device__ float ggx_D(const float tan2_m) {
    float result = C_one_over_pi / sqr((1.0f + tan2_m));
    return result;
}

inline __device__ float ggx_lambda(const float a2) {
    float result = 0.5f * (-1.0f + sqrtf(1.0f + 1.0f / a2));
    return result;
}

inline __device__ float eval_lambda(const float3 omega, const float alpha_x,
                                    const float alpha_y) {
    const float cos2_theta = sqr(omega.z);
    // these two are multipled by sin^2 theta for convenience
    const float cos2_phi_s2t = sqr(omega.x * alpha_x);
    const float sin2_phi_s2t = sqr(omega.y * alpha_y);

    float result = ggx_lambda(cos2_theta / (cos2_phi_s2t + sin2_phi_s2t));
    return result;
}

inline __device__ float eval_G2(float lambda_o, float lambda_i) {
    float result = 1.0f / (lambda_o + lambda_i + 1.0f);
    return result;
}

inline __device__ float eval_G1(float lambda) { return 1.0f / (lambda + 1.0f); }

inline __device__ float eval_D(const float3 m, const float alpha_x,
                               const float alpha_y) {
    float result = 0.0f;
    const float cos_theta_m = m.z;

    if (cos_theta_m > 0.0f) {
        const float cos2_phi = sqr(m.x / alpha_x);
        const float sin2_phi = sqr(m.y / alpha_y);
        const float cos2_theta_m = sqr(cos_theta_m);
        const float cos4_theta_m = sqr(cos2_theta_m);

        const float tan2_theta_m = (cos2_phi + sin2_phi) / cos2_theta_m;

        result = ggx_D(tan2_theta_m) / (alpha_x * alpha_y * cos4_theta_m);
    }

    return result;
}

inline __device__ float schlick_fresnel(const float r0, const float cos_theta) {
    return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
}

inline __device__ float3 ggx_eval(const float3 omega_o_l,
                                  const float3 omega_i_l, const float alpha_x,
                                  const float alpha_y, float& pdf) {
    // if both directions are front-facing
    if (omega_o_l.z > 0.0f && omega_i_l.z > 0.0f) {
        const float3 m = normalize(omega_o_l + omega_i_l);
        const float D = eval_D(m, alpha_x, alpha_y);
        const float lambda_o = eval_lambda(omega_o_l, alpha_x, alpha_y);
        const float lambda_i = eval_lambda(omega_i_l, alpha_x, alpha_y);
        const float G2 = eval_G2(lambda_o, lambda_i);
        const float G1 = eval_G1(lambda_o);
        const float cos_theta_o = omega_o_l.z;
        const float eta = 1.0f / ior;
        const float r0 = sqr(eta - 1) / sqr(eta + 1);
        const float kr = schlick_fresnel(r0, dot(m, omega_i_l));

        pdf = (G1 * D * dot(m, omega_i_l)) /
              (4 * dot(m, omega_i_l) * omega_i_l.z);

        return strength * color * G2 * D * kr / (4 * omega_i_l.z * omega_o_l.z);
    } else {
        return make_float3(0);
    }
}

inline __device__ float2 ggx_sample11(float cos_theta_i, float u1, float u2) {
    float2 slope;
    // normaal incidence
    if (cos_theta_i > 0.9999f) {
        float r = sqrtf(u1 / (1.0f - u1));
        float phi = C_two_pi * u2;
        slope = make_float2(r * cosf(phi), r * sinf(phi));
        return slope;
    }

    float sin_theta = sqrtf(max(0.0f, 1.0f - sqr(cos_theta_i)));
    float tan_theta_i = sin_theta / cos_theta_i;
    float a = 1.0f / tan_theta_i;
    float G1 = 2.0f / (1.0f + sqrtf(1.0f + 1.0f / sqr(a)));

    // sample slope_x
    float A = 2.0f * u1 / G1 - 1.0f;
    float tmp = 1.0f / (sqr(A) - 1.0f);
    if (tmp > 1e10f) tmp = 1e10f;
    float B = tan_theta_i;
    float D = sqrtf(max(0.0f, sqr(B) * sqr(tmp) - (sqr(A) - sqr(B)) * tmp));
    float slope_x_1 = B * tmp - D;
    float slope_x_2 = B * tmp + D;
    slope.x = (A < 0 || slope_x_2 > 1.0f / tan_theta_i) ? slope_x_1 : slope_x_2;

    // sample slope y
    float S;
    if (u2 > 0.5f) {
        S = 1.0f;
        u2 = 2.0f * (u2 - 0.5f);
    } else {
        S = -1.0f;
        u2 = 2.0f * (0.5f - u2);
    }
    float z = (u2 * (u2 * (u2 * 0.27385f - 0.73369f) + 0.46341f)) /
              (u2 * (u2 * (u2 * 0.093073f + 0.309420f) - 1.0f) + 0.597999f);
    slope.y = S * z * sqrtf(1.0f + sqr(slope.x));

    return slope;
}

inline __device__ float3 ggx_sample(float3 omega_o_l, float alpha_x,
                                    float alpha_y, float u1, float u2) {
    // stretch the outgoing direction
    float3 omega_o_l_s = normalize(
        make_float3(alpha_x * omega_o_l.x, alpha_y * omega_o_l.y, omega_o_l.z));

    // sample the slope
    float2 slope = ggx_sample11(omega_o_l_s.z, u1, u2);

    // rotate
    float cos2_theta = sqr(omega_o_l_s.z);
    float sin_theta = sqrtf(1.0f - cos2_theta);
    float cos_phi = 0.0f;
    float sin_phi = 1.0f;
    if (sin_theta != 0.0f) {
        cos_phi = omega_o_l_s.x / sin_theta;
        sin_phi = omega_o_l_s.y / sin_theta;
    }
    float tmp = cos_phi * slope.x - sin_phi * slope.y;
    slope.y = sin_phi * slope.x + cos_phi * slope.y;
    slope.x = tmp;

    // unstretch
    slope.x = alpha_x * slope.x;
    slope.y = alpha_y * slope.y;

    // compute normal
    return normalize(make_float3(-slope.x, -slope.y, 1.0f));
}

/*#define SAMPLE_HEMI*/

RT_CALLABLE_PROGRAM BsdfSample bsdf_eval(ShadingFrame frame,
                                         const float3 omega_i) {
    BsdfSample bs;
    bs.omega_i = omega_i;
    const float3 omega_o_l = frame.world_to_local(frame.omega_o);
    const float3 omega_i_l = frame.world_to_local(omega_i);
    const float alpha_x = sqr(roughness);
    const float alpha_y = alpha_x;
    bs.f = ggx_eval(omega_o_l, omega_i_l, alpha_x, alpha_y, bs.pdf);
#ifdef SAMPLE_HEMI
    bs.pdf = dot(bs.omega_i, frame.n) / M_PIf;
#endif
    return bs;
}

RT_CALLABLE_PROGRAM BsdfSample bsdf_sample(ShadingFrame frame, const float u1,
                                           const float u2) {
    BsdfSample bs;
    const float3 omega_o_l = frame.world_to_local(frame.omega_o);
    const float alpha_x = sqr(roughness);
    const float alpha_y = alpha_x;
#ifdef SAMPLE_HEMI
    float3 omega_i_l;
    cosine_sample_hemisphere(u1, u2, omega_i_l);
    const bool flip = false;
#else
    const bool flip = omega_o_l.z < 0;
    const float3 m =
        ggx_sample(flip ? -omega_o_l : omega_o_l, alpha_x, alpha_y, u1, u2);
    const float3 omega_i_l = reflect(omega_o_l, m);

    bs.omega_i = normalize(omega_i_l.z * frame.n + omega_i_l.x * frame.u +
                           omega_i_l.y * frame.v);
#endif

    bs.f = ggx_eval(omega_o_l, omega_i_l, alpha_x, alpha_y, bs.pdf);
    if (flip) bs.omega_i = -bs.omega_i;

#ifdef SAMPLE_HEMI
    bs.pdf = dot(bs.omega_i, frame.n) / M_PIf;
#endif
    return bs;
}

RT_CALLABLE_PROGRAM float bsdf_pdf(ShadingFrame frame, const float3 omega_i) {
    return dot(omega_i, frame.n) / M_PIf;
}
