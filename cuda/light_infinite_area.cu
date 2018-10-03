#include "lights.cuh"
#include "raydata.cuh"
#include "shader_globals.cuh"

using namespace optix;

rtDeclareVariable(Matrix4x4, light_to_world, , );
rtDeclareVariable(Matrix4x4, world_to_light, , );
rtDeclareVariable(float3, L_e, , );
rtDeclareVariable(int, tex_id, , );
rtBuffer<float2, 2> buf_conditional;
rtBuffer<float2, 1> buf_marginal;

__device__ __forceinline__ uint find_interval(const float2* cdf,
                                              const uint size, const float u) {
    uint first = 0, len = size;
    while (len > 0) {
        uint half = len >> 1, middle = first + half;
        if (cdf[middle].y <= u) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }

    return clamp(first - 1, 0u, size - 2);
}

__device__ __forceinline__ void sample_continuous_1d(const float2* cdf,
                                                     const uint size,
                                                     const float u, float& x,
                                                     float& pdf, uint& offset) {
    offset = find_interval(cdf, size, u);

    float du = u - cdf[offset].y;
    if (cdf[offset + 1].y - cdf[offset].y > 0) {
        du /= (cdf[offset + 1].y - cdf[offset].y);
    }

    pdf = cdf[offset].x / cdf[size - 1].x;
    x = (float(offset) + du) / float(size - 1);
}

//#define COSINE_SAMPLING
RT_CALLABLE_PROGRAM LightSample light_sample(ShadingFrame frame, const float u1,
                                             const float u2) {
    LightSample ls;
    if (tex_id != RT_TEXTURE_ID_NULL) {
#ifdef COSINE_SAMPLING
        float3 omega_l;
        cosine_sample_hemisphere(u1, u2, omega_l);
        ls.omega_i = frame.local_to_world(omega_l);
        float4 dir_w = make_float4(ls.omega_i, 0.0f);
        float4 dir = world_to_light * dir_w;
        float theta = acosf(dir.y) / M_PIf;
        float phi = atan2f(dir.x, -dir.z);
        if (phi < 0.0f)
            phi += 2.0f * M_PIf;
        phi /= (2.0f * M_PIf);
        float4 tx = rtTex2D<float4>(tex_id, phi, theta);
        ls.L_i = make_float3(tx.x, tx.y, tx.z);
        ls.pdf = dot(ls.omega_i, frame.n) / M_PIf;
#else
        float map_u, map_v;
        float pdf0, pdf1;
        uint offset0, offset1;
        sample_continuous_1d(&buf_marginal[0], buf_marginal.size(), u2, map_v,
                             pdf1, offset1);
        sample_continuous_1d(&buf_conditional[make_uint2(0, offset1)],
                             buf_conditional.size().x, u1, map_u, pdf0,
                             offset0);
        float map_pdf = pdf0 * pdf1;

        float4 tx = rtTex2D<float4>(tex_id, map_u, map_v);
        ls.L_i = make_float3(tx.x, tx.y, tx.z);

        float theta = map_v * M_PIf;
        float phi = map_u * 2.0f * M_PIf;
    
        float sin_theta = sinf(theta);
        float cos_theta = cosf(theta);
        float4 dir = make_float4(sin_theta * cosf(phi), cos_theta,
                                 sin_theta * -sinf(phi), 0.0f);
        float4 dir_w = light_to_world * dir;
        ls.omega_i = make_float3(dir_w.z, dir_w.y, dir_w.x);
        ls.pdf = map_pdf / (2.0f * M_PIf * M_PIf * sin_theta);
#endif
    } else {
        float3 omega_l;
        cosine_sample_hemisphere(u1, u2, omega_l);
        ls.omega_i = frame.local_to_world(omega_l);
        ls.L_i = L_e;
        ls.pdf = dot(ls.omega_i, frame.n) / M_PIf;
    }
    return ls;
}

RT_CALLABLE_PROGRAM LightSample light_eval(const float3 p_s,
                                           const float3 omega_i) {
    LightSample ls;
    ls.omega_i = omega_i;
    float4 dir_w = make_float4(ls.omega_i, 0.0f);
    float4 dir = world_to_light * dir_w;
    float theta = acosf(dir.y);
    float map_v = theta / M_PIf;
    float sin_theta = sinf(theta);
    if (sin_theta == 0.0f) {
        ls.L_i = make_float3(0.0f);
        ls.pdf = 0.0f;
        return ls;
    }
    float phi = atan2f(dir.x, -dir.z);
    if (phi < 0.0f)
        phi += 2.0f * M_PIf;
    float map_u = phi / (2.0f * M_PIf);
    float4 tx = rtTex2D<float4>(tex_id, map_u, map_v);
    ls.L_i = make_float3(tx.x, tx.y, tx.z);

    int iu = clamp(int(map_u * (buf_conditional.size().x - 1)), 0,
                   buf_conditional.size().x - 2);
    int iv = clamp(int(map_v * (buf_marginal.size() - 1)), 0,
                   buf_marginal.size() - 2);
    float map_pdf = buf_conditional[make_uint2(iu, iv)].x /
                    buf_marginal[buf_marginal.size() - 1].x;

    ls.pdf = map_pdf / (2.0f * M_PIf * M_PIf * sin_theta);

    return ls;
}

RT_CALLABLE_PROGRAM float light_pdf(const float3 p_s, const float3 p_l) {
    return 1.0f;
}
