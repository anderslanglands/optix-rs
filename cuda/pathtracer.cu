#include <optix.h>
#include <optix_world.h>
#include "random.cuh"

using namespace optix;

struct PerRayData_radiance
{
    optix::float3 L_e;
    bool done;
    int depth;
    float z;
    uint seed;
    float3 p;
    float3 w;
    float3 f;
    float g_c;
    float g_nee;
};

struct PerRayData_shadow
{
    optix::float3 attenuation;
};

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(rtObject, scene_root, , );
rtBuffer<float4, 2> result_buffer;

rtDeclareVariable(Matrix4x4, camera_to_world, , );
rtDeclareVariable(Matrix4x4, raster_to_camera, , );

rtDeclareVariable(unsigned int, progression, , );

// a simple screen generator
RT_PROGRAM void generate_ray() {
    PerRayData_radiance prd;
    prd.z = 1000.0f;
    // prd.pixel = launch_index;
    prd.seed = tea<16>(launch_index.y * launch_dim.x + launch_index.x, progression);
    float u0 = rnd(prd.seed);
    float u1 = rnd(prd.seed);

    float x = float(launch_index.x) + u0;
    float y = float(launch_index.y) + u1;

    float3 result = make_float3(0.0f);

    float4 Pras4 = make_float4(x, y, 0.0f, 1.0f);
    float4 Pcam4 = raster_to_camera * Pras4;
    float4 Dcam4 = make_float4(Pcam4.x, Pcam4.y, -1 * Pcam4.z, 0.0f);
    float4 P_origin_cam = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 P_origin_world = camera_to_world * P_origin_cam;
    float4 D_world = camera_to_world * Dcam4;
    float4 P_world = camera_to_world * Pcam4;
    auto origin = make_float3(P_origin_world);
    auto direction = normalize(make_float3(Pcam4));
    auto throughput = make_float3(1.0f);

    for (int i = 0; i < 32; ++i) {
        optix::Ray ray = optix::make_Ray(origin, direction, 0, 1e-5f, RT_DEFAULT_MAX);
        prd.depth = 0;
        prd.done = false;
        prd.L_e = make_float3(0.0f);
        rtTrace(scene_root, ray, prd);

        throughput *= prd.f;
        result += prd.L_e * throughput * prd.g_nee;
        throughput *= prd.g_c;

        if (i > 2) {
            float p = fmaxf(throughput);
            if (rnd(prd.seed) >= p) {
                break;
            } else {
                throughput /= p;
            }
        }

        if (prd.done) {
            break;
        }

        origin = prd.p;
        direction = prd.w;
    }
    float alpha = prd.depth > 0 ? 1.0f : 0.0f;

    result_buffer[launch_index] = make_float4(result, alpha);
}

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );

RT_PROGRAM void miss() {
    prd_radiance.L_e = make_float3(0.f);
}

RT_PROGRAM void miss_shadow() { prd_shadow.attenuation = make_float3(1); }


rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(float3, in_diffuse_albedo, , );
// Diffuse surface
RT_PROGRAM void mtl_ch_diffuse() {
    float3 n =
        normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

    float3 hit_point = ray.origin + ray.direction * t_hit;

    float3 light_center = make_float3(277.5, 554, -277.5);
    float3 light_dim = make_float3(100.0f, 0.0f, 100.0f);
    float3 light_min = light_center - light_dim / 2;

    float3 light_origin = light_min + light_dim * make_float3(
        rnd(prd_radiance.seed), 0.0f, rnd(prd_radiance.seed)
    );
    float3 light_normal = make_float3(0.0f, -1.0f, 0.0f);


    float3 w_o = -ray.direction;
    float3 W_i = light_origin - hit_point;
    float d2 = dot(W_i, W_i);
    float3 w_i = normalize(W_i);

    float geo_l = max(0.0f, dot(w_i, n));

    PerRayData_shadow prd_shadow;
    prd_shadow.attenuation = make_float3(1.0f);
    if (geo_l > 0) {
        float d = sqrtf(d2);
        Ray ray_shadow = make_Ray(hit_point, w_i, 1, 1e-3f, d - 1e-3f);
        rtTrace(scene_root, ray_shadow, prd_shadow);
    }

    prd_radiance.L_e = 
        make_float3(1.0f) 
        * prd_shadow.attenuation;
    prd_radiance.g_nee = geo_l;    

    prd_radiance.z = t_hit;
    prd_radiance.depth = 1;

    optix::Onb onb(n);
    auto u0 = rnd(prd_radiance.seed);
    auto u1 = rnd(prd_radiance.seed);
    float3 w;
    cosine_sample_hemisphere(u0, u1, w);
    onb.inverse_transform(w);
    prd_radiance.w = w;
    prd_radiance.p = hit_point;
    prd_radiance.f = in_diffuse_albedo; 
    prd_radiance.g_c = dot(w, n);
}

RT_PROGRAM void mtl_ah_shadow() {
    prd_shadow.attenuation = optix::make_float3(0.0f);
    rtTerminateRay();
}

RT_PROGRAM void mtl_ch_emission() {
    prd_radiance.L_e = make_float3(3.0f);
    prd_radiance.f = make_float3(1.0f);
    prd_radiance.done = true;
}

// intersection
rtBuffer<float3> vertex_buffer;
rtBuffer<float3> normal_buffer;
rtBuffer<float2> texcoord_buffer;
rtBuffer<int3> index_buffer;
rtBuffer<int> material_buffer;

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
