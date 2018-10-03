#include <optix.h>
#include <optix_world.h>
#include "random.cuh"
#include "raydata.cuh"

using namespace optix;

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(rtObject, scene_root, , );
rtBuffer<float4, 2> result_buffer;
rtBuffer<float4, 2> result_buffer_qr;
rtBuffer<float, 2> depth_buffer;
rtBuffer<float, 2> depth_buffer_qr;
rtDeclareVariable(Matrix4x4, camera_to_world, , );
rtDeclareVariable(Matrix4x4, raster_to_camera, , );
rtDeclareVariable(float, inv_near_clip, , );
rtDeclareVariable(float, inv_far_clip, , );
rtDeclareVariable(int, progression, , );

rtBuffer<float, 2> sample_buffer;
rtBuffer<uint, 2> blue_noise_buffer;

RT_PROGRAM void generate_ray() {
    float x = float(launch_index.x);
    float y = float(launch_index.y);
    int prg = progression;
    if (prg < 0) {
        prg = 0;
        x *= 4;
        y *= 4;
    }

    float3 result = make_float3(0.0f);
    float result_z = 0.0f;

    unsigned int seed =
        tea<16>(launch_dim.x * launch_index.y + launch_index.x, prg);

    unsigned int pixel_offset =
        tea<16>(launch_dim.x * launch_index.y + launch_index.x, 0);

    unsigned int bn_offset = blue_noise_buffer[make_uint2(
        launch_index.x % 128, launch_index.y % 128)];

    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.z = 1000.0f;
    prd.sampler.seed = seed;
    prd.sampler.index = (prg + bn_offset) % 16384;
    prd.sampler.dim = 0;
    prd.sampler.offset = pixel_offset % 256;
    prd.pixel = launch_index;

    float u1, u2;
    NEXT_SAMPLE(prd.sampler, u1);
    NEXT_SAMPLE(prd.sampler, u2);

    float4 Pras4 = make_float4(x + u1, y + u2, 0.0f, 1.0f);
    float4 Pcam4 = raster_to_camera * Pras4;
    float4 Dcam4 = normalize(make_float4(Pcam4.x, Pcam4.y, -1 * Pcam4.z, 0.0f));
    float4 P_origin_cam = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    float4 P_origin_world = camera_to_world * P_origin_cam;
    float4 D_world = camera_to_world * Dcam4;
    float3 origin =
        make_float3(P_origin_world.x, P_origin_world.y, P_origin_world.z);
    float3 direction = normalize(make_float3(D_world.x, D_world.y, D_world.z));

    optix::Ray ray =
        optix::make_Ray(origin, direction, 0, 1.0e-5f, RT_DEFAULT_MAX);

    rtTrace(scene_root, ray, prd);

    seed = prd.sampler.seed;

    float z = 1.0f / (prd.z);
    result_z = max(
        clamp((z - inv_near_clip) / (inv_far_clip - inv_near_clip), 0.0f, 1.0f),
        result_z);
    result = prd.result;
    float alpha = prd.depth > 0 ? 1.0f : 0.0f;

    if (progression < 0) {
        result_buffer_qr[launch_index] = make_float4(result, alpha);
        depth_buffer_qr[launch_index] = result_z;
    } else {
        result_buffer[launch_index] = make_float4(result, alpha);
        depth_buffer[launch_index] = result_z;
    }
}

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(int, env_tex_id, , );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
RT_PROGRAM void miss() {
    prd_radiance.result = make_float3(0.0f);
    /*float3 dir = normalize(ray.direction);*/
    /*float theta = acosf(dir.y) / M_PIf;*/
    /*float phi = atan2f(dir.z, dir.x);*/
    /*if (phi < 0.0f) phi += 2.0f * M_PIf;*/
    /*phi /= (2.0f * M_PIf);*/
    /*float4 tx = rtTex2D<float4>(env_tex_id, phi, theta);*/
    /*prd_radiance.result = make_float3(tx.x, tx.y, tx.z);*/

    prd_radiance.z = 100.0f;
}

RT_PROGRAM void miss_shadow() { prd_shadow.attenuation = make_float3(1); }
