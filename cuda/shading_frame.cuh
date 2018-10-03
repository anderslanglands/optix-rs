#ifndef RT_SHADING_FRAME_H
#define RT_SHADING_FRAME_H

#include <optix.h>
#include <optix_world.h>
#include <optixu/optixu_vector_types.h>
#include <optixu/optixu_math_namespace.h>


struct ShadingFrame
{
    inline __device__ ShadingFrame(const float3& p_, const float3& normal, const float3& wo)
    {
        p = p_;
        n = normal;
        omega_o = wo;

        if( fabs(n.x) > fabs(n.z) )
        {
            v.x = -n.y;
            v.y =  n.x;
            v.z =  0;
        }
        else
        {
            v.x =  0;
            v.y = -n.z;
            v.z =  n.y;
        }

        v = optix::normalize(v);
        u = optix::cross( v, n );
    }

    inline __device__ ShadingFrame(const float3& p_, const float3& normal, const float3& wo, const float3& tangent)
    {
        p = p_;
        n = normal;
        omega_o = wo;
        u = tangent;
        v = optix::cross(u, n);
    }

    inline __device__ float3 local_to_world(const float3 p) const
    {
        return p.x*u + p.y*v + p.z*n;
    }

    inline __device__ float3 world_to_local(const float3 p) const
    {
        return make_float3(optix::dot(u, p), optix::dot(v, p), optix::dot(n, p));
    }

    optix::float3 p;
    optix::float3 n;
    optix::float3 u;
    optix::float3 v;
    optix::float3 omega_o;
};

#endif
