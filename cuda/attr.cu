#include <optix.h>
#include <optixu/optixu_math_namespace.h>

rtBuffer<float3> vertex_buffer;
rtBuffer<int3>   vindex_buffer;

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

RT_PROGRAM void attribute_program()
{
    const unsigned int primIdx = rtGetPrimitiveIndex();

    const float3 v0    = vertex_buffer[v_idx.x];
    const float3 v1    = vertex_buffer[v_idx.y];
    const float3 v2    = vertex_buffer[v_idx.z];
    const float3 Ng    = optix::cross( v1 - v0, v2 - v0 );
    geometric_normal = optix::normalize( Ng );
    shading_normal = geometric_normal;
}