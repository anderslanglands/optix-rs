#include "shader_globals.cuh"
#include <optixu/optixu_math_namespace.h>
using namespace optix;

rtDeclareVariable(float3, in_color, , );
rtDeclareVariable(int, tex_id, , );
RT_CALLABLE_PROGRAM float3 get_color(ShaderGlobals sg) {
    float4 tx = rtTex2D<float4>(tex_id, sg.u, sg.v);
    return make_float3(tx.x, tx.y, tx.z);
    /*float3 ret = in_color;*/
    /*return ret;*/
}
