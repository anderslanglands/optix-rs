#pragma once

#include "../common/gdt/gdt/math/vec.h"

namespace osc {
using namespace gdt;

struct LaunchParams {
    int frame_id{0};
    float4* color_buffer;
    vec2i fb_size;
};

} // namespace osc
