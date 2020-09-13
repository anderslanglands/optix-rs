#pragma once

#include "../common/gdt/gdt/math/vec.h"

namespace osc {
using namespace gdt;

struct LaunchParams {
    int frameID{0};
    uint32_t* colorBuffer;
    vec2i fbSize;
};

} // namespace osc
