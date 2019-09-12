#include "types.h"

namespace osc {
/*! simple 24-bit linear congruence generator */
template <u32 N = 16> struct LCG {

    inline DEVICE
    LCG() { /* intentionally empty so we can use it in device vars that
               don't allow dynamic initialization (ie, PRD) */
    }
    inline DEVICE LCG(u32 val0, u32 val1) { init(val0, val1); }

    inline DEVICE void init(u32 val0, u32 val1) {
        u32 v0 = val0;
        u32 v1 = val1;
        u32 s0 = 0;

        for (u32 n = 0; n < N; n++) {
            s0 += 0x9e3779b9;
            v0 +=
                ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
            v1 +=
                ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
        }
        state = v0;
    }

    // Generate random u32 in [0, 2^24)
    inline DEVICE float operator()() {
        const u32 LCG_A = 1664525u;
        const u32 LCG_C = 1013904223u;
        state = (LCG_A * state + LCG_C);
        return (state & 0x00FFFFFF) / (float)0x01000000;
    }

    u32 state;
};

} // namespace osc
