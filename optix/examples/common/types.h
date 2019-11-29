#pragma once

#if __CUDACC__
#define DEVICE __device__
#define FORCEINLINE __forceinline__
#else
#define DEVICE
#define FORCEINLINE
#endif

namespace osc {
using f32 = float;
using f64 = double;

using i8 = char;
using i16 = short;
using i32 = int;
using i64 = long long;

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
} // namespace osc