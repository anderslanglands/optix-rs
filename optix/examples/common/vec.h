#pragma once

#include "types.h"

namespace osc {

using u32x2 = uint2;
using u32x3 = uint3;
using u32x4 = uint4;

using i32x2 = int2;
using i32x3 = int3;
using i32x4 = int4;

using f32x2 = float2;
using f32x3 = float3;
using f32x4 = float4;

using ::copysignf;
using ::fmaxf;
using ::fminf;
using ::max;
using ::min;

FORCEINLINE DEVICE float lerp(const float a, const float b, const float t) {
    return a + t * (b - a);
}

/** bilerp */
FORCEINLINE DEVICE float bilerp(const float x00, const float x10,
                                const float x01, const float x11, const float u,
                                const float v) {
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp */
FORCEINLINE DEVICE float clamp(const float f, const float a, const float b) {
    return fmaxf(a, fminf(f, b));
}

FORCEINLINE DEVICE float saturate(const float x) { return ::__saturatef(x); }

FORCEINLINE DEVICE f32 fract(const f32 x) { return ::modff(x, nullptr); }

/* f32x2 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
FORCEINLINE DEVICE f32x2 make_f32x2(const float a, const float b) {
    return ::make_float2(a, b);
}
FORCEINLINE DEVICE f32x2 make_f32x2(const float s) { return make_f32x2(s, s); }
FORCEINLINE DEVICE f32x2 make_f32x2(const int2& a) {
    return make_f32x2(float(a.x), float(a.y));
}
FORCEINLINE DEVICE f32x2 make_f32x2(const uint2& a) {
    return make_f32x2(float(a.x), float(a.y));
}
/** @} */

/** negate */
FORCEINLINE DEVICE f32x2 operator-(const f32x2& a) {
    return make_f32x2(-a.x, -a.y);
}

/** min
 * @{
 */
FORCEINLINE DEVICE f32x2 fminf(const f32x2& a, const f32x2& b) {
    return make_f32x2(fminf(a.x, b.x), fminf(a.y, b.y));
}
FORCEINLINE DEVICE float fminf(const f32x2& a) { return fminf(a.x, a.y); }
/** @} */

/** max
 * @{
 */
FORCEINLINE DEVICE f32x2 fmaxf(const f32x2& a, const f32x2& b) {
    return make_f32x2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
FORCEINLINE DEVICE float fmaxf(const f32x2& a) { return fmaxf(a.x, a.y); }
/** @} */

/** add
 * @{
 */
FORCEINLINE DEVICE f32x2 operator+(const f32x2& a, const f32x2& b) {
    return make_f32x2(a.x + b.x, a.y + b.y);
}
FORCEINLINE DEVICE f32x2 operator+(const f32x2& a, const float b) {
    return make_f32x2(a.x + b, a.y + b);
}
FORCEINLINE DEVICE f32x2 operator+(const float a, const f32x2& b) {
    return make_f32x2(a + b.x, a + b.y);
}
FORCEINLINE DEVICE void operator+=(f32x2& a, const f32x2& b) {
    a.x += b.x;
    a.y += b.y;
}
/** @} */

/** subtract
 * @{
 */
FORCEINLINE DEVICE f32x2 operator-(const f32x2& a, const f32x2& b) {
    return make_f32x2(a.x - b.x, a.y - b.y);
}
FORCEINLINE DEVICE f32x2 operator-(const f32x2& a, const float b) {
    return make_f32x2(a.x - b, a.y - b);
}
FORCEINLINE DEVICE f32x2 operator-(const float a, const f32x2& b) {
    return make_f32x2(a - b.x, a - b.y);
}
FORCEINLINE DEVICE void operator-=(f32x2& a, const f32x2& b) {
    a.x -= b.x;
    a.y -= b.y;
}
/** @} */

/** multiply
 * @{
 */
FORCEINLINE DEVICE f32x2 operator*(const f32x2& a, const f32x2& b) {
    return make_f32x2(a.x * b.x, a.y * b.y);
}
FORCEINLINE DEVICE f32x2 operator*(const f32x2& a, const float s) {
    return make_f32x2(a.x * s, a.y * s);
}
FORCEINLINE DEVICE f32x2 operator*(const float s, const f32x2& a) {
    return make_f32x2(a.x * s, a.y * s);
}
FORCEINLINE DEVICE void operator*=(f32x2& a, const f32x2& s) {
    a.x *= s.x;
    a.y *= s.y;
}
FORCEINLINE DEVICE void operator*=(f32x2& a, const float s) {
    a.x *= s;
    a.y *= s;
}
/** @} */

/** divide
 * @{
 */
FORCEINLINE DEVICE f32x2 operator/(const f32x2& a, const f32x2& b) {
    return make_f32x2(a.x / b.x, a.y / b.y);
}
FORCEINLINE DEVICE f32x2 operator/(const f32x2& a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
}
FORCEINLINE DEVICE f32x2 operator/(const float s, const f32x2& a) {
    return make_f32x2(s / a.x, s / a.y);
}
FORCEINLINE DEVICE void operator/=(f32x2& a, const float s) {
    float inv = 1.0f / s;
    a *= inv;
}
/** @} */

/** lerp */
FORCEINLINE DEVICE f32x2 lerp(const f32x2& a, const f32x2& b, const float t) {
    return a + t * (b - a);
}

/** bilerp */
FORCEINLINE DEVICE f32x2 bilerp(const f32x2& x00, const f32x2& x10,
                                const f32x2& x01, const f32x2& x11,
                                const float u, const float v) {
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
 * @{
 */
FORCEINLINE DEVICE f32x2 clamp(const f32x2& v, const float a, const float b) {
    return make_f32x2(clamp(v.x, a, b), clamp(v.y, a, b));
}

FORCEINLINE DEVICE f32x2 clamp(const f32x2& v, const f32x2& a, const f32x2& b) {
    return make_f32x2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** dot product */
FORCEINLINE DEVICE float dot(const f32x2& a, const f32x2& b) {
    return a.x * b.x + a.y * b.y;
}

/** length */
FORCEINLINE DEVICE float length(const f32x2& v) { return sqrtf(dot(v, v)); }

/** normalize */
FORCEINLINE DEVICE f32x2 normalize(const f32x2& v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

/** floor */
FORCEINLINE DEVICE f32x2 floor(const f32x2& v) {
    return make_f32x2(::floorf(v.x), ::floorf(v.y));
}

/** reflect */
FORCEINLINE DEVICE f32x2 reflect(const f32x2& i, const f32x2& n) {
    return i - 2.0f * n * dot(n, i);
}

/** Faceforward
 * Returns N if dot(i, nref) > 0; else -N;
 * Typical usage is N = faceforward(N, -ray.dir, N);
 * Note that this is opposite of what faceforward does in Cg and GLSL */
FORCEINLINE DEVICE f32x2 faceforward(const f32x2& n, const f32x2& i,
                                     const f32x2& nref) {
    return n * copysignf(1.0f, dot(i, nref));
}

/** exp */
FORCEINLINE DEVICE f32x2 expf(const f32x2& v) {
    return make_f32x2(::expf(v.x), ::expf(v.y));
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE float getByIndex(const f32x2& v, int i) {
    return ((float*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE void setByIndex(f32x2& v, int i, float x) {
    ((float*)(&v))[i] = x;
}

/* f32x3 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */

FORCEINLINE DEVICE f32x3 make_f32x3(const float a, const float b,
                                    const float c) {
    return ::make_float3(a, b, c);
}

FORCEINLINE DEVICE f32x3 make_f32x3(const float s) {
    return ::make_float3(s, s, s);
}
FORCEINLINE DEVICE f32x3 make_f32x3(const f32x2& a) {
    return ::make_float3(a.x, a.y, 0.0f);
}
FORCEINLINE DEVICE f32x3 make_f32x3(const int3& a) {
    return ::make_float3(float(a.x), float(a.y), float(a.z));
}
FORCEINLINE DEVICE f32x3 make_f32x3(const uint3& a) {
    return ::make_float3(float(a.x), float(a.y), float(a.z));
}
/** @} */

/** negate */
FORCEINLINE DEVICE f32x3 operator-(const f32x3& a) {
    return ::make_float3(-a.x, -a.y, -a.z);
}

/** min
 * @{
 */
FORCEINLINE DEVICE f32x3 fminf(const f32x3& a, const f32x3& b) {
    return make_f32x3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
FORCEINLINE DEVICE float fminf(const f32x3& a) {
    return fminf(fminf(a.x, a.y), a.z);
}
/** @} */

/** max
 * @{
 */
FORCEINLINE DEVICE f32x3 fmaxf(const f32x3& a, const f32x3& b) {
    return make_f32x3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
FORCEINLINE DEVICE float fmaxf(const f32x3& a) {
    return fmaxf(fmaxf(a.x, a.y), a.z);
}
/** @} */

/** add
 * @{
 */
FORCEINLINE DEVICE f32x3 operator+(const f32x3& a, const f32x3& b) {
    return make_f32x3(a.x + b.x, a.y + b.y, a.z + b.z);
}
FORCEINLINE DEVICE f32x3 operator+(const f32x3& a, const float b) {
    return make_f32x3(a.x + b, a.y + b, a.z + b);
}
FORCEINLINE DEVICE f32x3 operator+(const float a, const f32x3& b) {
    return make_f32x3(a + b.x, a + b.y, a + b.z);
}
FORCEINLINE DEVICE void operator+=(f32x3& a, const f32x3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
/** @} */

/** subtract
 * @{
 */
FORCEINLINE DEVICE f32x3 operator-(const f32x3& a, const f32x3& b) {
    return make_f32x3(a.x - b.x, a.y - b.y, a.z - b.z);
}
FORCEINLINE DEVICE f32x3 operator-(const f32x3& a, const float b) {
    return make_f32x3(a.x - b, a.y - b, a.z - b);
}
FORCEINLINE DEVICE f32x3 operator-(const float a, const f32x3& b) {
    return make_f32x3(a - b.x, a - b.y, a - b.z);
}
FORCEINLINE DEVICE void operator-=(f32x3& a, const f32x3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
/** @} */

/** multiply
 * @{
 */
FORCEINLINE DEVICE f32x3 operator*(const f32x3& a, const f32x3& b) {
    return make_f32x3(a.x * b.x, a.y * b.y, a.z * b.z);
}
FORCEINLINE DEVICE f32x3 operator*(const f32x3& a, const float s) {
    return make_f32x3(a.x * s, a.y * s, a.z * s);
}
FORCEINLINE DEVICE f32x3 operator*(const float s, const f32x3& a) {
    return make_f32x3(a.x * s, a.y * s, a.z * s);
}
FORCEINLINE DEVICE void operator*=(f32x3& a, const f32x3& s) {
    a.x *= s.x;
    a.y *= s.y;
    a.z *= s.z;
}
FORCEINLINE DEVICE void operator*=(f32x3& a, const float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
}
/** @} */

/** divide
 * @{
 */
FORCEINLINE DEVICE f32x3 operator/(const f32x3& a, const f32x3& b) {
    return make_f32x3(a.x / b.x, a.y / b.y, a.z / b.z);
}
FORCEINLINE DEVICE f32x3 operator/(const f32x3& a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
}
FORCEINLINE DEVICE f32x3 operator/(const float s, const f32x3& a) {
    return make_f32x3(s / a.x, s / a.y, s / a.z);
}
FORCEINLINE DEVICE void operator/=(f32x3& a, const float s) {
    float inv = 1.0f / s;
    a *= inv;
}
/** @} */

/** lerp */
FORCEINLINE DEVICE f32x3 lerp(const f32x3& a, const f32x3& b, const float t) {
    return a + t * (b - a);
}

/** bilerp */
FORCEINLINE DEVICE f32x3 bilerp(const f32x3& x00, const f32x3& x10,
                                const f32x3& x01, const f32x3& x11,
                                const float u, const float v) {
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
 * @{
 */
FORCEINLINE DEVICE f32x3 clamp(const f32x3& v, const float a, const float b) {
    return make_f32x3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

FORCEINLINE DEVICE f32x3 clamp(const f32x3& v, const f32x3& a, const f32x3& b) {
    return make_f32x3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
                      clamp(v.z, a.z, b.z));
}
/** @} */

/** dot product */
FORCEINLINE DEVICE float dot(const f32x3& a, const f32x3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** cross product */
FORCEINLINE DEVICE f32x3 cross(const f32x3& a, const f32x3& b) {
    return make_f32x3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x);
}

/** length */
FORCEINLINE DEVICE float length(const f32x3& v) { return sqrtf(dot(v, v)); }

/** normalize */
FORCEINLINE DEVICE f32x3 normalize(const f32x3& v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

/** floor */
FORCEINLINE DEVICE f32x3 floor(const f32x3& v) {
    return make_f32x3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z));
}

/** reflect */
FORCEINLINE DEVICE f32x3 reflect(const f32x3& i, const f32x3& n) {
    return i - 2.0f * n * dot(n, i);
}

/** Faceforward
 * Returns N if dot(i, nref) > 0; else -N;
 * Typical usage is N = faceforward(N, -ray.dir, N);
 * Note that this is opposite of what faceforward does in Cg and GLSL */
FORCEINLINE DEVICE f32x3 faceforward(const f32x3& n, const f32x3& i,
                                     const f32x3& nref) {
    return n * copysignf(1.0f, dot(i, nref));
}

/** exp */
FORCEINLINE DEVICE f32x3 expf(const f32x3& v) {
    return make_f32x3(::expf(v.x), ::expf(v.y), ::expf(v.z));
}

/** sin */
FORCEINLINE DEVICE f32x3 sinf(const f32x3& v) {
    return make_f32x3(::sinf(v.x), ::sinf(v.y), ::sinf(v.z));
}

/// Return the fractional part of each component
FORCEINLINE DEVICE f32x3 fract(const f32x3& v) {
    return make_f32x3(fract(v.x), fract(v.y), fract(v.z));
}

/// Break each element down into its fractional and integer components
FORCEINLINE DEVICE f32x3 modff(const f32x3& v, f32x3& i) {
    f32 ix, iy, iz;
    f32x3 result =
        make_f32x3(::modff(v.x, &ix), ::modff(v.y, &iy), ::modff(v.z, &iz));
    i = make_f32x3(ix, iy, iz);
    return result;
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE float getByIndex(const f32x3& v, int i) {
    return ((float*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE void setByIndex(f32x3& v, int i, float x) {
    ((float*)(&v))[i] = x;
}

/* f32x4 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
FORCEINLINE DEVICE f32x4 make_f32x4(const float a, const float b, const float c,
                                    const float d) {
    return ::make_float4(a, b, c, d);
}

FORCEINLINE DEVICE f32x4 make_f32x4(const float s) {
    return ::make_float4(s, s, s, s);
}
FORCEINLINE DEVICE f32x4 make_f32x4(const f32x3& a) {
    return ::make_float4(a.x, a.y, a.z, 0.0f);
}
FORCEINLINE DEVICE f32x4 make_f32x4(const i32x4& a) {
    return ::make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
FORCEINLINE DEVICE f32x4 make_f32x4(const u32x4& a) {
    return ::make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
/** @} */

/** negate */
FORCEINLINE DEVICE f32x4 operator-(const f32x4& a) {
    return make_f32x4(-a.x, -a.y, -a.z, -a.w);
}

/** min
 * @{
 */
FORCEINLINE DEVICE f32x4 fminf(const f32x4& a, const f32x4& b) {
    return make_f32x4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z),
                      fminf(a.w, b.w));
}
FORCEINLINE DEVICE float fminf(const f32x4& a) {
    return fminf(fminf(a.x, a.y), fminf(a.z, a.w));
}
/** @} */

/** max
 * @{
 */
FORCEINLINE DEVICE f32x4 fmaxf(const f32x4& a, const f32x4& b) {
    return make_f32x4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z),
                      fmaxf(a.w, b.w));
}
FORCEINLINE DEVICE float fmaxf(const f32x4& a) {
    return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}
/** @} */

/** add
 * @{
 */
FORCEINLINE DEVICE f32x4 operator+(const f32x4& a, const f32x4& b) {
    return make_f32x4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
FORCEINLINE DEVICE f32x4 operator+(const f32x4& a, const float b) {
    return make_f32x4(a.x + b, a.y + b, a.z + b, a.w + b);
}
FORCEINLINE DEVICE f32x4 operator+(const float a, const f32x4& b) {
    return make_f32x4(a + b.x, a + b.y, a + b.z, a + b.w);
}
FORCEINLINE DEVICE void operator+=(f32x4& a, const f32x4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
/** @} */

/** subtract
 * @{
 */
FORCEINLINE DEVICE f32x4 operator-(const f32x4& a, const f32x4& b) {
    return make_f32x4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
FORCEINLINE DEVICE f32x4 operator-(const f32x4& a, const float b) {
    return make_f32x4(a.x - b, a.y - b, a.z - b, a.w - b);
}
FORCEINLINE DEVICE f32x4 operator-(const float a, const f32x4& b) {
    return make_f32x4(a - b.x, a - b.y, a - b.z, a - b.w);
}
FORCEINLINE DEVICE void operator-=(f32x4& a, const f32x4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
/** @} */

/** multiply
 * @{
 */
FORCEINLINE DEVICE f32x4 operator*(const f32x4& a, const f32x4& s) {
    return make_f32x4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}
FORCEINLINE DEVICE f32x4 operator*(const f32x4& a, const float s) {
    return make_f32x4(a.x * s, a.y * s, a.z * s, a.w * s);
}
FORCEINLINE DEVICE f32x4 operator*(const float s, const f32x4& a) {
    return make_f32x4(a.x * s, a.y * s, a.z * s, a.w * s);
}
FORCEINLINE DEVICE void operator*=(f32x4& a, const f32x4& s) {
    a.x *= s.x;
    a.y *= s.y;
    a.z *= s.z;
    a.w *= s.w;
}
FORCEINLINE DEVICE void operator*=(f32x4& a, const float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    a.w *= s;
}
/** @} */

/** divide
 * @{
 */
FORCEINLINE DEVICE f32x4 operator/(const f32x4& a, const f32x4& b) {
    return make_f32x4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
FORCEINLINE DEVICE f32x4 operator/(const f32x4& a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
}
FORCEINLINE DEVICE f32x4 operator/(const float s, const f32x4& a) {
    return make_f32x4(s / a.x, s / a.y, s / a.z, s / a.w);
}
FORCEINLINE DEVICE void operator/=(f32x4& a, const float s) {
    float inv = 1.0f / s;
    a *= inv;
}
/** @} */

/** lerp */
FORCEINLINE DEVICE f32x4 lerp(const f32x4& a, const f32x4& b, const float t) {
    return a + t * (b - a);
}

/** bilerp */
FORCEINLINE DEVICE f32x4 bilerp(const f32x4& x00, const f32x4& x10,
                                const f32x4& x01, const f32x4& x11,
                                const float u, const float v) {
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
 * @{
 */
FORCEINLINE DEVICE f32x4 clamp(const f32x4& v, const float a, const float b) {
    return make_f32x4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b),
                      clamp(v.w, a, b));
}

FORCEINLINE DEVICE f32x4 clamp(const f32x4& v, const f32x4& a, const f32x4& b) {
    return make_f32x4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
                      clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

FORCEINLINE DEVICE f32x4 saturate(const f32x4& v) {
    return make_f32x4(saturate(v.x), saturate(v.y), saturate(v.z),
                      saturate(v.w));
} /** @} */

/** dot product */
FORCEINLINE DEVICE float dot(const f32x4& a, const f32x4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/** length */
FORCEINLINE DEVICE float length(const f32x4& r) { return sqrtf(dot(r, r)); }

/** normalize */
FORCEINLINE DEVICE f32x4 normalize(const f32x4& v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

/** floor */
FORCEINLINE DEVICE f32x4 floor(const f32x4& v) {
    return make_f32x4(::floorf(v.x), ::floorf(v.y), ::floorf(v.z),
                      ::floorf(v.w));
}

/** reflect */
FORCEINLINE DEVICE f32x4 reflect(const f32x4& i, const f32x4& n) {
    return i - 2.0f * n * dot(n, i);
}

/**
 * Faceforward
 * Returns N if dot(i, nref) > 0; else -N;
 * Typical usage is N = faceforward(N, -ray.dir, N);
 * Note that this is opposite of what faceforward does in Cg and GLSL
 */
FORCEINLINE DEVICE f32x4 faceforward(const f32x4& n, const f32x4& i,
                                     const f32x4& nref) {
    return n * copysignf(1.0f, dot(i, nref));
}

/** exp */
FORCEINLINE DEVICE f32x4 expf(const f32x4& v) {
    return make_f32x4(::expf(v.x), ::expf(v.y), ::expf(v.z), ::expf(v.w));
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE float getByIndex(const f32x4& v, int i) {
    return ((float*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE void setByIndex(f32x4& v, int i, float x) {
    ((float*)(&v))[i] = x;
}

/* int functions */
/******************************************************************************/

/** clamp */
FORCEINLINE DEVICE int clamp(const int f, const int a, const int b) {
    return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE int getByIndex(const int1& v, int i) {
    return ((int*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE void setByIndex(int1& v, int i, int x) {
    ((int*)(&v))[i] = x;
}

/* int2 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
FORCEINLINE DEVICE i32x2 make_i32x2(const int a, const int b) {
    return ::make_int2(a, b);
}

FORCEINLINE DEVICE i32x2 make_i32x2(const int s) { return make_int2(s, s); }
FORCEINLINE DEVICE i32x2 make_i32x2(const f32x2& a) {
    return make_int2(int(a.x), int(a.y));
}
/** @} */

/** negate */
FORCEINLINE DEVICE int2 operator-(const int2& a) {
    return make_int2(-a.x, -a.y);
}

/** min */
FORCEINLINE DEVICE int2 min(const int2& a, const int2& b) {
    return make_int2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
FORCEINLINE DEVICE int2 max(const int2& a, const int2& b) {
    return make_int2(max(a.x, b.x), max(a.y, b.y));
}

/** add
 * @{
 */
FORCEINLINE DEVICE int2 operator+(const int2& a, const int2& b) {
    return make_int2(a.x + b.x, a.y + b.y);
}
FORCEINLINE DEVICE void operator+=(int2& a, const int2& b) {
    a.x += b.x;
    a.y += b.y;
}
/** @} */

/** subtract
 * @{
 */
FORCEINLINE DEVICE int2 operator-(const int2& a, const int2& b) {
    return make_int2(a.x - b.x, a.y - b.y);
}
FORCEINLINE DEVICE int2 operator-(const int2& a, const int b) {
    return make_int2(a.x - b, a.y - b);
}
FORCEINLINE DEVICE void operator-=(int2& a, const int2& b) {
    a.x -= b.x;
    a.y -= b.y;
}
/** @} */

/** multiply
 * @{
 */
FORCEINLINE DEVICE int2 operator*(const int2& a, const int2& b) {
    return make_int2(a.x * b.x, a.y * b.y);
}
FORCEINLINE DEVICE int2 operator*(const int2& a, const int s) {
    return make_int2(a.x * s, a.y * s);
}
FORCEINLINE DEVICE int2 operator*(const int s, const int2& a) {
    return make_int2(a.x * s, a.y * s);
}
FORCEINLINE DEVICE void operator*=(int2& a, const int s) {
    a.x *= s;
    a.y *= s;
}
/** @} */

/** clamp
 * @{
 */
FORCEINLINE DEVICE int2 clamp(const int2& v, const int a, const int b) {
    return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}

FORCEINLINE DEVICE int2 clamp(const int2& v, const int2& a, const int2& b) {
    return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
 * @{
 */
FORCEINLINE DEVICE bool operator==(const int2& a, const int2& b) {
    return a.x == b.x && a.y == b.y;
}

FORCEINLINE DEVICE bool operator!=(const int2& a, const int2& b) {
    return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE int getByIndex(const int2& v, int i) {
    return ((int*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE void setByIndex(int2& v, int i, int x) {
    ((int*)(&v))[i] = x;
}

/* int3 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
FORCEINLINE DEVICE i32x3 make_i32x3(const int a, const int b, const int c) {
    return ::make_int3(a, b, c);
}

FORCEINLINE DEVICE i32x3 make_i32x3(const int s) { return make_int3(s, s, s); }
FORCEINLINE DEVICE i32x3 make_i32x3(const f32x3& a) {
    return make_int3(int(a.x), int(a.y), int(a.z));
}
/** @} */

/** negate */
FORCEINLINE DEVICE int3 operator-(const int3& a) {
    return make_int3(-a.x, -a.y, -a.z);
}

/** min */
FORCEINLINE DEVICE int3 min(const int3& a, const int3& b) {
    return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
FORCEINLINE DEVICE int3 max(const int3& a, const int3& b) {
    return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
 * @{
 */
FORCEINLINE DEVICE int3 operator+(const int3& a, const int3& b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
FORCEINLINE DEVICE void operator+=(int3& a, const int3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
/** @} */

/** subtract
 * @{
 */
FORCEINLINE DEVICE int3 operator-(const int3& a, const int3& b) {
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

FORCEINLINE DEVICE void operator-=(int3& a, const int3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
/** @} */

/** multiply
 * @{
 */
FORCEINLINE DEVICE int3 operator*(const int3& a, const int3& b) {
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
FORCEINLINE DEVICE int3 operator*(const int3& a, const int s) {
    return make_int3(a.x * s, a.y * s, a.z * s);
}
FORCEINLINE DEVICE int3 operator*(const int s, const int3& a) {
    return make_int3(a.x * s, a.y * s, a.z * s);
}
FORCEINLINE DEVICE void operator*=(int3& a, const int s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
}
/** @} */

/** divide
 * @{
 */
FORCEINLINE DEVICE int3 operator/(const int3& a, const int3& b) {
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
FORCEINLINE DEVICE int3 operator/(const int3& a, const int s) {
    return make_int3(a.x / s, a.y / s, a.z / s);
}
FORCEINLINE DEVICE int3 operator/(const int s, const int3& a) {
    return make_int3(s / a.x, s / a.y, s / a.z);
}
FORCEINLINE DEVICE void operator/=(int3& a, const int s) {
    a.x /= s;
    a.y /= s;
    a.z /= s;
}
/** @} */

/** clamp
 * @{
 */
FORCEINLINE DEVICE int3 clamp(const int3& v, const int a, const int b) {
    return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

FORCEINLINE DEVICE int3 clamp(const int3& v, const int3& a, const int3& b) {
    return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
                     clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
 * @{
 */
FORCEINLINE DEVICE bool operator==(const int3& a, const int3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

FORCEINLINE DEVICE bool operator!=(const int3& a, const int3& b) {
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE int getByIndex(const int3& v, int i) {
    return ((int*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE void setByIndex(int3& v, int i, int x) {
    ((int*)(&v))[i] = x;
}

/* int4 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
FORCEINLINE DEVICE i32x4 make_i32x4(const int a, const int b, const int c,
                                    const int d) {
    return ::make_int4(a, b, c, d);
}

FORCEINLINE DEVICE i32x4 make_i32x4(const int s) {
    return make_int4(s, s, s, s);
}

FORCEINLINE DEVICE i32x4 make_i32x4(const f32x4& a) {
    return make_int4((int)a.x, (int)a.y, (int)a.z, (int)a.w);
}
/** @} */

/** negate */
FORCEINLINE DEVICE int4 operator-(const int4& a) {
    return make_int4(-a.x, -a.y, -a.z, -a.w);
}

/** min */
FORCEINLINE DEVICE int4 min(const int4& a, const int4& b) {
    return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z),
                     min(a.w, b.w));
}

/** max */
FORCEINLINE DEVICE int4 max(const int4& a, const int4& b) {
    return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z),
                     max(a.w, b.w));
}

/** add
 * @{
 */
FORCEINLINE DEVICE int4 operator+(const int4& a, const int4& b) {
    return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
FORCEINLINE DEVICE void operator+=(int4& a, const int4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
/** @} */

/** subtract
 * @{
 */
FORCEINLINE DEVICE int4 operator-(const int4& a, const int4& b) {
    return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

FORCEINLINE DEVICE void operator-=(int4& a, const int4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
/** @} */

/** multiply
 * @{
 */
FORCEINLINE DEVICE int4 operator*(const int4& a, const int4& b) {
    return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
FORCEINLINE DEVICE int4 operator*(const int4& a, const int s) {
    return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
FORCEINLINE DEVICE int4 operator*(const int s, const int4& a) {
    return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
FORCEINLINE DEVICE void operator*=(int4& a, const int s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    a.w *= s;
}
/** @} */

/** divide
 * @{
 */
FORCEINLINE DEVICE int4 operator/(const int4& a, const int4& b) {
    return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
FORCEINLINE DEVICE int4 operator/(const int4& a, const int s) {
    return make_int4(a.x / s, a.y / s, a.z / s, a.w / s);
}
FORCEINLINE DEVICE int4 operator/(const int s, const int4& a) {
    return make_int4(s / a.x, s / a.y, s / a.z, s / a.w);
}
FORCEINLINE DEVICE void operator/=(int4& a, const int s) {
    a.x /= s;
    a.y /= s;
    a.z /= s;
    a.w /= s;
}
/** @} */

/** clamp
 * @{
 */
FORCEINLINE DEVICE int4 clamp(const int4& v, const int a, const int b) {
    return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b),
                     clamp(v.w, a, b));
}

FORCEINLINE DEVICE int4 clamp(const int4& v, const int4& a, const int4& b) {
    return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
                     clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
 * @{
 */
FORCEINLINE DEVICE bool operator==(const int4& a, const int4& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

FORCEINLINE DEVICE bool operator!=(const int4& a, const int4& b) {
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE int getByIndex(const int4& v, int i) {
    return ((int*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE void setByIndex(int4& v, int i, int x) {
    ((int*)(&v))[i] = x;
}

/* uint functions */
/******************************************************************************/

/** clamp */
FORCEINLINE DEVICE u32 clamp(const u32 f, const u32 a, const u32 b) {
    return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE u32 getByIndex(const uint1& v, u32 i) {
    return ((u32*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE void setByIndex(uint1& v, int i, u32 x) {
    ((u32*)(&v))[i] = x;
}

/* uint2 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
FORCEINLINE DEVICE u32x2 make_u32x2(const u32 a, const u32 b) {
    return ::make_uint2(a, b);
}

FORCEINLINE DEVICE u32x2 make_u32x2(const u32 s) { return make_uint2(s, s); }

FORCEINLINE DEVICE u32x2 make_u32x2(const f32x2& a) {
    return make_uint2((u32)a.x, (u32)a.y);
}
/** @} */

/** min */
FORCEINLINE DEVICE uint2 min(const uint2& a, const uint2& b) {
    return make_uint2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
FORCEINLINE DEVICE uint2 max(const uint2& a, const uint2& b) {
    return make_uint2(max(a.x, b.x), max(a.y, b.y));
}

/** add
 * @{
 */
FORCEINLINE DEVICE uint2 operator+(const uint2& a, const uint2& b) {
    return make_uint2(a.x + b.x, a.y + b.y);
}
FORCEINLINE DEVICE void operator+=(uint2& a, const uint2& b) {
    a.x += b.x;
    a.y += b.y;
}
/** @} */

/** subtract
 * @{
 */
FORCEINLINE DEVICE uint2 operator-(const uint2& a, const uint2& b) {
    return make_uint2(a.x - b.x, a.y - b.y);
}
FORCEINLINE DEVICE uint2 operator-(const uint2& a, const u32 b) {
    return make_uint2(a.x - b, a.y - b);
}
FORCEINLINE DEVICE void operator-=(uint2& a, const uint2& b) {
    a.x -= b.x;
    a.y -= b.y;
}
/** @} */

/** multiply
 * @{
 */
FORCEINLINE DEVICE uint2 operator*(const uint2& a, const uint2& b) {
    return make_uint2(a.x * b.x, a.y * b.y);
}
FORCEINLINE DEVICE uint2 operator*(const uint2& a, const u32 s) {
    return make_uint2(a.x * s, a.y * s);
}
FORCEINLINE DEVICE uint2 operator*(const u32 s, const uint2& a) {
    return make_uint2(a.x * s, a.y * s);
}
FORCEINLINE DEVICE void operator*=(uint2& a, const u32 s) {
    a.x *= s;
    a.y *= s;
}
/** @} */

/** clamp
 * @{
 */
FORCEINLINE DEVICE uint2 clamp(const uint2& v, const u32 a, const u32 b) {
    return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}

FORCEINLINE DEVICE uint2 clamp(const uint2& v, const uint2& a, const uint2& b) {
    return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
 * @{
 */
FORCEINLINE DEVICE bool operator==(const uint2& a, const uint2& b) {
    return a.x == b.x && a.y == b.y;
}

FORCEINLINE DEVICE bool operator!=(const uint2& a, const uint2& b) {
    return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE u32 getByIndex(const uint2& v, u32 i) {
    return ((u32*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
FORCEINLINE DEVICE void setByIndex(uint2& v, int i, u32 x) {
    ((u32*)(&v))[i] = x;
}

/* uint3 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
FORCEINLINE DEVICE u32x3 make_u32x3(const u32 a, const u32 b, const u32 c) {
    return make_uint3(a, b, c);
}

FORCEINLINE DEVICE u32x3 make_u32x3(const u32 s) { return make_uint3(s, s, s); }
FORCEINLINE DEVICE u32x3 make_u32x3(const f32x3& a) {
    return make_uint3((u32)a.x, (u32)a.y, (u32)a.z);
}
/** @} */

/** min */
FORCEINLINE DEVICE uint3 min(const uint3& a, const uint3& b) {
    return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
FORCEINLINE DEVICE uint3 max(const uint3& a, const uint3& b) {
    return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
 * @{
 */
FORCEINLINE DEVICE uint3 operator+(const uint3& a, const uint3& b) {
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
FORCEINLINE DEVICE void operator+=(uint3& a, const uint3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
/** @} */

/** subtract
 * @{
 */
FORCEINLINE DEVICE uint3 operator-(const uint3& a, const uint3& b) {
    return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

FORCEINLINE DEVICE void operator-=(uint3& a, const uint3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
/** @} */

/** multiply
 * @{
 */
FORCEINLINE DEVICE uint3 operator*(const uint3& a, const uint3& b) {
    return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
FORCEINLINE DEVICE uint3 operator*(const uint3& a, const u32 s) {
    return make_uint3(a.x * s, a.y * s, a.z * s);
}
FORCEINLINE DEVICE uint3 operator*(const u32 s, const uint3& a) {
    return make_uint3(a.x * s, a.y * s, a.z * s);
}
FORCEINLINE DEVICE void operator*=(uint3& a, const u32 s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
}
/** @} */

/** divide
 * @{
 */
FORCEINLINE DEVICE uint3 operator/(const uint3& a, const uint3& b) {
    return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
FORCEINLINE DEVICE uint3 operator/(const uint3& a, const u32 s) {
    return make_uint3(a.x / s, a.y / s, a.z / s);
}
FORCEINLINE DEVICE uint3 operator/(const u32 s, const uint3& a) {
    return make_uint3(s / a.x, s / a.y, s / a.z);
}
FORCEINLINE DEVICE void operator/=(uint3& a, const u32 s) {
    a.x /= s;
    a.y /= s;
    a.z /= s;
}
/** @} */

/** clamp
 * @{
 */
FORCEINLINE DEVICE uint3 clamp(const uint3& v, const u32 a, const u32 b) {
    return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

FORCEINLINE DEVICE uint3 clamp(const uint3& v, const uint3& a, const uint3& b) {
    return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
                      clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
 * @{
 */
FORCEINLINE DEVICE bool operator==(const uint3& a, const uint3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

FORCEINLINE DEVICE bool operator!=(const uint3& a, const uint3& b) {
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
 */
FORCEINLINE DEVICE u32 getByIndex(const uint3& v, u32 i) {
    return ((u32*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
 */
FORCEINLINE DEVICE void setByIndex(uint3& v, int i, u32 x) {
    ((u32*)(&v))[i] = x;
}

/* uint4 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
FORCEINLINE DEVICE u32x4 make_u32x4(const u32 a, const u32 b, const u32 c,
                                    const u32 d) {
    return ::make_uint4(a, b, c, d);
}

FORCEINLINE DEVICE u32x4 make_u32x4(const u32 s) {
    return make_uint4(s, s, s, s);
}

FORCEINLINE DEVICE u32x4 make_u32x4(const f32x4& a) {
    return make_uint4((u32)a.x, (u32)a.y, (u32)a.z, (u32)a.w);
}
/** @} */

/** min
 * @{
 */
FORCEINLINE DEVICE uint4 min(const uint4& a, const uint4& b) {
    return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z),
                      min(a.w, b.w));
}
/** @} */

/** max
 * @{
 */
FORCEINLINE DEVICE uint4 max(const uint4& a, const uint4& b) {
    return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z),
                      max(a.w, b.w));
}
/** @} */

/** add
 * @{
 */
FORCEINLINE DEVICE uint4 operator+(const uint4& a, const uint4& b) {
    return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
FORCEINLINE DEVICE void operator+=(uint4& a, const uint4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
/** @} */

/** subtract
 * @{
 */
FORCEINLINE DEVICE uint4 operator-(const uint4& a, const uint4& b) {
    return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

FORCEINLINE DEVICE void operator-=(uint4& a, const uint4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
/** @} */

/** multiply
 * @{
 */
FORCEINLINE DEVICE uint4 operator*(const uint4& a, const uint4& b) {
    return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
FORCEINLINE DEVICE uint4 operator*(const uint4& a, const u32 s) {
    return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
FORCEINLINE DEVICE uint4 operator*(const u32 s, const uint4& a) {
    return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
FORCEINLINE DEVICE void operator*=(uint4& a, const u32 s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    a.w *= s;
}
/** @} */

/** divide
 * @{
 */
FORCEINLINE DEVICE uint4 operator/(const uint4& a, const uint4& b) {
    return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
FORCEINLINE DEVICE uint4 operator/(const uint4& a, const u32 s) {
    return make_uint4(a.x / s, a.y / s, a.z / s, a.w / s);
}
FORCEINLINE DEVICE uint4 operator/(const u32 s, const uint4& a) {
    return make_uint4(s / a.x, s / a.y, s / a.z, s / a.w);
}
FORCEINLINE DEVICE void operator/=(uint4& a, const u32 s) {
    a.x /= s;
    a.y /= s;
    a.z /= s;
    a.w /= s;
}
/** @} */

/** clamp
 * @{
 */
FORCEINLINE DEVICE uint4 clamp(const uint4& v, const u32 a, const u32 b) {
    return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b),
                      clamp(v.w, a, b));
}

FORCEINLINE DEVICE uint4 clamp(const uint4& v, const uint4& a, const uint4& b) {
    return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y),
                      clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

/** @} */

/** equality
 * @{
 */
FORCEINLINE DEVICE bool operator==(const uint4& a, const uint4& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

FORCEINLINE DEVICE bool operator!=(const uint4& a, const uint4& b) {
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
 */
FORCEINLINE DEVICE u32 getByIndex(const uint4& v, u32 i) {
    return ((u32*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
 */
FORCEINLINE DEVICE void setByIndex(uint4& v, int i, u32 x) {
    ((u32*)(&v))[i] = x;
}

/******************************************************************************/

/** Narrowing functions
 * @{
 */
FORCEINLINE DEVICE i32x2 make_i32x2(const int3& v0) {
    return make_int2(v0.x, v0.y);
}
FORCEINLINE DEVICE i32x2 make_i32x2(const int4& v0) {
    return make_int2(v0.x, v0.y);
}
FORCEINLINE DEVICE i32x3 make_i32x3(const int4& v0) {
    return make_int3(v0.x, v0.y, v0.z);
}
FORCEINLINE DEVICE u32x2 make_u32x2(const uint3& v0) {
    return make_uint2(v0.x, v0.y);
}
FORCEINLINE DEVICE u32x2 make_u32x2(const uint4& v0) {
    return make_uint2(v0.x, v0.y);
}
FORCEINLINE DEVICE u32x3 make_u32x3(const uint4& v0) {
    return make_uint3(v0.x, v0.y, v0.z);
}

FORCEINLINE DEVICE f32x2 make_f32x2(const f32x3& v0) {
    return make_f32x2(v0.x, v0.y);
}
FORCEINLINE DEVICE f32x2 make_f32x2(const f32x4& v0) {
    return make_f32x2(v0.x, v0.y);
}
FORCEINLINE DEVICE f32x3 make_f32x3(const f32x4& v0) {
    return make_f32x3(v0.x, v0.y, v0.z);
}
/** @} */

/** Assemble functions from smaller vectors
 * @{
 */
FORCEINLINE DEVICE i32x3 make_i32x3(const int v0, const int2& v1) {
    return make_int3(v0, v1.x, v1.y);
}
FORCEINLINE DEVICE i32x3 make_i32x3(const int2& v0, const int v1) {
    return make_int3(v0.x, v0.y, v1);
}
FORCEINLINE DEVICE i32x4 make_i32x4(const int v0, const int v1,
                                    const int2& v2) {
    return make_int4(v0, v1, v2.x, v2.y);
}
FORCEINLINE DEVICE i32x4 make_i32x4(const int v0, const int2& v1,
                                    const int v2) {
    return make_int4(v0, v1.x, v1.y, v2);
}

FORCEINLINE DEVICE i32x4 make_i32x4(const int2& v0, const int v1,
                                    const int v2) {
    return make_int4(v0.x, v0.y, v1, v2);
}

FORCEINLINE DEVICE i32x4 make_i32x4(const int v0, const int3& v1) {
    return make_int4(v0, v1.x, v1.y, v1.z);
}

FORCEINLINE DEVICE i32x4 make_i32x4(const int3& v0, const int v1) {
    return make_int4(v0.x, v0.y, v0.z, v1);
}

FORCEINLINE DEVICE i32x4 make_i32x4(const int2& v0, const int2& v1) {
    return make_int4(v0.x, v0.y, v1.x, v1.y);
}

FORCEINLINE DEVICE u32x3 make_u32x3(const u32 v0, const uint2& v1) {
    return make_uint3(v0, v1.x, v1.y);
}

FORCEINLINE DEVICE u32x3 make_u32x3(const uint2& v0, const u32 v1) {
    return make_uint3(v0.x, v0.y, v1);
}

FORCEINLINE DEVICE u32x4 make_u32x4(const u32 v0, const u32 v1,
                                    const uint2& v2) {
    return make_uint4(v0, v1, v2.x, v2.y);
}

FORCEINLINE DEVICE u32x4 make_u32x4(const u32 v0, const uint2& v1,
                                    const u32 v2) {
    return make_uint4(v0, v1.x, v1.y, v2);
}

FORCEINLINE DEVICE u32x4 make_u32x4(const uint2& v0, const u32 v1,
                                    const u32 v2) {
    return make_uint4(v0.x, v0.y, v1, v2);
}

FORCEINLINE DEVICE u32x4 make_u32x4(const u32 v0, const uint3& v1) {
    return make_uint4(v0, v1.x, v1.y, v1.z);
}

FORCEINLINE DEVICE u32x4 make_u32x4(const uint3& v0, const u32 v1) {
    return make_uint4(v0.x, v0.y, v0.z, v1);
}

FORCEINLINE DEVICE u32x4 make_u32x4(const uint2& v0, const uint2& v1) {
    return make_uint4(v0.x, v0.y, v1.x, v1.y);
}

FORCEINLINE DEVICE f32x3 make_f32x3(const f32x2& v0, const float v1) {
    return make_f32x3(v0.x, v0.y, v1);
}
FORCEINLINE DEVICE f32x3 make_f32x3(const float v0, const f32x2& v1) {
    return make_f32x3(v0, v1.x, v1.y);
}

FORCEINLINE DEVICE f32x4 make_f32x4(const float v0, const float v1,
                                    const f32x2& v2) {
    return make_f32x4(v0, v1, v2.x, v2.y);
}

FORCEINLINE DEVICE f32x4 make_f32x4(const float v0, const f32x2& v1,
                                    const float v2) {
    return make_f32x4(v0, v1.x, v1.y, v2);
}

FORCEINLINE DEVICE f32x4 make_f32x4(const f32x2& v0, const float v1,
                                    const float v2) {
    return make_f32x4(v0.x, v0.y, v1, v2);
}

FORCEINLINE DEVICE f32x4 make_f32x4(const float v0, const f32x3& v1) {
    return make_f32x4(v0, v1.x, v1.y, v1.z);
}

FORCEINLINE DEVICE f32x4 make_f32x4(const f32x3& v0, const float v1) {
    return make_f32x4(v0.x, v0.y, v0.z, v1);
}

FORCEINLINE DEVICE f32x4 make_f32x4(const f32x2& v0, const f32x2& v1) {
    return make_f32x4(v0.x, v0.y, v1.x, v1.y);
}
/** @} */

/* Common helper functions */
/******************************************************************************/

/** Return a smooth value in [0,1], where the transition from 0
 *   to 1 takes place for values of x in [edge0,edge1].
 */
FORCEINLINE DEVICE float smoothstep(const float edge0, const float edge1,
                                    const float x) {
    /** assert( edge1 > edge0 ); */
    const float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/** Simple mapping from [0,1] to a temperature-like RGB color.
 */
FORCEINLINE DEVICE f32x3 temperature(const float t) {
    const float b = t < 0.25f ? smoothstep(-0.25f, 0.25f, t)
                              : 1.0f - smoothstep(0.25f, 0.5f, t);
    const float g =
        t < 0.5f ? smoothstep(0.0f, 0.5f, t)
                 : (t < 0.75f ? 1.0f : 1.0f - smoothstep(0.75f, 1.0f, t));
    const float r = smoothstep(0.5f, 0.75f, t);
    return make_f32x3(r, g, b);
}

/**
 *  Calculates refraction direction
 *  r   : refraction vector
 *  i   : incident vector
 *  n   : surface normal
 *  ior : index of refraction ( n2 / n1 )
 *  returns false in case of total internal reflection, in that case r is
 *          initialized to (0,0,0).
 */
FORCEINLINE DEVICE bool refract(f32x3& r, const f32x3& i, const f32x3& n,
                                const float ior) {
    f32x3 nn = n;
    float negNdotV = dot(i, nn);
    float eta;

    if (negNdotV > 0.0f) {
        eta = ior;
        nn = -n;
        negNdotV = -negNdotV;
    } else {
        eta = 1.f / ior;
    }

    const float k = 1.f - eta * eta * (1.f - negNdotV * negNdotV);

    if (k < 0.0f) {
        // Initialize this value, so that r always leaves this function
        // initialized.
        r = make_f32x3(0.f);
        return false;
    } else {
        r = normalize(eta * i - (eta * negNdotV + sqrtf(k)) * nn);
        return true;
    }
}

/** Schlick approximation of Fresnel reflectance
 */
FORCEINLINE DEVICE float fresnel_schlick(const float cos_theta,
                                         const float exponent = 5.0f,
                                         const float minimum = 0.0f,
                                         const float maximum = 1.0f) {
    /**
      Clamp the result of the arithmetic due to floating point precision:
      the result should lie strictly within [minimum, maximum]
      return clamp(minimum + (maximum - minimum) * powf(1.0f - cos_theta,
      exponent), minimum, maximum);
    */

    /** The max doesn't seem like it should be necessary, but without it you get
        annoying broken pixels at the center of reflective spheres where
       cos_theta ~ 1.
    */
    return clamp(minimum + (maximum - minimum) *
                               powf(fmaxf(0.0f, 1.0f - cos_theta), exponent),
                 minimum, maximum);
}

FORCEINLINE DEVICE f32x3 fresnel_schlick(const float cos_theta,
                                         const float exponent,
                                         const f32x3& minimum,
                                         const f32x3& maximum) {
    return make_f32x3(
        fresnel_schlick(cos_theta, exponent, minimum.x, maximum.x),
        fresnel_schlick(cos_theta, exponent, minimum.y, maximum.y),
        fresnel_schlick(cos_theta, exponent, minimum.z, maximum.z));
}

/** Calculate the NTSC luminance value of an rgb triple
 */
FORCEINLINE DEVICE float luminance(const f32x3& rgb) {
    const f32x3 ntsc_luminance = {0.30f, 0.59f, 0.11f};
    return dot(rgb, ntsc_luminance);
}

/** Calculate the CIE luminance value of an rgb triple
 */
FORCEINLINE DEVICE float luminanceCIE(const f32x3& rgb) {
    const f32x3 cie_luminance = {0.2126f, 0.7152f, 0.0722f};
    return dot(rgb, cie_luminance);
}

} // namespace osc
