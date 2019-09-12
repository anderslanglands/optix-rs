// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <optix_device.h>
#include <vec.h>

namespace osc {

enum { RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

/*
struct TriangleMeshSBTData {
    V3f32 color;
    V3f32* vertex;
    V3f32* normal;
    V2f32* texcoord;
    V3i32* index;
    bool hasTexture;
    cudaTextureObject_t texture;
};

struct LaunchParams {
    struct {
        float4* colorBuffer;
        V2i32 size;
        i32 accumID{0};
    } frame;

    struct {
        V3f32 position;
        V3f32 direction;
        V3f32 horizontal;
        V3f32 vertical;
    } camera;

    struct {
        V3f32 origin, du, dv, power;
    } light;

    OptixTraversableHandle traversable;
};
*/

struct TriangleMeshSBTData {
    V3f32 color;
    V3f32* vertex;
    V3f32* normal;
    V2f32* texcoord;
    V3i32* index;
    bool has_texture;
    cudaTextureObject_t texture;
};
struct LaunchParams {
    struct {
        V4f32* color_buffer;
        V2i32 size;
        int accum_id;
    } frame;
    struct {
        V3f32 position;
        V3f32 direction;
        V3f32 horizontal;
        V3f32 vertical;
    } camera;
    struct {
        V3f32 origin;
        V3f32 du;
        V3f32 dv;
        V3f32 power;
    } light;
    OptixTraversableHandle traversable;
};

} // namespace osc
