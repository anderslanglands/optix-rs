use crate::impl_device_copy_align;
use crate::{DeviceCopy, Vertex, IndexTriple, VertexFormat, IndicesFormat};

use al_math::*;

impl_device_copy_align!(
    V2i32:8
    V3i32:4
    V4i32:16

    V2f32:8
    V3f32:4
    V4f32:16
);

impl Vertex for V3f32 {
    const FORMAT: VertexFormat = VertexFormat::Float3;
}

impl IndexTriple for V3i32 {
    const FORMAT: IndicesFormat = IndicesFormat::Int3;
}