use crate::{
    sys, DeviceCopy, DeviceStorage, TypedBuffer, BufferSlice
};
use super::{GeometryFlags};
use cu::DeviceAllocRef;
use smallvec::SmallVec;

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum VertexFormat {
    None = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_NONE as u32,
    Float3 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_FLOAT3 as u32,
    Float2 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_FLOAT2 as u32,
    Half3 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_HALF3 as u32,
    Half2 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_HALF2 as u32,
    SNorm16 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_SNORM16_3 as u32,
    SNorm32 = sys::OptixVertexFormat_OPTIX_VERTEX_FORMAT_SNORM16_2 as u32,
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum IndicesFormat {
    None = sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_NONE as u32,
    Short3 =
        sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 as u32,
    Int3 = sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_UNSIGNED_INT3 as u32,
}

#[repr(u32)]
#[derive(Copy, Clone, PartialEq)]
pub enum TransformFormat {
    None = sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_NONE,
    MatrixFloat12 =
        sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12,
}

pub trait Vertex: DeviceCopy {
    const FORMAT: VertexFormat;
    const STRIDE: u32 = 0;
}

pub trait IndexTriple: DeviceCopy {
    const FORMAT: IndicesFormat;
    const STRIDE: u32 = 0;
}

pub trait BuildInputTriangleArray {
    fn to_sys(&self) -> sys::OptixBuildInputTriangleArray;
}

impl BuildInputTriangleArray for () {
    fn to_sys(&self) -> sys::OptixBuildInputTriangleArray {
        unreachable!()
    }
}

pub struct TriangleArray<'v, 'vs, 'g, V: Vertex, A: DeviceAllocRef> {
    vertex_buffers: &'vs [BufferSlice<'v, V, A>],
    d_vertex_buffers: SmallVec<[cu::sys::CUdeviceptr; 8]>,
    geometry_flags: &'g [GeometryFlags],
}

impl<'v, 'vs, 'g, V: Vertex, A: DeviceAllocRef> BuildInputTriangleArray
    for TriangleArray<'v, 'vs, 'g, V, A>
{
    fn to_sys(&self) -> sys::OptixBuildInputTriangleArray {
        sys::OptixBuildInputTriangleArray {
            vertexBuffers: self.d_vertex_buffers.as_ptr(),
            numVertices: self.vertex_buffers[0].len() as u32,
            vertexFormat: V::FORMAT as u32,
            vertexStrideInBytes: V::STRIDE,
            indexBuffer: 0,
            numIndexTriplets: 0,
            indexFormat: sys::OptixIndicesFormat_OPTIX_INDICES_FORMAT_NONE,
            indexStrideInBytes: 0,
            preTransform: 0,
            flags: self.geometry_flags.as_ptr() as *const _,
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0,
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
            transformFormat:
                sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_NONE,
        }
    }
}

impl<'v, 'vs, 'g, V: Vertex, A: DeviceAllocRef> TriangleArray<'v, 'vs, 'g, V, A> {
    pub fn new(
        vertex_buffers: &'vs [BufferSlice<'v, V, A>],
        geometry_flags: &'g [GeometryFlags],
    ) -> Self {
        let d_vertex_buffers: SmallVec<[cu::sys::CUdeviceptr; 8]> = 
            vertex_buffers
            .iter()
            .map(|b| b.device_ptr().ptr())
            .collect();

        TriangleArray {
            vertex_buffers,
            d_vertex_buffers,
            geometry_flags,
        }
    }

    pub fn index_buffer<'i, I: IndexTriple, A2: DeviceAllocRef>(
        self,
        index_buffer: BufferSlice<'i, I, A2>,
    ) -> IndexedTriangleArray<'v, 'vs, 'g, 'i, V, I, A, A2> {
        IndexedTriangleArray {
            vertex_buffers: self.vertex_buffers,
            d_vertex_buffers: self.d_vertex_buffers,
            index_buffer,
            geometry_flags: self.geometry_flags,
        }
    }
}

#[doc(hidden)]
pub struct IndexedTriangleArray<
    'v,
    'vs,
    'g,
    'i,
    V: Vertex,
    I: IndexTriple,
    A1: DeviceAllocRef,
    A2: DeviceAllocRef,
> {
    // vertex_buffers: &'v [TypedBuffer<V, A1>],
    vertex_buffers: &'vs [BufferSlice<'v, V, A1>],
    d_vertex_buffers: SmallVec<[cu::sys::CUdeviceptr; 8]>,
    // index_buffer: &'i TypedBuffer<I, A2>,
    index_buffer: BufferSlice<'i, I, A2>,
    geometry_flags: &'g [GeometryFlags],
}

impl<
        'v,
        'vs,
        'g,
        'i,
        V: Vertex,
        I: IndexTriple,
        A: DeviceAllocRef,
        A2: DeviceAllocRef,
    > BuildInputTriangleArray
    for IndexedTriangleArray<'v, 'vs, 'i, 'g, V, I, A, A2>
{
    fn to_sys(&self) -> sys::OptixBuildInputTriangleArray {
        sys::OptixBuildInputTriangleArray {
            vertexBuffers: self.d_vertex_buffers.as_ptr(),
            numVertices: self.vertex_buffers[0].len() as u32,
            vertexFormat: V::FORMAT as u32,
            vertexStrideInBytes: V::STRIDE,
            indexBuffer: self.index_buffer.device_ptr().ptr(),
            numIndexTriplets: self.index_buffer.len() as u32,
            indexFormat: I::FORMAT as u32,
            indexStrideInBytes: I::STRIDE,
            preTransform: 0,
            flags: self.geometry_flags.as_ptr() as *const _,
            numSbtRecords: 1,
            sbtIndexOffsetBuffer: 0,
            sbtIndexOffsetSizeInBytes: 0,
            sbtIndexOffsetStrideInBytes: 0,
            primitiveIndexOffset: 0,
            transformFormat:
                sys::OptixTransformFormat_OPTIX_TRANSFORM_FORMAT_NONE,
        }
    }
}

