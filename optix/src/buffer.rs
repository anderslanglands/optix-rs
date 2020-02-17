use super::cuda::{self, Allocator};
use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

use super::DeviceShareable;

/*
/// Runtime-typed buffer
pub struct DynamicBuffer {
    buffer: cuda::Buffer,
    count: usize,
    format: BufferFormat,
}

impl DynamicBuffer {
    pub fn new<T>(data: &[T]) -> Result<DynamicBuffer>
    where
        T: BufferElement,
    {
        let buffer = cuda::Buffer::with_data(data)?;

        Ok(DynamicBuffer {
            buffer,
            count: data.len(),
            format: T::FORMAT,
        })
    }

    pub fn with_format<T>(
        data: &[T],
        format: BufferFormat,
    ) -> Result<DynamicBuffer> {
        let buffer = cuda::Buffer::with_data(data)?;

        Ok(DynamicBuffer {
            buffer,
            count: data.len(),
            format,
        })
    }

    pub fn as_ptr(&self) -> *const std::os::raw::c_void {
        self.buffer.as_ptr()
    }

    pub fn as_device_ptr(&self) -> cuda::CUdeviceptr {
        self.buffer.as_device_ptr()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn format(&self) -> BufferFormat {
        self.format
    }

    pub fn byte_size(&self) -> usize {
        self.buffer.byte_size()
    }
}

impl DeviceShareable for DynamicBuffer {
    type Target = BufferD;
    fn to_device(&self) -> Self::Target {
        BufferD {
            ptr: self.buffer.as_device_ptr(),
            len: self.len(),
        }
    }
    fn cuda_type() -> String {
        "DynamicBuffer".into()
    }
    fn cuda_decl() -> String {
        "struct DynamicBuffer { void* ptr; size_t len; };".into()
    }
}
*/

pub struct Buffer<'a, AllocT, T>
where
    AllocT: Allocator,
    T: BufferElement,
{
    buffer: cuda::Buffer<'a, AllocT>,
    count: usize,
    _t: std::marker::PhantomData<T>,
}

impl<'a, AllocT, T> Buffer<'a, AllocT, T>
where
    AllocT: Allocator,
    T: BufferElement,
{
    pub fn new(
        data: &[T],
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<Buffer<'a, AllocT, T>> {
        let buffer =
            cuda::Buffer::with_data(data, T::ALIGNMENT, tag, allocator)?;

        Ok(Buffer {
            buffer,
            count: data.len(),
            _t: std::marker::PhantomData::<T> {},
        })
    }

    pub fn uninitialized(
        count: usize,
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<Buffer<'a, AllocT, T>>
    where
        AllocT: Allocator,
    {
        let buffer = cuda::Buffer::new(
            count * std::mem::size_of::<T>(),
            T::ALIGNMENT,
            tag,
            allocator,
        )?;

        Ok(Buffer {
            buffer,
            count,
            _t: std::marker::PhantomData::<T> {},
        })
    }

    pub fn as_ptr(&self) -> *const std::os::raw::c_void {
        self.buffer.as_ptr()
    }

    pub fn as_device_ptr(&self) -> cuda::CUdeviceptr {
        self.buffer.as_device_ptr()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn format(&self) -> BufferFormat {
        T::FORMAT
    }

    pub fn byte_size(&self) -> usize {
        self.buffer.byte_size()
    }

    pub fn download(&self, dst: &mut [T]) -> Result<()> {
        Ok(self.buffer.download(dst)?)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BufferD {
    ptr: cuda::CUdeviceptr,
    len: usize,
}

impl<'a, AllocT, T> DeviceShareable for Buffer<'a, AllocT, T>
where
    AllocT: Allocator,
    T: BufferElement,
{
    type Target = BufferD;
    fn to_device(&self) -> Self::Target {
        BufferD {
            ptr: self.buffer.as_device_ptr(),
            len: self.len(),
        }
    }
    fn cuda_type() -> String {
        format!("Buffer<{}>", T::FORMAT.device_name())
    }

    fn cuda_decl() -> String {
        r#"
template <typename ElemT> 
struct Buffer { 
    ElemT* ptr; size_t len; 

    const ElemT& operator[](size_t i) const {
        return ptr[i];
    } 

    ElemT& operator[](size_t i) {
        return ptr[i];
    } 

    bool is_null() const {
        return ptr == nullptr;
    }

    bool is_empty() const {
        return len == 0;
    }
};
"#
        .into()
    }

    fn zero() -> Self::Target {
        BufferD { ptr: 0, len: 0 }
    }
}

pub struct Buffer2d<'a, AllocT, T>
where
    AllocT: Allocator,
    T: BufferElement,
{
    buffer: cuda::Buffer<'a, AllocT>,
    width: usize,
    height: usize,
    _t: std::marker::PhantomData<T>,
}

impl<'a, AllocT, T> Buffer2d<'a, AllocT, T>
where
    AllocT: Allocator,
    T: BufferElement,
{
    pub fn new(
        data: &[T],
        width: usize,
        height: usize,
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<Buffer2d<'a, AllocT, T>> {
        let buffer =
            cuda::Buffer::with_data(data, T::ALIGNMENT, tag, allocator)?;

        if width * height != data.len() {
            panic!(
                "bad buffer dimensions: [{}] {}x{}",
                data.len(),
                width,
                height
            );
        }

        Ok(Buffer2d {
            buffer,
            width,
            height,
            _t: std::marker::PhantomData::<T> {},
        })
    }

    pub fn uninitialized(
        width: usize,
        height: usize,
        tag: u64,
        allocator: &'a AllocT,
    ) -> Result<Buffer2d<'a, AllocT, T>>
    where
        AllocT: Allocator,
    {
        let buffer = cuda::Buffer::new(
            width * height * std::mem::size_of::<T>(),
            T::ALIGNMENT,
            tag,
            allocator,
        )?;

        Ok(Buffer2d {
            buffer,
            width,
            height,
            _t: std::marker::PhantomData::<T> {},
        })
    }

    pub fn as_ptr(&self) -> *const std::os::raw::c_void {
        self.buffer.as_ptr()
    }

    pub fn as_device_ptr(&self) -> cuda::CUdeviceptr {
        self.buffer.as_device_ptr()
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn format(&self) -> BufferFormat {
        T::FORMAT
    }

    pub fn byte_size(&self) -> usize {
        self.buffer.byte_size()
    }

    pub fn download(&self, dst: &mut [T]) -> Result<()> {
        Ok(self.buffer.download(dst)?)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Buffer2dD {
    ptr: cuda::CUdeviceptr,
    width: usize,
    height: usize,
}

impl<'a, AllocT, T> DeviceShareable for Buffer2d<'a, AllocT, T>
where
    AllocT: Allocator,
    T: BufferElement,
{
    type Target = Buffer2dD;
    fn to_device(&self) -> Self::Target {
        Buffer2dD {
            ptr: self.buffer.as_device_ptr(),
            width: self.width(),
            height: self.height(),
        }
    }
    fn cuda_type() -> String {
        format!("Buffer2d<{}>", T::FORMAT.device_name())
    }

    fn cuda_decl() -> String {
        r#"
template <typename ElemT> 
struct Buffer2d { 
    ElemT* ptr; size_t width; size_t height; 

    const ElemT& operator[](uint2 i) const {
        return ptr[i.y * width + i.x];
    } 

    ElemT& operator[](uint2 i) {
        return ptr[i.y * width + i.x];
    } 

    bool is_null() const {
        return ptr == nullptr;
    }

    bool is_empty() const {
        return len == 0;
    }

};
"#
        .into()
    }

    fn zero() -> Self::Target {
        Buffer2dD {
            ptr: 0,
            width: 0,
            height: 0,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BufferFormat {
    U8,
    U8x2,
    U8x3,
    U8x4,
    U16,
    U16x2,
    U16x3,
    U16x4,
    F16,
    F16x2,
    F16x3,
    F16x4,
    F32,
    F32x2,
    F32x3,
    F32x4,
    I32,
    I32x2,
    I32x3,
    I32x4,
    U32,
    U32x2,
    U32x3,
    U32x4,
}

impl BufferFormat {
    pub fn byte_size(&self) -> usize {
        match self {
            BufferFormat::U8 => 1,
            BufferFormat::U8x2 => 2,
            BufferFormat::U8x3 => 3,
            BufferFormat::U8x4 => 4,
            BufferFormat::U16 => 2,
            BufferFormat::U16x2 => 4,
            BufferFormat::U16x3 => 6,
            BufferFormat::U16x4 => 8,
            BufferFormat::F16 => 2,
            BufferFormat::F16x2 => 4,
            BufferFormat::F16x3 => 6,
            BufferFormat::F16x4 => 8,
            BufferFormat::F32 => 4,
            BufferFormat::F32x2 => 8,
            BufferFormat::F32x3 => 12,
            BufferFormat::F32x4 => 16,
            BufferFormat::I32 => 4,
            BufferFormat::I32x2 => 8,
            BufferFormat::I32x3 => 12,
            BufferFormat::I32x4 => 16,
            BufferFormat::U32 => 4,
            BufferFormat::U32x2 => 8,
            BufferFormat::U32x3 => 12,
            BufferFormat::U32x4 => 16,
        }
    }

    pub fn device_name(&self) -> &'static str {
        match self {
            BufferFormat::U8 => "u8",
            BufferFormat::U8x2 => "u8x2",
            BufferFormat::U8x3 => "u8x3",
            BufferFormat::U8x4 => "u8x4",
            BufferFormat::U16 => "u16",
            BufferFormat::U16x2 => "u16x2",
            BufferFormat::U16x3 => "u16x3",
            BufferFormat::U16x4 => "u16x4",
            BufferFormat::F16 => "f16",
            BufferFormat::F16x2 => "f16x2",
            BufferFormat::F16x3 => "f16x3",
            BufferFormat::F16x4 => "f16x4",
            BufferFormat::F32 => "f32",
            BufferFormat::F32x2 => "f32x2",
            BufferFormat::F32x3 => "f32x3",
            BufferFormat::F32x4 => "f32x4",
            BufferFormat::I32 => "i32",
            BufferFormat::I32x2 => "i32x2",
            BufferFormat::I32x3 => "i32x3",
            BufferFormat::I32x4 => "i32x4",
            BufferFormat::U32 => "u32",
            BufferFormat::U32x2 => "u32x2",
            BufferFormat::U32x3 => "u32x3",
            BufferFormat::U32x4 => "u32x4",
        }
    }
}

pub trait BufferElement {
    const FORMAT: BufferFormat;
    const COMPONENTS: usize;
    const ALIGNMENT: usize;
    type ComponentType;
}

impl BufferElement for u8 {
    const FORMAT: BufferFormat = BufferFormat::U8;
    const COMPONENTS: usize = 1;
    const ALIGNMENT: usize = 1;
    type ComponentType = u8;
}

impl BufferElement for [u8; 2] {
    const FORMAT: BufferFormat = BufferFormat::U8x2;
    const COMPONENTS: usize = 2;
    const ALIGNMENT: usize = 2;
    type ComponentType = u8;
}

impl BufferElement for [u8; 3] {
    const FORMAT: BufferFormat = BufferFormat::U8x3;
    const COMPONENTS: usize = 3;
    const ALIGNMENT: usize = 1;
    type ComponentType = u8;
}

impl BufferElement for [u8; 4] {
    const FORMAT: BufferFormat = BufferFormat::U8x4;
    const COMPONENTS: usize = 4;
    const ALIGNMENT: usize = 4;
    type ComponentType = u8;
}

impl BufferElement for u16 {
    const FORMAT: BufferFormat = BufferFormat::U16;
    const COMPONENTS: usize = 1;
    const ALIGNMENT: usize = 2;
    type ComponentType = u16;
}

impl BufferElement for [u16; 2] {
    const FORMAT: BufferFormat = BufferFormat::U16x2;
    const COMPONENTS: usize = 2;
    const ALIGNMENT: usize = 4;
    type ComponentType = u16;
}

impl BufferElement for [u16; 3] {
    const FORMAT: BufferFormat = BufferFormat::U16x3;
    const COMPONENTS: usize = 3;
    const ALIGNMENT: usize = 2;
    type ComponentType = u16;
}

impl BufferElement for [u16; 4] {
    const FORMAT: BufferFormat = BufferFormat::U16x4;
    const COMPONENTS: usize = 4;
    const ALIGNMENT: usize = 8;
    type ComponentType = u16;
}

impl BufferElement for i32 {
    const FORMAT: BufferFormat = BufferFormat::I32;
    const COMPONENTS: usize = 1;
    const ALIGNMENT: usize = 4;
    type ComponentType = i32;
}

impl BufferElement for [i32; 2] {
    const FORMAT: BufferFormat = BufferFormat::I32x2;
    const COMPONENTS: usize = 2;
    const ALIGNMENT: usize = 8;
    type ComponentType = i32;
}

impl BufferElement for [i32; 3] {
    const FORMAT: BufferFormat = BufferFormat::I32x3;
    const COMPONENTS: usize = 3;
    const ALIGNMENT: usize = 4;
    type ComponentType = i32;
}

impl BufferElement for [i32; 4] {
    const FORMAT: BufferFormat = BufferFormat::I32x4;
    const COMPONENTS: usize = 4;
    const ALIGNMENT: usize = 16;
    type ComponentType = i32;
}

impl BufferElement for u32 {
    const FORMAT: BufferFormat = BufferFormat::U32;
    const COMPONENTS: usize = 1;
    const ALIGNMENT: usize = 4;
    type ComponentType = u32;
}

impl BufferElement for [u32; 2] {
    const FORMAT: BufferFormat = BufferFormat::U32x2;
    const COMPONENTS: usize = 2;
    const ALIGNMENT: usize = 8;
    type ComponentType = u32;
}

impl BufferElement for [u32; 3] {
    const FORMAT: BufferFormat = BufferFormat::U32x3;
    const COMPONENTS: usize = 3;
    const ALIGNMENT: usize = 4;
    type ComponentType = u32;
}

impl BufferElement for [u32; 4] {
    const FORMAT: BufferFormat = BufferFormat::U32x4;
    const COMPONENTS: usize = 4;
    const ALIGNMENT: usize = 16;
    type ComponentType = u32;
}

impl BufferElement for f32 {
    const FORMAT: BufferFormat = BufferFormat::F32;
    const COMPONENTS: usize = 1;
    const ALIGNMENT: usize = 4;
    type ComponentType = f32;
}

impl BufferElement for [f32; 2] {
    const FORMAT: BufferFormat = BufferFormat::F32x2;
    const COMPONENTS: usize = 2;
    const ALIGNMENT: usize = 8;
    type ComponentType = f32;
}

impl BufferElement for [f32; 3] {
    const FORMAT: BufferFormat = BufferFormat::F32x3;
    const COMPONENTS: usize = 3;
    const ALIGNMENT: usize = 4;
    type ComponentType = f32;
}

impl BufferElement for [f32; 4] {
    const FORMAT: BufferFormat = BufferFormat::F32x4;
    const COMPONENTS: usize = 4;
    const ALIGNMENT: usize = 16;
    type ComponentType = f32;
}
