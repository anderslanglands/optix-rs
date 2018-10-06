use gl;
use gl::types::{GLchar, GLenum, GLint, GLsizeiptr, GLuint, GLvoid};
use std::ffi::{CStr, CString};

pub struct Shader {
    id: GLuint,
}

impl Shader {
    pub fn from_source(
        source: &CStr,
        shader_type: GLenum,
    ) -> Result<Shader, String> {
        let id = unsafe { gl::CreateShader(shader_type) };

        unsafe {
            gl::ShaderSource(id, 1, &source.as_ptr(), std::ptr::null());
            gl::CompileShader(id);
        }

        let mut success: GLint = 1;
        unsafe {
            gl::GetShaderiv(id, gl::COMPILE_STATUS, &mut success);
        }

        if success == 0 {
            let mut len: GLint = 0;
            unsafe {
                gl::GetShaderiv(id, gl::INFO_LOG_LENGTH, &mut len);
            }
            let error = create_whitespace_cstring(len as usize);
            unsafe {
                gl::GetShaderInfoLog(
                    id,
                    len,
                    std::ptr::null_mut(),
                    error.as_ptr() as *mut GLchar,
                );
            }
            Err(error.to_string_lossy().into_owned())
        } else {
            Ok(Shader { id })
        }
    }

    pub fn vertex_from_source(source: &CStr) -> Result<Shader, String> {
        Shader::from_source(source, gl::VERTEX_SHADER)
    }

    pub fn fragment_from_source(source: &CStr) -> Result<Shader, String> {
        Shader::from_source(source, gl::FRAGMENT_SHADER)
    }

    pub fn id(&self) -> GLuint {
        self.id
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { gl::DeleteShader(self.id) };
    }
}

pub struct Program {
    id: GLuint,
}

impl Program {
    pub fn from_shaders(shaders: &[Shader]) -> Result<Program, String> {
        let id = unsafe { gl::CreateProgram() };

        for shader in shaders {
            unsafe { gl::AttachShader(id, shader.id()) };
        }

        unsafe { gl::LinkProgram(id) };

        let mut success: GLint = 1;
        unsafe {
            gl::GetProgramiv(id, gl::LINK_STATUS, &mut success);
        }

        if success == 0 {
            let mut len: GLint = 0;
            unsafe {
                gl::GetProgramiv(id, gl::INFO_LOG_LENGTH, &mut len);
            }
            let error = create_whitespace_cstring(len as usize);
            unsafe {
                gl::GetProgramInfoLog(
                    id,
                    len,
                    std::ptr::null_mut(),
                    error.as_ptr() as *mut GLchar,
                );
            }
            return Err(error.to_string_lossy().into_owned());
        }

        for shader in shaders {
            unsafe { gl::DetachShader(id, shader.id()) }
        }

        Ok(Program { id })
    }

    pub fn id(&self) -> GLuint {
        self.id
    }

    pub fn use_program(&self) {
        unsafe {
            gl::UseProgram(self.id);
        }
    }
}

fn create_whitespace_cstring(len: usize) -> CString {
    let mut buffer: Vec<u8> = Vec::with_capacity(len as usize + 1);
    buffer.extend([b' '].iter().cycle().take(len as usize));
    unsafe { CString::from_vec_unchecked(buffer) }
}

#[repr(u32)]
#[derive(Copy, Clone)]
pub enum BufferType {
    ArrayBuffer = gl::ARRAY_BUFFER,
}

#[repr(u32)]
#[derive(Copy, Clone)]
pub enum BufferUsage {
    StaticDraw = gl::STATIC_DRAW,
    StreamDraw = gl::STREAM_DRAW,
}

pub struct Buffer<T> {
    id: GLuint,
    buffer_type: BufferType,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Buffer<T> {
    pub fn new(buffer_type: BufferType) -> Buffer<T> {
        let mut id: GLuint = 0;
        unsafe {
            gl::GenBuffers(1, &mut id);
        }
        Buffer {
            id,
            buffer_type,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn id(&self) -> GLuint {
        self.id
    }

    pub fn buffer_data(&self, data: &[T], usage: BufferUsage) {
        unsafe {
            gl::BindBuffer(self.buffer_type as GLuint, self.id);
            gl::BufferData(
                self.buffer_type as GLuint,
                (data.len() * std::mem::size_of::<T>()) as GLsizeiptr,
                data.as_ptr() as *const GLvoid,
                usage as GLenum,
            );
            gl::BindBuffer(self.buffer_type as GLuint, 0);
        }
    }

    pub fn bind(&self) {
        unsafe {
            gl::BindBuffer(self.buffer_type as GLuint, self.id);
        }
    }

    pub fn unbind(&self) {
        unsafe {
            gl::BindBuffer(self.buffer_type as GLuint, 0);
        }
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.id as *const GLuint);
        }
    }
}

pub struct VertexArray {
    id: GLuint,
}

impl VertexArray {
    pub fn new() -> VertexArray {
        let mut id: GLuint = 0;
        unsafe {
            gl::GenVertexArrays(1, &mut id);
        }

        VertexArray { id }
    } 
    
    pub fn id(&self) -> GLuint {
        self.id
    }

    pub fn bind(&self) {
        unsafe {gl::BindVertexArray(self.id);}
    }

    pub fn unbind(&self) {
        unsafe {gl::BindVertexArray(0);}
    }
}

impl Drop for VertexArray {
    fn drop(&mut self) {
        unsafe { gl::DeleteVertexArrays(1, &self.id as *const GLuint);}
    }
}

#[allow(non_camel_case_types)]
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct f32x2 {
    x: f32,
    y: f32,
}

impl f32x2 {
    pub fn new(x: f32, y: f32) -> f32x2 {
        f32x2{x, y}
    }

    pub fn num_components() -> usize {
        2
    }
}

#[allow(non_camel_case_types)]
#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
pub struct f32x3 {
    x: f32,
    y: f32,
    z: f32
}

impl f32x3 {
    pub fn new(x: f32, y: f32, z: f32) -> f32x3 {
        f32x3{x, y, z}
    }

    pub fn num_components() -> usize {
        3
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct Vertex {
    p: f32x3,
    st: f32x2,
}
impl Vertex {
    pub fn new(p: f32x3, st: f32x2) -> Vertex {
        Vertex { p, st }
    }

    unsafe fn vertex_attrib_pointer(
        num_components: usize,
        stride: usize,
        location: usize,
        offset: usize,
    ) {
        gl::EnableVertexAttribArray(location as gl::types::GLuint); // location(0)
        gl::VertexAttribPointer(
            location as gl::types::GLuint, // index of the vertex attribute
            num_components as gl::types::GLint, /* number of components per
                                            * vertex attrib */
            gl::FLOAT,
            gl::FALSE, // normalized (int-to-float conversion),
            stride as gl::types::GLint, /* byte stride between
                                            * successive elements */
            offset as *const gl::types::GLvoid, /* offset of the first
                                                    * element */
        );
    }

    pub fn vertex_attrib_pointers() {
        let stride = std::mem::size_of::<Self>();

        let location = 0;
        let offset = 0;

        // and configure the vertex array
        unsafe {
            Vertex::vertex_attrib_pointer(
                f32x3::num_components(),
                stride,
                location,
                offset,
            );
        }

        let location = location + 1;
        let offset = offset + std::mem::size_of::<f32x3>();

        // and configure the st array
        unsafe {
            Vertex::vertex_attrib_pointer(
                f32x2::num_components(),
                stride,
                location,
                offset,
            );
        }
    }
}