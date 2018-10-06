use glfw::{Action, Context, Key};
use std::ffi::{CStr, CString};

pub mod gl_util;
use crate::gl_util::*;

fn main() -> Result<(), String> {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 1));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    let width = 512;
    let height = 512;

    let (mut window, events) = glfw
        .create_window(
            width,
            height,
            "path tracer example",
            glfw::WindowMode::Windowed,
        ).expect("failed to create glfw window");

    window.set_key_polling(true);
    window.make_current();

    let (fb_width, fb_height) = window.get_framebuffer_size();
    println!("fb_width: {}, fb_height: {}", fb_width, fb_height);

    let gl = gl::load_with(|s| {
        glfw.get_proc_address_raw(s) as *const std::os::raw::c_void
    });

    unsafe {
        gl::Viewport(0, 0, fb_width, fb_height);
        gl::ClearColor(0.3, 0.3, 0.5, 1.0)
    };

    let vert_shader = Shader::vertex_from_source(
        CStr::from_bytes_with_nul(
            b"
        #version 330 core
        layout (location = 0) in vec3 _p;
        layout (location = 1) in vec2 _st;
        out vec2 st;
        void main() {
            gl_Position = vec4(_p, 1.0);
            st = _st;
        }
    \0",
        ).unwrap(),
    )?;

    let frag_shader = Shader::fragment_from_source(
        CStr::from_bytes_with_nul(
            b"
        #version 330 core
        in vec2 st;
        out vec4 Color;

        uniform sampler2D smp2d_0;

        void main() {
            Color = texture(smp2d_0, st);
        }
    \0",
        ).unwrap(),
    )?;

    let program = Program::from_shaders(&[vert_shader, frag_shader])?;
    program.use_program();

    let vertices: Vec<Vertex> = vec![
        Vertex::new(f32x3::new(-1.0, -1.0, 0.0), f32x2::new(0.0, 0.0)),
        Vertex::new(f32x3::new(1.0, -1.0, 0.0), f32x2::new(1.0, 0.0)),
        Vertex::new(f32x3::new(1.0, 1.0, 0.0), f32x2::new(1.0, 1.0)),
        Vertex::new(f32x3::new(-1.0, -1.0, 0.0), f32x2::new(0.0, 0.0)),
        Vertex::new(f32x3::new(1.0, 1.0, 0.0), f32x2::new(1.0, 1.0)),
        Vertex::new(f32x3::new(-1.0, 1.0, 0.0), f32x2::new(0.0, 1.0)),
    ];
    let vertex_buffer = Buffer::<Vertex>::new(BufferType::ArrayBuffer);
    vertex_buffer.buffer_data(&vertices, BufferUsage::StaticDraw);

    // Generate and bind the VAO
    let vertex_array = VertexArray::new();

    vertex_array.bind();
    // Re-bind the VBO to associate the two. We could just have left it bound
    // earlier and let the association happen when we configure the VAO but
    // this way at least makes the connection between the two seem more
    // explicit, despite the magical state machine hiding in OpenGL
    vertex_buffer.bind();

    // Set up the vertex attribute pointers for all locations
    Vertex::vertex_attrib_pointers();

    // now unbind both the vbo and vao to keep everything cleaner
    vertex_buffer.unbind();
    vertex_array.unbind();

    // generate test texture data using the image width rather than the
    // framebuffer width
    let mut tex_data = Vec::new();
    for y in 0..height {
        for x in 0..width {
            tex_data.push(f32x3::new(
                (x as f32) / width as f32,
                (y as f32) / height as f32,
                1.0,
            ));
        }
    }

    // generate the texture for the quad
    let mut texture_id: gl::types::GLuint = 0;
    unsafe {
        gl::GenTextures(1, &mut texture_id);
        gl::ActiveTexture(gl::TEXTURE0);
        gl::Enable(gl::TEXTURE_2D);
        gl::BindTexture(gl::TEXTURE_2D, texture_id);
        gl::TexParameteri(
            gl::TEXTURE_2D,
            gl::TEXTURE_WRAP_S,
            gl::CLAMP_TO_BORDER as gl::types::GLint,
        );
        gl::TexParameteri(
            gl::TEXTURE_2D,
            gl::TEXTURE_WRAP_T,
            gl::CLAMP_TO_BORDER as gl::types::GLint,
        );
        gl::TexParameteri(
            gl::TEXTURE_2D,
            gl::TEXTURE_MIN_FILTER,
            gl::NEAREST as gl::types::GLint,
        );
        gl::TexParameteri(
            gl::TEXTURE_2D,
            gl::TEXTURE_MAG_FILTER,
            gl::NEAREST as gl::types::GLint,
        );
        gl::TexImage2D(
            gl::TEXTURE_2D,
            0,
            gl::RGB32F as gl::types::GLint,
            width as gl::types::GLint,
            height as gl::types::GLint,
            0,
            gl::RGB,
            gl::FLOAT,
            tex_data.as_ptr() as *const gl::types::GLvoid,
        );
    }

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        unsafe { gl::Clear(gl::COLOR_BUFFER_BIT) };

        program.use_program();
        vertex_array.bind();
        unsafe {
            gl::DrawArrays(
                gl::TRIANGLES,
                0, // starting index in the enabled array
                6, // number of indices to draw
            )
        }

        window.swap_buffers();
    }

    Ok(())
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true)
        }
        _ => {}
    }
}
