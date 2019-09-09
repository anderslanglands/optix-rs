#[macro_use]
extern crate derive_more;

mod sample_renderer;
use sample_renderer::*;

use glfw::{Action, Context, Key};
pub mod gl_util;
use crate::gl_util::*;

use imath::*;

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 1));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    let mut width = 960u32;
    let mut height = 540u32;

    let mut meshes = Vec::new();
    let mut mesh = TriangleMesh::new(v3f32(0.8, 0.2, 0.1));
    mesh.add_cube(v3f32(0.0, -1.5, 0.0), v3f32(10.0, 1.0, 10.0));
    meshes.push(mesh);
    let mut mesh = TriangleMesh::new(v3f32(0.2, 0.8, 0.1));
    mesh.add_cube(v3f32(0.0, 0.0, 0.0), v3f32(2.0, 2.0, 2.0));
    meshes.push(mesh);

    let camera = Camera {
        from: v3f32(-10.0, 2.0, -12.0),
        at: v3f32(0.0, 0.0, 0.0),
        up: v3f32(0.0, 1.0, 0.0),
    };

    let mut sample =
        SampleRenderer::new(v2i32(width as i32, height as i32), camera, meshes)
            .unwrap();

    let (mut window, events) = glfw
        .create_window(
            width,
            height,
            "Example 06: multiple meshes",
            glfw::WindowMode::Windowed,
        )
        .expect("failed to create glfw window");

    window.set_key_polling(true);
    window.make_current();

    // retina displays will return a higher res for the framebuffer
    // which we need to use for the viewport
    let (fb_width, fb_height) = window.get_framebuffer_size();

    gl::load_with(|s| {
        glfw.get_proc_address_raw(s) as *const std::os::raw::c_void
    });

    let mut fsq = FullscreenQuad::new(width, height).unwrap();

    let mut image_data =
        vec![v4f32(0.0, 0.0, 0.0, 0.0); (width * height) as usize];

    unsafe {
        gl::Viewport(0, 0, fb_width, fb_height);
    };

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        let (w, h) = window.get_framebuffer_size();
        let w = w as u32;
        let h = h as u32;
        if w != width || h != height {
            fsq.resize(w, h);
            sample.resize(v2i32(w as i32, h as i32));
            width = w;
            height = h;
            image_data
                .resize((width * height) as usize, v4f32(0.0, 0.0, 0.0, 0.0));
        }

        sample.render();
        sample.download_pixels(&mut image_data).unwrap();
        fsq.update_texture(&image_data);
        fsq.set_progression(1);

        // draw the quad
        fsq.draw();

        window.swap_buffers();
    }
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true)
        }
        _ => {}
    }
}
