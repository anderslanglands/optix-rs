mod renderer;
use renderer::{Renderer, Camera, TriangleMesh};

pub mod vector;
use vector::*;
mod gl_util;
use gl_util::FullscreenQuad;
use glfw::{Action, Context, Key};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 1));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    let mut width = 960;
    let mut height = 540;

    let mut mesh = TriangleMesh::new(v3f32(0.2, 0.8, 0.2));
    mesh.add_cube(v3f32(0.0, -1.5, 0.0), v3f32(10.0, 0.1, 10.0));
    mesh.add_cube(v3f32(0.0, 0.0, 0.0), v3f32(2.0, 2.0, 2.0));

    let camera = Camera {
        from: v3f32(-10.0, 2.0, -12.0),
        at: v3f32(0.0, 0.0, 0.0),
        up: v3f32(0.0, 1.0, 0.0),
    };

    let mut renderer = Renderer::new(width, height, camera, mesh)?;

    let (mut window, events) = glfw
        .create_window(
            width,
            height,
            "Example 05: first sbt data",
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

    println!("creating miage data of {}", width * height);
    let mut image_data =
        vec![v4f32(0.0, 0.0, 0.0, 0.0); (width * height) as usize];

    unsafe {
        gl::Viewport(0, 0, width as i32, height as i32);
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
            renderer.resize(w, h)?;
            println!("Resizing to {}x{}", w, h);
            width = w;
            height = h;
            image_data
                .resize((width * height) as usize, v4f32(0.0, 0.0, 0.0, 0.0));
            unsafe {
                gl::Viewport(0, 0, w as i32, h as i32);
            };

        }

        renderer.render()?;
        renderer.download_pixels(&mut image_data)?;
        fsq.update_texture(&image_data);
        fsq.set_progression(1);

        // draw the quad
        fsq.draw();

        window.swap_buffers();
    }

    renderer.render()?;
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
