#[macro_use]
extern crate derive_more;

mod sample_renderer;
use sample_renderer::*;

use glfw::{Action, Context, Key};
pub mod gl_util;
use crate::gl_util::*;

use optix::math::*;

fn load_model(path: &std::path::Path) -> Model {
    let (models, materials) = tobj::load_obj(path).unwrap();

    let mut bounds = Box3f32::make_empty();
    let meshes = models
        .into_iter()
        .map(|model| {
            let diffuse = if let Some(material_id) = model.mesh.material_id {
                materials[material_id].diffuse.into()
            } else {
                v3f32(0.8, 0.8, 0.8)
            };

            let vertex: Vec<V3f32> = model
                .mesh
                .positions
                .chunks(3)
                .map(|c| {
                    let p = v3f32(c[0], c[1], c[2]);
                    bounds.extend_by_pnt(p);
                    p
                })
                .collect();

            let normal: Vec<V3f32> = model
                .mesh
                .normals
                .chunks(3)
                .map(|c| v3f32(c[0], c[1], c[2]))
                .collect();

            let texcoord: Vec<V2f32> = model
                .mesh
                .texcoords
                .chunks(2)
                .map(|c| v2f32(c[0], c[1]))
                .collect();

            let index: Vec<V3i32> = model
                .mesh
                .indices
                .chunks(3)
                .map(|c| v3i32(c[0] as i32, c[1] as i32, c[2] as i32))
                .collect();

            Mesh {
                vertex,
                normal,
                texcoord,
                index,
                diffuse,
            }
        })
        .collect::<Vec<_>>();

    Model { meshes, bounds }
}

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 1));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    let mut width = 960u32;
    let mut height = 540u32;

    let model = load_model(&std::path::Path::new(&format!(
        "{}/examples/data/sponza.obj",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    )));

    let camera = Camera {
        from: v3f32(-1293.07, 154.681, -0.7304),
        at: model.bounds.center() - v3f32(0.0, 400.0, 0.0),
        up: v3f32(0.0, 1.0, 0.0),
    };

    let mut sample =
        SampleRenderer::new(v2i32(width as i32, height as i32), camera, model)
            .unwrap();

    let (mut window, events) = glfw
        .create_window(
            width,
            height,
            "Example 07: First Real Model",
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
