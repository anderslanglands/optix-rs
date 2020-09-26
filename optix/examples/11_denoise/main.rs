mod renderer;
use renderer::{Renderer, Camera, TriangleMesh, Mesh, Model, Texture, Light};

pub mod vector;
use vector::*;
mod gl_util;
use gl_util::FullscreenQuad;
use glfw::{Action, Context, Key};

use anyhow::{Result, anyhow};

fn main() -> Result<()> {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();
    glfw.window_hint(glfw::WindowHint::ContextVersion(4, 1));
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    let mut width = 960;
    let mut height = 540;

    let model = load_model(&std::path::Path::new(&format!(
        "{}/examples/data/sponza.obj",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    )));

    let camera = Camera {
        from: v3f32(-1293.07, 154.681, -0.7304),
        at: model.bounds.center() - v3f32(0.0, 400.0, 0.0),
        up: v3f32(0.0, 1.0, 0.0),
    };

    let light_size = 200.0;
    let power = 1000000.0;
    let light = Light {
        origin: v3f32(-1000.0 - light_size, 800.0, -light_size),
        du: v3f32(2.0 * light_size, 0.0, 0.0),
        dv: v3f32(0.0, 0.0, 2.0 * light_size),
        power: v3f32(power, power, power),
    };

    let mut renderer = Renderer::new(width, height, camera, model, light)
        .map_err(|error| {
            for cause in error.chain() {
                eprintln!("ERROR: {}", cause)
            }
            anyhow!("Renderer creation failed")
        })?;

    renderer::alloc_mem_report();

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

    renderer.resize(width, height)?;
    image_data
        .resize((width * height) as usize, v4f32(0.0, 0.0, 0.0, 0.0));

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

fn load_texture(path: &std::path::Path) -> Option<Texture> {
    let im = match image::open(path) {
        Ok(im) => im.to_rgba(),
        Err(e) => {
            println!("{}", e);
            return None;
        }
    };

    let dim = im.dimensions();
    println!("{}x{}", dim.0, dim.1);

    Some(Texture {
        resolution: v2i32(dim.0 as i32, dim.1 as i32),
        pixels: im.into_raw(),
    })
}

fn load_model(path: &std::path::Path) -> Model {
    let (models, materials) = tobj::load_obj(path, true).unwrap();

    let mut bounds = Box3f32::make_empty();
    let mut loaded_texture_ids = std::collections::HashMap::new();
    let mut textures = Vec::new();
    let meshes = models
        .into_iter()
        .map(|model| {
            let (diffuse, diffuse_texture_id) = if let Some(material_id) =
                model.mesh.material_id
            {
                let diffuse = materials[material_id].diffuse.into();

                // load the diffuse texture if there is one
                let diffuse_texture = &materials[material_id].diffuse_texture;
                let diffuse_texture_id = if diffuse_texture != "" {
                    // if our texture cache has an entry for this texture
                    // just return the (maybe) texture_id
                    if loaded_texture_ids.contains_key(&diffuse_texture) {
                        loaded_texture_ids[&diffuse_texture]
                    } else {
                        // If we don't have an entry, try and load the texture
                        let full_texture_path = std::path::PathBuf::from(
                            std::env::var("CARGO_MANIFEST_DIR").unwrap(),
                        )
                        .join("examples/data")
                        .join(diffuse_texture.replace("\\", "/"));

                        println!(
                            "Loading texture {}",
                            full_texture_path.display()
                        );

                        match load_texture(&full_texture_path) {
                            Some(texture) => {
                                textures.push(texture);
                                let texture_id = Some(textures.len() - 1);
                                loaded_texture_ids
                                    .insert(diffuse_texture, texture_id);
                                texture_id
                            }
                            None => {
                                loaded_texture_ids
                                    .insert(diffuse_texture, None);
                                None
                            }
                        }
                    }
                } else {
                    None
                };

                (diffuse, diffuse_texture_id)
            } else {
                (v3f32(0.8, 0.8, 0.8), None)
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
                diffuse_texture_id,
            }
        })
        .collect::<Vec<_>>();

    Model {
        meshes,
        textures,
        bounds,
    }
}