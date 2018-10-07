use crossbeam_channel as channel;
use glfw::{Action, Context, Key};
use rand::distributions::{Distribution, Uniform};
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

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

    // retina displays will return a higher res for the framebuffer
    // which we need to use for the viewport
    let (fb_width, fb_height) = window.get_framebuffer_size();

    let gl = gl::load_with(|s| {
        glfw.get_proc_address_raw(s) as *const std::os::raw::c_void
    });

    let fsq = FullscreenQuad::new(width, height)?;

    let mut image_data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            image_data.push(f32x4::new(
                (x as f32) / width as f32,
                (y as f32) / height as f32,
                0.0,
                1.0,
            ));
        }
    }

    unsafe {
        gl::Viewport(0, 0, fb_width, fb_height);
    };

    let (tx_render, rx_render) = channel::unbounded();
    let main_progression_counter = Arc::new(AtomicUsize::new(0));
    let render_progression_counter = Arc::clone(&main_progression_counter);
    let mut last_progression = 0;
    let mtx_image_data = Arc::new(Mutex::new(image_data));
    let mtx_image_data_render = Arc::clone(&mtx_image_data);
    let render_thread = thread::spawn(move || {
        let range = Uniform::new(0.0f32, 1.0);
        let mut rng = rand::thread_rng();
        let mut buffer = Vec::with_capacity((width * height) as usize);
        buffer.resize((width * height) as usize, f32x4::zero());
        'outer: loop {
            match rx_render.recv() {
                Some(MsgMaster::StartRender) => 'inner: loop {
                    match rx_render.try_recv() {
                        Some(MsgMaster::StartRender) => {
                            println!(
                                "ERROR: request to start an in-progress render"
                            );
                            break 'outer;
                        }
                        Some(MsgMaster::StopRender) => break 'outer,
                        None => (),
                    }

                    // perform a "render progression" by sleeping and setting a
                    // random colour
                    thread::sleep(std::time::Duration::from_millis(1000));
                    let r: f32 = range.sample(&mut rng);
                    let g: f32 = range.sample(&mut rng);
                    let b: f32 = range.sample(&mut rng);
                    for p in buffer.iter_mut() {
                        *p = f32x4::new(r, g, b, 1.0);
                    }

                    // update the shared buffer
                    {
                        let mut output = mtx_image_data_render.lock().unwrap();
                        output.clone_from_slice(&buffer);
                    }
                    // let the ui thread know there's new image data to display
                    render_progression_counter.fetch_add(1, Ordering::SeqCst);
                },
                Some(MsgMaster::StopRender) => break 'outer,
                _ => (),
            }
        }
    });

    tx_render.send(MsgMaster::StartRender);

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        // If the render has progressed since we last updated our display, then
        // synchronize and update the diplay
        let current_progression =
            main_progression_counter.load(Ordering::SeqCst);
        if current_progression > last_progression {
            last_progression = current_progression;
            {
                let buffer = mtx_image_data.lock().unwrap();
                fsq.update_texture(&buffer);
            }
        }

        // draw the quad
        fsq.draw();

        window.swap_buffers();
    }
    // user requested close. Shutdown the render thread and exit
    tx_render.send(MsgMaster::StopRender);
    render_thread.join().unwrap();

    Ok(())
}

enum MsgMaster {
    StartRender,
    StopRender,
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true)
        }
        _ => {}
    }
}
