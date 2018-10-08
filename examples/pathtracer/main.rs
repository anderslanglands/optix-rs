use crossbeam_channel as channel;
use glfw::{Action, Context, Key};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

pub mod gl_util;
use crate::gl_util::*;

use optix as rt;
use optix::math::*;

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
            image_data.push(v4f(
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

    let (ctx, entry_point) = create_context().unwrap();

    let (tx_render, rx_render) = channel::unbounded();
    let main_progression_counter = Arc::new(AtomicUsize::new(0));
    let render_progression_counter = Arc::clone(&main_progression_counter);
    let mut last_progression = 0;
    let mtx_image_data = Arc::new(Mutex::new(image_data));
    let mtx_image_data_render = Arc::clone(&mtx_image_data);
    let render_thread = thread::spawn(move || {
        'outer: loop {
            let mut time_min = std::time::Duration::from_secs(99999999);
            let mut time_max = std::time::Duration::from_secs(0);
            let mut time_total = std::time::Duration::from_secs(0);
            let mut time_samples = 0;

            // block until we receive a command message
            match rx_render.recv() {
                Some(MsgMaster::StartRender(mut ctx, entry_point)) => {
                    let result_buffer = ctx
                        .buffer_create_2d::<V4f32>(
                            width as usize,
                            height as usize,
                            rt::BufferType::OUTPUT,
                            rt::BufferFlag::NONE,
                        ).expect("Could not create result buffer");

                    ctx.set_variable(
                        "result_buffer",
                        rt::ObjectHandle::Buffer2d(result_buffer),
                    ).expect("Setting buffer2d variable failed");

                    'inner: loop {
                        let now = std::time::Instant::now();
                        match rx_render.try_recv() {
                            Some(MsgMaster::StartRender(_, _)) => {
                                println!(
                                    "ERROR: request to start an in-progress render"
                                );
                                break 'outer;
                            }
                            Some(MsgMaster::StopRender) => {
                                time_total /= time_samples;
                                println!(
                                    "Min : {}s {}ms",
                                    time_min.as_secs(),
                                    time_min.subsec_millis()
                                );
                                println!(
                                    "Max : {}s {}ms",
                                    time_max.as_secs(),
                                    time_max.subsec_nanos()
                                );
                                println!(
                                    "Mean: {}s {}ms",
                                    time_total.as_secs(),
                                    time_total.subsec_nanos()
                                );
                                break 'outer;
                            }
                            None => (),
                        }

                        match ctx.launch_2d(
                            entry_point,
                            width as usize,
                            height as usize,
                        ) {
                            Ok(()) => (),
                            Err(_) => println!("ERROR!!!!!!111!!!11!"),
                        }

                        // update the shared buffer
                        {
                            let buffer_map = ctx
                                .buffer_map_2d::<V4f32>(result_buffer)
                                .unwrap();
                            let mut output =
                                mtx_image_data_render.lock().unwrap();
                            output.clone_from_slice(buffer_map.as_slice());
                        }
                        // let the ui thread know there's new image data to
                        // display
                        render_progression_counter
                            .fetch_add(1, Ordering::SeqCst);

                        let duration = now.elapsed();
                        if duration < time_min {
                            time_min = duration;
                        }
                        if duration > time_max {
                            time_max = duration;
                        }
                        time_total += duration;
                        time_samples += 1;
                    }
                }
                Some(MsgMaster::StopRender) => break 'outer,
                _ => (),
            }
        }
    });

    tx_render.send(MsgMaster::StartRender(ctx, entry_point));

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
    StartRender(rt::Context, rt::EntryPointHandle),
    StopRender,
}

fn create_context() -> Result<(rt::Context, rt::EntryPointHandle), rt::Error> {
    let mut ctx = rt::Context::new();
    ctx.set_search_path(rt::SearchPath::from_config_file(
        "ptx_path", "ptx_path",
    ));

    let prg_cam_screen =
        ctx.program_create_from_ptx_file("cam_screen.ptx", "generate_ray")?;
    let prg_miss =
        ctx.program_create_from_ptx_file("cam_screen.ptx", "miss")?;
    let prg_mesh_intersect = ctx.program_create_from_ptx_file(
        "triangle_mesh.ptx",
        "mesh_intersect_refine",
    )?;
    let prg_mesh_bound =
        ctx.program_create_from_ptx_file("triangle_mesh.ptx", "bound")?;
    let prg_material_constant_closest =
        ctx.program_create_from_ptx_file("mtl_constant.ptx", "closest_hit")?;
    let prg_material_constant_any =
        ctx.program_create_from_ptx_file("mtl_constant.ptx", "any_hit")?;

    ctx.program_set_variable(
        prg_material_constant_closest,
        "in_diffuse_albedo",
        v3f(0.2, 0.2, 0.8),
    )?;

    let geo_triangle = create_quad(
        &mut ctx,
        [
            v3f(-0.3, -0.3, -10.0),
            v3f(0.3, -0.3, -10.0),
            v3f(0.3, 0.3, -10.0),
            v3f(-0.3, 0.3, -10.0),
        ],
        prg_mesh_bound,
        prg_mesh_intersect,
    )?;
    let materials = vec![0];
    let buf_material = ctx.buffer_create_from_slice_1d(
        &materials,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;
    ctx.geometry_set_variable(
        geo_triangle,
        "material_buffer",
        rt::ObjectHandle::Buffer1d(buf_material),
    )?;

    let raytype_camera = ctx.set_ray_type(0, "camera")?;

    ctx.set_miss_program(raytype_camera, prg_miss)?;

    let mut map_mtl_camera_any = HashMap::new();
    let mut map_mtl_camera_closest = HashMap::new();
    map_mtl_camera_any.insert(raytype_camera, prg_material_constant_any);
    map_mtl_camera_closest
        .insert(raytype_camera, prg_material_constant_closest);

    let mtl_constant =
        ctx.material_create(map_mtl_camera_any, map_mtl_camera_closest)?;

    let geo_inst = ctx.geometry_instance_create(
        rt::GeometryType::Geometry(geo_triangle),
        vec![mtl_constant],
    )?;

    let acc = ctx.acceleration_create(rt::Builder::NoAccel)?;

    let geo_group = ctx.geometry_group_create(acc, vec![geo_inst])?;

    let mtx_t = m4f_translation(0.5, 0.5, 0.0);
    let mtx_r = m4f_rotation(v3f(0.0, 0.0, 1.0), std::f32::consts::PI / 4.0);
    let mtx = mtx_t * mtx_r;

    let xform = ctx.transform_create(
        rt::MatrixDirection::Forward(mtx),
        rt::MatrixFormat::ColumnMajor,
        rt::TransformChild::GeometryGroup(geo_group),
    )?;

    ctx.program_set_variable(
        prg_cam_screen,
        "scene_root",
        // rt::ObjectHandle::GeometryGroup(geo_group),
        rt::ObjectHandle::Transform(xform),
    )?;

    let entry_point = ctx.add_entry_point(prg_cam_screen, None)?;

    Ok((ctx, entry_point))
}

pub fn create_quad(
    ctx: &mut rt::Context,
    vertices: [V3f32; 4],
    prg_mesh_bound: rt::ProgramHandle,
    prg_mesh_intersect: rt::ProgramHandle,
) -> Result<rt::GeometryHandle, rt::Error> {
    let buf_vertex = ctx.buffer_create_from_slice_1d(
        &vertices,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;
    let indices = [v3i(0, 1, 2), v3i(0, 2, 3)];
    let buf_indices = ctx.buffer_create_from_slice_1d(
        &indices,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;
    let buf_normal = ctx.buffer_create_1d::<V3f32>(
        0,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;
    let buf_texcoord = ctx.buffer_create_1d::<V2f32>(
        0,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;
    let geo_triangle =
        ctx.geometry_create(prg_mesh_bound, prg_mesh_intersect)?;
    ctx.geometry_set_primitive_count(geo_triangle, indices.len() as u32)?;
    ctx.geometry_set_variable(
        geo_triangle,
        "vertex_buffer",
        rt::ObjectHandle::Buffer1d(buf_vertex),
    )?;
    ctx.geometry_set_variable(
        geo_triangle,
        "index_buffer",
        rt::ObjectHandle::Buffer1d(buf_indices),
    )?;
    ctx.geometry_set_variable(
        geo_triangle,
        "normal_buffer",
        rt::ObjectHandle::Buffer1d(buf_normal),
    )?;
    ctx.geometry_set_variable(
        geo_triangle,
        "texcoord_buffer",
        rt::ObjectHandle::Buffer1d(buf_texcoord),
    )?;

    Ok(geo_triangle)
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true)
        }
        _ => {}
    }
}
