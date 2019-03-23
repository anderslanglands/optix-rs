// This example shows how to use the crate to generate a simple, progressive,
// Cornell box.
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
        )
        .expect("failed to create glfw window");

    window.set_key_polling(true);
    window.make_current();

    // retina displays will return a higher res for the framebuffer
    // which we need to use for the viewport
    let (fb_width, fb_height) = window.get_framebuffer_size();

    let _gl = gl::load_with(|s| {
        glfw.get_proc_address_raw(s) as *const std::os::raw::c_void
    });

    let fsq = FullscreenQuad::new(width, height)?;

    let image_data = vec![v4f32(0.0, 0.0, 0.0, 0.0); (width * height) as usize];

    unsafe {
        gl::Viewport(0, 0, fb_width, fb_height);
    };

    let (ctx, entry_point) = create_context(width, height).unwrap();

    // We'll create a second, render thread and pass messages to it in order to
    // control rendering, sharing an image buffer for the rendered frames and
    // using an atomic counter to signal when the context has finished each
    // render progression.
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
                        .buffer_create_2d(
                            width as usize,
                            height as usize,
                            rt::Format::FLOAT4,
                            rt::BufferType::OUTPUT,
                            rt::BufferFlag::NONE,
                        )
                        .expect("Could not create result buffer");

                    ctx.set_variable(
                        "result_buffer",
                        rt::ObjectHandle::Buffer2d(result_buffer),
                    )
                    .expect("Setting buffer2d variable failed");

                    let mut progression = 0u32;

                    'inner: loop {
                        let now = std::time::Instant::now();
                        match rx_render.try_recv() {
                            Some(MsgMaster::StartRender(_, _)) => {
                                println!("ERROR: request to start an in-progress render");
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

                        ctx.set_variable("progression", progression).unwrap();
                        progression += 1;

                        match ctx.launch_2d(
                            entry_point,
                            width as usize,
                            height as usize,
                        ) {
                            Ok(()) => (),
                            Err(rt::Error::Optix(e)) => {
                                println!("[Optix ERROR]: {}", e.1);
                            }
                            Err(_) => {
                                println!("ERROR!!!!!!!!");
                            }
                        }

                        // update the shared buffer
                        {
                            let buffer_map = ctx
                                .buffer_map_2d::<V4f32>(result_buffer)
                                .unwrap();
                            let mut output =
                                mtx_image_data_render.lock().unwrap();

                            // output.clone_from_slice(buffer_map.as_slice());
                            for i in 0..output.len() {
                                output[i] += buffer_map.as_slice()[i];
                            }
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

        // If the render thread has progressed since we last updated our display,
        // synchronize and update the display
        let current_progression =
            main_progression_counter.load(Ordering::SeqCst);
        if current_progression > last_progression {
            last_progression = current_progression;
            {
                let buffer = mtx_image_data.lock().unwrap();
                fsq.update_texture(&buffer);
            }
            fsq.set_progression(current_progression as i32);
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

fn create_context(
    width: u32,
    height: u32,
) -> Result<(rt::Context, rt::EntryPointHandle), rt::Error> {
    // The context owns everything
    let mut ctx = rt::Context::new();

    // The search path is used for finding ptx files. In this case the path to
    // the generated ptx is known by the build script, which outputs a config
    // file for us so we know where to look...
    ctx.set_search_path(rt::SearchPath::from_config_file(
        "ptx_path", "ptx_path",
    ));

    let prg_cam_screen =
        ctx.program_create_from_ptx_file("pathtracer.ptx", "generate_ray")?;
    let prg_miss =
        ctx.program_create_from_ptx_file("pathtracer.ptx", "miss")?;

    // generate camera matrices
    let swn = v2f32(-1.0, -1.0);
    let swx = v2f32(1.0, 1.0);
    let mtx_screen_to_raster = m4f32_scaling(width as f32, height as f32, 1.0)
        * m4f32_scaling(1.0 / (swx.x - swn.x), 1.0 / (swx.y - swn.y), 1.0)
        * m4f32_translation(-swn.x, -swn.y, 0.0);
    let mtx_raster_to_screen = mtx_screen_to_raster.try_inverse().unwrap();
    let mtx_camera_to_screen = M4f32::new_perspective(
        (width as f32) / (height as f32),
        34.0f32.to_radians(),
        0.1,
        1.0e7,
    );
    let mtx_screen_to_camera = mtx_camera_to_screen.try_inverse().unwrap();
    let mtx_camera_to_world = m4f32_translation(275.0, 275.0, 900.0);
    let mtx_raster_to_camera = mtx_screen_to_camera * mtx_raster_to_screen;
    ctx.program_set_variable(
        prg_cam_screen,
        "raster_to_camera",
        rt::MatrixFormat::ColumnMajor(mtx_raster_to_camera),
    )?;

    ctx.program_set_variable(
        prg_cam_screen,
        "camera_to_world",
        rt::MatrixFormat::ColumnMajor(mtx_camera_to_world),
    )?;

    let prg_mesh_intersect =
        ctx.program_create_from_ptx_file("pathtracer.ptx", "mesh_intersect")?;
    let prg_mesh_bound =
        ctx.program_create_from_ptx_file("pathtracer.ptx", "bound")?;
    let prg_material_constant_closest =
        ctx.program_create_from_ptx_file("pathtracer.ptx", "mtl_ch_diffuse")?;
    let prg_material_constant_any =
        ctx.program_create_from_ptx_file("pathtracer.ptx", "mtl_ah_shadow")?;

    let prg_material_emission =
        ctx.program_create_from_ptx_file("pathtracer.ptx", "mtl_ch_emission")?;

    let raytype_camera = ctx.set_ray_type(0, "camera")?;
    let raytype_shadow = ctx.set_ray_type(1, "shadow")?;
    ctx.set_miss_program(raytype_camera, prg_miss)?;

    let mut map_mtl_camera_programs = HashMap::new();
    map_mtl_camera_programs.insert(
        raytype_camera,
        rt::MaterialProgram {
            closest: Some(prg_material_constant_closest),
            any: None,
        },
    );
    map_mtl_camera_programs.insert(
        raytype_shadow,
        rt::MaterialProgram {
            closest: None,
            any: Some(prg_material_constant_any),
        },
    );

    let mtl_constant = ctx.material_create(map_mtl_camera_programs)?;

    let mut map_mtl_emission = HashMap::new();
    map_mtl_emission.insert(
        raytype_camera,
        rt::MaterialProgram {
            closest: Some(prg_material_emission),
            any: None,
        },
    );
    map_mtl_emission.insert(
        raytype_shadow,
        rt::MaterialProgram {
            closest: None,
            any: Some(prg_material_constant_any),
        },
    );
    let mtl_emission = ctx.material_create(map_mtl_emission)?;

    let materials = vec![0];
    let buf_material = ctx.buffer_create_from_slice_1d(
        &materials,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;

    let geo_floor = create_quad(
        &mut ctx,
        [
            v3f32(0.0, 0.0, 0.0),
            v3f32(555.0, 0.0, 0.0),
            v3f32(555.0, 0.0, -555.0),
            v3f32(0.0, 0.0, -555.0),
        ],
        prg_mesh_bound,
        prg_mesh_intersect,
    )?;
    let geo_ceiling = create_quad(
        &mut ctx,
        [
            v3f32(0.0, 555.0, -555.0),
            v3f32(555.0, 555.0, -555.0),
            v3f32(555.0, 555.0, 0.0),
            v3f32(0.0, 555.0, 0.0),
        ],
        prg_mesh_bound,
        prg_mesh_intersect,
    )?;
    let geo_wall_back = create_quad(
        &mut ctx,
        [
            v3f32(0.0, 0.0, -555.0),
            v3f32(555.0, 0.0, -555.0),
            v3f32(555.0, 555.0, -555.0),
            v3f32(0.0, 555.0, -555.0),
        ],
        prg_mesh_bound,
        prg_mesh_intersect,
    )?;
    let geo_wall_left = create_quad(
        &mut ctx,
        [
            v3f32(0.0, 0.0, 0.0),
            v3f32(0.0, 0.0, -555.0),
            v3f32(0.0, 555.0, -555.0),
            v3f32(0.0, 555.0, 0.0),
        ],
        prg_mesh_bound,
        prg_mesh_intersect,
    )?;
    let geo_wall_right = create_quad(
        &mut ctx,
        [
            v3f32(555.0, 0.0, -555.0),
            v3f32(555.0, 0.0, 0.0),
            v3f32(555.0, 555.0, 0.0),
            v3f32(555.0, 555.0, -555.0),
        ],
        prg_mesh_bound,
        prg_mesh_intersect,
    )?;

    let light_center = v3f32(277.5, 554.0, -277.5);
    let light_dim = v3f32(100.0, 0.0, 100.0);
    let light_min = light_center - light_dim / 2.0f32;
    let light_max = light_min + light_dim;
    let geo_light = create_quad(
        &mut ctx,
        [
            v3f32(light_min.x, light_min.y, light_min.z),
            v3f32(light_max.x, light_min.y, light_min.z),
            v3f32(light_max.x, light_min.y, light_max.z),
            v3f32(light_min.x, light_min.y, light_max.z),
        ],
        prg_mesh_bound,
        prg_mesh_intersect,
    )?;

    let geo_tall_box = create_box(
        &mut ctx,
        v3f32(0.0, 0.0, 0.0),
        v3f32(165.0, 330.0, 165.0),
        prg_mesh_bound,
        prg_mesh_intersect,
    )?;
    let geo_short_box = create_box(
        &mut ctx,
        v3f32(0.0, 0.0, 0.0),
        v3f32(165.0, 165.0, 165.0),
        prg_mesh_bound,
        prg_mesh_intersect,
    )?;

    ctx.geometry_set_variable(
        geo_floor,
        "material_buffer",
        rt::ObjectHandle::Buffer1d(buf_material),
    )?;
    ctx.geometry_set_variable(
        geo_ceiling,
        "material_buffer",
        rt::ObjectHandle::Buffer1d(buf_material),
    )?;
    ctx.geometry_set_variable(
        geo_wall_back,
        "material_buffer",
        rt::ObjectHandle::Buffer1d(buf_material),
    )?;
    ctx.geometry_set_variable(
        geo_wall_left,
        "material_buffer",
        rt::ObjectHandle::Buffer1d(buf_material),
    )?;
    ctx.geometry_set_variable(
        geo_wall_right,
        "material_buffer",
        rt::ObjectHandle::Buffer1d(buf_material),
    )?;
    ctx.geometry_set_variable(
        geo_tall_box,
        "material_buffer",
        rt::ObjectHandle::Buffer1d(buf_material),
    )?;
    ctx.geometry_set_variable(
        geo_short_box,
        "material_buffer",
        rt::ObjectHandle::Buffer1d(buf_material),
    )?;

    ctx.geometry_set_variable(
        geo_light,
        "material_buffer",
        rt::ObjectHandle::Buffer1d(buf_material),
    )?;

    let geo_inst_floor = ctx.geometry_instance_create(
        rt::GeometryType::Geometry(geo_floor),
        vec![mtl_constant],
    )?;
    let geo_inst_ceiling = ctx.geometry_instance_create(
        rt::GeometryType::Geometry(geo_ceiling),
        vec![mtl_constant],
    )?;
    let geo_inst_wall_back = ctx.geometry_instance_create(
        rt::GeometryType::Geometry(geo_wall_back),
        vec![mtl_constant],
    )?;
    let geo_inst_wall_left = ctx.geometry_instance_create(
        rt::GeometryType::Geometry(geo_wall_left),
        vec![mtl_constant],
    )?;
    let geo_inst_wall_right = ctx.geometry_instance_create(
        rt::GeometryType::Geometry(geo_wall_right),
        vec![mtl_constant],
    )?;
    let geo_inst_tall_box = ctx.geometry_instance_create(
        rt::GeometryType::Geometry(geo_tall_box),
        vec![mtl_constant],
    )?;
    let geo_inst_short_box = ctx.geometry_instance_create(
        rt::GeometryType::Geometry(geo_short_box),
        vec![mtl_constant],
    )?;

    let geo_inst_light = ctx.geometry_instance_create(
        rt::GeometryType::Geometry(geo_light),
        vec![mtl_emission],
    )?;

    let col_white = v3f32(0.76, 0.75, 0.5);
    let col_red = v3f32(0.63, 0.06, 0.04);
    let col_green = v3f32(0.15, 0.48, 0.09);

    ctx.geometry_instance_set_variable(
        geo_inst_floor,
        "in_diffuse_albedo",
        col_white,
    )?;
    ctx.geometry_instance_set_variable(
        geo_inst_ceiling,
        "in_diffuse_albedo",
        col_white,
    )?;
    ctx.geometry_instance_set_variable(
        geo_inst_wall_back,
        "in_diffuse_albedo",
        col_white,
    )?;
    ctx.geometry_instance_set_variable(
        geo_inst_wall_left,
        "in_diffuse_albedo",
        col_red,
    )?;
    ctx.geometry_instance_set_variable(
        geo_inst_wall_right,
        "in_diffuse_albedo",
        col_green,
    )?;
    ctx.geometry_instance_set_variable(
        geo_inst_tall_box,
        "in_diffuse_albedo",
        col_white,
    )?;
    ctx.geometry_instance_set_variable(
        geo_inst_short_box,
        "in_diffuse_albedo",
        col_white,
    )?;

    let acc_main_box = ctx.acceleration_create(rt::Builder::Trbvh)?;

    let geo_group = ctx.geometry_group_create(
        acc_main_box,
        vec![
            geo_inst_floor,
            geo_inst_ceiling,
            geo_inst_wall_back,
            geo_inst_wall_left,
            geo_inst_wall_right,
            geo_inst_light,
        ],
    )?;

    let acc_tb = ctx.acceleration_create(rt::Builder::NoAccel)?;
    let geo_group_tall_box =
        ctx.geometry_group_create(acc_tb, vec![geo_inst_tall_box])?;

    let acc_sb = ctx.acceleration_create(rt::Builder::NoAccel)?;
    let geo_group_short_box =
        ctx.geometry_group_create(acc_sb, vec![geo_inst_short_box])?;

    let mtx_tall_box = m4f32_translation(70.0, 0.0, -385.0)
        * m4f32_rotation(v3f32(0.0, 1.0, 0.0), 0.3925);
    let mtx_short_box = m4f32_translation(320.0, 0.0, -255.0)
        * m4f32_rotation(v3f32(0.0, 1.0, 0.0), -0.314);

    let xform_tall_box = ctx.transform_create(
        rt::MatrixFormat::ColumnMajor(mtx_tall_box),
        rt::TransformChild::GeometryGroup(geo_group_tall_box),
    )?;
    let xform_short_box = ctx.transform_create(
        rt::MatrixFormat::ColumnMajor(mtx_short_box),
        rt::TransformChild::GeometryGroup(geo_group_short_box),
    )?;

    let mtx = M4f32::identity();

    let xform = ctx.transform_create(
        rt::MatrixFormat::ColumnMajor(mtx),
        rt::TransformChild::GeometryGroup(geo_group),
    )?;

    let acc_grp = ctx.acceleration_create(rt::Builder::NoAccel)?;
    let grp_all = ctx.group_create(
        acc_grp,
        vec![
            rt::GroupChild::Transform(xform),
            rt::GroupChild::Transform(xform_tall_box),
            rt::GroupChild::Transform(xform_short_box),
        ],
    )?;

    ctx.set_variable("scene_root", rt::ObjectHandle::Group(grp_all))?;

    let entry_point = ctx.add_entry_point(prg_cam_screen, None)?;

    ctx.set_print_enabled(false)?;

    Ok((ctx, entry_point))
}

pub fn create_box(
    ctx: &mut rt::Context,
    min: V3f32,
    max: V3f32,
    prg_mesh_bound: rt::ProgramHandle,
    prg_mesh_intersect: rt::ProgramHandle,
) -> Result<rt::GeometryHandle, rt::Error> {
    let buf_vertex = ctx.buffer_create_from_slice_1d(
        &[
            // BLF - 0
            v3f32(min.x, min.y, min.z),
            // BRF - 1
            v3f32(max.x, min.y, min.z),
            // TRF - 2
            v3f32(max.x, max.y, min.z),
            // TLF - 3
            v3f32(min.x, max.y, min.z),
            // BLB - 4
            v3f32(min.x, min.y, max.z),
            // BRB - 5
            v3f32(max.x, min.y, max.z),
            // TRB - 6
            v3f32(max.x, max.y, max.z),
            // TLB - 7
            v3f32(min.x, max.y, max.z),
        ],
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;

    let indices = [
        // FRONT
        v3i32(2, 1, 0),
        v3i32(3, 2, 0),
        // BACK
        v3i32(4, 5, 6),
        v3i32(4, 6, 7),
        // LEFT
        v3i32(0, 4, 7),
        v3i32(0, 7, 3),
        // RIGHT
        v3i32(5, 1, 2),
        v3i32(5, 2, 6),
        // TOP
        v3i32(6, 2, 3),
        v3i32(7, 6, 3),
        // BOTTOM
        v3i32(1, 5, 4),
        v3i32(0, 1, 4),
    ];

    let buf_indices = ctx.buffer_create_from_slice_1d(
        &indices,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;

    let buf_normal = ctx.buffer_create_1d(
        0,
        rt::Format::FLOAT3,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;

    let buf_texcoord = ctx.buffer_create_1d(
        0,
        rt::Format::FLOAT2,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;

    let geo_box = ctx.geometry_create(prg_mesh_bound, prg_mesh_intersect)?;
    ctx.geometry_set_primitive_count(geo_box, indices.len() as u32)?;
    ctx.geometry_set_variable(
        geo_box,
        "vertex_buffer",
        rt::ObjectHandle::Buffer1d(buf_vertex),
    )?;
    ctx.geometry_set_variable(
        geo_box,
        "index_buffer",
        rt::ObjectHandle::Buffer1d(buf_indices),
    )?;
    ctx.geometry_set_variable(
        geo_box,
        "normal_buffer",
        rt::ObjectHandle::Buffer1d(buf_normal),
    )?;
    ctx.geometry_set_variable(
        geo_box,
        "texcoord_buffer",
        rt::ObjectHandle::Buffer1d(buf_texcoord),
    )?;

    Ok(geo_box)
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
    let indices = [v3i32(0, 1, 2), v3i32(0, 2, 3)];
    let buf_indices = ctx.buffer_create_from_slice_1d(
        &indices,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;
    let buf_normal = ctx.buffer_create_1d(
        0,
        rt::Format::FLOAT3,
        rt::BufferType::INPUT,
        rt::BufferFlag::NONE,
    )?;
    let buf_texcoord = ctx.buffer_create_1d(
        0,
        rt::Format::FLOAT2,
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
