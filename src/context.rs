pub use crate::error::*;
use std::collections::HashMap;

use slotmap::*;

pub use crate::math::*;
use crate::optix_bindings::*;
pub use crate::optix_bindings::{BufferFlag, BufferType, Format};
pub use crate::search_path::SearchPath;

pub mod program;
use self::program::*;
pub mod geometry;
use self::geometry::*;
pub mod material;
use self::material::*;
pub mod geometry_instance;
use self::geometry_instance::*;
pub mod buffer;
use self::buffer::*;
pub mod variable;
use self::variable::*;
pub mod acceleration;
use self::acceleration::*;
pub mod geometry_group;
use self::geometry_group::*;
pub mod transform;
use self::transform::*;
pub mod group;
use self::group::*;
pub mod texture_sampler;
use self::texture_sampler::*;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct RayType {
    ray_type: u32,
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct EntryPointHandle {
    index: u32,
}

#[derive(Debug, Copy, Clone)]
pub struct EntryPoint {
    ray_generation_program: ProgramHandle,
    exception_program: Option<ProgramHandle>,
}

/// An OptiX context provides an interface for controlling the setup and
/// subsequent launch of the ray tracing engine.
///
/// Contexts are created with the
/// `new` method. A context object encapsulates all OptiX resources
/// — textures, geometry, user-defined programs, etc. The destruction of a
/// context, via the rtContextDestroy function, will clean up all of these
/// resources and invalidate any existing handles to them. The functions
/// rtContextLaunch1D, rtContextLaunch2D and rtContextLaunch3D (collectively
/// known as rtContextLaunch) serve as entry points to ray engine computation.
/// The launch function takes an entry point parameter, discussed in “Entry
/// points”, as well as one, two or three grid dimension parameters. The
/// dimensions establish a logical computation grid. Upon a call to
/// rtContextLaunch, any necessary preprocessing is performed and then the ray
/// generation program associated with the provided entry point index is
/// invoked once per computational grid cell. The launch precomputation
/// includes state validation and, if necessary, acceleration structure
/// generation and kernel compilation. Output from the launch is passed back
/// via OptiX buffers, typically but not necessarily of the same dimensionality
/// as the computation grid.
pub struct Context {
    rt_ctx: RTcontext,
    ray_types: HashMap<RayType, String>,
    max_ray_type: u32,
    entry_points: Vec<EntryPoint>,
    miss_programs: HashMap<RayType, ProgramHandle>,
    search_path: SearchPath,

    context_variables: HashMap<String, Variable>,

    ga_acceleration_obj: SlotMap<AccelerationHandle, RTacceleration>,

    ga_buffer1d_obj: SlotMap<Buffer1dHandle, RTbuffer>,
    ga_buffer2d_obj: SlotMap<Buffer2dHandle, RTbuffer>,
    _ga_buffer3d_obj: SlotMap<Buffer3dHandle, RTbuffer>,

    ga_program_obj: SlotMap<ProgramHandle, RTprogram>,
    gd_program_variables:
        SecondaryMap<ProgramHandle, HashMap<String, Variable>>,

    ga_geometry_obj: SlotMap<GeometryHandle, RTgeometry>,
    gd_geometry_variables:
        SecondaryMap<GeometryHandle, HashMap<String, Variable>>,
    gd_geometry_bounding_box: SecondaryMap<GeometryHandle, ProgramHandle>,
    gd_geometry_intersection: SecondaryMap<GeometryHandle, ProgramHandle>,

    ga_geometry_instance_obj:
        SlotMap<GeometryInstanceHandle, RTgeometryinstance>,
    gd_geometry_instance_variables:
        SecondaryMap<GeometryInstanceHandle, HashMap<String, Variable>>,
    gd_geometry_instance_geometry:
        SecondaryMap<GeometryInstanceHandle, GeometryType>,
    gd_geometry_instance_materials:
        SecondaryMap<GeometryInstanceHandle, Vec<MaterialHandle>>,

    ga_geometry_group_obj: SlotMap<GeometryGroupHandle, RTgeometrygroup>,
    gd_geometry_group_acceleration:
        SecondaryMap<GeometryGroupHandle, AccelerationHandle>,
    gd_geometry_group_children:
        SecondaryMap<GeometryGroupHandle, Vec<GeometryInstanceHandle>>,

    ga_material_obj: SlotMap<MaterialHandle, RTmaterial>,
    gd_material_variables:
        SecondaryMap<MaterialHandle, HashMap<String, Variable>>,
    gd_material_programs:
        SecondaryMap<MaterialHandle, HashMap<RayType, MaterialProgram>>,

    ga_transform_obj: SlotMap<TransformHandle, RTtransform>,
    gd_transform_child: SecondaryMap<TransformHandle, TransformChild>,

    ga_group_obj: SlotMap<GroupHandle, RTgroup>,
    gd_group_acceleration: SecondaryMap<GroupHandle, AccelerationHandle>,
    gd_group_children: SecondaryMap<GroupHandle, Vec<GroupChild>>,

    ga_texture_sampler_obj: SlotMap<TextureSamplerHandle, RTtexturesampler>,
    gd_texture_sampler_buffer: SecondaryMap<TextureSamplerHandle, BufferHandle>,
}

unsafe impl Send for Context {}

impl Context {
    pub fn new() -> Context {
        let ga_acceleration_obj = SlotMap::with_key();

        let ga_buffer1d_obj = SlotMap::with_key();
        let ga_buffer2d_obj = SlotMap::with_key();
        let _ga_buffer3d_obj = SlotMap::with_key();

        let ga_program_obj = SlotMap::with_key();
        let gd_program_variables = SecondaryMap::new();

        let ga_geometry_obj = SlotMap::with_key();
        let gd_geometry_variables = SecondaryMap::new();
        let gd_geometry_bounding_box = SecondaryMap::new();
        let gd_geometry_intersection = SecondaryMap::new();

        let ga_geometry_instance_obj = SlotMap::with_key();
        let gd_geometry_instance_variables = SecondaryMap::new();
        let gd_geometry_instance_geometry = SecondaryMap::new();
        let gd_geometry_instance_materials = SecondaryMap::new();

        let ga_geometry_group_obj = SlotMap::with_key();
        let gd_geometry_group_acceleration = SecondaryMap::new();
        let gd_geometry_group_children = SecondaryMap::new();

        let ga_material_obj = SlotMap::with_key();
        let gd_material_variables = SecondaryMap::new();
        let gd_material_programs = SecondaryMap::new();

        let ga_transform_obj = SlotMap::with_key();
        let gd_transform_child = SecondaryMap::new();

        let ga_group_obj = SlotMap::with_key();
        let gd_group_acceleration = SecondaryMap::new();
        let gd_group_children = SecondaryMap::new();

        let (rt_ctx, result) = unsafe {
            let mut rt_ctx: RTcontext = std::mem::uninitialized();
            let result = rtContextCreate(&mut rt_ctx);
            (rt_ctx, result)
        };
        if result != RtResult::SUCCESS {
            panic!("Could not create RTcontext. Cannot continue");
        }

        Context {
            rt_ctx,
            ray_types: HashMap::new(),
            max_ray_type: 0,
            entry_points: Vec::new(),
            miss_programs: HashMap::new(),
            search_path: SearchPath::new(),

            context_variables: HashMap::new(),

            ga_acceleration_obj,

            ga_buffer1d_obj,
            ga_buffer2d_obj,
            _ga_buffer3d_obj,

            ga_program_obj,
            gd_program_variables,

            ga_geometry_obj,
            gd_geometry_variables,
            gd_geometry_bounding_box,
            gd_geometry_intersection,

            ga_geometry_instance_obj,
            gd_geometry_instance_variables,
            gd_geometry_instance_geometry,
            gd_geometry_instance_materials,

            ga_geometry_group_obj,
            gd_geometry_group_acceleration,
            gd_geometry_group_children,

            ga_material_obj,
            gd_material_variables,
            gd_material_programs,

            ga_transform_obj,
            gd_transform_child,

            ga_group_obj,
            gd_group_acceleration,
            gd_group_children,

            ga_texture_sampler_obj: SlotMap::with_key(),
            gd_texture_sampler_buffer: SecondaryMap::new(),
        }
    }

    /// Set the `SearchPath` this `Context` will use for finding `Programs`.
    pub fn set_search_path(&mut self, search_path: SearchPath) {
        self.search_path = search_path;
    }

    /// Register the ray type of the given `index` and `name` with the
    /// `Context`. The returned `RayType` struct can be used to assign
    /// `Materials` and miss `Programs`.
    pub fn set_ray_type(&mut self, index: u32, name: &str) -> Result<RayType> {
        if index >= self.max_ray_type {
            self.max_ray_type = index + 1;
            let result = unsafe {
                rtContextSetRayTypeCount(self.rt_ctx, self.max_ray_type)
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtContextSetRayTypeCount", result));
            }
        }

        let ray_type = RayType { ray_type: index };
        self.ray_types.insert(ray_type, name.to_owned());

        Ok(ray_type)
    }

    /// Sets the `Program` that handles what happens when the given `RayType`
    /// does not hit any `Geometry`.
    pub fn set_miss_program(
        &mut self,
        ray_type: RayType,
        prg: ProgramHandle,
    ) -> Result<()> {
        self.program_validate(prg)?;

        let rt_prg = self.ga_program_obj.get(prg).unwrap();
        let result = unsafe {
            rtContextSetMissProgram(self.rt_ctx, ray_type.ray_type, *rt_prg)
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtContextSetMissProgram", result))
        } else {
            self.miss_programs.insert(ray_type, prg);

            Ok(())
        }
    }

    /// Checks the the given `Context` and all of its associated OptiX objects
    /// for a valid state. These checks include tests for presence of necessary
    /// `Programs` (e.g. an intersection `Program` for a `Geometry` node),
    /// invalid internal state, and presence of variables required by all
    /// specified programs.
    pub fn validate(&self) -> Result<()> {
        let result = unsafe { rtContextValidate(self.rt_ctx) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtContextValidate", result))
        } else {
            Ok(())
        }
    }

    /// Creates a new `EntryPoint` into the `Context` consisting of a ray
    /// generation `Program` and an optional exception `Program`. The returned
    /// `EntryPointHandle` can be given to the `Launcher` to launch kernels.
    pub fn add_entry_point(
        &mut self,
        ray_generation_program: ProgramHandle,
        exception_program: Option<ProgramHandle>,
    ) -> Result<EntryPointHandle> {
        // Make sure the programs are good before trying to set them
        self.program_validate(ray_generation_program)?;
        if let Some(ep) = exception_program {
            self.program_validate(ep)?;
        }

        let result = unsafe {
            rtContextSetEntryPointCount(
                self.rt_ctx,
                self.entry_points.len() as u32 + 1,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtContextSetEntryPointCount", result));
        }

        let index = self.entry_points.len() as u32;

        let rt_prg_raygen = self
            .ga_program_obj
            .get(ray_generation_program)
            .expect(&format!(
                "Could not get RTprogram object from handle: {:?}",
                ray_generation_program
            ));
        let result = unsafe {
            rtContextSetRayGenerationProgram(self.rt_ctx, index, *rt_prg_raygen)
        };
        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtContextSetRayGenerationProgram", result)
            );
        }
        if let Some(ep) = exception_program {
            let rt_prg_except = self.ga_program_obj.get(ep).expect(&format!(
                "Could not get RTprogram object from handle: {:?}",
                ep
            ));
            let result = unsafe {
                rtContextSetExceptionProgram(self.rt_ctx, index, *rt_prg_except)
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtContextSetExceptionProgram", result)
                );
            }
        }

        self.entry_points.push(EntryPoint {
            ray_generation_program,
            exception_program,
        });

        Ok(EntryPointHandle { index })
    }

    /// Get a ref to this `Context's` `SearchPath`
    pub fn search_path(&self) -> &SearchPath {
        &self.search_path
    }

    /// Create a new `OptixError` by querying the `Context` for the error
    /// string for the given `result`.
    pub fn optix_error(&self, msg: &str, result: RtResult) -> Error {
        Error::Optix((
            result,
            format!("{}: {}", msg, get_error_string(self.rt_ctx, result)),
        ))
    }

    /// Set the Variable referred to by `name` to the given `data`. Any objects
    /// previously assigned to the variable will be destroyed.
    pub fn set_variable<T: VariableStorable>(
        &mut self,
        name: &str,
        data: T,
    ) -> Result<()> {
        // check if the variable exists first
        if let Some(old_variable) = self.context_variables.get(name) {
            let var = match old_variable {
                Variable::Pod(vp) => vp.var,
                Variable::Object(vo) => vo.var,
            };
            let new_variable = data.set_optix_variable(self, var)?;
            // destroy any resources the existing variable holds
            if let Some(old_variable) =
                self.context_variables.insert(name.to_owned(), new_variable)
            {
                self.destroy_variable(old_variable);
            };

            Ok(())
        } else {
            let (var, result) = unsafe {
                let mut var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtContextDeclareVariable(
                    self.rt_ctx,
                    c_name.as_ptr(),
                    &mut var,
                );
                (var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtDeclareVariable", result));
            }

            let variable = data.set_optix_variable(self, var)?;
            self.context_variables.insert(name.to_owned(), variable);

            Ok(())
        }
    }

    fn destroy_variable(&mut self, var: Variable) {
        match var {
            Variable::Pod(_) => (),
            Variable::Object(vo) => match vo.object_handle {
                ObjectHandle::BufferId(_) => (),
                ObjectHandle::Buffer1d(h) => {
                    self.buffer_destroy_1d(h);
                }
                ObjectHandle::Buffer2d(h) => {
                    self.buffer_destroy_2d(h);
                }
                ObjectHandle::Group(h) => {
                    self.group_destroy(h);
                }
                ObjectHandle::GeometryGroup(h) => {
                    self.geometry_group_destroy(h);
                }
                ObjectHandle::Program(h) => {
                    self.program_destroy(h);
                }
                ObjectHandle::Transform(h) => {
                    self.transform_destroy(h);
                }
                ObjectHandle::TextureSampler(h) => {
                    self.texture_sampler_destroy(h);
                }
            },
        }
    }

    pub fn launch_2d(
        &self,
        entry_point: EntryPointHandle,
        width: usize,
        height: usize,
    ) -> Result<()> {
        let result = unsafe {
            rtContextLaunch2D(
                self.rt_ctx,
                entry_point.index,
                width as RTsize,
                height as RTsize,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtContextLaunch2D", result));
        }
        Ok(())
    }

    pub fn set_print_enabled(&mut self, enabled: bool) -> Result<()> {
        let result = unsafe {
            let en = if enabled { 1 } else { 0 };
            rtContextSetPrintEnabled(self.rt_ctx, en)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtContextSetPrintEnabled", result));
        }
        Ok(())
    }

    pub fn set_print_launch_index(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
    ) -> Result<()> {
        let result =
            unsafe { rtContextSetPrintLaunchIndex(self.rt_ctx, x, y, z) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtContextSetPrintLaunchIndex", result));
        }
        Ok(())
    }

    pub fn set_usage_report_callback(
        &mut self,
        callback: extern "C" fn(
            i32,
            *const std::os::raw::c_char,
            *const std::os::raw::c_char,
            *mut std::os::raw::c_void,
        ),
        verbosity: i32,
    ) -> Result<()> {
        let result = unsafe {
            rtContextSetUsageReportCallback(
                self.rt_ctx,
                Some(callback),
                verbosity,
                std::ptr::null_mut(),
            )
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtContextSetUsageReportCallback", result)
            );
        }

        Ok(())
    }
}

impl Drop for Context {
    /// Destroys the `Context`, destroying all objects attached to it.
    fn drop(&mut self) {
        // destroy all stuffs here
        unsafe {
            rtContextDestroy(self.rt_ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    fn v4f32_buffer_to_u8(buf: &ScopedBufMap2d<V4f32>) -> Vec<u8> {
        let mut result: Vec<u8> = Vec::with_capacity(buf.len() * 4);
        for yi in (0..buf.height()).rev() {
            for x in 0..buf.width() {
                let elem = buf[(x, yi)];
                result.push((elem[0] * 255.) as u8);
                result.push((elem[1] * 255.) as u8);
                result.push((elem[2] * 255.) as u8);
                result.push((elem[3] * 255.) as u8);
            }
        }

        result
    }

    fn write_scoped_buf_map_v4f32(
        filename: &str,
        buf: &ScopedBufMap2d<V4f32>,
    ) -> Result<()> {
        use image::{save_buffer, ColorType};
        let buf_u8 = v4f32_buffer_to_u8(&buf);
        Ok(save_buffer(
            std::path::Path::new(filename),
            &buf_u8,
            buf.width() as u32,
            buf.height() as u32,
            ColorType::RGBA(8),
        )?)
    }

    use super::*;
    #[test]
    fn draw_solid_color() -> Result<()> {
        let mut ctx = Context::new();
        ctx.set_search_path(SearchPath::from_config_file(
            "ptx_path", "ptx_path",
        ));

        let hprg_draw_solid_color = ctx
            .program_create_from_ptx_file("draw_color.ptx", "draw_solid_color")
            .expect("Failed to load draw_solid_color from draw_color.ptx");

        let entry_point =
            ctx.add_entry_point(hprg_draw_solid_color, None).unwrap();

        let result_buffer = ctx
            .buffer_create_2d(
                256,
                128,
                Format::FLOAT4,
                BufferType::OUTPUT,
                BufferFlag::NONE,
            )
            .expect("Could not create result buffer");

        ctx.set_variable(
            "result_buffer",
            ObjectHandle::Buffer2d(result_buffer),
        )
        .expect("Setting buffer2d variable failed");

        ctx.validate().expect("Context validation failed");

        ctx.launch_2d(entry_point, 256, 128)?;

        {
            let buffer_map = ctx
                .buffer_map_2d::<V4f32>(result_buffer)
                .expect("Buffer map failed");

            assert_eq!(buffer_map[(0, 0)], v4f32(0., 0., 0., 0.));
            assert_eq!(buffer_map.width(), 256);
            assert_eq!(buffer_map.height(), 128);
            assert_eq!(
                buffer_map[(255, 127)],
                V4f32::new(255f32 / 256f32, 127f32 / 128f32, 0f32, 0f32)
            );
            write_scoped_buf_map_v4f32("solid_color.png", &buffer_map)?;
        }

        Ok(())
    }

    enum Message {
        Done(Context),
    }

    #[test]
    fn draw_solid_color_mt() -> Result<()> {
        use std::sync::mpsc;
        use std::thread;

        let mut ctx = Context::new();
        ctx.set_search_path(SearchPath::from_config_file(
            "ptx_path", "ptx_path",
        ));

        let hprg_draw_solid_color = ctx
            .program_create_from_ptx_file("draw_color.ptx", "draw_solid_color")
            .expect("Failed to load draw_solid_color from draw_color.ptx");

        let entry_point =
            ctx.add_entry_point(hprg_draw_solid_color, None).unwrap();

        let result_buffer = ctx
            .buffer_create_2d(
                256,
                128,
                Format::FLOAT4,
                BufferType::OUTPUT,
                BufferFlag::NONE,
            )
            .expect("Could not create result buffer");

        ctx.set_variable(
            "result_buffer",
            ObjectHandle::Buffer2d(result_buffer),
        )
        .expect("Setting buffer2d variable failed");

        ctx.validate().expect("Context validation failed");

        let (tx, rx) = mpsc::channel();

        println!("starting thread");
        thread::spawn(move || {
            ctx.launch_2d(entry_point, 256, 128).unwrap();
            tx.send(Message::Done(ctx)).unwrap();
        });

        let thread_result = rx.recv().unwrap();
        let ctx = match thread_result {
            Message::Done(ctx) => ctx,
        };

        {
            let buffer_map = ctx
                .buffer_map_2d::<V4f32>(result_buffer)
                .expect("Buffer map failed");

            assert_eq!(buffer_map[(0, 0)], v4f32(0., 0., 0., 0.));
            assert_eq!(buffer_map.width(), 256);
            assert_eq!(buffer_map.height(), 128);
            assert_eq!(
                buffer_map[(255, 127)],
                V4f32::new(255f32 / 256f32, 127f32 / 128f32, 0f32, 0f32)
            );
            write_scoped_buf_map_v4f32("solid_color_mt.png", &buffer_map)?;
        }

        Ok(())
    }

    extern "C" fn usage_callback(
        verbosity: i32,
        tag: *const std::os::raw::c_char,
        msg: *const std::os::raw::c_char,
        _cbdata: *mut std::os::raw::c_void,
    ) {
        let tag = unsafe {
            std::ffi::CStr::from_ptr(tag).to_string_lossy().into_owned()
        };
        let msg = unsafe {
            std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned()
        };

        print!("{} {:>16} {}", verbosity, tag, msg);
    }

    #[test]
    fn single_triangle_mt() -> Result<()> {
        use std::sync::mpsc;
        use std::thread;

        let mut ctx = Context::new();
        ctx.set_search_path(SearchPath::from_config_file(
            "ptx_path", "ptx_path",
        ));

        ctx.set_usage_report_callback(usage_callback, 1)?;

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
        let prg_material_constant_closest = ctx
            .program_create_from_ptx_file("mtl_constant.ptx", "closest_hit")?;

        ctx.program_set_variable(
            prg_material_constant_closest,
            "in_diffuse_albedo",
            v3f32(0.2, 0.2, 0.8),
        )?;

        // Create mesh data
        let buf_vertex = ctx.buffer_create_from_slice_1d(
            &[
                v3f32(0.2, 0.2, -10.),
                v3f32(0.8, 0.2, -10.),
                v3f32(0.5, 0.8, -10.),
            ],
            BufferType::INPUT,
            BufferFlag::NONE,
        )?;
        let buf_indices = ctx.buffer_create_from_slice_1d(
            &[v3i32(0, 1, 2)],
            BufferType::INPUT,
            BufferFlag::NONE,
        )?;
        let buf_normal = ctx.buffer_create_1d(
            0,
            Format::FLOAT3,
            BufferType::INPUT,
            BufferFlag::NONE,
        )?;
        let buf_texcoord = ctx.buffer_create_1d(
            0,
            Format::FLOAT2,
            BufferType::INPUT,
            BufferFlag::NONE,
        )?;
        let geo_triangle =
            ctx.geometry_create(prg_mesh_bound, prg_mesh_intersect)?;
        ctx.geometry_set_primitive_count(geo_triangle, 1)?;
        ctx.geometry_set_variable(
            geo_triangle,
            "vertex_buffer",
            ObjectHandle::Buffer1d(buf_vertex),
        )?;
        ctx.geometry_set_variable(
            geo_triangle,
            "index_buffer",
            ObjectHandle::Buffer1d(buf_indices),
        )?;
        ctx.geometry_set_variable(
            geo_triangle,
            "normal_buffer",
            ObjectHandle::Buffer1d(buf_normal),
        )?;
        ctx.geometry_set_variable(
            geo_triangle,
            "texcoord_buffer",
            ObjectHandle::Buffer1d(buf_texcoord),
        )?;
        let materials = vec![0];
        let buf_material = ctx.buffer_create_from_slice_1d(
            &materials,
            BufferType::INPUT,
            BufferFlag::NONE,
        )?;
        ctx.geometry_set_variable(
            geo_triangle,
            "material_buffer",
            ObjectHandle::Buffer1d(buf_material),
        )?;

        let raytype_camera = ctx.set_ray_type(0, "camera")?;

        ctx.set_miss_program(raytype_camera, prg_miss)?;

        // let mut map_mtl_camera_any = HashMap::new();
        // let mut map_mtl_camera_closest = HashMap::new();
        // map_mtl_camera_any.insert(raytype_camera, prg_material_constant_any);
        // map_mtl_camera_closest
        //     .insert(raytype_camera, prg_material_constant_closest);

        // let mtl_constant =
        //     ctx.material_create(map_mtl_camera_any, map_mtl_camera_closest)?;
        let mut mtl_camera_programs = HashMap::new();
        mtl_camera_programs.insert(
            raytype_camera,
            MaterialProgram {
                closest: Some(prg_material_constant_closest),
                any: None,
            },
        );
        let mtl_constant = ctx.material_create(mtl_camera_programs)?;

        let geo_inst = ctx.geometry_instance_create(
            GeometryType::Geometry(geo_triangle),
            vec![mtl_constant],
        )?;

        let acc = ctx.acceleration_create(Builder::Trbvh)?;

        let geo_group = ctx.geometry_group_create(acc, vec![geo_inst])?;

        ctx.program_set_variable(
            prg_cam_screen,
            "scene_root",
            ObjectHandle::GeometryGroup(geo_group),
        )?;

        let entry_point = ctx.add_entry_point(prg_cam_screen, None)?;

        let result_buffer = ctx
            .buffer_create_2d(
                256,
                128,
                Format::FLOAT4,
                BufferType::OUTPUT,
                BufferFlag::NONE,
            )
            .expect("Could not create result buffer");

        ctx.set_variable(
            "result_buffer",
            ObjectHandle::Buffer2d(result_buffer),
        )
        .expect("Setting buffer2d variable failed");

        ctx.validate().expect("Context validation failed");

        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            ctx.launch_2d(entry_point, 256, 128).unwrap();
            ctx.launch_2d(entry_point, 256, 128).unwrap();
            ctx.launch_2d(entry_point, 256, 128).unwrap();
            ctx.launch_2d(entry_point, 256, 128).unwrap();
            ctx.launch_2d(entry_point, 256, 128).unwrap();
            tx.send(Message::Done(ctx)).unwrap();
        });

        let thread_result = rx.recv().unwrap();
        let ctx = match thread_result {
            Message::Done(ctx) => ctx,
        };

        {
            let buffer_map = ctx
                .buffer_map_2d::<V4f32>(result_buffer)
                .expect("Buffer map failed");
            write_scoped_buf_map_v4f32("single_triangle_mt.png", &buffer_map)?;
        }

        Ok(())
    }
}
