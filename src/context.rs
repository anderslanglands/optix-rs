use crate::error::*;
use std::collections::HashMap;

use crate::ginallocator::{GinAllocator, GinAllocatorChild, Marker};
use crate::math::*;
use crate::optix_bindings::*;
use crate::search_path::SearchPath;

mod program;
use self::program::*;
mod geometry;
use self::geometry::*;
mod material;
use self::material::*;
mod geometry_instance;
use self::geometry_instance::*;
mod buffer;
use self::buffer::*;
mod variable;
use self::variable::*;

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
    entry_points: Vec<EntryPoint>,
    search_path: SearchPath,

    context_variables: HashMap<String, Variable>,

    ga_buffer1d_obj: GinAllocator<RTbuffer, Buffer1dMarker>,
    ga_buffer2d_obj: GinAllocator<RTbuffer, Buffer2dMarker>,
    ga_buffer3d_obj: GinAllocator<RTbuffer, Buffer3dMarker>,

    ga_program_obj: GinAllocator<RTprogram, ProgramMarker>,
    gd_program_variables:
        GinAllocatorChild<HashMap<String, Variable>, ProgramMarker>,

    ga_geometry_obj: GinAllocator<RTgeometry, GeometryMarker>,
    gd_geometry_variables:
        GinAllocatorChild<HashMap<String, Variable>, GeometryMarker>,
    gd_geometry_bounding_box: GinAllocatorChild<ProgramHandle, GeometryMarker>,
    gd_geometry_intersection: GinAllocatorChild<ProgramHandle, GeometryMarker>,

    ga_geometry_instance_obj:
        GinAllocator<RTgeometryinstance, GeometryInstanceMarker>,
    gd_geometry_instance_variables:
        GinAllocatorChild<HashMap<String, Variable>, GeometryInstanceMarker>,
    gd_geometry_instance_geometry:
        GinAllocatorChild<GeometryHandle, GeometryInstanceMarker>,
    gd_geometry_instance_materials:
        GinAllocatorChild<Vec<MaterialHandle>, GeometryInstanceMarker>,

    ga_material_obj: GinAllocator<RTmaterial, MaterialMarker>,
    gd_material_variables:
        GinAllocatorChild<HashMap<String, Variable>, MaterialMarker>,
    gd_material_any_hit:
        GinAllocatorChild<HashMap<RayType, ProgramHandle>, MaterialMarker>,
    gd_material_closest_hit:
        GinAllocatorChild<HashMap<RayType, ProgramHandle>, MaterialMarker>,
}

/// The `Launcher` is used to launch the raytracing engine. In order to obtain a
/// `Launcher` you must consume the `Context` with the `get_launcher` method, 
/// which returns a `Launcher` and a `Silo`.
pub struct Launcher {
    rt_ctx: RTcontext,
}

impl Launcher {
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

    pub fn optix_error(&self, msg: &str, result: RtResult) -> Error {
        Error::Optix((
            result,
            format!("{}: {}", msg, get_error_string(self.rt_ctx, result)),
        ))
    }
}

pub struct Silo {
    ctx: Context,
}

/// The `Silo` is storage for the `Context` while the `Launcher` is active. In 
/// this way it is impossible to modify the scene graph while a launch is in 
/// progress.
impl Silo {
    /// Consume self and the given `Launcher` to get back the original `Context`
    /// and be able to make scene edits again
    pub fn to_context(self, _launcher: Launcher) -> Context {
        self.ctx
    }
}

/// We impl Send here as because you can only get a Launcher by consuming a
/// a Context, and we can't get a Context again until we consume the launcher,
/// we know it's impossible to mutate the Context while the Launcher is alive
unsafe impl Send for Launcher {}

impl Context {
    pub fn new() -> Context {
        let ga_buffer1d_obj = GinAllocator::<RTbuffer, Buffer1dMarker>::new();
        let ga_buffer2d_obj = GinAllocator::<RTbuffer, Buffer2dMarker>::new();
        let ga_buffer3d_obj = GinAllocator::<RTbuffer, Buffer3dMarker>::new();

        let ga_program_obj = GinAllocator::<RTprogram, ProgramMarker>::new();
        let gd_program_variables = ga_program_obj.create_child();

        let ga_geometry_obj = GinAllocator::<RTgeometry, GeometryMarker>::new();
        let gd_geometry_variables = ga_geometry_obj.create_child();
        let gd_geometry_bounding_box = ga_geometry_obj.create_child();
        let gd_geometry_intersection = ga_geometry_obj.create_child();

        let ga_geometry_instance_obj =
            GinAllocator::<RTgeometryinstance, GeometryInstanceMarker>::new();
        let gd_geometry_instance_variables =
            ga_geometry_instance_obj.create_child();
        let gd_geometry_instance_geometry =
            ga_geometry_instance_obj.create_child();
        let gd_geometry_instance_materials =
            ga_geometry_instance_obj.create_child();

        let ga_material_obj = GinAllocator::<RTmaterial, MaterialMarker>::new();
        let gd_material_variables = ga_material_obj.create_child();
        let gd_material_any_hit = ga_material_obj.create_child();
        let gd_material_closest_hit = ga_material_obj.create_child();

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
            entry_points: Vec::new(),
            search_path: SearchPath::new(),

            context_variables: HashMap::new(),

            ga_buffer1d_obj,
            ga_buffer2d_obj,
            ga_buffer3d_obj,

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

            ga_material_obj,
            gd_material_variables,
            gd_material_any_hit,
            gd_material_closest_hit,
        }
    }

    pub fn set_search_path(&mut self, search_path: SearchPath) {
        self.search_path = search_path;
    }

    pub fn add_ray_type(&mut self, index: u32, name: &str) -> RayType {
        let ray_type = RayType { ray_type: index };
        self.ray_types.insert(ray_type, name.to_owned());
        ray_type
    }

    pub fn validate(&self) -> Result<()> {
        let result = unsafe { rtContextValidate(self.rt_ctx) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtContextValidate", result))
        } else {
            Ok(())
        }
    }

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
                "Could not get RTprogram object from handle: {}",
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
                "Could not get RTprogram object from handle: {}",
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

    pub fn to_launcher(self) -> (Launcher, Silo) {
        (
            Launcher {
                rt_ctx: self.rt_ctx,
            },
            Silo { ctx: self },
        )
    }

    pub fn search_path(&self) -> &SearchPath {
        &self.search_path
    }

    pub fn optix_error(&self, msg: &str, result: RtResult) -> Error {
        Error::Optix((
            result,
            format!("{}: {}", msg, get_error_string(self.rt_ctx, result)),
        ))
    }

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
                match old_variable {
                    Variable::Pod(_vp) => (),
                    Variable::Object(vo) => match vo.object_handle {
                        ObjectHandle::Buffer2d(bh) => {
                            self.buffer_destroy_2d(bh)
                        }
                        ObjectHandle::Program(ph) => self.program_destroy(ph),
                    },
                };
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
}

#[cfg(test)]
mod tests {
    fn v4f_buffer_to_u8(buf: &ScopedBufMap2d<V4f32>) -> Vec<u8> {
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

    fn write_scoped_buf_map_v4f(
        filename: &str,
        buf: &ScopedBufMap2d<V4f32>,
    ) -> Result<()> {
        use image::{save_buffer, ColorType};
        let buf_u8 = v4f_buffer_to_u8(&buf);
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
            .buffer_create_2d::<V4f32>(
                256,
                128,
                BufferType::OUTPUT,
                BufferFlag::NONE,
            ).expect("Could not create result buffer");

        ctx.set_variable(
            "result_buffer",
            ObjectHandle::Buffer2d(result_buffer),
        ).expect("Setting buffer2d variable failed");

        ctx.validate().expect("Context validation failed");

        let (launcher, silo) = ctx.to_launcher();
        launcher.launch_2d(entry_point, 256, 128)?;
        ctx = silo.to_context(launcher);

        // try destroying the buffer... the refcounting should allow the
        // buffer to survive and the map and write to succeed without error
        ctx.buffer_destroy_2d(result_buffer);

        {
            let buffer_map = ctx
                .buffer_map_2d::<V4f32>(result_buffer)
                .expect("Buffer map failed");

            assert_eq!(buffer_map[(0, 0)], v4f(0., 0., 0., 0.));
            assert_eq!(buffer_map.width(), 256);
            assert_eq!(buffer_map.height(), 128);
            assert_eq!(
                buffer_map[(255, 127)],
                V4f32::new(255f32 / 256f32, 127f32 / 128f32, 0f32, 0f32)
            );
            write_scoped_buf_map_v4f("solid_color.png", &buffer_map)?;
        }

        Ok(())
    }

    enum Message {
        Done(Launcher),
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
            .buffer_create_2d::<V4f32>(
                256,
                128,
                BufferType::OUTPUT,
                BufferFlag::NONE,
            ).expect("Could not create result buffer");

        ctx.set_variable(
            "result_buffer",
            ObjectHandle::Buffer2d(result_buffer),
        ).expect("Setting buffer2d variable failed");

        ctx.validate().expect("Context validation failed");

        let (launcher, silo) = ctx.to_launcher();

        let (tx, rx) = mpsc::channel();

        thread::spawn(move || {
            launcher.launch_2d(entry_point, 256, 128).unwrap();
            tx.send(Message::Done(launcher)).unwrap();
        });

        // The compiler will stop us from trying to edit the scene while a
        // launcher is active...
        // ctx.buffer_destroy_2d(result_buffer); <- Error: borrow of moved value

        let thread_result = rx.recv().unwrap();
        let launcher = match thread_result {
            Message::Done(l) => l,
        };

        ctx = silo.to_context(launcher);

        // try destroying the buffer... the refcounting should allow the
        // buffer to survive and the map and write to succeed without error
        ctx.buffer_destroy_2d(result_buffer);

        {
            let buffer_map = ctx
                .buffer_map_2d::<V4f32>(result_buffer)
                .expect("Buffer map failed");

            assert_eq!(buffer_map[(0, 0)], v4f(0., 0., 0., 0.));
            assert_eq!(buffer_map.width(), 256);
            assert_eq!(buffer_map.height(), 128);
            assert_eq!(
                buffer_map[(255, 127)],
                V4f32::new(255f32 / 256f32, 127f32 / 128f32, 0f32, 0f32)
            );
            write_scoped_buf_map_v4f("solid_color_mt.png", &buffer_map)?;
        }

        Ok(())
    }
}
