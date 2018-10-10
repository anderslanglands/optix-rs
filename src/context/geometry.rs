use crate::context::*;
use crate::ginallocator::*;
use std::collections::HashMap;
use std::marker::PhantomData;

#[derive(Default, Copy, Clone)]
pub struct GeometryMarker;
impl Marker for GeometryMarker {
    const ID: &'static str = "Geometry";
}
pub type GeometryHandle = Handle<GeometryMarker>;

pub enum GeometryType {
    Geometry(GeometryHandle),
    // GeometryTriangles(GeometryTrianglesHandle),
}

impl Default for GeometryType {
    fn default() -> GeometryType {
        GeometryType::Geometry(GeometryHandle {
            index: 0,
            generation: 0,
            phantom: PhantomData,
        })
    }
}

impl Context {
    pub fn geometry_create(
        &mut self,
        prg_bounding_box: ProgramHandle,
        prg_intersection: ProgramHandle,
    ) -> Result<GeometryHandle> {
        self.program_validate(prg_bounding_box)?;
        self.program_validate(prg_intersection)?;

        let (geo, result) = unsafe {
            let mut geo: RTgeometry = std::mem::zeroed();
            let result = rtGeometryCreate(self.rt_ctx, &mut geo);
            (geo, result)
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryCreate", result))
        } else {
            let rt_prg_bound =
                self.ga_program_obj.get(prg_bounding_box).unwrap();
            let rt_prg_intersection =
                self.ga_program_obj.get(prg_intersection).unwrap();

            let result =
                unsafe { rtGeometrySetBoundingBoxProgram(geo, *rt_prg_bound) };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtGeometryBoundingBoxProgram", result)
                );
            }

            let result = unsafe {
                rtGeometrySetIntersectionProgram(geo, *rt_prg_intersection)
            };
            if result != RtResult::SUCCESS {
                return Err(self
                    .optix_error("rtGeometrySetIntersectionProgram", result));
            }

            let vars = HashMap::<String, Variable>::new();
            let hnd = self.ga_geometry_obj.insert(geo);
            let chnd = self.ga_geometry_obj.check_handle(hnd).unwrap();
            self.gd_geometry_variables.insert(&chnd, vars);
            self.ga_program_obj.incref(prg_bounding_box);
            self.gd_geometry_bounding_box
                .insert(&chnd, prg_bounding_box);
            self.ga_program_obj.incref(prg_intersection);
            self.gd_geometry_intersection
                .insert(&chnd, prg_intersection);

            Ok(hnd)
        }
    }

    pub fn geometry_destroy(&mut self, geo: GeometryHandle) {
        let cgeo = self.ga_geometry_obj.check_handle(geo).unwrap();

        let vars = self.gd_geometry_variables.remove(cgeo);
        self.destroy_variables(vars);

        let prg_bounding_box = self.gd_geometry_bounding_box.get(cgeo);
        self.program_destroy(*prg_bounding_box);
        let prg_intersection = self.gd_geometry_intersection.get(cgeo);
        self.program_destroy(*prg_intersection);

        match self.ga_geometry_obj.destroy(geo) {
            DestroyResult::StillAlive => (),
            DestroyResult::ShouldDrop => {
                let rt_geo = *self.ga_geometry_obj.get(geo).unwrap();
                if unsafe {rtGeometryDestroy(rt_geo)} != RtResult::SUCCESS {
                    panic!("Error destroying Geometry");
                }
            }
        }
    }

    pub fn geometry_validate(&self, geo: GeometryHandle) -> Result<()> {
        let rt_geo = self.ga_geometry_obj.get(geo).unwrap();
        let result = unsafe { rtGeometryValidate(*rt_geo) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryValidate", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_set_primitive_count(
        &mut self,
        geo: GeometryHandle,
        num_primitives: u32,
    ) -> Result<()> {
        let rt_geo = self.ga_geometry_obj.get_mut(geo).unwrap();
        let result =
            unsafe { rtGeometrySetPrimitiveCount(*rt_geo, num_primitives) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometrySetPrimitiveCount", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_set_variable<T: VariableStorable>(
        &mut self,
        geo: GeometryHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        let cgeo = self
            .ga_geometry_obj
            .check_handle(geo)
            .expect("Tried to access an invalid geometry handle");
        let rt_geo = self.ga_geometry_obj.get(geo).unwrap();

        if let Some(old_variable) =
            self.gd_geometry_variables.get(cgeo).get(name)
        {
            let var = match old_variable {
                Variable::Pod(vp) => vp.var,
                Variable::Object(vo) => vo.var,
            };
            let new_variable = data.set_optix_variable(self, var)?;
            // destroy any resources the existing variable holds
            if let Some(old_variable) = self
                .gd_geometry_variables
                .get_mut(cgeo)
                .insert(name.to_owned(), new_variable)
            {
                match old_variable {
                    Variable::Pod(_vp) => (),
                    Variable::Object(vo) => match vo.object_handle {
                        ObjectHandle::Buffer1d(bh) => {
                            self.buffer_destroy_1d(bh)
                        }
                        ObjectHandle::Buffer2d(bh) => {
                            self.buffer_destroy_2d(bh)
                        }
                        ObjectHandle::Group(ggh) => {
                            self.group_destroy(ggh)
                        }
                        ObjectHandle::GeometryGroup(ggh) => {
                            self.geometry_group_destroy(ggh)
                        }
                        ObjectHandle::Program(ph) => self.program_destroy(ph),
                        ObjectHandle::Transform(th) => self.transform_destroy(th),
                    },
                };
            };

            Ok(())
        } else {
            let (var, result) = unsafe {
                let mut var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtGeometryDeclareVariable(
                    *rt_geo,
                    c_name.as_ptr(),
                    &mut var,
                );
                (var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtGeometryDeclareVariable", result)
                );
            }

            let variable = data.set_optix_variable(self, var)?;
            self.gd_geometry_variables
                .get_mut(cgeo)
                .insert(name.to_owned(), variable);

            Ok(())
        }
    }
}
