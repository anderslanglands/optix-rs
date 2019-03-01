use crate::context::*;
use std::collections::HashMap;

new_key_type! { pub struct GeometryHandle; }

pub enum GeometryType {
    Geometry(GeometryHandle),
    // GeometryTriangles(GeometryTrianglesHandle),
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
            let rt_prg_bound = self.ga_program_obj.get(prg_bounding_box).unwrap();
            let rt_prg_intersection = self.ga_program_obj.get(prg_intersection).unwrap();

            let result = unsafe { rtGeometrySetBoundingBoxProgram(geo, *rt_prg_bound) };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtGeometryBoundingBoxProgram", result));
            }

            let result = unsafe { rtGeometrySetIntersectionProgram(geo, *rt_prg_intersection) };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtGeometrySetIntersectionProgram", result));
            }

            let vars = HashMap::<String, Variable>::new();
            let hnd = self.ga_geometry_obj.insert(geo);
            self.gd_geometry_variables.insert(hnd, vars);
            self.gd_geometry_bounding_box.insert(hnd, prg_bounding_box);
            self.gd_geometry_intersection.insert(hnd, prg_intersection);

            Ok(hnd)
        }
    }

    pub fn geometry_destroy(&mut self, geo: GeometryHandle) {
        let _vars = self.gd_geometry_variables.remove(geo);

        let _prg_bounding_box = self.gd_geometry_bounding_box.remove(geo);
        let _prg_intersection = self.gd_geometry_intersection.remove(geo);

        let rt_geo = self.ga_geometry_obj.remove(geo).unwrap();
        if unsafe { rtGeometryDestroy(rt_geo) } != RtResult::SUCCESS {
            panic!("Error destroying Geometry");
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
        let result = unsafe { rtGeometrySetPrimitiveCount(*rt_geo, num_primitives) };
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
        let rt_geo = self.ga_geometry_obj.get(geo).unwrap();

        if let Some(old_variable) = self.gd_geometry_variables.get(geo).unwrap().get(name) {
            let var = match old_variable {
                Variable::Pod(vp) => vp.var,
                Variable::Object(vo) => vo.var,
            };
            let new_variable = data.set_optix_variable(self, var)?;
            // destroy any resources the existing variable holds
            self.gd_geometry_variables
                .get_mut(geo)
                .unwrap()
                .insert(name.to_owned(), new_variable);

            Ok(())
        } else {
            let (var, result) = unsafe {
                let mut var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtGeometryDeclareVariable(*rt_geo, c_name.as_ptr(), &mut var);
                (var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtGeometryDeclareVariable", result));
            }

            let variable = data.set_optix_variable(self, var)?;
            self.gd_geometry_variables
                .get_mut(geo)
                .unwrap()
                .insert(name.to_owned(), variable);

            Ok(())
        }
    }
}
