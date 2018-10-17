use crate::context::*;
use crate::ginallocator::*;
use std::collections::HashMap;

#[derive(Default, Copy, Clone)]
pub struct GeometryInstanceMarker;
impl Marker for GeometryInstanceMarker {
    const ID: &'static str = "GeometryInstance";
}
pub type GeometryInstanceHandle = Handle<GeometryInstanceMarker>;

impl Context {
    pub fn geometry_instance_create(
        &mut self,
        geometry_type: GeometryType,
        materials: Vec<MaterialHandle>,
    ) -> Result<GeometryInstanceHandle> {
        for mat in &materials {
            self.material_validate(*mat)?;
        }

        match geometry_type {
            GeometryType::Geometry(geo) => {
                self.geometry_validate(geo)?;

                let (geoinst, result) = unsafe {
                    let mut geoinst: RTgeometryinstance = std::mem::zeroed();
                    let result =
                        rtGeometryInstanceCreate(self.rt_ctx, &mut geoinst);
                    (geoinst, result)
                };
                if result != RtResult::SUCCESS {
                    Err(self.optix_error("rtGeometryCreate", result))
                } else {
                    let hnd = self.ga_geometry_instance_obj.insert(geoinst);
                    let chnd = self
                        .ga_geometry_instance_obj
                        .check_handle(hnd)
                        .unwrap();

                    let rt_geo = self.ga_geometry_obj.get(geo).unwrap();
                    let result = unsafe {
                        rtGeometryInstanceSetGeometry(geoinst, *rt_geo)
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error(
                            "rtGeometryInstanceSetGeometry",
                            result,
                        ));
                    } else {
                        self.ga_geometry_obj.incref(geo);
                        self.gd_geometry_instance_geometry
                            .insert(&chnd, GeometryType::Geometry(geo));
                    }

                    let result = unsafe {
                        rtGeometryInstanceSetMaterialCount(
                            geoinst,
                            materials.len() as u32,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error(
                            "rtGeometryInstanceSetMaterialCount",
                            result,
                        ));
                    };
                    for (i, mat) in materials.iter().enumerate() {
                        let rt_mat = self.ga_material_obj.get(*mat).unwrap();
                        let result = unsafe {
                            rtGeometryInstanceSetMaterial(
                                geoinst, i as u32, *rt_mat,
                            )
                        };
                        if result != RtResult::SUCCESS {
                            return Err(self.optix_error(
                                "rtGeometryInstanceSetMaterial",
                                result,
                            ));
                        } else {
                            self.ga_material_obj.incref(*mat);
                        }
                    }

                    let vars = HashMap::<String, Variable>::new();
                    self.gd_geometry_instance_variables.insert(&chnd, vars);
                    self.gd_geometry_instance_materials
                        .insert(&chnd, materials);

                    Ok(hnd)
                }
            }
        }
    }

    /// Set the Variable referred to by `name` to the given `data`. Any objects
    /// previously assigned to the variable will be destroyed.
    pub fn geometry_instance_set_variable<T: VariableStorable>(
        &mut self,
        geoinst: GeometryInstanceHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        let cgeoinst = self
            .ga_geometry_instance_obj
            .check_handle(geoinst)
            .expect("Tried to access an invalid geometry_instance handle");
        let rt_geoinst = self.ga_geometry_instance_obj.get(geoinst).unwrap();

        if let Some(old_variable) =
            self.gd_geometry_instance_variables.get(cgeoinst).get(name)
        {
            let var = match old_variable {
                Variable::Pod(vp) => vp.var,
                Variable::Object(vo) => vo.var,
            };
            let new_variable = data.set_optix_variable(self, var)?;
            // destroy any resources the existing variable holds
            if let Some(old_variable) = self
                .gd_geometry_instance_variables
                .get_mut(cgeoinst)
                .insert(name.to_owned(), new_variable)
            {
                self.destroy_variable(old_variable);
            };

            Ok(())

        } else {
            let (var, result) = unsafe {
                let mut var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtGeometryInstanceDeclareVariable(
                    *rt_geoinst,
                    c_name.as_ptr(),
                    &mut var,
                );
                (var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtGeometryInstanceDeclareVariable", result));
            }

            let variable = data.set_optix_variable(self, var)?;
            self.gd_geometry_instance_variables
                .get_mut(cgeoinst)
                .insert(name.to_owned(), variable);

            Ok(())
        }
    }

    pub fn geometry_instance_destroy(
        &mut self,
        geoinst: GeometryInstanceHandle,
    ) {
        let cgeoinst =
            self.ga_geometry_instance_obj.check_handle(geoinst).unwrap();

        let vars = self.gd_geometry_instance_variables.remove(cgeoinst);
        self.destroy_variables(vars);

        let geo_type = self.gd_geometry_instance_geometry.get(cgeoinst);
        match geo_type {
            GeometryType::Geometry(geo) => {
                self.geometry_destroy(*geo);
            }
        }

        let materials = self.gd_geometry_instance_materials.remove(cgeoinst);
        for mat in materials {
            self.material_destroy(mat);
        }

        match self.ga_geometry_instance_obj.destroy(geoinst) {
            DestroyResult::StillAlive => (),
            DestroyResult::ShouldDrop => {
                let rt_geoinst =
                    *self.ga_geometry_instance_obj.get(geoinst).unwrap();
                if unsafe { rtGeometryInstanceDestroy(rt_geoinst) }
                    != RtResult::SUCCESS
                {
                    panic!("Error destroying program {}", geoinst);
                }
            }
        }
    }

    pub fn geometry_instance_validate(
        &self,
        geoinst: GeometryInstanceHandle,
    ) -> Result<()> {
        let rt_geoinst = self.ga_geometry_instance_obj.get(geoinst).unwrap();
        let result = unsafe { rtGeometryInstanceValidate(*rt_geoinst) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryInstanceValidate", result))
        } else {
            Ok(())
        }
    }
}
