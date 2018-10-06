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
