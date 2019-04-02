use crate::context::*;
use std::collections::HashMap;

use std::cell::RefCell;
use std::rc::Rc;

pub struct GeometryInstance {
    pub(crate) rt_geoinst: RTgeometryinstance,
    pub(crate) variables: HashMap<String, VariableHandle>,
    pub(crate) geo: GeometryType,
    pub(crate) materials: Vec<MaterialHandle>,
}

pub type GeometryInstanceHandle = Rc<RefCell<GeometryInstance>>;

impl Context {
    /// Create a new `GeometryInstance` with the given geometry and materials.
    pub fn geometry_instance_create(
        &mut self,
        geometry_type: GeometryType,
        materials: Vec<MaterialHandle>,
    ) -> Result<GeometryInstanceHandle> {
        for mat in &materials {
            self.material_validate(mat)?;
        }

        match &geometry_type {
            GeometryType::Geometry(geo) => {
                self.geometry_validate(&geo)?;

                let (geoinst, result) = unsafe {
                    let mut geoinst: RTgeometryinstance = std::mem::zeroed();
                    let result =
                        rtGeometryInstanceCreate(self.rt_ctx, &mut geoinst);
                    (geoinst, result)
                };
                if result != RtResult::SUCCESS {
                    Err(self.optix_error("rtGeometryInstanceCreate", result))
                } else {
                    let result = unsafe {
                        rtGeometryInstanceSetGeometry(
                            geoinst,
                            geo.borrow().rt_geo,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error(
                            "rtGeometryInstanceSetGeometry",
                            result,
                        ));
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
                        let result = unsafe {
                            rtGeometryInstanceSetMaterial(
                                geoinst,
                                i as u32,
                                mat.borrow().rt_mat,
                            )
                        };
                        if result != RtResult::SUCCESS {
                            return Err(self.optix_error(
                                "rtGeometryInstanceSetMaterial",
                                result,
                            ));
                        }
                    }

                    let hnd = Rc::new(RefCell::new(GeometryInstance {
                        rt_geoinst: geoinst,
                        variables: HashMap::new(),
                        geo: geometry_type,
                        materials,
                    }));

                    self.geometry_instances.push(Rc::clone(&hnd));

                    Ok(hnd)
                }
            }
        }
    }

    /// Set the Variable referred to by `name` to the given `data`. Any objects
    /// previously assigned to the variable will be destroyed.
    pub fn geometry_instance_set_variable<T: VariableStorable>(
        &mut self,
        geoinst: &GeometryInstanceHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        // we have to remove the variable first so that we're not holding a borrow
        // further down here
        let ex_var = geoinst.borrow_mut().variables.remove(name);
        if let Some(ex_var) = ex_var {
            let var = {
                let ex_var_c = ex_var.borrow();
                match &*ex_var_c {
                    Variable::Pod(vp) => vp.var,
                    Variable::Object(vo) => vo.var,
                }
            };

            // replace the variable and reinsert it
            ex_var.replace(data.set_optix_variable(self, var)?);
            geoinst.borrow_mut().variables.insert(name.into(), ex_var);
        } else {
            let (rt_var, result) = unsafe {
                let mut rt_var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtGeometryInstanceDeclareVariable(
                    geoinst.borrow_mut().rt_geoinst,
                    c_name.as_ptr(),
                    &mut rt_var,
                );
                (rt_var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(self
                    .optix_error("rtGeometryInstanceDeclareVariable", result));
            }

            let var =
                Rc::new(RefCell::new(data.set_optix_variable(self, rt_var)?));
            geoinst
                .borrow_mut()
                .variables
                .insert(name.into(), Rc::clone(&var));

            // if it's a new variable, push it to the context storage
            self.variables.push(var)
        }

        Ok(())
    }

    pub fn geometry_instance_validate(
        &self,
        geoinst: &GeometryInstanceHandle,
    ) -> Result<()> {
        let result =
            unsafe { rtGeometryInstanceValidate(geoinst.borrow().rt_geoinst) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryInstanceValidate", result))
        } else {
            Ok(())
        }
    }
}
