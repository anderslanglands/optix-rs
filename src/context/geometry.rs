use crate::context::*;
use cfg_if::cfg_if;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct Geometry {
    pub(crate) rt_geo: RTgeometry,
    pub(crate) prg_bounding_box: ProgramHandle,
    pub(crate) prg_intersection: ProgramHandle,
    pub(crate) variables: HashMap<String, VariableHandle>,
}

pub type GeometryHandle = Rc<RefCell<Geometry>>;

cfg_if! {
if #[cfg(feature="optix5")] {
    pub enum GeometryType {
        Geometry(GeometryHandle),
    }
} else {
    pub enum GeometryType {
        Geometry(GeometryHandle),
        GeometryTriangles(GeometryTrianglesHandle),
    }

    impl GeometryType {
        pub fn clone(&self) -> GeometryType {
            match &self {
                GeometryType::Geometry(geo) => GeometryType::Geometry(Rc::clone(geo)),
                GeometryType::GeometryTriangles(geo) => GeometryType::GeometryTriangles(Rc::clone(geo)),
            }
        }
    }
}
}

impl Context {
    pub fn geometry_create(
        &mut self,
        prg_bounding_box: ProgramHandle,
        prg_intersection: ProgramHandle,
    ) -> Result<GeometryHandle> {
        self.program_validate(&prg_bounding_box)?;
        self.program_validate(&prg_intersection)?;

        let (geo, result) = unsafe {
            let mut geo: RTgeometry = std::mem::zeroed();
            let result = rtGeometryCreate(self.rt_ctx, &mut geo);
            (geo, result)
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryCreate", result))
        } else {
            let result = unsafe {
                rtGeometrySetBoundingBoxProgram(
                    geo,
                    prg_bounding_box.borrow().rt_prg,
                )
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtGeometryBoundingBoxProgram", result)
                );
            }

            let result = unsafe {
                rtGeometrySetIntersectionProgram(
                    geo,
                    prg_intersection.borrow().rt_prg,
                )
            };
            if result != RtResult::SUCCESS {
                return Err(self
                    .optix_error("rtGeometrySetIntersectionProgram", result));
            }

            let hnd = Rc::new(RefCell::new(Geometry {
                rt_geo: geo,
                prg_bounding_box,
                prg_intersection,
                variables: HashMap::new(),
            }));

            self.geometrys.push(Rc::clone(&hnd));

            Ok(hnd)
        }
    }

    pub fn geometry_validate(&self, geo: &GeometryHandle) -> Result<()> {
        let result = unsafe { rtGeometryValidate(geo.borrow().rt_geo) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryValidate", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_set_primitive_count(
        &mut self,
        geo: &GeometryHandle,
        num_primitives: u32,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometrySetPrimitiveCount(geo.borrow().rt_geo, num_primitives)
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometrySetPrimitiveCount", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_set_variable<T: VariableStorable>(
        &mut self,
        geo: &GeometryHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        // we have to remove the variable first so that we're not holding a borrow
        // further down here
        let ex_var = geo.borrow_mut().variables.remove(name);
        if let Some(ex_var) = ex_var {
            let var = {
                let ex_var_c = ex_var.borrow();
                match &*ex_var_c {
                    Variable::Pod(vp) => vp.var,
                    Variable::Object(vo) => vo.var,
                    Variable::User(vo) => vo.var,
                }
            };

            // replace the variable and reinsert it
            ex_var.replace(data.set_optix_variable(self, var)?);
            geo.borrow_mut().variables.insert(name.into(), ex_var);
        } else {
            let (rt_var, result) = unsafe {
                let mut rt_var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtGeometryDeclareVariable(
                    geo.borrow_mut().rt_geo,
                    c_name.as_ptr(),
                    &mut rt_var,
                );
                (rt_var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtGeometryDeclareVariable", result)
                );
            }

            let var =
                Rc::new(RefCell::new(data.set_optix_variable(self, rt_var)?));
            geo.borrow_mut()
                .variables
                .insert(name.into(), Rc::clone(&var));

            // if it's a new variable, push it to the context storage
            self.variables.push(var)
        }

        Ok(())
    }

    /// Set the Variable referred to by `name` to the given `data`. Any objects
    /// previously assigned to the variable will be destroyed.
    pub fn geometry_set_user_variable(
        &mut self,
        geo: &GeometryHandle,
        name: &str,
        data: Box<dyn UserVariable>,
    ) -> Result<()> {
        // check if the variable exists first
        if let Some(ex_var) = geo.borrow_mut().variables.remove(name) {
            let var = {
                let ex_var_c = ex_var.borrow();
                match &*ex_var_c {
                    Variable::Pod(vp) => vp.var,
                    Variable::Object(vo) => vo.var,
                    Variable::User(vu) => vu.var,
                }
            };

            data.set_user_variable(self, var)?;
            ex_var.replace(Variable::User(UserData { var, data }));
            geo.borrow_mut().variables.insert(name.into(), ex_var);
        } else {
            let (rt_var, result) = unsafe {
                let mut rt_var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtContextDeclareVariable(
                    self.rt_ctx,
                    c_name.as_ptr(),
                    &mut rt_var,
                );
                (rt_var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtContextDeclareVariable", result)
                );
            }

            data.set_user_variable(self, rt_var)?;
            let var = Rc::new(RefCell::new(Variable::User(UserData {
                var: rt_var,
                data,
            })));

            geo.borrow_mut()
                .variables
                .insert(name.into(), Rc::clone(&var));

            // if it's a new variable, push it to the context storage
            self.variables.push(var)
        }

        Ok(())
    }

    pub fn geometry_set_motion_steps(
        &mut self,
        geo: &GeometryHandle,
        steps: u32,
    ) -> Result<()> {
        let result =
            unsafe { rtGeometrySetMotionSteps(geo.borrow().rt_geo, steps) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometrySetMotionSteps", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_set_motion_range(
        &mut self,
        geo: &GeometryHandle,
        time_begin: f32,
        time_end: f32,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometrySetMotionRange(geo.borrow().rt_geo, time_begin, time_end)
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometrySetMotionRange", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_set_motion_border_mode(
        &mut self,
        geo: &GeometryHandle,
        begin_mode: MotionBorderMode,
        end_mode: MotionBorderMode,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometrySetMotionBorderMode(
                geo.borrow().rt_geo,
                begin_mode,
                end_mode,
            )
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometrySetMotionBorderMode", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_set_intersection_program(
        &mut self,
        geo: &GeometryHandle,
        prg: &ProgramHandle,
    ) -> Result<()> {
        self.program_validate(&prg)?;
        let result = unsafe {
            rtGeometrySetIntersectionProgram(
                geo.borrow().rt_geo,
                prg.borrow().rt_prg,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtGeometrySetIntersectionProgram", result)
            );
        }

        geo.borrow_mut().prg_intersection = Rc::clone(prg);

        Ok(())
    }

    pub fn geometry_set_bounding_box_program(
        &mut self,
        geo: &GeometryHandle,
        prg: &ProgramHandle,
    ) -> Result<()> {
        self.program_validate(&prg)?;
        let result = unsafe {
            rtGeometrySetBoundingBoxProgram(
                geo.borrow().rt_geo,
                prg.borrow().rt_prg,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtGeometrySetBoundingBoxProgram", result)
            );
        }

        geo.borrow_mut().prg_bounding_box = Rc::clone(prg);

        Ok(())
    }
}
