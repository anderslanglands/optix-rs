//! Materials contain the Programs that decide what to do when a ray/primitive
//! intersection is found.
use crate::context::*;
use std::collections::HashMap;

use std::cell::RefCell;
use std::rc::Rc;

pub struct Material {
    pub(crate) rt_mat: RTmaterial,
    pub(crate) variables: HashMap<String, VariableHandle>,
    pub programs: HashMap<RayType, MaterialProgram>,
}

pub type MaterialHandle = Rc<RefCell<Material>>;

/// Struct to hold the Programs associated with a particular Material and
/// RayType.
pub struct MaterialProgram {
    pub closest: Option<ProgramHandle>,
    pub any: Option<ProgramHandle>,
}

impl Context {
    /// Creates a new Material on this Context, returning a handle that can be
    /// used to access it later.
    ///
    /// # Errors
    /// If any of the programs in the `programs` map are invalid
    pub fn material_create(
        &mut self,
        programs: HashMap<RayType, MaterialProgram>,
    ) -> Result<MaterialHandle> {
        // First check that the programs are well-defined
        for (_, program) in &programs {
            if let Some(prg) = &program.closest {
                self.program_validate(prg)?
            };
            if let Some(prg) = &program.any {
                self.program_validate(prg)?
            };
        }

        let (mat, result) = unsafe {
            let mut mat: RTmaterial = std::mem::zeroed();
            let result = rtMaterialCreate(self.rt_ctx, &mut mat);
            (mat, result)
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtMaterialCreate", result))
        } else {
            for (raytype, program) in &programs {
                if let Some(prg) = &program.closest {
                    let result = unsafe {
                        rtMaterialSetClosestHitProgram(
                            mat,
                            raytype.ray_type,
                            prg.borrow().rt_prg,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error(
                            "rtMaterialSetClosestHitProgram",
                            result,
                        ));
                    }
                }
                if let Some(prg) = &program.any {
                    let result = unsafe {
                        rtMaterialSetAnyHitProgram(
                            mat,
                            raytype.ray_type,
                            prg.borrow().rt_prg,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error(
                            "rtMaterialSetAnyHitProgram",
                            result,
                        ));
                    }
                }
            }

            let hnd = Rc::new(RefCell::new(Material {
                rt_mat: mat,
                variables: HashMap::new(),
                programs,
            }));

            self.materials.push(Rc::clone(&hnd));

            Ok(hnd)
        }
    }

    /// Check that the Material and all objects attached to it are correctly
    /// set up.
    /// # Panics
    /// If mat is not a valid MaterialHandle
    pub fn material_validate(&self, mat: &MaterialHandle) -> Result<()> {
        let result = unsafe { rtMaterialValidate(mat.borrow().rt_mat) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtMaterialValidate", result))
        } else {
            Ok(())
        }
    }

    /// Set the Variable named `name` to `data`. If a Variable named `name` does
    /// not exist then it is created.
    ///
    /// # Panics
    /// If mat is not a valid MaterialHandle
    ///
    /// # Errors
    /// If `data` is a different type than a pre-existing Variable called `name`
    pub fn material_set_variable<T: VariableStorable>(
        &mut self,
        mat: &MaterialHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        // we have to remove the variable first so that we're not holding a borrow
        // further down here
        let ex_var = mat.borrow_mut().variables.remove(name);
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
            mat.borrow_mut().variables.insert(name.into(), ex_var);
        } else {
            let (rt_var, result) = unsafe {
                let mut rt_var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtMaterialDeclareVariable(
                    mat.borrow_mut().rt_mat,
                    c_name.as_ptr(),
                    &mut rt_var,
                );
                (rt_var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtMaterialDeclareVariable", result)
                );
            }

            let var =
                Rc::new(RefCell::new(data.set_optix_variable(self, rt_var)?));
            mat.borrow_mut()
                .variables
                .insert(name.into(), Rc::clone(&var));

            // if it's a new variable, push it to the context storage
            self.variables.push(var)
        }

        Ok(())
    }

    /// Set the Variable referred to by `name` to the given `data`. Any objects
    /// previously assigned to the variable will be destroyed.
    pub fn material_set_user_variable(
        &mut self,
        mat: &MaterialHandle,
        name: &str,
        data: Box<dyn UserVariable>,
    ) -> Result<()> {
        // check if the variable exists first
        let ex_var = mat.borrow_mut().variables.remove(name);
        if let Some(ex_var) = ex_var {
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
            mat.borrow_mut().variables.insert(name.into(), ex_var);
        } else {
            let (rt_var, result) = unsafe {
                let mut rt_var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtMaterialDeclareVariable(
                    mat.borrow_mut().rt_mat,
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

            mat.borrow_mut()
                .variables
                .insert(name.into(), Rc::clone(&var));

            // if it's a new variable, push it to the context storage
            self.variables.push(var)
        }

        Ok(())
    }

    pub fn material_set_any_hit_program(
        &mut self,
        mat: &MaterialHandle,
        ray_type: RayType,
        prg: ProgramHandle,
    ) -> Result<()> {
        self.program_validate(&prg)?;

        let result = unsafe {
            rtMaterialSetAnyHitProgram(
                mat.borrow().rt_mat,
                ray_type.ray_type,
                prg.borrow().rt_prg,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtMaterialSetAnyHitProgram", result));
        }

        if let Some(matprg) = mat.borrow_mut().programs.get_mut(&ray_type) {
            matprg.any = Some(prg);
        } else {
            mat.borrow_mut().programs.insert(
                ray_type,
                MaterialProgram {
                    closest: None,
                    any: Some(prg),
                },
            );
        }

        Ok(())
    }

    pub fn material_set_closest_hit_program(
        &mut self,
        mat: &MaterialHandle,
        ray_type: RayType,
        prg: ProgramHandle,
    ) -> Result<()> {
        self.program_validate(&prg)?;

        let result = unsafe {
            rtMaterialSetClosestHitProgram(
                mat.borrow().rt_mat,
                ray_type.ray_type,
                prg.borrow().rt_prg,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtMaterialSetClosestHitProgram", result)
            );
        }

        if let Some(matprg) = mat.borrow_mut().programs.get_mut(&ray_type) {
            matprg.closest = Some(prg);
        } else {
            mat.borrow_mut().programs.insert(
                ray_type,
                MaterialProgram {
                    any: None,
                    closest: Some(prg),
                },
            );
        }

        Ok(())
    }
}
