//! Materials contain the Programs that decide what to do when a ray/primitive
//! intersection is found.

use crate::context::*;
use std::collections::HashMap;

new_key_type! { pub struct MaterialHandle; }

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
            if let Some(prg) = program.closest {
                self.program_validate(prg)?
            };
            if let Some(prg) = program.any {
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
                if let Some(prg) = program.closest {
                    let rt_prg = self.ga_program_obj.get(prg).unwrap();
                    let result =
                        unsafe { rtMaterialSetClosestHitProgram(mat, raytype.ray_type, *rt_prg) };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtMaterialSetClosestHitProgram", result));
                    }
                }
                if let Some(prg) = program.any {
                    let rt_prg = self.ga_program_obj.get(prg).unwrap();
                    let result =
                        unsafe { rtMaterialSetAnyHitProgram(mat, raytype.ray_type, *rt_prg) };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtMaterialSetAnyHitProgram", result));
                    }
                }
            }

            let vars = HashMap::<String, Variable>::new();
            let hnd = self.ga_material_obj.insert(mat);
            self.gd_material_variables.insert(hnd, vars);
            self.gd_material_programs.insert(hnd, programs);

            Ok(hnd)
        }
    }

    /// Destroys this Material an all objects attached to it. Note that the
    /// Material will not actually be destroyed until all references to it from
    /// other scene graph objects are released.
    /// # Panics
    /// If mat is not a valid MaterialHandle
    pub fn material_destroy(&mut self, mat: MaterialHandle) {
        let _vars = self.gd_material_variables.remove(mat);

        // destroy material programs
        let _programs = self.gd_material_programs.remove(mat);

        let rt_mat = self.ga_material_obj.remove(mat).unwrap();
        if unsafe { rtMaterialDestroy(rt_mat) } != RtResult::SUCCESS {
            panic!("Error destroying material {:?}", mat);
        }
    }

    /// Check that the Material and all objects attached to it are correctly
    /// set up.
    /// # Panics
    /// If mat is not a valid MaterialHandle
    pub fn material_validate(&self, mat: MaterialHandle) -> Result<()> {
        let rt_mat = self.ga_material_obj.get(mat).unwrap();
        let result = unsafe { rtMaterialValidate(*rt_mat) };
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
        mat: MaterialHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        let rt_mat = self.ga_material_obj.get(mat).unwrap();

        if let Some(old_variable) = self.gd_material_variables.get(mat).unwrap().get(name) {
            let var = match old_variable {
                Variable::Pod(vp) => vp.var,
                Variable::Object(vo) => vo.var,
            };
            let new_variable = data.set_optix_variable(self, var)?;
            // destroy any resources the existing variable holds
            self.gd_material_variables
                .get_mut(mat)
                .unwrap()
                .insert(name.to_owned(), new_variable);

            Ok(())
        } else {
            let (var, result) = unsafe {
                let mut var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtMaterialDeclareVariable(*rt_mat, c_name.as_ptr(), &mut var);
                (var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtMaterialDeclareVariable", result));
            }

            let variable = data.set_optix_variable(self, var)?;
            self.gd_material_variables
                .get_mut(mat)
                .unwrap()
                .insert(name.to_owned(), variable);

            Ok(())
        }
    }
}
