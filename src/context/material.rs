use crate::context::*;
use crate::ginallocator::*;
use std::collections::HashMap;

#[derive(Default, Copy, Clone)]
pub struct MaterialMarker;
impl Marker for MaterialMarker {
    const ID: &'static str = "Material";
}
pub type MaterialHandle = Handle<MaterialMarker>;

pub enum MaterialProgram {
    ClosestHit(ProgramHandle),
    AnyHit(ProgramHandle),
}

impl Context {
    pub fn material_create(
        &mut self,
        programs: HashMap<RayType, MaterialProgram>,
    ) -> Result<MaterialHandle> {
        // First check that the programs are well-defined
        for (_, program) in &programs {
            match program {
                MaterialProgram::ClosestHit(prg) => self.program_validate(*prg)?,
                MaterialProgram::AnyHit(prg) => self.program_validate(*prg)?,
            }
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
                match program {
                    MaterialProgram::ClosestHit(prg) => {
                        let rt_prg = self.ga_program_obj.get(*prg).unwrap();
                        let result = unsafe {
                            rtMaterialSetClosestHitProgram(mat, raytype.ray_type, *rt_prg)
                        };
                        if result != RtResult::SUCCESS {
                            return Err(
                                self.optix_error("rtMaterialSetClosestHitProgram", result)
                            );
                        } else {
                            self.ga_program_obj.incref(*prg);
                        }
                    }
                    MaterialProgram::AnyHit(prg) => {
                        let rt_prg = self.ga_program_obj.get(*prg).unwrap();
                        let result = unsafe {
                            rtMaterialSetAnyHitProgram(mat, raytype.ray_type, *rt_prg)
                        };
                        if result != RtResult::SUCCESS {
                            return Err(
                                self.optix_error("rtMaterialSetAnyHitProgram", result)
                            );
                        } else {
                            self.ga_program_obj.incref(*prg);
                        }
                    }
                }
            }

            let vars = HashMap::<String, Variable>::new();
            let hnd = self.ga_material_obj.insert(mat);
            let chnd = self.ga_material_obj.check_handle(hnd).unwrap();
            self.gd_material_variables.insert(&chnd, vars);
            self.gd_material_programs.insert(&chnd, programs);

            Ok(hnd)
        }
    }

    ///
    pub fn material_destroy(&mut self, mat: MaterialHandle) {
        if let Some(cmat) = self.ga_material_obj.check_handle(mat) {

            let vars = self.gd_material_variables.remove(cmat);
            self.destroy_variables(vars);

            // destroy material programs
            let programs = self.gd_material_programs.remove(cmat);
            for (_, program) in programs {
                match program {
                    MaterialProgram::ClosestHit(prg) => self.program_destroy(prg),
                    MaterialProgram::AnyHit(prg) => self.program_destroy(prg),
                }
            }

            let rt_mat = *self.ga_material_obj.get(mat).unwrap();
            match self.ga_material_obj.destroy(mat) {
                DestroyResult::StillAlive => (),
                DestroyResult::ShouldDrop => {
                    if unsafe { rtMaterialDestroy(rt_mat) } != RtResult::SUCCESS
                    {
                        panic!("Error destroying material {}", mat);
                    }
                }
            }
        } else {
            panic!("Tried to destroy an invalid MaterialHandle: {}", mat);
        }
    }

    pub fn material_validate(&self, mat: MaterialHandle) -> Result<()> {
        let rt_mat = self.ga_material_obj.get(mat).unwrap();
        let result = unsafe { rtMaterialValidate(*rt_mat) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtMaterialValidate", result))
        } else {
            Ok(())
        }
    }

    pub fn material_set_variable<T: VariableStorable>(
        &mut self,
        mat: MaterialHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        let cmat = self
            .ga_material_obj
            .check_handle(mat)
            .expect("Tried to access an invalid material handle");
        let rt_mat = self.ga_material_obj.get(mat).unwrap();

        if let Some(old_variable) =
            self.gd_material_variables.get(cmat).get(name)
        {
            let var = match old_variable {
                Variable::Pod(vp) => vp.var,
                Variable::Object(vo) => vo.var,
            };
            let new_variable = data.set_optix_variable(self, var)?;
            // destroy any resources the existing variable holds
            if let Some(old_variable) = self
                .gd_material_variables
                .get_mut(cmat)
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
                let result = rtMaterialDeclareVariable(
                    *rt_mat,
                    c_name.as_ptr(),
                    &mut var,
                );
                (var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtMaterialDeclareVariable", result)
                );
            }

            let variable = data.set_optix_variable(self, var)?;
            self.gd_material_variables
                .get_mut(cmat)
                .insert(name.to_owned(), variable);

            Ok(())
        }
    }
}
