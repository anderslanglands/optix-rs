use crate::context::*;
use crate::ginallocator::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::io::Read;

#[derive(Default, Debug, Copy, Clone)]
pub struct ProgramMarker;
impl Marker for ProgramMarker {
    const ID: &'static str = "Program";
}
pub type ProgramHandle = Handle<ProgramMarker>;

impl Context {
    /// Destroys the Program referred to by `prg` and all its attached objects.
    /// The underlying program will remain alive until all references to it
    /// have been destroyed.
    pub fn program_destroy(&mut self, prg: ProgramHandle) {
        let cprg = self.ga_program_obj.check_handle(prg).unwrap();
        let vars = self.gd_program_variables.remove(cprg);
        self.destroy_variables(vars);

        let rt_prg = *self.ga_program_obj.get(prg).unwrap();
        match self.ga_program_obj.destroy(prg) {
            DestroyResult::StillAlive => (),
            DestroyResult::ShouldDrop => {
                if unsafe { rtProgramDestroy(rt_prg) } != RtResult::SUCCESS {
                    panic!("Error destroying program {}", prg);
                }
            }
        }
    }

    fn program_create_from_obj(&mut self, rt_prg: RTprogram) -> ProgramHandle {
        let vars = HashMap::<String, Variable>::new();
        let hnd = self.ga_program_obj.insert(rt_prg);
        let chnd = self.ga_program_obj.check_handle(hnd).unwrap();
        self.gd_program_variables.insert(&chnd, vars);

        hnd
    }

    /// Create a new Program from the PTX in `ptx_str`, with the given
    /// `entry_point`.  
    pub fn program_create_from_ptx_string(
        &mut self,
        ptx_str: &str,
        entry_point_name: &str,
    ) -> Result<ProgramHandle> {
        let c_ptx = CString::new(ptx_str).unwrap();
        let c_entry_point_name = CString::new(entry_point_name).unwrap();
        let (prg, result) = unsafe {
            let mut prg: RTprogram = std::mem::zeroed();
            let result = rtProgramCreateFromPTXString(
                self.rt_ctx,
                c_ptx.as_ptr(),
                c_entry_point_name.as_ptr(),
                &mut prg,
            );
            (prg, result)
        };

        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtProgramCreateFromPTXString", result))
        } else {
            Ok(self.program_create_from_obj(prg))
        }
    }

    /// Create a new Program by loading the given `ptx_file` and passing its
    /// contents to `program_create_from_ptx_string`.
    pub fn program_create_from_ptx_file(
        &mut self,
        ptx_file: &str,
        entry_point_name: &str,
    ) -> Result<ProgramHandle> {
        let mut ptx_str = String::new();
        self.search_path
            .open(ptx_file)?
            .read_to_string(&mut ptx_str)?;
        self.program_create_from_ptx_string(&ptx_str, entry_point_name)
    }

    /// Validate the Program. If the program referred to by `handle` or any of
    /// its attached objects are invalid, return an OptixError.
    pub fn program_validate(&mut self, handle: ProgramHandle) -> Result<()> {
        let rt_prg = self.ga_program_obj.get(handle).expect(&format!(
            "Tried to validate an invalid handle: {}",
            handle
        ));
        unsafe {
            let result = rtProgramValidate(*rt_prg);
            if result == RtResult::SUCCESS {
                Ok(())
            } else {
                Err(self.optix_error("rtProgramValidate", result))
            }
        }
    }

    /// Set the Variable referred to by `name` to the given `data`. Any objects
    /// previously assigned to the variable will be destroyed.
    pub fn program_set_variable<T: VariableStorable>(
        &mut self,
        prg: ProgramHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        let cprg = self
            .ga_program_obj
            .check_handle(prg)
            .expect("Tried to access an invalid program handle");
        let rt_prg = self.ga_program_obj.get(prg).unwrap();

        if let Some(old_variable) =
            self.gd_program_variables.get(cprg).get(name)
        {
            let var = match old_variable {
                Variable::Pod(vp) => vp.var,
                Variable::Object(vo) => vo.var,
            };
            let new_variable = data.set_optix_variable(self, var)?;
            // destroy any resources the existing variable holds
            if let Some(old_variable) = self
                .gd_program_variables
                .get_mut(cprg)
                .insert(name.to_owned(), new_variable)
            {
                self.destroy_variable(old_variable);
            };

            Ok(())

        } else {
            let (var, result) = unsafe {
                let mut var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtProgramDeclareVariable(
                    *rt_prg,
                    c_name.as_ptr(),
                    &mut var,
                );
                (var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtProgramDeclareVariable", result));
            }

            let variable = data.set_optix_variable(self, var)?;
            self.gd_program_variables
                .get_mut(cprg)
                .insert(name.to_owned(), variable);

            Ok(())
        }
    }
}
