use crate::context::*;
use std::collections::HashMap;
use std::ffi::CString;
use std::io::Read;
use crate::nvrtc;

new_key_type! { pub struct ProgramHandle; }

impl Context {
    /// Destroys the Program referred to by `prg` and all its attached objects.
    /// The underlying program will remain alive until all references to it
    /// have been destroyed.
    pub fn program_destroy(&mut self, prg: ProgramHandle) {
        let _vars = self.gd_program_variables.remove(prg);

        let rt_prg = self.ga_program_obj.remove(prg).unwrap();
        if unsafe { rtProgramDestroy(rt_prg) } != RtResult::SUCCESS {
            panic!("Error destroying program {:?}", prg);
        }
    }

    fn program_create_from_obj(&mut self, rt_prg: RTprogram) -> ProgramHandle {
        let vars = HashMap::<String, Variable>::new();
        let hnd = self.ga_program_obj.insert(rt_prg);
        self.gd_program_variables.insert(hnd, vars);

        hnd
    }

    /// Create a new Program from the PTX in `ptx_str`, with the given
    /// `entry_point`.  
    pub fn program_create_from_ptx_string(
        &mut self,
        ptx_str: &str,
        entry_point_name: &str,
    ) -> Result<ProgramHandle> {
        let c_ptx = CString::new(ptx_str);
        if let Err(nerr) = c_ptx {
            return Err(Error::NulError(nerr.nul_position()));
        }
        let c_ptx = c_ptx.unwrap();
        let c_entry_point_name = CString::new(entry_point_name)?;
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

    /// Create a new Program by loading the given `cuda_file` and compiling it
    /// with nvrtc
    pub fn program_create_from_cuda_file(
        &mut self,
        cuda_file: &str,
        entry_point_name: &str,
        compile_options: &Vec<String>,
    ) -> Result<ProgramHandle> {
        let mut cuda_str = String::new();
        self.search_path
            .open(cuda_file)?
            .read_to_string(&mut cuda_str)?;

        let mut prg = nvrtc::Program::new(&cuda_str, entry_point_name, vec![])?;
        prg.compile_program(compile_options)?;
        let ptx_str = prg.get_ptx()?;

        self.program_create_from_ptx_string(&ptx_str, entry_point_name)
    }

    /// Validate the Program. If the program referred to by `handle` or any of
    /// its attached objects are invalid, return an OptixError.
    pub fn program_validate(&mut self, handle: ProgramHandle) -> Result<()> {
        let rt_prg = self.ga_program_obj.get(handle).expect(&format!(
            "Tried to validate an invalid handle: {:?}",
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
        let rt_prg = self.ga_program_obj.get(prg).unwrap();

        if let Some(old_variable) = self.gd_program_variables.get(prg).unwrap().get(name) {
            let var = match old_variable {
                Variable::Pod(vp) => vp.var,
                Variable::Object(vo) => vo.var,
            };
            let new_variable = data.set_optix_variable(self, var)?;
            // destroy any resources the existing variable holds
            self.gd_program_variables
                .get_mut(prg)
                .unwrap()
                .insert(name.to_owned(), new_variable);

            Ok(())
        } else {
            let (var, result) = unsafe {
                let mut var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtProgramDeclareVariable(*rt_prg, c_name.as_ptr(), &mut var);
                (var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtProgramDeclareVariable", result));
            }

            let variable = data.set_optix_variable(self, var)?;
            self.gd_program_variables
                .get_mut(prg)
                .unwrap()
                .insert(name.to_owned(), variable);

            Ok(())
        }
    }
}
