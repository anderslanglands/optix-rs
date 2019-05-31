use crate::context::*;
use crate::nvrtc;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::io::Read;
use std::rc::Rc;

pub struct Program {
    pub(crate) rt_prg: RTprogram,
    pub(crate) variables: HashMap<String, VariableHandle>,
}

pub type ProgramHandle = Rc<RefCell<Program>>;

pub struct ProgramID {
    pub prg: ProgramHandle,
    pub id: i32,
}

impl Clone for ProgramID {
    fn clone(&self) -> ProgramID {
        ProgramID {
            prg: Rc::clone(&self.prg),
            id: self.id,
        }
    }
}

impl Drop for ProgramID {
    fn drop(&mut self) {
        println!("DROPPING PROGRAM ID {}", self.id);
    }
}

impl From<ProgramHandle> for ProgramID {
    fn from(prg: ProgramHandle) -> ProgramID {
        unsafe {
            let mut id: i32 = 0;
            let result = rtProgramGetId(prg.borrow().rt_prg, &mut id);
            if result == RtResult::SUCCESS {
                ProgramID { prg, id }
            } else {
                panic!("Could nto convert handle to id");
            }
        }
    }
}

impl Context {
    /// Destroys the Program referred to by `prg` and all its attached objects.
    /// The underlying program will remain alive until all references to it
    /// have been destroyed.
    /*
    pub fn program_destroy(&mut self, prg: ProgramHandle) {
        let _vars = self.gd_program_variables.remove(prg);

        let rt_prg = self.ga_program_obj.remove(prg).unwrap();
        if unsafe { rtProgramDestroy(rt_prg) } != RtResult::SUCCESS {
            panic!("Error destroying program {:?}", prg);
        }
    }
    */

    fn program_create_from_obj(&mut self, rt_prg: RTprogram) -> ProgramHandle {
        let variables = HashMap::<String, VariableHandle>::new();
        let prg = Rc::new(RefCell::new(Program { rt_prg, variables }));
        self.programs.push(Rc::clone(&prg));
        prg
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
    pub fn program_validate(&mut self, handle: &ProgramHandle) -> Result<()> {
        unsafe {
            let result = rtProgramValidate(handle.borrow().rt_prg);
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
        prg: &ProgramHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        // we have to remove the variable first so that we're not holding a borrow
        // further down here
        let ex_var = prg.borrow_mut().variables.remove(name);
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
            prg.borrow_mut().variables.insert(name.into(), ex_var);
        } else {
            let (rt_var, result) = unsafe {
                let mut rt_var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtProgramDeclareVariable(
                    prg.borrow_mut().rt_prg,
                    c_name.as_ptr(),
                    &mut rt_var,
                );
                (rt_var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtProgramDeclareVariable", result)
                );
            }

            let var =
                Rc::new(RefCell::new(data.set_optix_variable(self, rt_var)?));
            prg.borrow_mut()
                .variables
                .insert(name.into(), Rc::clone(&var));

            // if it's a new variable, push it to the context storage
            self.variables.push(var)
        }

        Ok(())
    }

    /// Set the Variable referred to by `name` to the given `data`. Any objects
    /// previously assigned to the variable will be destroyed.
    pub fn program_set_user_variable(
        &mut self,
        prg: &ProgramHandle,
        name: &str,
        data: Box<dyn UserVariable>,
    ) -> Result<()> {
        // check if the variable exists first
        if let Some(ex_var) = prg.borrow_mut().variables.remove(name) {
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
            prg.borrow_mut().variables.insert(name.into(), ex_var);
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

            prg.borrow_mut()
                .variables
                .insert(name.into(), Rc::clone(&var));

            // if it's a new variable, push it to the context storage
            self.variables.push(var)
        }

        Ok(())
    }

    pub fn program_get_id(
        &mut self,
        handle: &ProgramHandle,
    ) -> Result<ProgramID> {
        unsafe {
            let mut id: i32 = 0;
            let result = rtProgramGetId(handle.borrow().rt_prg, &mut id);
            if result == RtResult::SUCCESS {
                Ok(ProgramID {
                    prg: Rc::clone(handle),
                    id,
                })
            } else {
                Err(self.optix_error("rtProgramGetId", result))
            }
        }
    }
}
