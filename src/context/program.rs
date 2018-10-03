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
    pub fn program_destroy(&mut self, prg: ProgramHandle) {
        let rt_prg = *self.ga_program_obj.get(prg).unwrap();
        match self.ga_program_obj.destroy(prg) {
            DestroyResult::StillAlive => (),
            DestroyResult::ShouldDrop => {
                if unsafe {
                    rtProgramDestroy(rt_prg)
                } != RtResult::SUCCESS {
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

    pub fn program_validate(&mut self, handle: ProgramHandle) -> Result<()> {
        let rt_prg = self
            .ga_program_obj
            .get(handle)
            .expect(&format!("Tried to validate an invalid handle: {}", handle));
        unsafe {
            let result = rtProgramValidate(*rt_prg);
            if result == RtResult::SUCCESS {
                Ok(())
            } else {
                Err(self.optix_error("rtProgramValidate", result))
            }
        }
    }
}
