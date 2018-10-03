use std::collections::HashMap;
use crate::context::*;
use crate::ginallocator::*;

#[derive(Default, Copy, Clone)]
pub struct MaterialMarker;
impl Marker for MaterialMarker {
    const ID: &'static str = "Material";
}
pub type MaterialHandle = Handle<MaterialMarker>;

impl Context {
    pub fn create_material(
        &mut self,
        prg_any_hit: HashMap<RayType, ProgramHandle>,
        prg_closest_hit: HashMap<RayType, ProgramHandle>,
    ) -> MaterialHandle {
        let obj = std::ptr::null_mut();
        let vars = HashMap::<String, Variable>::new();

        let hnd = self.ga_material_obj.insert(obj);
        let chnd = self.ga_material_obj.check_handle(hnd).unwrap();
        self.gd_material_variables.insert(&chnd, vars);
        for (_, prg) in &prg_any_hit {
            self.ga_program_obj.incref(*prg);
        }
        self.gd_material_any_hit.insert(&chnd, prg_any_hit);
        for (_, prg) in &prg_closest_hit {
            self.ga_program_obj.incref(*prg);
        }
        self.gd_material_closest_hit.insert(&chnd, prg_closest_hit);

        hnd
    }

    ///
    pub fn material_destroy(&mut self, mat: MaterialHandle) {
        if let Some(cmat) = self.ga_material_obj.check_handle(mat) {
            // destroy material programs
            let prg_any_hit = self.gd_material_any_hit.remove(cmat);
            for (_, prg) in prg_any_hit {
                self.program_destroy(prg);
            }
            let prg_closest_hit = self.gd_material_closest_hit.remove(cmat);
            for (_, prg) in prg_closest_hit {
                self.program_destroy(prg);
            }
            let rt_mat = *self.ga_material_obj.get(mat).unwrap();
            match self.ga_material_obj.destroy(mat) {
                DestroyResult::StillAlive => (),
                DestroyResult::ShouldDrop => {
                    if unsafe {
                        rtMaterialDestroy(rt_mat)
                    } != RtResult::SUCCESS {
                        panic!("Error destroying program {}", mat);
                    }
                }
            }
        } else {
            panic!("Tried to destroy an invalid MaterialHandle: {}", mat);
        }
    }
}
