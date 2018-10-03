use std::collections::HashMap;
use crate::context::*;
use crate::ginallocator::*;

#[derive(Default, Copy, Clone)]
pub struct GeometryMarker;
impl Marker for GeometryMarker {
    const ID: &'static str = "Geometry";
}
pub type GeometryHandle = Handle<GeometryMarker>;

impl Context {
    pub fn create_geometry(
        &mut self,
        prg_bounding_box: ProgramHandle,
        prg_intersection: ProgramHandle,
    ) -> GeometryHandle {
        let obj = std::ptr::null_mut();
        let vars = HashMap::<String, Variable>::new();

        let hnd = self.ga_geometry_obj.insert(obj);
        let chnd = self.ga_geometry_obj.check_handle(hnd).unwrap();
        self.gd_geometry_variables.insert(&chnd, vars);
        self.ga_program_obj.incref(prg_bounding_box);
        self.gd_geometry_bounding_box
            .insert(&chnd, prg_bounding_box);
        self.ga_program_obj.incref(prg_intersection);
        self.gd_geometry_intersection
            .insert(&chnd, prg_intersection);

        hnd
    }
}
