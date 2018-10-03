use std::collections::HashMap;
use crate::context::*;
use crate::ginallocator::*;

#[derive(Default, Copy, Clone)]
pub struct GeometryInstanceMarker;
impl Marker for GeometryInstanceMarker {
    const ID: &'static str = "GeometryInstance";
}
pub type GeometryInstanceHandle = Handle<GeometryInstanceMarker>;

impl Context {
    pub fn create_geometry_instance(
        &mut self,
        geometry: GeometryHandle,
        materials: Vec<MaterialHandle>,
    ) -> GeometryInstanceHandle {
        let obj = std::ptr::null_mut();
        let vars = HashMap::<String, Variable>::new();

        let hnd = self.ga_geometry_instance_obj.insert(obj);
        let chnd = self.ga_geometry_instance_obj.check_handle(hnd).unwrap();
        self.gd_geometry_instance_variables.insert(&chnd, vars);
        self.ga_geometry_obj.incref(geometry);
        self.gd_geometry_instance_geometry.insert(&chnd, geometry);
        for mat in &materials {
            self.ga_material_obj.incref(*mat);
        }
        self.gd_geometry_instance_materials.insert(&chnd, materials);

        hnd
    }
}