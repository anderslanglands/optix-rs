use crate::context::*;
use crate::ginallocator::*;

#[derive(Default, Debug, Copy, Clone)]
pub struct GeometryGroupMarker;
impl Marker for GeometryGroupMarker {
    const ID: &'static str = "GeometryGroup";
}
pub type GeometryGroupHandle = Handle<GeometryGroupMarker>;

impl Context {
    pub fn geometry_group_create(
        &mut self,
        acc: AccelerationHandle,
        children: Vec<GeometryInstanceHandle>,
    ) -> Result<GeometryGroupHandle> {
        self.acceleration_validate(acc)?;
        for child in &children {
            self.geometry_instance_validate(*child)?;
        }

        let (rt_geogrp, result) = unsafe {
            let mut rt_geogrp: RTgeometrygroup = std::mem::zeroed();
            let result = rtGeometryGroupCreate(self.rt_ctx, &mut rt_geogrp);
            (rt_geogrp, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGeometryGroupCreate", result));
        }

        let geogrp = self.ga_geometry_group_obj.insert(rt_geogrp);
        let cgeogrp = self.ga_geometry_group_obj.check_handle(geogrp).unwrap();

        let rt_acc = *self.ga_acceleration_obj.get(acc).unwrap();
        let result =
            unsafe { rtGeometryGroupSetAcceleration(rt_geogrp, rt_acc) };
        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtGeometryGroupSetAcceleration", result)
            );
        } else {
            self.gd_geometry_group_acceleration.insert(&cgeogrp, acc);
            self.ga_acceleration_obj.incref(acc);
        }

        let result = unsafe {
            rtGeometryGroupSetChildCount(rt_geogrp, children.len() as u32)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGeometryGroupSetChildCount", result));
        }

        for (i, geoinst) in children.iter().enumerate() {
            let rt_geoinst =
                *self.ga_geometry_instance_obj.get(*geoinst).unwrap();
            let result = unsafe {
                rtGeometryGroupSetChild(rt_geogrp, i as u32, rt_geoinst)
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtGeometryGroupSetChild", result));
            } else {
                self.ga_geometry_instance_obj.incref(*geoinst);
            }
        }
        self.gd_geometry_group_children.insert(&cgeogrp, children);

        Ok(geogrp)
    }

    pub fn geometry_group_destroy(&mut self, geogrp: GeometryGroupHandle) {
        let rt_geogrp = *self.ga_geometry_group_obj.get(geogrp).unwrap();
        let cgeogrp = self.ga_geometry_group_obj.check_handle(geogrp).unwrap();
        let acc = self.gd_geometry_group_acceleration.get(cgeogrp);
        self.acceleration_destroy(*acc);
        let children = self.gd_geometry_group_children.remove(cgeogrp);
        for child in children {
            self.geometry_instance_destroy(child);
        }

        match self.ga_geometry_group_obj.destroy(geogrp) {
            DestroyResult::StillAlive => (),
            DestroyResult::ShouldDrop => {
                if unsafe { rtGeometryGroupDestroy(rt_geogrp) }
                    != RtResult::SUCCESS
                {
                    panic!("Error destroying geometry_group {}", geogrp);
                }
            }
        }
    }

    pub fn geometry_group_validate(
        &self,
        geogrp: GeometryGroupHandle,
    ) -> Result<()> {
        let rt_geogrp = *self.ga_geometry_group_obj.get(geogrp).unwrap();
        let result = unsafe { rtGeometryGroupValidate(rt_geogrp) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryGroupValidate", result))
        } else {
            Ok(())
        }
    }
}
