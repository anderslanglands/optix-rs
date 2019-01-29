use crate::context::*;

use slotmap::*;

new_key_type! { pub struct GeometryGroupHandle; }

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

        let rt_acc = self.ga_acceleration_obj.get(acc).unwrap();
        let result = unsafe { rtGeometryGroupSetAcceleration(rt_geogrp, *rt_acc) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGeometryGroupSetAcceleration", result));
        } else {
            self.gd_geometry_group_acceleration.insert(geogrp, acc);
        }

        let result = unsafe { rtGeometryGroupSetChildCount(rt_geogrp, children.len() as u32) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGeometryGroupSetChildCount", result));
        }

        for (i, geoinst) in children.iter().enumerate() {
            let rt_geoinst = self.ga_geometry_instance_obj.get(*geoinst).unwrap();
            let result = unsafe { rtGeometryGroupSetChild(rt_geogrp, i as u32, *rt_geoinst) };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtGeometryGroupSetChild", result));
            }
        }
        self.gd_geometry_group_children.insert(geogrp, children);

        Ok(geogrp)
    }

    pub fn geometry_group_destroy(&mut self, geogrp: GeometryGroupHandle) {
        let rt_geogrp = self.ga_geometry_group_obj.remove(geogrp).unwrap();
        // acceleration will just dangle here
        let acc = self.gd_geometry_group_acceleration.remove(geogrp).unwrap();
        let children = self.gd_geometry_group_children.remove(geogrp).unwrap();
        for child in children {
            self.geometry_instance_destroy(child);
        }

        if unsafe { rtGeometryGroupDestroy(rt_geogrp) } != RtResult::SUCCESS {
            panic!("Error destroying geometry_group {:?}", geogrp);
        }
    }

    pub fn geometry_group_validate(&self, geogrp: GeometryGroupHandle) -> Result<()> {
        let rt_geogrp = *self.ga_geometry_group_obj.get(geogrp).unwrap();
        let result = unsafe { rtGeometryGroupValidate(rt_geogrp) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryGroupValidate", result))
        } else {
            Ok(())
        }
    }
}
