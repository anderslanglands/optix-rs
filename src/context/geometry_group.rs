use crate::context::*;

use std::cell::RefCell;
use std::rc::Rc;

pub struct GeometryGroup {
    pub(crate) rt_geogrp: RTgeometrygroup,
    pub(crate) acc: AccelerationHandle,
    pub(crate) children: Vec<GeometryInstanceHandle>,
}

pub type GeometryGroupHandle = Rc<RefCell<GeometryGroup>>;

impl Context {
    pub fn geometry_group_create(
        &mut self,
        acc: AccelerationHandle,
        children: Vec<GeometryInstanceHandle>,
    ) -> Result<GeometryGroupHandle> {
        self.acceleration_validate(&acc)?;
        for child in &children {
            self.geometry_instance_validate(child)?;
        }

        let (rt_geogrp, result) = unsafe {
            let mut rt_geogrp: RTgeometrygroup = std::mem::zeroed();
            let result = rtGeometryGroupCreate(self.rt_ctx, &mut rt_geogrp);
            (rt_geogrp, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGeometryGroupCreate", result));
        }

        let result = unsafe {
            rtGeometryGroupSetAcceleration(rt_geogrp, acc.borrow().rt_acc)
        };
        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtGeometryGroupSetAcceleration", result)
            );
        }

        let result = unsafe {
            rtGeometryGroupSetChildCount(rt_geogrp, children.len() as u32)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGeometryGroupSetChildCount", result));
        }

        for (i, geoinst) in children.iter().enumerate() {
            let result = unsafe {
                rtGeometryGroupSetChild(
                    rt_geogrp,
                    i as u32,
                    geoinst.borrow().rt_geoinst,
                )
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtGeometryGroupSetChild", result));
            }
        }

        let geogrp = Rc::new(RefCell::new(GeometryGroup {
            rt_geogrp,
            acc,
            children,
        }));

        self.geometry_groups.push(Rc::clone(&geogrp));

        Ok(geogrp)
    }

    /*
    pub fn geometry_group_destroy(&mut self, geogrp: GeometryGroupHandle) {
        let rt_geogrp = self.ga_geometry_group_obj.remove(geogrp).unwrap();
        // acceleration will just dangle here
        let _acc = self.gd_geometry_group_acceleration.remove(geogrp).unwrap();
        let children = self.gd_geometry_group_children.remove(geogrp).unwrap();
        for child in children {
            self.geometry_instance_destroy(child);
        }

        if unsafe { rtGeometryGroupDestroy(rt_geogrp) } != RtResult::SUCCESS {
            panic!("Error destroying geometry_group {:?}", geogrp);
        }
    }
    */

    pub fn geometry_group_validate(
        &self,
        geogrp: &GeometryGroupHandle,
    ) -> Result<()> {
        let result =
            unsafe { rtGeometryGroupValidate(geogrp.borrow().rt_geogrp) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryGroupValidate", result))
        } else {
            Ok(())
        }
    }
}
