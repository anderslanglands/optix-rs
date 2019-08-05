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
            return Err(
                self.optix_error("rtGeometryGroupSetChildCount", result)
            );
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

    pub fn geometry_group_set_acceleration(
        &mut self,
        geogrp: &GeometryGroupHandle,
        accel: AccelerationHandle,
    ) -> Result<()> {
        self.acceleration_validate(&accel)?;

        let result = unsafe {
            rtGeometryGroupSetAcceleration(
                geogrp.borrow().rt_geogrp,
                accel.borrow().rt_acc,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtGeometryGroupSetAcceleration", result)
            );
        }

        geogrp.borrow_mut().acc = accel;

        Ok(())
    }

    pub fn geometry_group_set_children(
        &mut self,
        geogrp: &GeometryGroupHandle,
        children: Vec<GeometryInstanceHandle>,
    ) -> Result<()> {
        for child in &children {
            self.geometry_instance_validate(child)?;
        }

        let result = unsafe {
            rtGeometryGroupSetChildCount(
                geogrp.borrow().rt_geogrp,
                children.len() as u32,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtGeometryGroupSetChildCount", result)
            );
        }

        for (i, geoinst) in children.iter().enumerate() {
            let result = unsafe {
                rtGeometryGroupSetChild(
                    geogrp.borrow().rt_geogrp,
                    i as u32,
                    geoinst.borrow().rt_geoinst,
                )
            };
            if result != RtResult::SUCCESS {
                return Err(self.optix_error("rtGeometryGroupSetChild", result));
            }
        }

        geogrp.borrow_mut().children = children;

        Ok(())
    }

    pub fn geometry_group_set_visibility_mask(
        &mut self,
        geogrp: &GeometryGroupHandle,
        visibility: u32,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometryGroupSetVisibilityMask(
                geogrp.borrow().rt_geogrp,
                visibility,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtGeometryGroupSetVisibilityMask", result)
            );
        }

        Ok(())
    }
}
