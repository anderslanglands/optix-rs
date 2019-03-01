use crate::context::*;

new_key_type! { pub struct GroupHandle; }

pub enum GroupChild {
    None,
    GeometryGroup(GeometryGroupHandle),
    Group(GroupHandle),
    // Selector(SelectorHandle),
    Transform(TransformHandle),
}

impl Context {
    pub fn group_create(
        &mut self,
        acc: AccelerationHandle,
        children: Vec<GroupChild>,
    ) -> Result<GroupHandle> {
        for child in &children {
            match child {
                GroupChild::GeometryGroup(h) => self.geometry_group_validate(*h)?,
                GroupChild::Group(h) => self.group_validate(*h)?,
                GroupChild::Transform(h) => self.transform_validate(*h)?,
                GroupChild::None => unreachable!(),
            }
        }

        let (rt_grp, result) = unsafe {
            let mut rt_grp: RTgroup = std::mem::zeroed();
            let result = rtGroupCreate(self.rt_ctx, &mut rt_grp);
            (rt_grp, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupCreate", result));
        }

        let grp = self.ga_group_obj.insert(rt_grp);

        // Add the acceleration
        let rt_acc = self.ga_acceleration_obj.get(acc).unwrap();
        let result = unsafe { rtGroupSetAcceleration(rt_grp, *rt_acc) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetAcceleration", result));
        }
        self.gd_group_acceleration.insert(grp, acc);

        // Add the children
        let result = unsafe { rtGroupSetChildCount(rt_grp, children.len() as u32) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetChildCount", result));
        }
        for (i, child) in children.iter().enumerate() {
            match child {
                GroupChild::GeometryGroup(h) => {
                    let c_rt_geogrp = *self.ga_geometry_group_obj.get(*h).unwrap();
                    let result =
                        unsafe { rtGroupSetChild(rt_grp, i as u32, c_rt_geogrp as RTobject) };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                }
                GroupChild::Group(h) => {
                    let c_rt_grp = *self.ga_group_obj.get(*h).unwrap();
                    let result = unsafe { rtGroupSetChild(rt_grp, i as u32, c_rt_grp as RTobject) };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                }
                GroupChild::Transform(h) => {
                    let c_rt_xform = *self.ga_transform_obj.get(*h).unwrap();
                    let result =
                        unsafe { rtGroupSetChild(rt_grp, i as u32, c_rt_xform as RTobject) };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                }
                GroupChild::None => unreachable!(),
            }
        }

        self.gd_group_children.insert(grp, children);

        Ok(grp)
    }

    pub fn group_add_child(&mut self, grp: GroupHandle, child: GroupChild) -> Result<()> {
        match child {
            GroupChild::GeometryGroup(h) => self.geometry_group_validate(h)?,
            GroupChild::Group(h) => self.group_validate(h)?,
            GroupChild::Transform(h) => self.transform_validate(h)?,
            GroupChild::None => unreachable!(),
        }

        let rt_grp = *self.ga_group_obj.get(grp).unwrap();
        let children = self.gd_group_children.get_mut(grp).unwrap();
        let index = children.len() as u32;

        let result = unsafe { rtGroupSetChildCount(rt_grp, index + 1) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetChildCount", result));
        }

        match child {
            GroupChild::GeometryGroup(h) => {
                let c_rt_geogrp = *self.ga_geometry_group_obj.get(h).unwrap();
                let result = unsafe { rtGroupSetChild(rt_grp, index, c_rt_geogrp as RTobject) };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtGroupSetChild", result));
                }
            }
            GroupChild::Group(h) => {
                let c_rt_grp = *self.ga_group_obj.get(h).unwrap();
                let result = unsafe { rtGroupSetChild(rt_grp, index, c_rt_grp as RTobject) };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtGroupSetChild", result));
                }
            }
            GroupChild::Transform(h) => {
                let c_rt_xform = *self.ga_transform_obj.get(h).unwrap();
                let result = unsafe { rtGroupSetChild(rt_grp, index, c_rt_xform as RTobject) };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtGroupSetChild", result));
                }
            }
            GroupChild::None => unreachable!(),
        }

        children.push(child);

        Ok(())
    }

    pub fn group_destroy(&mut self, grp: GroupHandle) {
        let rt_grp = self.ga_group_obj.remove(grp).unwrap();

        let _acc = self.gd_group_acceleration.remove(grp);

        let _children = self.gd_group_children.remove(grp);

        if unsafe { rtGroupDestroy(rt_grp) } != RtResult::SUCCESS {
            panic!("Error destroying group {:?}", grp);
        }
    }

    pub fn group_validate(&self, grp: GroupHandle) -> Result<()> {
        let rt_grp = *self.ga_group_obj.get(grp).unwrap();
        let result = unsafe { rtGroupValidate(rt_grp) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGroupValidate", result))
        } else {
            Ok(())
        }
    }
}
