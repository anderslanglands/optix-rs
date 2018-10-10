use crate::context::*;
use crate::ginallocator::*;

#[derive(Default, Debug, Copy, Clone)]
pub struct GroupMarker;
impl Marker for GroupMarker {
    const ID: &'static str = "Group";
}
pub type GroupHandle = Handle<GroupMarker>;

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
                GroupChild::GeometryGroup(h) => {
                    self.geometry_group_validate(*h)?
                },
                GroupChild::Group(h) => {
                    self.group_validate(*h)?
                },
                GroupChild::Transform(h) => {
                    self.transform_validate(*h)?
                },
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
        let cgrp = self.ga_group_obj.check_handle(grp).unwrap();

        // Add the acceleration
        let rt_acc = self.ga_acceleration_obj.get(acc).unwrap();
        let result = unsafe {
            rtGroupSetAcceleration(rt_grp, *rt_acc)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetAcceleration", result));
        }
        self.ga_acceleration_obj.incref(acc);
        self.gd_group_acceleration.insert(&cgrp, acc);

        // Add the children
        let result = unsafe {
            rtGroupSetChildCount(rt_grp, children.len() as u32)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetChildCount", result));
        }
        for (i, child) in children.iter().enumerate() {
            match child {
                GroupChild::GeometryGroup(h) => {
                    let c_rt_geogrp = *self.ga_geometry_group_obj.get(*h).unwrap();
                    let result = unsafe {
                        rtGroupSetChild(rt_grp, i as u32, c_rt_geogrp as RTobject)
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                    self.ga_geometry_group_obj.incref(*h);
                },
                GroupChild::Group(h) => {
                    let c_rt_grp = *self.ga_group_obj.get(*h).unwrap();
                    let result = unsafe {
                        rtGroupSetChild(rt_grp, i as u32, c_rt_grp as RTobject)
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                    self.ga_group_obj.incref(*h);
                },
                GroupChild::Transform(h) => {
                    let c_rt_xform = *self.ga_transform_obj.get(*h).unwrap();
                    let result = unsafe {
                        rtGroupSetChild(rt_grp, i as u32, c_rt_xform as RTobject)
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                    self.ga_transform_obj.incref(*h);
                },
                GroupChild::None => unreachable!(),
            }
        }

        self.gd_group_children.insert(&cgrp, children);

        Ok(grp)
    }

    pub fn group_destroy(&mut self, grp: GroupHandle) {
        let rt_grp = *self.ga_group_obj.get(grp).unwrap();
        let cgrp = self.ga_group_obj.check_handle(grp).unwrap();
        
        let acc = self.gd_group_acceleration.get(cgrp);
        self.acceleration_destroy(*acc);

        let children = self.gd_group_children.remove(cgrp);
        for child in children {
            match child {
                GroupChild::GeometryGroup(h) => {
                    self.geometry_group_destroy(h);
                },
                GroupChild::Group(h) => {
                    self.group_destroy(h);
                },
                GroupChild::Transform(h) => {
                    self.transform_destroy(h);
                },
                GroupChild::None => unreachable!(),
            }
        }

        match self.ga_group_obj.destroy(grp) {
            DestroyResult::StillAlive => (),
            DestroyResult::ShouldDrop => {
                if unsafe { rtGroupDestroy(rt_grp) } != RtResult::SUCCESS {
                    panic!("Error destroying group {}", grp);
                }
            }   
        }
    }

    pub fn group_validate(&self, grp: GroupHandle) -> Result<()> {
        let rt_grp = *self.ga_group_obj.get(grp).unwrap();
        let result = unsafe {
            rtGroupValidate(rt_grp)
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGroupValidate", result))
        } else {
            Ok(())
        }
    }
}

