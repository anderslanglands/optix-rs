use crate::context::*;

use std::cell::RefCell;
use std::rc::Rc;

pub struct Group {
    pub(crate) rt_grp: RTgroup,
    pub(crate) acc: AccelerationHandle,
    pub(crate) children: Vec<GroupChild>,
}

pub type GroupHandle = Rc<RefCell<Group>>;

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
                    self.geometry_group_validate(h)?
                }
                GroupChild::Group(h) => self.group_validate(h)?,
                GroupChild::Transform(h) => self.transform_validate(h)?,
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

        // Add the acceleration
        let result =
            unsafe { rtGroupSetAcceleration(rt_grp, acc.borrow().rt_acc) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetAcceleration", result));
        }

        // Add the children
        let result =
            unsafe { rtGroupSetChildCount(rt_grp, children.len() as u32) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetChildCount", result));
        }
        for (i, child) in children.iter().enumerate() {
            match child {
                GroupChild::GeometryGroup(h) => {
                    let result = unsafe {
                        rtGroupSetChild(
                            rt_grp,
                            i as u32,
                            h.borrow().rt_geogrp as RTobject,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                }
                GroupChild::Group(h) => {
                    let result = unsafe {
                        rtGroupSetChild(
                            rt_grp,
                            i as u32,
                            h.borrow().rt_grp as RTobject,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                }
                GroupChild::Transform(h) => {
                    let result = unsafe {
                        rtGroupSetChild(
                            rt_grp,
                            i as u32,
                            h.borrow().rt_xform as RTobject,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                }
                GroupChild::None => unreachable!(),
            }
        }

        let grp = Rc::new(RefCell::new(Group {
            rt_grp,
            acc,
            children,
        }));

        self.groups.push(Rc::clone(&grp));

        Ok(grp)
    }

    pub fn group_set_children(
        &mut self,
        grp: &GroupHandle,
        children: Vec<GroupChild>,
    ) -> Result<()> {
        for child in &children {
            match child {
                GroupChild::GeometryGroup(h) => {
                    self.geometry_group_validate(h)?
                }
                GroupChild::Group(h) => self.group_validate(h)?,
                GroupChild::Transform(h) => self.transform_validate(h)?,
                GroupChild::None => unreachable!(),
            }
        }

        // Add the children
        let result = unsafe {
            rtGroupSetChildCount(grp.borrow().rt_grp, children.len() as u32)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetChildCount", result));
        }
        for (i, child) in children.iter().enumerate() {
            match child {
                GroupChild::GeometryGroup(h) => {
                    let result = unsafe {
                        rtGroupSetChild(
                            grp.borrow().rt_grp,
                            i as u32,
                            h.borrow().rt_geogrp as RTobject,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                }
                GroupChild::Group(h) => {
                    let result = unsafe {
                        rtGroupSetChild(
                            grp.borrow().rt_grp,
                            i as u32,
                            h.borrow().rt_grp as RTobject,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                }
                GroupChild::Transform(h) => {
                    let result = unsafe {
                        rtGroupSetChild(
                            grp.borrow().rt_grp,
                            i as u32,
                            h.borrow().rt_xform as RTobject,
                        )
                    };
                    if result != RtResult::SUCCESS {
                        return Err(self.optix_error("rtGroupSetChild", result));
                    }
                }
                GroupChild::None => unreachable!(),
            }
        }

        grp.borrow_mut().children = children;

        Ok(())
    }

    pub fn group_add_child(
        &mut self,
        grp: &GroupHandle,
        child: GroupChild,
    ) -> Result<()> {
        match &child {
            GroupChild::GeometryGroup(h) => self.geometry_group_validate(&h)?,
            GroupChild::Group(h) => self.group_validate(&h)?,
            GroupChild::Transform(h) => self.transform_validate(&h)?,
            GroupChild::None => unreachable!(),
        }

        let index = grp.borrow().children.len() as u32;

        let result =
            unsafe { rtGroupSetChildCount(grp.borrow().rt_grp, index + 1) };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetChildCount", result));
        }

        match &child {
            GroupChild::GeometryGroup(h) => {
                let result = unsafe {
                    rtGroupSetChild(
                        grp.borrow().rt_grp,
                        index,
                        h.borrow().rt_geogrp as RTobject,
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtGroupSetChild", result));
                }
            }
            GroupChild::Group(h) => {
                let result = unsafe {
                    rtGroupSetChild(
                        grp.borrow().rt_grp,
                        index,
                        h.borrow().rt_grp as RTobject,
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtGroupSetChild", result));
                }
            }
            GroupChild::Transform(h) => {
                let result = unsafe {
                    rtGroupSetChild(
                        grp.borrow().rt_grp,
                        index,
                        h.borrow().rt_xform as RTobject,
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtGroupSetChild", result));
                }
            }
            GroupChild::None => unreachable!(),
        }

        grp.borrow_mut().children.push(child);

        Ok(())
    }

    pub fn group_validate(&self, grp: &GroupHandle) -> Result<()> {
        let result = unsafe { rtGroupValidate(grp.borrow().rt_grp) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGroupValidate", result))
        } else {
            Ok(())
        }
    }

    pub fn group_set_acceleration(
        &self,
        grp: &GroupHandle,
        acc: AccelerationHandle,
    ) -> Result<()> {
        // Add the acceleration
        let result = unsafe {
            rtGroupSetAcceleration(grp.borrow().rt_grp, acc.borrow().rt_acc)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGroupSetAcceleration", result));
        }

        grp.borrow_mut().acc = acc;

        Ok(())
    }
}
