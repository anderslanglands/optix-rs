use crate::context::*;
use crate::math::*;

use slotmap::*;

new_key_type! { pub struct TransformHandle; }

pub enum TransformChild {
    None,
    GeometryGroup(GeometryGroupHandle),
    Group(GroupHandle),
    // Selector(SelectorHandle),
    Transform(TransformHandle),
}

impl Default for TransformChild {
    fn default() -> TransformChild {
        TransformChild::None
    }
}

pub enum MatrixFormat {
    RowMajor(M4f32),
    ColumnMajor(M4f32),
}

impl Context {
    pub fn transform_create(
        &mut self,
        matrix: MatrixFormat,
        child: TransformChild,
    ) -> Result<TransformHandle> {
        // validate first
        match child {
            TransformChild::Group(h) => self.group_validate(h)?,
            TransformChild::GeometryGroup(ggh) => self.geometry_group_validate(ggh)?,
            TransformChild::Transform(th) => self.transform_validate(th)?,
            TransformChild::None => unreachable!(),
        }

        let (rt_xform, result) = unsafe {
            let mut rt_xform: RTtransform = std::mem::zeroed();
            let result = rtTransformCreate(self.rt_ctx, &mut rt_xform);
            (rt_xform, result)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTransformCreate", result));
        }

        let xform = self.ga_transform_obj.insert(rt_xform);

        match matrix {
            MatrixFormat::RowMajor(m) => {
                let result = unsafe {
                    rtTransformSetMatrix(
                        rt_xform,
                        0,
                        &m as *const M4f32 as *const f32,
                        std::ptr::null(),
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetMatrix", result));
                }
            }
            MatrixFormat::ColumnMajor(m) => {
                let result = unsafe {
                    rtTransformSetMatrix(
                        rt_xform,
                        1,
                        &m as *const M4f32 as *const f32,
                        std::ptr::null(),
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetMatrix", result));
                }
            }
        }

        match child {
            TransformChild::Group(h) => {
                let child_rt_grp = self.ga_group_obj.get(h).unwrap();
                let result = unsafe { rtTransformSetChild(rt_xform, *child_rt_grp as RTobject) };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetChild", result));
                }
            }
            TransformChild::GeometryGroup(ggh) => {
                let child_rt_geogrp = self.ga_geometry_group_obj.get(ggh).unwrap();
                let result = unsafe { rtTransformSetChild(rt_xform, *child_rt_geogrp as RTobject) };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetChild", result));
                }
            }
            TransformChild::Transform(th) => {
                let child_rt_xf = self.ga_transform_obj.get(th).unwrap();
                let result = unsafe { rtTransformSetChild(rt_xform, *child_rt_xf as RTobject) };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetChild", result));
                }
            }
            TransformChild::None => unreachable!(),
        }

        self.gd_transform_child.insert(xform, child);

        Ok(xform)
    }

    pub fn transform_destroy(&mut self, xform: TransformHandle) {
        let rt_xform = self.ga_transform_obj.remove(xform).unwrap();
        let child = self.gd_transform_child.remove(xform);
        if unsafe { rtTransformDestroy(rt_xform) } != RtResult::SUCCESS {
            panic!("Error destroying transform {:?}", xform);
        }
    }

    pub fn transform_validate(&self, xform: TransformHandle) -> Result<()> {
        let rt_xform = *self.ga_transform_obj.get(xform).unwrap();
        let result = unsafe { rtTransformValidate(rt_xform) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtTransformValidate", result))
        } else {
            Ok(())
        }
    }
}
