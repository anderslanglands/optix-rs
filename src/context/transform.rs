use crate::context::*;

use std::cell::RefCell;
use std::rc::Rc;

pub struct Transform {
    pub(crate) rt_xform: RTtransform,
    pub child: TransformChild,
}

pub type TransformHandle = Rc<RefCell<Transform>>;

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
        match &child {
            TransformChild::Group(h) => self.group_validate(&h)?,
            TransformChild::GeometryGroup(ggh) => {
                self.geometry_group_validate(&ggh)?
            }
            TransformChild::Transform(th) => self.transform_validate(&th)?,
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
                    return Err(
                        self.optix_error("rtTransformSetMatrix", result)
                    );
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
                    return Err(
                        self.optix_error("rtTransformSetMatrix", result)
                    );
                }
            }
        }

        match &child {
            TransformChild::Group(h) => {
                let result = unsafe {
                    rtTransformSetChild(rt_xform, h.borrow().rt_grp as RTobject)
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetChild", result));
                }
            }
            TransformChild::GeometryGroup(ggh) => {
                let result = unsafe {
                    rtTransformSetChild(
                        rt_xform,
                        ggh.borrow().rt_geogrp as RTobject,
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetChild", result));
                }
            }
            TransformChild::Transform(th) => {
                let result = unsafe {
                    rtTransformSetChild(
                        rt_xform,
                        th.borrow().rt_xform as RTobject,
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetChild", result));
                }
            }
            TransformChild::None => unreachable!(),
        }

        let xform = Rc::new(RefCell::new(Transform { rt_xform, child }));

        self.transforms.push(Rc::clone(&xform));

        Ok(xform)
    }

    pub fn transform_create_with_srt_motion(
        &mut self,
        motion_keys: &[f32],
        time_begin: f32,
        time_end: f32,
        child: TransformChild,
    ) -> Result<TransformHandle> {
        // validate first
        match &child {
            TransformChild::Group(h) => self.group_validate(&h)?,
            TransformChild::GeometryGroup(ggh) => {
                self.geometry_group_validate(&ggh)?
            }
            TransformChild::Transform(th) => self.transform_validate(&th)?,
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

        let n = motion_keys.len() / 16;
        let result = unsafe {
            rtTransformSetMotionKeys(
                rt_xform,
                n as u32,
                MotionKeyType::SRT_FLOAT16,
                motion_keys.as_ptr() as *const f32,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTransformSetMotionKeys", result));
        }

        let result = unsafe {
            rtTransformSetMotionRange(rt_xform, time_begin, time_end)
        };
        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtTransformSetMotionRange", result));
        }

        match &child {
            TransformChild::Group(h) => {
                let result = unsafe {
                    rtTransformSetChild(rt_xform, h.borrow().rt_grp as RTobject)
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetChild", result));
                }
            }
            TransformChild::GeometryGroup(ggh) => {
                let result = unsafe {
                    rtTransformSetChild(
                        rt_xform,
                        ggh.borrow().rt_geogrp as RTobject,
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetChild", result));
                }
            }
            TransformChild::Transform(th) => {
                let result = unsafe {
                    rtTransformSetChild(
                        rt_xform,
                        th.borrow().rt_xform as RTobject,
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(self.optix_error("rtTransformSetChild", result));
                }
            }
            TransformChild::None => unreachable!(),
        }

        let xform = Rc::new(RefCell::new(Transform { rt_xform, child }));

        self.transforms.push(Rc::clone(&xform));

        Ok(xform)
    }

    pub fn transform_validate(&self, xform: &TransformHandle) -> Result<()> {
        let result = unsafe { rtTransformValidate(xform.borrow().rt_xform) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtTransformValidate", result))
        } else {
            Ok(())
        }
    }

    pub fn transform_set_matrix(
        &mut self,
        xform: &TransformHandle,
        matrix: MatrixFormat,
    ) -> Result<()> {
        match matrix {
            MatrixFormat::RowMajor(m) => {
                let result = unsafe {
                    rtTransformSetMatrix(
                        xform.borrow().rt_xform,
                        0,
                        &m as *const M4f32 as *const f32,
                        std::ptr::null(),
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(
                        self.optix_error("rtTransformSetMatrix", result)
                    );
                }
            }
            MatrixFormat::ColumnMajor(m) => {
                let result = unsafe {
                    rtTransformSetMatrix(
                        xform.borrow().rt_xform,
                        1,
                        &m as *const M4f32 as *const f32,
                        std::ptr::null(),
                    )
                };
                if result != RtResult::SUCCESS {
                    return Err(
                        self.optix_error("rtTransformSetMatrix", result)
                    );
                }
            }
        }

        Ok(())
    }

    pub fn transform_set_motion_keys(
        &mut self,
        xform: &TransformHandle,
        n: u32,
        key_type: MotionKeyType,
        keys: &[f32],
    ) -> Result<()> {
        // check that the length of the keys array matches the key type and
        // number of samples
        match key_type {
            MotionKeyType::MATRIX_FLOAT12 => {
                let expected = n * 12;
                if keys.len() as u32 != expected {
                    return Err(Error::MotionKeyLength {
                        got: keys.len() as u32,
                        expected,
                    });
                }
            }
            MotionKeyType::SRT_FLOAT16 => {
                let expected = n * 16;
                if keys.len() as u32 != expected {
                    return Err(Error::MotionKeyLength {
                        got: keys.len() as u32,
                        expected,
                    });
                }
            }
            MotionKeyType::NONE => {
                return Err(Error::MotionKeyType);
            }
        }

        let result = unsafe {
            rtTransformSetMotionKeys(
                xform.borrow().rt_xform,
                n,
                key_type,
                keys.as_ptr() as *const f32,
            )
        };

        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtTransformSetMotionKeys", result))
        } else {
            Ok(())
        }
    }

    pub fn transform_set_motion_range(
        &mut self,
        xform: &TransformHandle,
        time_begin: f32,
        time_end: f32,
    ) -> Result<()> {
        let result = unsafe {
            rtTransformSetMotionRange(
                xform.borrow().rt_xform,
                time_begin,
                time_end,
            )
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtTransformSetMotionRange", result))
        } else {
            Ok(())
        }
    }

    pub fn transform_set_motion_border_mode(
        &mut self,
        xform: &TransformHandle,
        begin_mode: MotionBorderMode,
        end_mode: MotionBorderMode,
    ) -> Result<()> {
        let result = unsafe {
            rtTransformSetMotionBorderMode(
                xform.borrow().rt_xform,
                begin_mode,
                end_mode,
            )
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtTransformSetMotionBorderMode", result))
        } else {
            Ok(())
        }
    }
}
