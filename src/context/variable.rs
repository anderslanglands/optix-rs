use crate::context::*;

#[cfg(feature = "colorspace_support")]
use colorspace::rgb::RGBf32;

pub enum ObjectHandle {
    BufferId(BufferId),
    Buffer1d(Buffer1dHandle),
    Buffer2d(Buffer2dHandle),
    // Buffer3d(Buffer3dHandle),
    GeometryGroup(GeometryGroupHandle),
    Group(GroupHandle),
    Program(ProgramHandle),
    // Selector(SelectorHandle),
    TextureSampler(TextureSamplerHandle),
    Transform(TransformHandle),
}

pub struct VariableObject {
    pub(crate) var: RTvariable,
    pub(crate) object_handle: ObjectHandle,
}

pub struct VariablePod {
    pub(crate) var: RTvariable,
}

pub enum Variable {
    Pod(VariablePod),
    Object(VariableObject),
}

pub trait VariableStorable {
    fn set_optix_variable(self, ctx: &mut Context, variable: RTvariable) -> Result<Variable>;
}

impl VariableStorable for ObjectHandle {
    fn set_optix_variable(self, ctx: &mut Context, variable: RTvariable) -> Result<Variable> {
        match self {
            ObjectHandle::BufferId(ref buf_id) => unsafe {
                let result = rtVariableSetUserData(
                    variable,
                    std::mem::size_of::<i32>() as RTsize,
                    &buf_id.0 as *const i32 as *const std::ffi::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        &format!("rtVariableSetUserData<BufferId>  {:?}", buf_id.0),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::Buffer1d(buffer_handle) => unsafe {
                let buf = ctx.ga_buffer1d_obj.get(buffer_handle).expect(&format!(
                    "Could not get buffer object for handle {:?}",
                    buffer_handle
                ));
                let result = rtVariableSetObject(variable, *buf as *mut ::std::os::raw::c_void);
                if result != RtResult::SUCCESS {
                    return Err(ctx
                        .optix_error(&format!("rtVariableSetObject {:?}", buffer_handle), result));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::Buffer2d(buffer_handle) => unsafe {
                let buf = ctx.ga_buffer2d_obj.get(buffer_handle).expect(&format!(
                    "Could not get buffer object for handle {:?}",
                    buffer_handle
                ));
                let result = rtVariableSetObject(variable, *buf as *mut ::std::os::raw::c_void);
                if result != RtResult::SUCCESS {
                    return Err(ctx
                        .optix_error(&format!("rtVariableSetObject {:?}", buffer_handle), result));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::Group(group_handle) => unsafe {
                let rt_grp = ctx.ga_group_obj.get(group_handle).expect(&format!(
                    "Could not get group object for handle {:?}",
                    group_handle
                ));
                let result = rtVariableSetObject(variable, *rt_grp as *mut ::std::os::raw::c_void);
                if result != RtResult::SUCCESS {
                    return Err(
                        ctx.optix_error(&format!("rtVariableSetObject {:?}", group_handle), result)
                    );
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::GeometryGroup(geometry_group_handle) => unsafe {
                let prg = ctx
                    .ga_geometry_group_obj
                    .get(geometry_group_handle)
                    .expect(&format!(
                        "Could not get geometry_group object for handle {:?}",
                        geometry_group_handle
                    ));
                let result = rtVariableSetObject(variable, *prg as *mut ::std::os::raw::c_void);
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        &format!("rtVariableSetObject {:?}", geometry_group_handle),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::Program(program_handle) => unsafe {
                let prg = ctx.ga_program_obj.get(program_handle).expect(&format!(
                    "Could not get program object for handle {:?}",
                    program_handle
                ));
                let result = rtVariableSetObject(variable, *prg as *mut ::std::os::raw::c_void);
                if result != RtResult::SUCCESS {
                    return Err(ctx
                        .optix_error(&format!("rtVariableSetObject {:?}", program_handle), result));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::Transform(transform_handle) => unsafe {
                let prg = ctx.ga_transform_obj.get(transform_handle).expect(&format!(
                    "Could not get transform object for handle {:?}",
                    transform_handle
                ));
                let result = rtVariableSetObject(variable, *prg as *mut ::std::os::raw::c_void);
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        &format!("rtVariableSetObject {:?}", transform_handle),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::TextureSampler(ts_handle) => unsafe {
                let ts = ctx.ga_texture_sampler_obj.get(ts_handle).expect(&format!(
                    "Could not get texture sampler for handle {:?}",
                    ts_handle
                ));
                let result = rtVariableSetObject(variable, *ts as *mut ::std::os::raw::c_void);
                if result != RtResult::SUCCESS {
                    return Err(
                        ctx.optix_error(&format!("rtVariableSetObject {:?}", ts_handle), result)
                    );
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
        };
    }
}

impl VariableStorable for u32 {
    fn set_optix_variable(self, ctx: &mut Context, variable: RTvariable) -> Result<Variable> {
        let result = unsafe { rtVariableSet1ui(variable, self) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet1ui", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

impl VariableStorable for f32 {
    fn set_optix_variable(self, ctx: &mut Context, variable: RTvariable) -> Result<Variable> {
        let result = unsafe { rtVariableSet1f(variable, self) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet1f", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

impl VariableStorable for V3f32 {
    fn set_optix_variable(self, ctx: &mut Context, variable: RTvariable) -> Result<Variable> {
        let result = unsafe { rtVariableSet3f(variable, self.x, self.y, self.z) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet3f", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

#[cfg(feature = "colorspace")]
impl VariableStorable for RGBf32 {
    fn set_optix_variable(self, ctx: &mut Context, variable: RTvariable) -> Result<Variable> {
        let result = unsafe { rtVariableSet3f(variable, self.r, self.g, self.b) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet3f", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

impl VariableStorable for MatrixFormat {
    fn set_optix_variable(self, ctx: &mut Context, variable: RTvariable) -> Result<Variable> {
        let result = match self {
            MatrixFormat::RowMajor(mtx) => unsafe {
                rtVariableSetMatrix4x4fv(variable, 0, &mtx as *const M4f32 as *const f32)
            },
            MatrixFormat::ColumnMajor(mtx) => unsafe {
                rtVariableSetMatrix4x4fv(variable, 1, &mtx as *const M4f32 as *const f32)
            },
        };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet3f", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}
