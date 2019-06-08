use crate::context::*;

use std::cell::RefCell;
use std::rc::Rc;

#[cfg(feature = "colorspace_support")]
use colorspace::rgb::RGBf32;

pub enum ObjectHandle {
    BufferID(BufferId),
    Buffer1d(Buffer1dHandle),
    Buffer2d(Buffer2dHandle),
    // Buffer3d(Buffer3dHandle),
    GeometryGroup(GeometryGroupHandle),
    Group(GroupHandle),
    Program(ProgramHandle),
    ProgramID(ProgramID),
    // Selector(SelectorHandle),
    TextureSampler(TextureSamplerHandle),
    TextureID(TextureID),
    Transform(TransformHandle),
}

pub struct VariableObject {
    pub(crate) var: RTvariable,
    pub(crate) _object_handle: ObjectHandle,
}

pub struct VariablePod {
    pub(crate) var: RTvariable,
}

impl VariablePod {
    pub fn new(var: RTvariable) -> VariablePod {
        VariablePod { var }
    }
}

pub struct UserData {
    pub var: RTvariable,
    pub data: Box<dyn UserVariable>,
}

pub enum Variable {
    Pod(VariablePod),
    Object(VariableObject),
    User(UserData),
}

pub trait UserVariable {
    fn set_user_variable(
        &self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<()>;
}

pub type VariableHandle = Rc<RefCell<Variable>>;

impl Context {
    pub fn variable_set_user_data<T: Sized>(
        &self,
        var: RTvariable,
        data: &T,
    ) -> Result<()> {
        let result = unsafe {
            rtVariableSetUserData(
                var,
                std::mem::size_of::<T>() as RTsize,
                data as *const T as *const std::os::raw::c_void,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error(&format!("rtVariableSetUserData",), result)
            );
        }

        Ok(())
    }
}

pub trait VariableStorable {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable>;
}

impl VariableStorable for ObjectHandle {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable> {
        match &self {
            ObjectHandle::BufferID(ref buf_id) => unsafe {
                let result = rtVariableSetUserData(
                    variable,
                    std::mem::size_of::<i32>() as RTsize,
                    &buf_id.0 as *const i32 as *const std::ffi::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        &format!(
                            "rtVariableSetUserData<BufferId>  {:?}",
                            buf_id.0
                        ),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
            ObjectHandle::Buffer1d(buffer_handle) => unsafe {
                let result = rtVariableSetObject(
                    variable,
                    buffer_handle.borrow().rt_buf
                        as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        "rtVariableSetObject {Buffer1d}".into(),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
            ObjectHandle::Buffer2d(buffer_handle) => unsafe {
                let result = rtVariableSetObject(
                    variable,
                    buffer_handle.borrow().rt_buf
                        as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        "rtVariableSetObject {Buffer2d}".into(),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
            ObjectHandle::Group(group_handle) => unsafe {
                let result = rtVariableSetObject(
                    variable,
                    group_handle.borrow().rt_grp as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        "rtVariableSetObject(Group)".into(),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
            ObjectHandle::GeometryGroup(geometry_group_handle) => unsafe {
                let result = rtVariableSetObject(
                    variable,
                    geometry_group_handle.borrow().rt_geogrp
                        as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        "rtVariableSetObject {GeometryGroup}".into(),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
            ObjectHandle::Program(program_handle) => unsafe {
                let result = rtVariableSetObject(
                    variable,
                    program_handle.borrow().rt_prg
                        as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        "rtVariableSetObject(Program)".into(),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
            ObjectHandle::ProgramID(program_id) => unsafe {
                let result = rtVariableSet1i(variable, program_id.id);
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        "rtVariableSet1i(ProgramID)".into(),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
            ObjectHandle::Transform(transform_handle) => unsafe {
                let result = rtVariableSetObject(
                    variable,
                    transform_handle.borrow().rt_xform
                        as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        "rtVariableSetObject {Transform}".into(),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
            ObjectHandle::TextureSampler(ts_handle) => unsafe {
                let result = rtVariableSetObject(
                    variable,
                    ts_handle.borrow().rt_ts as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        "rtVariableSetObject {TextureSampler}".into(),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
            ObjectHandle::TextureID(ref tex_id) => unsafe {
                let result = rtVariableSetUserData(
                    variable,
                    std::mem::size_of::<i32>() as RTsize,
                    &tex_id.id as *const i32 as *const std::ffi::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        &format!(
                            "rtVariableSetUserData<TextureID>  {:?}",
                            tex_id.id
                        ),
                        result,
                    ));
                } else {
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        _object_handle: self,
                    }));
                }
            },
        };
    }
}

impl VariableStorable for u32 {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable> {
        let result = unsafe { rtVariableSet1ui(variable, self) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet1ui", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

impl VariableStorable for u64 {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable> {
        let result = unsafe { rtVariableSet1ull(variable, self) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet1ull", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

impl VariableStorable for f32 {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable> {
        let result = unsafe { rtVariableSet1f(variable, self) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet1f", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

impl VariableStorable for i32 {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable> {
        let result = unsafe { rtVariableSet1i(variable, self) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet1i", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

impl VariableStorable for V3f32 {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable> {
        let result =
            unsafe { rtVariableSet3f(variable, self.x, self.y, self.z) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet3f", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

#[cfg(feature = "colorspace")]
impl VariableStorable for RGBf32 {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable> {
        let result =
            unsafe { rtVariableSet3f(variable, self.r, self.g, self.b) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet3f", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}

impl VariableStorable for MatrixFormat {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable> {
        let result = match self {
            MatrixFormat::RowMajor(mtx) => unsafe {
                rtVariableSetMatrix4x4fv(
                    variable,
                    0,
                    &mtx as *const M4f32 as *const f32,
                )
            },
            MatrixFormat::ColumnMajor(mtx) => unsafe {
                rtVariableSetMatrix4x4fv(
                    variable,
                    1,
                    &mtx as *const M4f32 as *const f32,
                )
            },
        };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSetMatrix4x4fv", result))
        } else {
            Ok(Variable::Pod(VariablePod { var: variable }))
        }
    }
}
