use crate::context::*;
use crate::ginallocator::*;

pub enum ObjectHandle {
    Buffer1d(Buffer1dHandle),
    Buffer2d(Buffer2dHandle),
    // Buffer3d(Buffer3dHandle),
    GeometryGroup(GeometryGroupHandle),
    // Group(GroupHandle),
    Program(ProgramHandle),
    // Selector(SelectorHandle),
    // TextureSampler(TextureSamplerHandle),
    // Transform(TransformHandle),
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
        match self {
            ObjectHandle::Buffer1d(buffer_handle) => unsafe {
                let buf =
                    ctx.ga_buffer1d_obj.get(buffer_handle).expect(&format!(
                        "Could not get buffer object for handle {}",
                        buffer_handle
                    ));
                let result = rtVariableSetObject(
                    variable,
                    *buf as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        &format!("rtVariableSetObject {}", buffer_handle),
                        result,
                    ));
                } else {
                    ctx.ga_buffer1d_obj.incref(buffer_handle);
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::Buffer2d(buffer_handle) => unsafe {
                let buf =
                    ctx.ga_buffer2d_obj.get(buffer_handle).expect(&format!(
                        "Could not get buffer object for handle {}",
                        buffer_handle
                    ));
                let result = rtVariableSetObject(
                    variable,
                    *buf as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        &format!("rtVariableSetObject {}", buffer_handle),
                        result,
                    ));
                } else {
                    ctx.ga_buffer2d_obj.incref(buffer_handle);
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::GeometryGroup(geometry_group_handle) => unsafe {
                let prg =
                    ctx.ga_geometry_group_obj.get(geometry_group_handle).expect(&format!(
                        "Could not get geometry_group object for handle {}",
                        geometry_group_handle
                    ));
                let result = rtVariableSetObject(
                    variable,
                    *prg as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        &format!("rtVariableSetObject {}", geometry_group_handle),
                        result,
                    ));
                } else {
                    ctx.ga_geometry_group_obj.incref(geometry_group_handle);
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
            ObjectHandle::Program(program_handle) => unsafe {
                let prg =
                    ctx.ga_program_obj.get(program_handle).expect(&format!(
                        "Could not get program object for handle {}",
                        program_handle
                    ));
                let result = rtVariableSetObject(
                    variable,
                    *prg as *mut ::std::os::raw::c_void,
                );
                if result != RtResult::SUCCESS {
                    return Err(ctx.optix_error(
                        &format!("rtVariableSetObject {}", program_handle),
                        result,
                    ));
                } else {
                    ctx.ga_program_obj.incref(program_handle);
                    return Ok(Variable::Object(VariableObject {
                        var: variable,
                        object_handle: self,
                    }));
                }
            },
        };
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
            Ok(Variable::Pod(VariablePod{var: variable}))
        }
    }
}

impl VariableStorable for V3f32 {
    fn set_optix_variable(
        self,
        ctx: &mut Context,
        variable: RTvariable,
    ) -> Result<Variable> {
        let result = unsafe { rtVariableSet3f(variable, self.x, self.y, self.z) };
        if result != RtResult::SUCCESS {
            Err(ctx.optix_error("rtVariableSet3f", result))
        } else {
            Ok(Variable::Pod(VariablePod{var: variable}))
        }
    }
}
