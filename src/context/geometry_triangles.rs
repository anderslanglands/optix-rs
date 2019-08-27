use crate::context::*;
use crate::format_get_size;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct GeometryTriangles {
    pub(crate) rt_geotri: RTgeometrytriangles,
    pub(crate) buf_vertices: Vec<Buffer1dHandle>,
    pub(crate) buf_indices: Option<Buffer1dHandle>,
    pub(crate) prg_attribute: Option<ProgramHandle>,
    pub(crate) variables: HashMap<String, VariableHandle>,
}

pub type GeometryTrianglesHandle = Rc<RefCell<GeometryTriangles>>;

impl Context {
    pub fn geometry_triangles_create_soup(
        &mut self,
        vertices: Buffer1dHandle,
        prg_attribute: Option<ProgramHandle>,
    ) -> Result<GeometryTrianglesHandle> {
        if let Some(prg) = &prg_attribute {
            self.program_validate(prg)?;
        }

        let (rt_geotri, result) = unsafe {
            let mut rt_geotri: RTgeometrytriangles = std::mem::zeroed();
            let result = rtGeometryTrianglesCreate(self.rt_ctx, &mut rt_geotri);
            (rt_geotri, result)
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGeometryTriangles", result));
        }

        let format = self.buffer_get_format_1d(&vertices)?;
        let result = unsafe {
            rtGeometryTrianglesSetVertices(
                rt_geotri,
                self.buffer_get_size_1d(&vertices)? as u32,
                vertices.borrow().rt_buf,
                0 as RTsize,
                format_get_size(format) as RTsize,
                format,
            )
        };
        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtGeometryTrianglesSetVertices", result)
            );
        }

        if let Some(prg) = &prg_attribute {
            let result = unsafe {
                rtGeometryTrianglesSetAttributeProgram(
                    rt_geotri,
                    prg.borrow().rt_prg,
                )
            };

            if result != RtResult::SUCCESS {
                return Err(self.optix_error(
                    "rtGeometryTrianglesSetAttributeProgram",
                    result,
                ));
            }
        }

        let hnd = Rc::new(RefCell::new(GeometryTriangles {
            rt_geotri,
            buf_vertices: vec![Rc::clone(&vertices)],
            buf_indices: None,
            prg_attribute,
            variables: HashMap::new(),
        }));

        self.geometry_triangles.push(Rc::clone(&hnd));

        Ok(hnd)
    }

    pub fn geometry_triangles_create_indexed(
        &mut self,
        buf_vertices: Buffer1dHandle,
        buf_indices: Buffer1dHandle,
        prg_attribute: Option<ProgramHandle>,
    ) -> Result<GeometryTrianglesHandle> {
        let geotri = self.geometry_triangles_create_soup(
            Rc::clone(&buf_vertices),
            prg_attribute,
        )?;

        let format = self.buffer_get_format_1d(&buf_indices)?;
        let result = unsafe {
            rtGeometryTrianglesSetTriangleIndices(
                geotri.borrow().rt_geotri,
                buf_indices.borrow().rt_buf,
                0 as RTsize,
                format_get_size(format) as RTsize,
                format,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(self
                .optix_error("rtGeometryTrianglesSetTriangleIndices", result));
        }

        self.geometry_triangles_set_primitive_count(
            &geotri,
            (self.buffer_get_size_1d(&buf_indices)? / 3) as u32,
        )?;

        geotri.borrow_mut().buf_indices = Some(buf_indices);

        Ok(geotri)
    }

    pub fn geometry_triangles_set_indices(
        &mut self,
        geotri: &GeometryTrianglesHandle,
        buf_indices: Buffer1dHandle,
    ) -> Result<()> {
        let format = self.buffer_get_format_1d(&buf_indices)?;
        let result = unsafe {
            rtGeometryTrianglesSetTriangleIndices(
                geotri.borrow().rt_geotri,
                buf_indices.borrow().rt_buf,
                0 as RTsize,
                format_get_size(format) as RTsize,
                format,
            )
        };

        if result != RtResult::SUCCESS {
            Err(self
                .optix_error("rtGeometryTrianglesSetTriangleIndices", result))
        } else {
            geotri.borrow_mut().buf_indices = Some(buf_indices);
            Ok(())
        }
    }

    pub fn geometry_triangles_set_attribute_program(
        &mut self,
        geotri: &GeometryTrianglesHandle,
        prg_attribute: ProgramHandle,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometryTrianglesSetAttributeProgram(
                geotri.borrow().rt_geotri,
                prg_attribute.borrow().rt_prg,
            )
        };

        if result != RtResult::SUCCESS {
            Err(self
                .optix_error("rtGeometryTrianglesSetAttributeProgram", result))
        } else {
            geotri.borrow_mut().prg_attribute = Some(prg_attribute);
            Ok(())
        }
    }

    pub fn geometry_triangles_set_vertices(
        &mut self,
        geotri: &GeometryTrianglesHandle,
        vertices: Buffer1dHandle,
    ) -> Result<()> {
        let format = self.buffer_get_format_1d(&vertices)?;
        let result = unsafe {
            rtGeometryTrianglesSetVertices(
                geotri.borrow().rt_geotri,
                self.buffer_get_size_1d(&vertices)? as u32,
                vertices.borrow().rt_buf,
                0 as RTsize,
                format_get_size(format) as RTsize,
                format,
            )
        };

        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryTrianglesSetVertices", result))
        } else {
            geotri.borrow_mut().buf_vertices = vec![vertices];
            Ok(())
        }
    }

    pub fn geometry_triangles_set_primitive_count(
        &mut self,
        geotri: &GeometryTrianglesHandle,
        count: u32,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometryTrianglesSetPrimitiveCount(
                geotri.borrow().rt_geotri,
                count,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(self
                .optix_error("rtGeometryTrianglesSetPrimitiveCount", result));
        };

        Ok(())
    }

    pub fn geometry_triangles_set_material_count(
        &mut self,
        geotri: &GeometryTrianglesHandle,
        count: u32,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometryTrianglesSetMaterialCount(
                geotri.borrow().rt_geotri,
                count,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(
                self.optix_error("rtGeometryTrianglesSetMaterialCount", result)
            );
        };

        Ok(())
    }

    pub fn geometry_triangles_validate(
        &self,
        geo: &GeometryTrianglesHandle,
    ) -> Result<()> {
        let result =
            unsafe { rtGeometryTrianglesValidate(geo.borrow().rt_geotri) };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryTrianglesValidate", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_triangles_set_variable<T: VariableStorable>(
        &mut self,
        geotri: &GeometryTrianglesHandle,
        name: &str,
        data: T,
    ) -> Result<()> {
        // we have to remove the variable first so that we're not holding a borrow
        // further down here
        let ex_var = geotri.borrow_mut().variables.remove(name);
        if let Some(ex_var) = ex_var {
            let var = {
                let ex_var_c = ex_var.borrow();
                match &*ex_var_c {
                    Variable::Pod(vp) => vp.var,
                    Variable::Object(vo) => vo.var,
                    Variable::User(vo) => vo.var,
                }
            };

            // replace the variable and reinsert it
            ex_var.replace(data.set_optix_variable(self, var)?);
            geotri.borrow_mut().variables.insert(name.into(), ex_var);
        } else {
            let (rt_var, result) = unsafe {
                let mut rt_var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtGeometryTrianglesDeclareVariable(
                    geotri.borrow_mut().rt_geotri,
                    c_name.as_ptr(),
                    &mut rt_var,
                );
                (rt_var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtGeometryDeclareVariable", result)
                );
            }

            let var =
                Rc::new(RefCell::new(data.set_optix_variable(self, rt_var)?));
            geotri
                .borrow_mut()
                .variables
                .insert(name.into(), Rc::clone(&var));

            // if it's a new variable, push it to the context storage
            self.variables.push(var)
        }

        Ok(())
    }

    /// Set the Variable referred to by `name` to the given `data`. Any objects
    /// previously assigned to the variable will be destroyed.
    pub fn geometry_triangles_set_user_variable(
        &mut self,
        geotri: &GeometryTrianglesHandle,
        name: &str,
        data: Rc<dyn UserVariable>,
    ) -> Result<()> {
        // check if the variable exists first
        let ex_var = geotri.borrow_mut().variables.remove(name);
        if let Some(ex_var) = ex_var {
            let var = {
                let ex_var_c = ex_var.borrow();
                match &*ex_var_c {
                    Variable::Pod(vp) => vp.var,
                    Variable::Object(vo) => vo.var,
                    Variable::User(vu) => vu.var,
                }
            };

            data.set_user_variable(self, var)?;
            ex_var.replace(Variable::User(UserData { var, data }));
            geotri.borrow_mut().variables.insert(name.into(), ex_var);
        } else {
            let (rt_var, result) = unsafe {
                let mut rt_var: RTvariable = ::std::mem::uninitialized();
                let c_name = std::ffi::CString::new(name).unwrap();
                let result = rtGeometryTrianglesDeclareVariable(
                    geotri.borrow_mut().rt_geotri,
                    c_name.as_ptr(),
                    &mut rt_var,
                );
                (rt_var, result)
            };
            if result != RtResult::SUCCESS {
                return Err(
                    self.optix_error("rtContextDeclareVariable", result)
                );
            }

            data.set_user_variable(self, rt_var)?;
            let var = Rc::new(RefCell::new(Variable::User(UserData {
                var: rt_var,
                data,
            })));

            geotri
                .borrow_mut()
                .variables
                .insert(name.into(), Rc::clone(&var));

            // if it's a new variable, push it to the context storage
            self.variables.push(var)
        }

        Ok(())
    }

    pub fn geometry_triangles_set_motion_steps(
        &mut self,
        geo: &GeometryTrianglesHandle,
        steps: u32,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometryTrianglesSetMotionSteps(geo.borrow().rt_geotri, steps)
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryTrianglesSetMotionSteps", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_triangles_set_motion_range(
        &mut self,
        geo: &GeometryTrianglesHandle,
        time_begin: f32,
        time_end: f32,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometryTrianglesSetMotionRange(
                geo.borrow().rt_geotri,
                time_begin,
                time_end,
            )
        };
        if result != RtResult::SUCCESS {
            Err(self.optix_error("rtGeometryTrianglesSetMotionRange", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_triangles_set_motion_border_mode(
        &mut self,
        geo: &GeometryTrianglesHandle,
        begin_mode: MotionBorderMode,
        end_mode: MotionBorderMode,
    ) -> Result<()> {
        let result = unsafe {
            rtGeometryTrianglesSetMotionBorderMode(
                geo.borrow().rt_geotri,
                begin_mode,
                end_mode,
            )
        };
        if result != RtResult::SUCCESS {
            Err(self
                .optix_error("rtGeometryTrianglesSetMotionBorderMode", result))
        } else {
            Ok(())
        }
    }

    pub fn geometry_triangles_set_motion_vertices_multi_buffer(
        &mut self,
        geo: &GeometryTrianglesHandle,
        vertex_buffers: Vec<Buffer1dHandle>,
    ) -> Result<()> {
        let rt_buffers = vertex_buffers
            .iter()
            .map(|bh| bh.borrow().rt_buf)
            .collect::<Vec<_>>();
        let format = self.buffer_get_format_1d(&vertex_buffers[0])?;
        let vertex_count = self.buffer_get_size_1d(&vertex_buffers[0])? as u32;
        let result = unsafe {
            rtGeometryTrianglesSetMotionVerticesMultiBuffer(
                geo.borrow().rt_geotri,
                vertex_count,
                rt_buffers.as_slice().as_ptr() as *const RTbuffer,
                vertex_buffers.len() as u32,
                0 as RTsize,
                format_get_size(format) as RTsize,
                format,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error(
                "rtGeometryTrianglesSetMotionVerticesMultiBuffer",
                result,
            ));
        }

        geo.borrow_mut().buf_vertices = vertex_buffers;

        Ok(())
    }

    pub fn geometry_triangles_create_motion(
        &mut self,
        vertex_buffers: Vec<Buffer1dHandle>,
        buf_indices: Buffer1dHandle,
        prg_attribute: Option<ProgramHandle>,
    ) -> Result<GeometryTrianglesHandle> {
        if let Some(prg) = &prg_attribute {
            self.program_validate(prg)?;
        }

        let (rt_geotri, result) = unsafe {
            let mut rt_geotri: RTgeometrytriangles = std::mem::zeroed();
            let result = rtGeometryTrianglesCreate(self.rt_ctx, &mut rt_geotri);
            (rt_geotri, result)
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error("rtGeometryTrianglesCreate", result));
        }

        let rt_buffers = vertex_buffers
            .iter()
            .map(|bh| bh.borrow().rt_buf)
            .collect::<Vec<_>>();
        let format = self.buffer_get_format_1d(&vertex_buffers[0])?;
        let vertex_count = self.buffer_get_size_1d(&vertex_buffers[0])? as u32;
        let result = unsafe {
            rtGeometryTrianglesSetMotionVerticesMultiBuffer(
                rt_geotri,
                vertex_count,
                rt_buffers.as_slice().as_ptr() as *const RTbuffer,
                vertex_buffers.len() as u32,
                0 as RTsize,
                format_get_size(format) as RTsize,
                format,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(self.optix_error(
                "rtGeometryTrianglesSetMotionVerticesMultiBuffer",
                result,
            ));
        }

        let format = self.buffer_get_format_1d(&buf_indices)?;
        let result = unsafe {
            rtGeometryTrianglesSetTriangleIndices(
                rt_geotri,
                buf_indices.borrow().rt_buf,
                0 as RTsize,
                format_get_size(format) as RTsize,
                format,
            )
        };

        if result != RtResult::SUCCESS {
            return Err(self
                .optix_error("rtGeometryTrianglesSetTriangleIndices", result));
        }

        if let Some(prg) = &prg_attribute {
            let result = unsafe {
                rtGeometryTrianglesSetAttributeProgram(
                    rt_geotri,
                    prg.borrow().rt_prg,
                )
            };

            if result != RtResult::SUCCESS {
                return Err(self.optix_error(
                    "rtGeometryTrianglesSetAttributeProgram",
                    result,
                ));
            }
        }

        let geo = Rc::new(RefCell::new(GeometryTriangles {
            rt_geotri,
            buf_vertices: vertex_buffers,
            buf_indices: Some(buf_indices),
            prg_attribute,
            variables: HashMap::new(),
        }));

        Ok(geo)
    }
}
