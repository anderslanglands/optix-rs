use crate::context::*;
use crate::format_get_size;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub struct GeometryTriangles {
    pub(crate) rt_geotri: RTgeometrytriangles,
    pub(crate) buf_vertices: Buffer1dHandle,
    pub(crate) buf_indices: Option<Buffer1dHandle>,
    pub(crate) prg_attribute: Option<ProgramHandle>,
    pub(crate) variables: HashMap<String, VariableHandle>,
}

pub type GeometryTrianglesHandle = Rc<RefCell<GeometryTriangles>>;

impl Context {
    pub fn geometry_triangles_create(
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
            buf_vertices: Rc::clone(&vertices),
            buf_indices: None,
            prg_attribute,
            variables: HashMap::new(),
        }));

        self.geometry_triangles_set_variable(
            &hnd,
            "vertex_buffer",
            ObjectHandle::Buffer1d(Rc::clone(&vertices)),
        )?;

        self.geometry_triangles.push(Rc::clone(&hnd));

        Ok(hnd)
    }

    pub fn geometry_triangles_create_indexed(
        &mut self,
        buf_vertices: Buffer1dHandle,
        buf_indices: Buffer1dHandle,
        prg_attribute: Option<ProgramHandle>,
    ) -> Result<GeometryTrianglesHandle> {
        let geotri = self.geometry_triangles_create(
            Rc::clone(&buf_vertices),
            prg_attribute,
        )?;

        self.geometry_triangles_set_variable(
            &geotri,
            "vertex_buffer",
            ObjectHandle::Buffer1d(Rc::clone(&buf_vertices)),
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

        self.geometry_triangles_set_variable(
            &geotri,
            "index_buffer",
            ObjectHandle::Buffer1d(Rc::clone(&buf_indices)),
        )?;

        geotri.borrow_mut().buf_indices = Some(buf_indices);

        Ok(geotri)
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
}
