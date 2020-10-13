use crate::{sys, DeviceContext, Error, Module};
type Result<T, E = Error> = std::result::Result<T, E>;

use ustr::Ustr;

use std::ffi::CStr;

#[derive(Clone)]
pub struct ProgramGroupModule<'m> {
    pub module: &'m Module,
    pub entry_function_name: Ustr,
}

pub enum ProgramGroupDesc<'m> {
    Raygen(ProgramGroupModule<'m>),
    Miss(ProgramGroupModule<'m>),
    Exception(ProgramGroupModule<'m>),
    Hitgroup {
        ch: Option<ProgramGroupModule<'m>>,
        ah: Option<ProgramGroupModule<'m>>,
        is: Option<ProgramGroupModule<'m>>,
    },
    Callables {
        dc: Option<ProgramGroupModule<'m>>,
        cc: Option<ProgramGroupModule<'m>>,
    },
}

impl<'m> ProgramGroupDesc<'m> {
    pub fn raygen(
        module: &'m Module,
        entry_function_name: Ustr,
    ) -> ProgramGroupDesc<'m> {
        ProgramGroupDesc::Raygen(ProgramGroupModule {
            module,
            entry_function_name,
        })
    }

    pub fn miss(
        module: &'m Module,
        entry_function_name: Ustr,
    ) -> ProgramGroupDesc<'m> {
        ProgramGroupDesc::Miss(ProgramGroupModule {
            module,
            entry_function_name,
        })
    }

    pub fn exception(
        module: &'m Module,
        entry_function_name: Ustr,
    ) -> ProgramGroupDesc<'m> {
        ProgramGroupDesc::Exception(ProgramGroupModule {
            module,
            entry_function_name,
        })
    }

    pub fn hitgroup(
        ch: Option<(&'m Module, Ustr)>,
        ah: Option<(&'m Module, Ustr)>,
        is: Option<(&'m Module, Ustr)>,
    ) -> ProgramGroupDesc<'m> {
        ProgramGroupDesc::Hitgroup {
            ch: ch.map(|(module, entry_function_name)| ProgramGroupModule {
                module,
                entry_function_name,
            }),
            ah: ah.map(|(module, entry_function_name)| ProgramGroupModule {
                module,
                entry_function_name,
            }),
            is: is.map(|(module, entry_function_name)| ProgramGroupModule {
                module,
                entry_function_name,
            }),
        }
    }
}

#[repr(transparent)]
#[derive(Clone)]
/// Modules can contain more than one program. The program in the module is
/// designated by its entry function name as part of the [ProgramGroupDesc]
/// struct passed to [DeviceContext::program_group_create()] and
/// [DeviceContext::program_group_create_single()], or specified directly in the
/// case of [DeviceContext::program_group_raygen()],
/// [DeviceContext::program_group_miss()] and
/// [DeviceContext::program_group_hitgroup()]
///
/// Four program groups can contain only a single program; only hitgroups can
/// designate up to three programs for the closest-hit, any-hit, and
/// intersection programs.
///
/// Programs from modules can be used in any number of [ProgramGroup] objects.
/// The resulting program groups can be used to fill in any number of
/// SBT records. Program groups can also be used across pipelines as long as the
/// compilation options match.
///
/// A hit group specifies the intersection program used to test whether a ray
/// intersects a primitive, together with the hit shaders to be executed when a
/// ray does intersect the primitive. For built-in primitive types, a built-in
/// intersection program should be obtained from
/// [DeviceContext::builtin_is_module_get()] and used in the hit group. As a
/// special case, the intersection program is not required – and is ignored –
/// for triangle primitives.
///
/// # Safety
/// The lifetime of a module must extend to the lifetime of any
/// OptixProgramGroup that references that module.
pub struct ProgramGroup {
    pub(crate) inner: sys::OptixProgramGroup,
}

impl ProgramGroup {
    /// Use this information to calculate the total required stack sizes for a
    /// particular call graph of NVIDIA OptiX programs.
    ///
    /// To set the stack sizes for a particular pipeline, use
    /// [Pipeline::set_stack_size()](crate::Pipeline::set_stack_size()).
    pub fn get_stack_size(&self) -> Result<StackSizes> {
        let mut stack_sizes = StackSizes::default();
        unsafe {
            sys::optixProgramGroupGetStackSize(
                self.inner,
                &mut stack_sizes as *mut _ as *mut _,
            )
            .to_result()
            .map(|_| stack_sizes)
            .map_err(|source| Error::ProgramGroupGetStackSizes { source })
        }
    }
}

impl PartialEq for ProgramGroup {
    fn eq(&self, rhs: &ProgramGroup) -> bool {
        self.inner == rhs.inner
    }
}

/// # Creating and destroying `ProgramGroup`s
impl DeviceContext {
    /// Create a [ProgramGroup] for each of the [ProgramGroupDesc] objects in 
    /// `desc`.
    pub fn program_group_create(
        &mut self,
        desc: &[ProgramGroupDesc],
    ) -> Result<(Vec<ProgramGroup>, String)> {
        let pg_options = sys::OptixProgramGroupOptions { placeholder: 0 };

        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let pg_desc: Vec<sys::OptixProgramGroupDesc> =
            desc.iter().map(|d| d.into()).collect();

        let mut inners = vec![std::ptr::null_mut(); pg_desc.len()];

        let res = unsafe {
            sys::optixProgramGroupCreate(
                self.inner,
                pg_desc.as_ptr(),
                pg_desc.len() as u32,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_len,
                inners.as_mut_ptr(),
            )
            .to_result()
        };

        let log = CStr::from_bytes_with_nul(&log[0..log_len])
            .unwrap()
            .to_string_lossy()
            .into_owned();

        match res {
            Ok(()) => Ok((
                inners
                    .iter()
                    .map(|inner| ProgramGroup { inner: *inner })
                    .collect(),
                log,
            )),
            Err(source) => Err(Error::ProgramGroupCreation { source, log }),
        }
    }

    /// Create a single [ProgramGroup] specified by `desc`.
    pub fn program_group_create_single(
        &mut self,
        desc: &ProgramGroupDesc,
    ) -> Result<(ProgramGroup, String)> {
        let pg_options = sys::OptixProgramGroupOptions { placeholder: 0 };

        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let pg_desc: sys::OptixProgramGroupDesc = desc.into();

        let mut inner = std::ptr::null_mut();

        let res = unsafe {
            sys::optixProgramGroupCreate(
                self.inner,
                &pg_desc,
                1,
                &pg_options,
                log.as_mut_ptr() as *mut i8,
                &mut log_len,
                &mut inner,
            )
            .to_result()
        };

        let log = CStr::from_bytes_with_nul(&log[0..log_len])
            .unwrap()
            .to_string_lossy()
            .into_owned();

        match res {
            Ok(()) => Ok((ProgramGroup { inner }, log)),
            Err(source) => Err(Error::ProgramGroupCreation { source, log }),
        }
    }

    /// Create a raygen [ProgramGroup] from `entry_function_name` in `module`.
    pub fn program_group_raygen(
        &mut self,
        module: &Module,
        entry_function_name: Ustr,
    ) -> Result<ProgramGroup> {
        let desc = ProgramGroupDesc::raygen(module, entry_function_name);
        Ok(self.program_group_create_single(&desc)?.0)
    }

    /// Create a miss [ProgramGroup] from `entry_function_name` in `module`.
    pub fn program_group_miss(
        &mut self,
        module: &Module,
        entry_function_name: Ustr,
    ) -> Result<ProgramGroup> {
        let desc = ProgramGroupDesc::miss(module, entry_function_name);
        Ok(self.program_group_create_single(&desc)?.0)
    }

    /// Create an exception [ProgramGroup] from `entry_function_name` in `module`.
    pub fn program_group_exception(
        &mut self,
        module: &Module,
        entry_function_name: Ustr,
    ) -> Result<ProgramGroup> {
        let desc = ProgramGroupDesc::exception(module, entry_function_name);
        Ok(self.program_group_create_single(&desc)?.0)
    }

    /// Create a hitgroup [ProgramGroup] from any combination of
    /// `(module, entry_function_name)` pairs.
    pub fn program_group_hitgroup(
        &mut self,
        closest_hit: Option<(&Module, Ustr)>,
        any_hit: Option<(&Module, Ustr)>,
        intersection: Option<(&Module, Ustr)>,
    ) -> Result<ProgramGroup> {
        let desc =
            ProgramGroupDesc::hitgroup(closest_hit, any_hit, intersection);
        Ok(self.program_group_create_single(&desc)?.0)
    }

    /// Destroy `program_group`
    /// 
    /// # Safety
    /// Thread safety: A program group must not be destroyed while it is still
    /// in use by concurrent API calls in other threads.
    pub fn program_group_destroy(
        &mut self,
        program_group: ProgramGroup,
    ) -> Result<()> {
        unsafe {
            sys::optixProgramGroupDestroy(program_group.inner)
                .to_result()
                .map_err(|source| Error::ProgramGroupDestroy { source })
        }
    }
}

impl<'m> From<&ProgramGroupDesc<'m>> for sys::OptixProgramGroupDesc {
    fn from(desc: &ProgramGroupDesc<'m>) -> sys::OptixProgramGroupDesc {
        unsafe {
            match &desc {
            ProgramGroupDesc::Raygen(ProgramGroupModule {
                module,
                entry_function_name,
            }) => sys::OptixProgramGroupDesc {
                kind:
                    sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                    raygen: sys::OptixProgramGroupSingleModule {
                        module: module.inner,
                        entryFunctionName: entry_function_name.as_char_ptr(),
                    },
                },
                flags: 0,
            },
            ProgramGroupDesc::Miss(ProgramGroupModule {
                module,
                entry_function_name,
            }) => sys::OptixProgramGroupDesc {
                kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
                __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                    miss: sys::OptixProgramGroupSingleModule {
                        module: module.inner,
                        entryFunctionName: entry_function_name.as_char_ptr(),
                    },
                },
                flags: 0,
            },
            ProgramGroupDesc::Exception(ProgramGroupModule {
                module,
                entry_function_name,
            }) => sys::OptixProgramGroupDesc {
                kind: sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_EXCEPTION,
                __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                    miss: sys::OptixProgramGroupSingleModule {
                        module: module.inner,
                        entryFunctionName: entry_function_name.as_char_ptr(),
                    },
                },
                flags: 0,
            },
            ProgramGroupDesc::Hitgroup { ch, ah, is } => {
                let mut efn_ch_ptr = std::ptr::null();
                let mut efn_ah_ptr = std::ptr::null();
                let mut efn_is_ptr = std::ptr::null();

                let module_ch = if let Some(pg_ch) = &ch {
                    efn_ch_ptr = pg_ch.entry_function_name.as_char_ptr();
                    pg_ch.module.inner
                } else {
                    std::ptr::null_mut()
                };

                let module_ah = if let Some(pg_ah) = &ah {
                    efn_ah_ptr = pg_ah.entry_function_name.as_char_ptr();
                    pg_ah.module.inner
                } else {
                    std::ptr::null_mut()
                };

                let module_is = if let Some(pg_is) = &is {
                    efn_is_ptr = pg_is.entry_function_name.as_char_ptr();
                    pg_is.module.inner
                } else {
                    std::ptr::null_mut()
                };

                sys::OptixProgramGroupDesc {
                    kind:
                        sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                    __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                        hitgroup: sys::OptixProgramGroupHitgroup {
                            moduleCH: module_ch,
                            entryFunctionNameCH: efn_ch_ptr,
                            moduleAH: module_ah,
                            entryFunctionNameAH: efn_ah_ptr,
                            moduleIS: module_is,
                            entryFunctionNameIS: efn_is_ptr,
                        },
                    },
                    flags: 0,
                }
            }
            ProgramGroupDesc::Callables { dc, cc } => {
                let (module_dc, efn_dc) = if let Some(pg_dc) = &dc {
                    (
                        pg_dc.module.inner,
                        pg_dc.entry_function_name.as_char_ptr(),
                    )
                } else {
                    (std::ptr::null_mut(), std::ptr::null())
                };

                let (module_cc, efn_cc) = if let Some(pg_cc) = &cc {
                    (
                        pg_cc.module.inner,
                        pg_cc.entry_function_name.as_char_ptr(),
                    )
                } else {
                    (std::ptr::null_mut(), std::ptr::null())
                };

                sys::OptixProgramGroupDesc {
                    kind:
                        sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_CALLABLES,
                    __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                        callables: sys::OptixProgramGroupCallables {
                            moduleDC: module_dc,
                            entryFunctionNameDC: efn_dc,
                            moduleCC: module_cc,
                            entryFunctionNameCC: efn_cc,
                        },
                        },
                    flags: 0,
                    }
                }
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct StackSizes {
    pub css_rg: u32,
    pub css_mg: u32,
    pub css_ch: u32,
    pub css_ah: u32,
    pub css_is: u32,
    pub css_cc: u32,
    pub css_dc: u32,
}
