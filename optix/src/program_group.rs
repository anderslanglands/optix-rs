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
pub struct ProgramGroup {
    pub(crate) inner: sys::OptixProgramGroup,
}

impl PartialEq for ProgramGroup {
    fn eq(&self, rhs: &ProgramGroup) -> bool {
        self.inner == rhs.inner
    }
}

impl DeviceContext {
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
