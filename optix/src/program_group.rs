use optix_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

use super::device_context::DeviceContext;
use super::module::ModuleRef;

use std::ffi::CStr;
use ustr::Ustr;

#[derive(Clone)]
pub struct ProgramGroupModule {
    pub module: ModuleRef,
    pub entry_function_name: Ustr,
}

pub enum ProgramGroupDesc {
    Raygen(ProgramGroupModule),
    Miss(ProgramGroupModule),
    Hitgroup {
        ch: Option<ProgramGroupModule>,
        ah: Option<ProgramGroupModule>,
        is: Option<ProgramGroupModule>,
    },
    Callables {
        dc: Option<ProgramGroupModule>,
        cc: Option<ProgramGroupModule>,
    },
}

pub struct ProgramGroup {
    pub(crate) pg: sys::OptixProgramGroup,
    _desc: ProgramGroupDesc,
}

impl PartialEq for ProgramGroup {
    fn eq(&self, rhs: &ProgramGroup) -> bool {
        self.pg == rhs.pg
    }
}

pub type ProgramGroupRef = super::Ref<ProgramGroup>;

impl ProgramGroup {
    pub fn sys_ptr(&self) -> sys::OptixProgramGroup {
        self.pg
    }
}

impl Drop for ProgramGroup {
    fn drop(&mut self) {
        unsafe {
            sys::optixProgramGroupDestroy(self.pg);
        }
    }
}

impl DeviceContext {
    pub fn program_group_create(
        &mut self,
        desc: ProgramGroupDesc,
    ) -> Result<(ProgramGroupRef, String)> {
        let pg_options = sys::OptixProgramGroupOptions { placeholder: 0 };

        let mut log = [0u8; 4096];
        let mut log_len = log.len();

        let mut pg = std::ptr::null_mut();

        unsafe {
            match &desc {
                ProgramGroupDesc::Raygen(ProgramGroupModule {
                    module,
                    entry_function_name,
                }) => {
                    let pg_desc = sys::OptixProgramGroupDesc {
                    kind:
                        sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                    __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                        raygen: sys::OptixProgramGroupSingleModule {
                            module: module.module,
                            entryFunctionName: entry_function_name.as_char_ptr(),
                        },
                    },
                    flags: 0,
                };
                    let res = sys::optixProgramGroupCreate(
                        self.ctx,
                        &pg_desc,
                        1,
                        &pg_options,
                        log.as_mut_ptr() as *mut i8,
                        &mut log_len,
                        &mut pg,
                    );

                    let log = CStr::from_bytes_with_nul(&log[0..log_len])
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();

                    if res != sys::OptixResult::OPTIX_SUCCESS {
                        return Err(Error::ProgramGroupCreationFailed {
                            source: res.into(),
                            log,
                        });
                    }

                    let pg = super::Ref::new(ProgramGroup { pg, _desc: desc });
                    // self.program_groups.push(super::Ref::clone(&pg));
                    Ok((pg, log))
                }
                ProgramGroupDesc::Miss(ProgramGroupModule {
                    module,
                    entry_function_name,
                }) => {
                    let pg_desc = sys::OptixProgramGroupDesc {
                    kind:
                        sys::OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_MISS,
                    __bindgen_anon_1: sys::OptixProgramGroupDesc__bindgen_ty_1 {
                        miss: sys::OptixProgramGroupSingleModule {
                            module: module.module,
                            entryFunctionName: entry_function_name.as_char_ptr(),
                        },
                    },
                    flags: 0,
                };
                    let res = sys::optixProgramGroupCreate(
                        self.ctx,
                        &pg_desc,
                        1,
                        &pg_options,
                        log.as_mut_ptr() as *mut i8,
                        &mut log_len,
                        &mut pg,
                    );

                    let log = CStr::from_bytes_with_nul(&log[0..log_len])
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();

                    if res != sys::OptixResult::OPTIX_SUCCESS {
                        return Err(Error::ProgramGroupCreationFailed {
                            source: res.into(),
                            log,
                        });
                    }

                    let pg = super::Ref::new(ProgramGroup { pg, _desc: desc });
                    // self.program_groups.push(super::Ref::clone(&pg));
                    Ok((pg, log))
                }
                ProgramGroupDesc::Hitgroup { ch, ah, is } => {
                    let mut efn_ch_ptr = std::ptr::null();
                    let mut efn_ah_ptr = std::ptr::null();
                    let mut efn_is_ptr = std::ptr::null();

                    let module_ch = if let Some(pg_ch) = &ch {
                        efn_ch_ptr = pg_ch.entry_function_name.as_char_ptr();
                        pg_ch.module.module
                    } else {
                        std::ptr::null_mut()
                    };

                    let module_ah = if let Some(pg_ah) = &ah {
                        efn_ah_ptr = pg_ah.entry_function_name.as_char_ptr();
                        pg_ah.module.module
                    } else {
                        std::ptr::null_mut()
                    };

                    let module_is = if let Some(pg_is) = &is {
                        efn_is_ptr = pg_is.entry_function_name.as_char_ptr();
                        pg_is.module.module
                    } else {
                        std::ptr::null_mut()
                    };

                    let pg_desc = sys::OptixProgramGroupDesc {
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
                };
                    let res = sys::optixProgramGroupCreate(
                        self.ctx,
                        &pg_desc,
                        1,
                        &pg_options,
                        log.as_mut_ptr() as *mut i8,
                        &mut log_len,
                        &mut pg,
                    );

                    let log = CStr::from_bytes_with_nul(&log[0..log_len])
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();

                    if res != sys::OptixResult::OPTIX_SUCCESS {
                        return Err(Error::ProgramGroupCreationFailed {
                            source: res.into(),
                            log,
                        });
                    }

                    let pg = super::Ref::new(ProgramGroup { pg, _desc: desc });
                    // self.program_groups.push(super::Ref::clone(&pg));
                    Ok((pg, log))
                }
                ProgramGroupDesc::Callables { dc, cc } => {
                    let (module_dc, efn_dc) = if let Some(pg_dc) = &dc {
                        (
                            pg_dc.module.module,
                            pg_dc.entry_function_name.as_char_ptr(),
                        )
                    } else {
                        (std::ptr::null_mut(), std::ptr::null())
                    };

                    let (module_cc, efn_cc) = if let Some(pg_cc) = &cc {
                        (
                            pg_cc.module.module,
                            pg_cc.entry_function_name.as_char_ptr(),
                        )
                    } else {
                        (std::ptr::null_mut(), std::ptr::null())
                    };

                    let pg_desc = sys::OptixProgramGroupDesc {
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
                    };

                    let res = sys::optixProgramGroupCreate(
                        self.ctx,
                        &pg_desc,
                        1,
                        &pg_options,
                        log.as_mut_ptr() as *mut i8,
                        &mut log_len,
                        &mut pg,
                    );

                    let log = CStr::from_bytes_with_nul(&log[0..log_len])
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();

                    if res != sys::OptixResult::OPTIX_SUCCESS {
                        return Err(Error::ProgramGroupCreationFailed {
                            source: res.into(),
                            log,
                        });
                    }

                    let pg = super::Ref::new(ProgramGroup { pg, _desc: desc });
                    Ok((pg, log))
                }
            }
        }
    }
}
