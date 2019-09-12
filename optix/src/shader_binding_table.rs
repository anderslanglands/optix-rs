use optix_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

use super::{DeviceShareable, ProgramGroupRef};

use std::any::Any;

pub struct ShaderBindingTable {
    pub(crate) sbt: sys::OptixShaderBindingTable,
    rg: cuda::Buffer,
    rec_rg: Box<dyn Any>,
    ex: Option<cuda::Buffer>,
    rec_ex: Option<Box<dyn Any>>,
    ms: Option<cuda::Buffer>,
    rec_ms: Vec<Box<dyn Any>>,
    hg: Option<cuda::Buffer>,
    rec_hg: Vec<Box<dyn Any>>,
    cl: Option<cuda::Buffer>,
    rec_cl: Vec<Box<dyn Any>>,
}

pub struct ShaderBindingTableBuilder {
    rg: cuda::Buffer,
    rec_rg: Box<dyn Any>,
    ex: Option<cuda::Buffer>,
    rec_ex: Option<Box<dyn Any>>,
    ms: Option<cuda::Buffer>,
    ms_stride: u32,
    ms_count: u32,
    rec_ms: Vec<Box<dyn Any>>,
    hg: Option<cuda::Buffer>,
    hg_stride: u32,
    hg_count: u32,
    rec_hg: Vec<Box<dyn Any>>,
    cl: Option<cuda::Buffer>,
    cl_stride: u32,
    cl_count: u32,
    rec_cl: Vec<Box<dyn Any>>,
}

impl ShaderBindingTable {
    pub fn new<T>(rec_rg: SbtRecord<T>) -> ShaderBindingTableBuilder
    where
        T: DeviceShareable + 'static,
    {
        ShaderBindingTableBuilder::new(rec_rg)
    }
}

impl ShaderBindingTableBuilder {
    pub fn new<T>(rec_rg: SbtRecord<T>) -> ShaderBindingTableBuilder
    where
        T: DeviceShareable + 'static,
    {
        let rec_rg_d = rec_rg.to_device_record();
        ShaderBindingTableBuilder {
            rg: cuda::Buffer::with_data(std::slice::from_ref(&rec_rg_d))
                .unwrap(),
            rec_rg: Box::new(rec_rg),
            ex: None,
            rec_ex: None,
            ms: None,
            ms_stride: 0,
            ms_count: 0,
            rec_ms: Vec::new(),
            hg: None,
            hg_stride: 0,
            hg_count: 0,
            rec_hg: Vec::new(),
            cl: None,
            cl_stride: 0,
            cl_count: 0,
            rec_cl: Vec::new(),
        }
    }

    pub fn exception_record<T>(
        mut self,
        rec_ex: SbtRecord<T>,
    ) -> ShaderBindingTableBuilder
    where
        T: DeviceShareable + 'static,
    {
        let rec_ex_d = rec_ex.to_device_record();
        self.ex = Some(
            cuda::Buffer::with_data(std::slice::from_ref(&rec_ex_d)).unwrap(),
        );
        self.rec_ex = Some(Box::new(rec_ex));

        self
    }

    pub fn miss_records<T>(
        mut self,
        rec_miss: Vec<SbtRecord<T>>,
    ) -> ShaderBindingTableBuilder
    where
        T: DeviceShareable + 'static,
    {
        let rec_miss_d: Vec<SbtRecordDevice<T::Target>> =
            rec_miss.iter().map(|r| r.to_device_record()).collect();
        self.ms = Some(cuda::Buffer::with_data(&rec_miss_d).unwrap());
        self.ms_stride =
            std::mem::size_of::<SbtRecordDevice<T::Target>>() as u32;
        self.ms_count = rec_miss.len() as u32;
        for r in rec_miss {
            self.rec_ms.push(Box::new(r));
        }

        self
    }

    pub fn hitgroup_records<T>(
        mut self,
        rec_hg: Vec<SbtRecord<T>>,
    ) -> ShaderBindingTableBuilder
    where
        T: DeviceShareable,
    {
        let rec_hg_d: Vec<SbtRecordDevice<T::Target>> =
            rec_hg.iter().map(|r| r.to_device_record()).collect();
        self.hg = Some(cuda::Buffer::with_data(&rec_hg_d).unwrap());
        self.hg_stride =
            std::mem::size_of::<SbtRecordDevice<T::Target>>() as u32;
        self.hg_count = rec_hg.len() as u32;
        for r in rec_hg {
            self.rec_hg.push(Box::new(r));
        }

        self
    }

    pub fn callables_records<T>(
        mut self,
        rec_cl: Vec<SbtRecord<T>>,
    ) -> ShaderBindingTableBuilder
    where
        T: DeviceShareable,
    {
        let rec_cl_d: Vec<SbtRecordDevice<T::Target>> =
            rec_cl.iter().map(|r| r.to_device_record()).collect();
        self.cl = Some(cuda::Buffer::with_data(&rec_cl_d).unwrap());
        self.cl_stride =
            std::mem::size_of::<SbtRecordDevice<T::Target>>() as u32;
        self.cl_count = rec_cl.len() as u32;
        for r in rec_cl {
            self.rec_cl.push(Box::new(r));
        }

        self
    }

    pub fn build(self) -> ShaderBindingTable {
        ShaderBindingTable {
            sbt: sys::OptixShaderBindingTable {
                raygenRecord: self.rg.as_device_ptr(),
                exceptionRecord: if let Some(ex) = &self.ex {
                    ex.as_device_ptr()
                } else {
                    0
                },
                missRecordBase: if let Some(ms) = &self.ms {
                    ms.as_device_ptr()
                } else {
                    0
                },
                missRecordStrideInBytes: self.ms_stride,
                missRecordCount: self.ms_count,
                hitgroupRecordBase: if let Some(hg) = &self.hg {
                    hg.as_device_ptr()
                } else {
                    0
                },
                hitgroupRecordStrideInBytes: self.hg_stride,
                hitgroupRecordCount: self.hg_count,
                callablesRecordBase: if let Some(cl) = &self.cl {
                    cl.as_device_ptr()
                } else {
                    0
                },
                callablesRecordStrideInBytes: self.cl_stride,
                callablesRecordCount: self.cl_count,
            },
            rg: self.rg,
            rec_rg: self.rec_rg,
            ex: self.ex,
            rec_ex: self.rec_ex,
            ms: self.ms,
            rec_ms: self.rec_ms,
            hg: self.hg,
            rec_hg: self.rec_hg,
            cl: self.cl,
            rec_cl: self.rec_cl,
        }
    }
}

pub struct SbtRecord<T>
where
    T: DeviceShareable + 'static,
{
    program_group: ProgramGroupRef,
    pub data: T,
}

impl<T> SbtRecord<T>
where
    T: DeviceShareable + 'static,
{
    pub fn new(data: T, program_group: ProgramGroupRef) -> SbtRecord<T> {
        SbtRecord {
            program_group,
            data,
        }
    }

    pub fn to_device_record(&self) -> SbtRecordDevice<T::Target> {
        let mut rec = SbtRecordDevice {
            header: [0u8; 32],
            data: self.data.to_device(),
        };

        let res = unsafe {
            optix_sys::optixSbtRecordPackHeader(
                self.program_group.sys_ptr(),
                rec.header.as_mut_ptr() as *mut std::os::raw::c_void,
            )
        };
        if res != optix_sys::OptixResult::OPTIX_SUCCESS {
            panic!("optixSbtRecordPackHeader failed");
        }

        rec
    }
}

#[repr(C)]
#[repr(align(16))]
pub struct SbtRecordDevice<T> {
    header: [u8; 32],
    data: T,
}
