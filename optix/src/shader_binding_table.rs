use optix_sys as sys;

use super::error::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

use super::SbtRecord;

pub struct ShaderBindingTable {
    pub(crate) sbt: sys::OptixShaderBindingTable,
    rg: cuda::Buffer,
    ex: Option<cuda::Buffer>,
    ms: Option<cuda::Buffer>,
    hg: Option<cuda::Buffer>,
    cl: Option<cuda::Buffer>,
}

pub struct ShaderBindingTableBuilder {
    rg: cuda::Buffer,
    ex: Option<cuda::Buffer>,
    ms: Option<cuda::Buffer>,
    ms_stride: u32,
    ms_count: u32,
    hg: Option<cuda::Buffer>,
    hg_stride: u32,
    hg_count: u32,
    cl: Option<cuda::Buffer>,
    cl_stride: u32,
    cl_count: u32,
}

impl ShaderBindingTable {
    pub fn new<RG: SbtRecord>(rec_rg: &RG) -> ShaderBindingTableBuilder {
        ShaderBindingTableBuilder::new(rec_rg)
    }
}

impl ShaderBindingTableBuilder {
    pub fn new<RG: SbtRecord>(rec_rg: &RG) -> ShaderBindingTableBuilder {
        ShaderBindingTableBuilder {
            rg: cuda::Buffer::with_data(std::slice::from_ref(rec_rg)).unwrap(),
            ex: None,
            ms: None,
            ms_stride: 0,
            ms_count: 0,
            hg: None,
            hg_stride: 0,
            hg_count: 0,
            cl: None,
            cl_stride: 0,
            cl_count: 0,
        }
    }

    pub fn exception_record<EX: SbtRecord>(
        mut self,
        rec_ex: &EX,
    ) -> ShaderBindingTableBuilder {
        self.ex = Some(
            cuda::Buffer::with_data(std::slice::from_ref(rec_ex)).unwrap(),
        );

        self
    }

    pub fn miss_records<MS: SbtRecord>(
        mut self,
        rec_miss: &[MS],
    ) -> ShaderBindingTableBuilder {
        self.ms = Some(cuda::Buffer::with_data(rec_miss).unwrap());
        self.ms_stride = std::mem::size_of::<MS>() as u32;
        self.ms_count = rec_miss.len() as u32;

        self
    }

    pub fn hitgroup_records<HG: SbtRecord>(
        mut self,
        rec_hg: &[HG],
    ) -> ShaderBindingTableBuilder {
        self.hg = Some(cuda::Buffer::with_data(rec_hg).unwrap());
        self.hg_stride = std::mem::size_of::<HG>() as u32;
        self.hg_count = rec_hg.len() as u32;

        self
    }

    pub fn callables_records<CL: SbtRecord>(
        mut self,
        rec_callables: &[CL],
    ) -> ShaderBindingTableBuilder {
        self.cl = Some(cuda::Buffer::with_data(rec_callables).unwrap());
        self.cl_stride = std::mem::size_of::<CL>() as u32;
        self.cl_count = rec_callables.len() as u32;

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
            ex: self.ex,
            ms: self.ms,
            hg: self.hg,
            cl: self.cl,
        }
    }
}
