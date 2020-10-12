use crate::{sys, Error, ProgramGroup, DeviceStorage, TypedBuffer, DeviceCopy};
type Result<T, E = Error> = std::result::Result<T, E>;

#[repr(C)]
#[repr(align(16))]
pub struct SbtRecord<T>
{
    header: sys::SbtRecordHeader,
    data: T,
}

impl<T> SbtRecord<T>
{
    pub fn pack(data: T, program_group: &ProgramGroup) -> Result<SbtRecord<T>> {
        let mut rec = SbtRecord {
            header: sys::SbtRecordHeader::default(),
            data,
        };

        unsafe {
            sys::optixSbtRecordPackHeader(
                program_group.inner,
                &mut rec as *mut _ as *mut std::os::raw::c_void,
            )
            .to_result()
            .map(|_| rec)
            .map_err(|source| Error::PackSbtRecord { source })
        }
    }
}

unsafe impl<T: DeviceCopy> DeviceCopy for SbtRecord<T> {}

#[repr(C)]
pub struct ShaderBindingTable {
    raygen_record: cu::sys::CUdeviceptr,
    exception_record: cu::sys::CUdeviceptr,
    miss_record_base: cu::sys::CUdeviceptr,
    miss_record_stride_in_bytes: u32,
    miss_record_count: u32,
    hitgroup_record_base: cu::sys::CUdeviceptr,
    hitgroup_record_stride_in_bytes: u32,
    hitgroup_record_count: u32,
    callables_record_base: cu::sys::CUdeviceptr,
    callables_record_stride_in_bytes: u32,
    callables_record_count: u32,
}

impl ShaderBindingTable {
    pub fn new<RG: DeviceCopy, A: cu::DeviceAllocRef>(
        buf_raygen_record: &TypedBuffer<SbtRecord<RG>, A>,
    ) -> Self {
        let raygen_record = buf_raygen_record.device_ptr().0;
        ShaderBindingTable {
            raygen_record,
            exception_record: 0,
            miss_record_base: 0,
            miss_record_stride_in_bytes: 0,
            miss_record_count: 0,
            hitgroup_record_base: 0,
            hitgroup_record_stride_in_bytes: 0,
            hitgroup_record_count: 0,
            callables_record_base: 0,
            callables_record_stride_in_bytes: 0,
            callables_record_count: 0,
        }
    }

    pub fn build(self) -> sys::OptixShaderBindingTable {
        unsafe { std::mem::transmute::<ShaderBindingTable, sys::OptixShaderBindingTable>(self) }
    }

    pub fn exception<EX: DeviceCopy, A: cu::DeviceAllocRef>(
        mut self,
        buf_exception_record: &TypedBuffer<SbtRecord<EX>, A>,
    ) -> Self {
        if buf_exception_record.len() != 1 {
            panic!(
                "SBT not psased single exception record",
            );
        }
        self.exception_record = buf_exception_record.device_ptr().0;
        self
    }

    pub fn miss<MS: DeviceCopy, A: cu::DeviceAllocRef>(
        mut self,
        buf_miss_records: &TypedBuffer<SbtRecord<MS>, A>,
    ) -> Self {
        if buf_miss_records.len() == 0 {
            panic!("SBT passed empty miss records");
        }
        self.miss_record_base = buf_miss_records.device_ptr().0;
        self.miss_record_stride_in_bytes = std::mem::size_of::<SbtRecord<MS>>() as u32;
        self.miss_record_count = buf_miss_records.len() as u32;
        self
    }

    pub fn hitgroup<HG: DeviceCopy, A: cu::DeviceAllocRef>(
        mut self,
        buf_hitgroup_records: &TypedBuffer<SbtRecord<HG>, A>,
    ) -> Self {
        if buf_hitgroup_records.len() == 0 {
            panic!("SBT passed empty hitgroup records");
        }
        self.hitgroup_record_base = buf_hitgroup_records.device_ptr().0;
        self.hitgroup_record_stride_in_bytes = std::mem::size_of::<SbtRecord<HG>>() as u32;
        self.hitgroup_record_count = buf_hitgroup_records.len() as u32;
        self
    }

    pub fn callables<CL: DeviceCopy, A: cu::DeviceAllocRef>(
        mut self,
        buf_callables_records: &TypedBuffer<SbtRecord<CL>, A>,
    ) -> Self {
        if buf_callables_records.len() == 0 {
            panic!("SBT passed empty callables records");
        }
        self.callables_record_base = buf_callables_records.device_ptr().0;
        self.callables_record_stride_in_bytes =
            std::mem::size_of::<SbtRecord<CL>>() as u32;
        self.callables_record_count = buf_callables_records.len() as u32;
        self
    }
}
