use std::ffi::CStr;

use crate::sys;

use crate::Error;
type Result<T, E = Error> = std::result::Result<T, E>;

pub struct Device {
    pub(crate) inner: sys::CUdevice,
}

impl Device {
    pub fn get(ordinal: i32) -> Result<Device> {
        let mut inner: sys::CUdevice = 0;
        unsafe {
            sys::cuDeviceGet(&mut inner, ordinal)
                .to_result()
                .map(|_| Device { inner })
        }
    }

    pub fn get_attribute(&self, attrib: DeviceAttribute) -> Result<i32> {
        let mut i = 0i32;
        unsafe {
            sys::cuDeviceGetAttribute(&mut i, attrib as u32, self.inner)
                .to_result()
                .map(|_| i)
        }
    }

    pub fn get_count() -> Result<i32> {
        let mut i = 0i32;
        unsafe { sys::cuDeviceGetCount(&mut i).to_result().map(|_| i) }
    }

    pub fn get_name(&self) -> Result<String> {
        let mut chars = [0i8; 256];
        unsafe {
            sys::cuDeviceGetName(chars.as_mut_ptr(), 256, self.inner)
                .to_result()
                .map(|_| {
                    CStr::from_ptr(chars.as_ptr()).to_string_lossy().to_string()
                })
        }
    }

    pub fn total_mem(&self) -> Result<usize> {
        let mut sz = 0usize;
        unsafe {
            sys::cuDeviceTotalMem_v2(
                &mut sz as *mut usize as *mut u64,
                self.inner,
            )
            .to_result()
            .map(|_| sz)
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DeviceAttribute {
    MaxThreadsPerBlock = 1,
    MaxBlockDimX = 2,
    MaxBlockDimY = 3,
    MaxBlockDimZ = 4,
    MaxGridDimX = 5,
    MaxGridDimY = 6,
    MaxGridDimZ = 7,
    MaxSharedMemoryPerBlock = 8,
    TotalConstantMemory = 9,
    WarpSize = 10,
    MaxPitch = 11,
    MaxRegistersPerBlock = 12,
    ClockRate = 13,
    TextureAlignment = 14,
    GpuOverlap = 15,
    MultiprocessorCount = 16,
    KernelExecTimeout = 17,
    Integrated = 18,
    CanMapHostMemory = 19,
    ComputeMode = 20,
    MaximumTexture1DWidth = 21,
    MaximumTexture2DWidth = 22,
    MaximumTexture2DHeight = 23,
    MaximumTexture3DWidth = 24,
    MaximumTexture3DHeight = 25,
    MaximumTexture3DDepth = 26,
    MaximumTexture2DLayeredWidth = 27,
    MaximumTexture2DLayeredHeight = 28,
    MaximumTexture2DLayeredLayers = 29,
    SurfaceAlignment = 30,
    ConcurrentKernels = 31,
    EccEnabled = 32,
    PciBusId = 33,
    PciDeviceId = 34,
    TccDriver = 35,
    MemoryClockRate = 36,
    GlobalMemoryBusWidth = 37,
    L2CacheSize = 38,
    MaxThreadsPerMultiprocessor = 39,
    AsyncEngineCount = 40,
    UnifiedAddressing = 41,
    MaximumTexture1DLayeredWidth = 42,
    MaximumTexture1DLayeredLayers = 43,
    CanTex2DGather = 44,
    MaximumTexture2DGatherWidth = 45,
    MaximumTexture2DGatherHeight = 46,
    MaximumTexture3DWidthAlternate = 47,
    MaximumTexture3DHeightAlternate = 48,
    MaximumTexture3DDepthAlternate = 49,
    PciDomainId = 50,
    TexturePitchAlignment = 51,
    MaximumTexturecubemapWidth = 52,
    MaximumTexturecubemapLayeredWidth = 53,
    MaximumTexturecubemapLayeredLayers = 54,
    MaximumSurface1DWidth = 55,
    MaximumSurface2DWidth = 56,
    MaximumSurface2DHeight = 57,
    MaximumSurface3DWidth = 58,
    MaximumSurface3DHeight = 59,
    MaximumSurface3DDepth = 60,
    MaximumSurface1DLayeredWidth = 61,
    MaximumSurface1DLayeredLayers = 62,
    MaximumSurface2DLayeredWidth = 63,
    MaximumSurface2DLayeredHeight = 64,
    MaximumSurface2DLayeredLayers = 65,
    MaximumSurfacecubemapWidth = 66,
    MaximumSurfacecubemapLayeredWidth = 67,
    MaximumSurfacecubemapLayeredLayers = 68,
    MaximumTexture1DLinearWidth = 69,
    MaximumTexture2DLinearWidth = 70,
    MaximumTexture2DLinearHeight = 71,
    MaximumTexture2DLinearPitch = 72,
    MaximumTexture2DMipmappedWidth = 73,
    MaximumTexture2DMipmappedHeight = 74,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    MaximumTexture1DMipmappedWidth = 77,
    StreamPrioritiesSupported = 78,
    GlobalL1CacheSupported = 79,
    LocalL1CacheSupported = 80,
    MaxSharedMemoryPerMultiprocessor = 81,
    MaxRegistersPerMultiprocessor = 82,
    ManagedMemory = 83,
    MultiGpuBoard = 84,
    MultiGpuBoardGroupId = 85,
    HostNativeAtomicSupported = 86,
    SingleToDoublePrecisionPerfRatio = 87,
    PageableMemoryAccess = 88,
    ConcurrentManagedAccess = 89,
    ComputePreemptionSupported = 90,
    CanUseHostPointerForRegisteredMem = 91,
    CanUseStreamMemOps = 92,
    CanUse_64BitStreamMemOps = 93,
    CanUseStreamWaitValueNor = 94,
    CooperativeLaunch = 95,
    CooperativeMultiDeviceLaunch = 96,
    MaxSharedMemoryPerBlockOptin = 97,
    CanFlushRemoteWrites = 98,
    HostRegisterSupported = 99,
    PageableMemoryAccessUsesHostPageTables = 100,
    DirectManagedMemAccessFromHost = 101,
    VirtualAddressManagementSupported = 102,
    HandleTypePosixFileDescriptorSupported = 103,
    HandleTypeWin32HandleSupported = 104,
    HandleTypeWin32KmtHandleSupported = 105,
    MaxBlocksPerMultiprocessor = 106,
    GenericCompressionSupported = 107,
    MaxPersistingL2CacheSize = 108,
    MaxAccessPolicyWindowSize = 109,
    GpuDirectRdmaWithCudaVmmSupported = 110,
    ReservedSharedMemoryPerBlock = 111,
    AttributeMax = 112,
}
