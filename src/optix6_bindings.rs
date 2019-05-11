#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

pub const OPTIX_VERSION: u32 = 60000;

#[repr(u32)]
/// OptiX formats
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Format {
   ///< Format unknown
   UNKNOWN = 256,
   ///< Float
   FLOAT = 257,
   ///< sizeof(float)*2
   FLOAT2 = 258,
   ///< sizeof(float)*3
   FLOAT3 = 259,
   ///< sizeof(float)*4
   FLOAT4 = 260,
   ///< BYTE
   BYTE = 261,
   ///< sizeof(CHAR)*2
   BYTE2 = 262,
   ///< sizeof(CHAR)*3
   BYTE3 = 263,
   ///< sizeof(CHAR)*4
   BYTE4 = 264,
   ///< UCHAR
   UNSIGNED_BYTE = 265,
   ///< sizeof(UCHAR)*2
   UNSIGNED_BYTE2 = 266,
   ///< sizeof(UCHAR)*3
   UNSIGNED_BYTE3 = 267,
   ///< sizeof(UCHAR)*4
   UNSIGNED_BYTE4 = 268,
   ///< SHORT
   SHORT = 269,
   ///< sizeof(SHORT)*2
   SHORT2 = 270,
   ///< sizeof(SHORT)*3
   SHORT3 = 271,
   ///< sizeof(SHORT)*4
   SHORT4 = 272,
   ///< USHORT
   UNSIGNED_SHORT = 273,
   ///< sizeof(USHORT)*2
   UNSIGNED_SHORT2 = 274,
   ///< sizeof(USHORT)*3
   UNSIGNED_SHORT3 = 275,
   ///< sizeof(USHORT)*4
   UNSIGNED_SHORT4 = 276,
   ///< INT
   INT = 277,
   ///< sizeof(INT)*2
   INT2 = 278,
   ///< sizeof(INT)*3
   INT3 = 279,
   ///< sizeof(INT)*4
   INT4 = 280,
   ///< sizeof(UINT)
   UNSIGNED_INT = 281,
   ///< sizeof(UINT)*2
   UNSIGNED_INT2 = 282,
   ///< sizeof(UINT)*3
   UNSIGNED_INT3 = 283,
   ///< sizeof(UINT)*4
   UNSIGNED_INT4 = 284,
   ///< User Format
   USER = 285,
   ///< Buffer Id
   BUFFER_ID = 286,
   ///< Program Id
   PROGRAM_ID = 287,
   ///< half float
   HALF = 288,
   ///< sizeof(half float)*2
   HALF2 = 289,
   ///< sizeof(half float)*3
   HALF3 = 290,
   ///< sizeof(half float)*4
   HALF4 = 291,
   ///< LONG_LONG
   LONG_LONG = 292,
   ///< sizeof(LONG_LONG)*2
   LONG_LONG2 = 293,
   ///< sizeof(LONG_LONG)*3
   LONG_LONG3 = 294,
   ///< sizeof(LONG_LONG)*4
   LONG_LONG4 = 295,
   ///< sizeof(ULONG_LONG)
   UNSIGNED_LONG_LONG = 296,
   ///< sizeof(ULONG_LONG)*2
   UNSIGNED_LONG_LONG2 = 297,
   ///< sizeof(ULONG_LONG)*3
   UNSIGNED_LONG_LONG3 = 298,
   ///< sizeof(ULONG_LONG)*4
   UNSIGNED_LONG_LONG4 = 299,
   ///< Block Compressed RGB + optional 1-bit alpha BC1,
   ///sizeof(UINT)*2
   UNSIGNED_BC1 = 300,
   ///< Block Compressed RGB + 4-bit alpha BC2,
   ///sizeof(UINT)*4
   UNSIGNED_BC2 = 301,
   ///< Block Compressed RGBA BC3,
   ///sizeof(UINT)*4
   UNSIGNED_BC3 = 302,
   ///< Block Compressed unsigned grayscale BC4,
   ///sizeof(UINT)*2
   UNSIGNED_BC4 = 303,
   ///< Block Compressed signed   grayscale BC4,
   ///sizeof(UINT)*2
   BC4 = 304,
   ///< Block Compressed unsigned 2 x grayscale BC5,
   ///sizeof(UINT)*4
   UNSIGNED_BC5 = 305,
   ///< Block compressed signed   2 x grayscale BC5,
   ///sizeof(UINT)*4
   BC5 = 306,
   ///< Block compressed BC6 unsigned half-float,
   ///sizeof(UINT)*4
   UNSIGNED_BC6H = 307,
   ///< Block compressed BC6 signed half-float,
   ///sizeof(UINT)*4
   BC6H = 308,
   ///< Block compressed BC7,
   ///sizeof(UINT)*4
   UNSIGNED_BC7 = 309,
}

#[repr(u32)]
/// OptiX Object Types
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ObjectType {
   ///< Object Type Unknown
   UNKNOWN = 512,
   ///< Group Type
   GROUP = 513,
   ///< Geometry Group Type
   GEOMETRY_GROUP = 514,
   ///< Transform Type
   TRANSFORM = 515,
   ///< Selector Type
   SELECTOR = 516,
   ///< Geometry Instance Type
   GEOMETRY_INSTANCE = 517,
   ///< Buffer Type
   BUFFER = 518,
   ///< Texture Sampler Type
   TEXTURE_SAMPLER = 519,
   ///< Object Type
   OBJECT = 520,
   ///< Matrix Float 2x2
   MATRIX_FLOAT2x2 = 521,
   ///< Matrix Float 2x3
   MATRIX_FLOAT2x3 = 522,
   ///< Matrix Float 2x4
   MATRIX_FLOAT2x4 = 523,
   ///< Matrix Float 3x2
   MATRIX_FLOAT3x2 = 524,
   ///< Matrix Float 3x3
   MATRIX_FLOAT3x3 = 525,
   ///< Matrix Float 3x4
   MATRIX_FLOAT3x4 = 526,
   ///< Matrix Float 4x2
   MATRIX_FLOAT4x2 = 527,
   ///< Matrix Float 4x3
   MATRIX_FLOAT4x3 = 528,
   ///< Matrix Float 4x4
   MATRIX_FLOAT4x4 = 529,
   ///< Float Type
   FLOAT = 530,
   ///< Float2 Type
   FLOAT2 = 531,
   ///< Float3 Type
   FLOAT3 = 532,
   ///< Float4 Type
   FLOAT4 = 533,
   ///< 32 Bit Integer Type
   INT = 534,
   ///< 32 Bit Integer2 Type
   INT2 = 535,
   ///< 32 Bit Integer3 Type
   INT3 = 536,
   ///< 32 Bit Integer4 Type
   INT4 = 537,
   ///< 32 Bit Unsigned Integer Type
   UNSIGNED_INT = 538,
   ///< 32 Bit Unsigned Integer2 Type
   UNSIGNED_INT2 = 539,
   ///< 32 Bit Unsigned Integer3 Type
   UNSIGNED_INT3 = 540,
   ///< 32 Bit Unsigned Integer4 Type
   UNSIGNED_INT4 = 541,
   ///< User Object Type
   USER = 542,
   ///< Object Type Program - Added in OptiX 3.0
   PROGRAM = 543,
   ///< Object Type Command List - Added in OptiX 5.0
   COMMANDLIST = 544,
   ///< Object Type Postprocessing Stage - Added in OptiX 5.0
   POSTPROCESSINGSTAGE = 545,
   ///< 64 Bit Integer Type - Added in Optix 6.0
   LONG_LONG = 546,
   ///< 64 Bit Integer2 Type - Added in Optix 6.0
   LONG_LONG2 = 547,
   ///< 64 Bit Integer3 Type - Added in Optix 6.0
   LONG_LONG3 = 548,
   ///< 64 Bit Integer4 Type - Added in Optix 6.0
   LONG_LONG4 = 549,
   ///< 64 Bit Unsigned Integer Type - Added in Optix 6.0
   UNSIGNED_LONG_LONG = 550,
   ///< 64 Bit Unsigned Integer2 Type - Added in Optix 6.0
   UNSIGNED_LONG_LONG2 = 551,
   ///< 64 Bit Unsigned Integer3 Type - Added in Optix 6.0
   UNSIGNED_LONG_LONG3 = 552,
   ///< 64 Bit Unsigned Integer4 Type - Added in Optix 6.0
   UNSIGNED_LONG_LONG4 = 553,
}

#[repr(u32)]
/// Wrap mode
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum WrapMode {
   ///< Wrap repeat
   REPEAT = 0,
   ///< Clamp to edge
   CLAMP_TO_EDGE = 1,
   ///< Mirror
   MIRROR = 2,
   ///< Clamp to border
   CLAMP_TO_BORDER = 3,
}

#[repr(u32)]
/// Filter mode
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FilterMode {
   ///< Nearest
   NEAREST = 0,
   ///< Linear
   LINEAR = 1,
   ///< No filter
   NONE = 2,
}

#[repr(u32)]
/// Texture read mode
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TextureReadMode {
   ///< Read element type
   ELEMENT_TYPE = 0,
   ///< Read normalized float
   NORMALIZED_FLOAT = 1,
   ///< Read element type and apply sRGB to linear conversion during texture read for 8-bit integer buffer formats
   ELEMENT_TYPE_SRGB = 2,
   ///< Read normalized float and apply sRGB to linear conversion during texture read for 8-bit integer buffer formats
   NORMALIZED_FLOAT_SRGB = 3,
}

#[repr(u32)]
/// GL Target
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GlTarget {
   ///< GL texture 2D
   GL_TEXTURE_2D = 0,
   ///< GL texture rectangle
   GL_TEXTURE_RECTANGLE = 1,
   ///< GL texture 3D
   GL_TEXTURE_3D = 2,
   ///< GL render buffer
   GL_RENDER_BUFFER = 3,
   ///< GL texture 1D
   GL_TEXTURE_1D = 4,
   ///< GL array of 1D textures
   GL_TEXTURE_1D_ARRAY = 5,
   ///< GL array of 2D textures
   GL_TEXTURE_2D_ARRAY = 6,
   ///< GL cube map texture
   GL_TEXTURE_CUBE_MAP = 7,
   ///< GL array of cube maps
   GL_TEXTURE_CUBE_MAP_ARRAY = 8,
}

#[repr(u32)]
/// Texture index mode
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TextureIndexMode {
   ///< Texture Index normalized coordinates
   NORMALIZED_COORDINATES = 0,
   ///< Texture Index Array
   ARRAY_INDEX = 1,
}

#[repr(u32)]
/// Buffer type
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BufferType {
   ///< Input buffer for the GPU
   INPUT = 1,
   ///< Output buffer for the GPU
   OUTPUT = 2,
   ///< Ouput/Input buffer for the GPU
   INPUT_OUTPUT = 3,
   ///< Progressive stream buffer
   PROGRESSIVE_STREAM = 16,
}

#[repr(u32)]
/// Buffer flags
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BufferFlag {
   NONE = 0,
   ///< An @ref RT_BUFFER_INPUT_OUTPUT has separate copies on each device that are not synchronized
   GPU_LOCAL = 4,
   ///< A CUDA Interop buffer will only be synchronized across devices when dirtied by @ref rtBufferMap or @ref rtBufferMarkDirty
   COPY_ON_DIRTY = 8,
   ///< An @ref INPUT for which a synchronize is forced on unmapping from host and the host memory is freed
   DISCARD_HOST_MEMORY = 32,
   ///< Depth specifies the number of layers, not the depth of a 3D array
   LAYERED = 2097152,
   ///< Enables creation of cubemaps. If this flag is set, Width must be equal to Height, and Depth must be six. If the @ref LAYERED flag is also set, then Depth must be a multiple of six
   CUBEMAP = 4194304,
}

#[repr(u32)]
/// Buffer mapping flags
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BufferMapFlag {
   ///< Map buffer memory for reading
   READ = 1,
   ///< Map buffer memory for both reading and writing
   READ_WRITE = 2,
   ///< Map buffer memory for writing
   WRITE = 4,
   ///< Map buffer memory for writing, with the previous contents being undefined
   WRITE_DISCARD = 8,
}

#[repr(u32)]
/// Exceptions
///
/// <B>See also</B>
/// @ref rtContextSetExceptionEnabled,
/// @ref rtContextGetExceptionEnabled,
/// @ref rtGetExceptionCode,
/// @ref rtThrow,
/// @ref rtPrintf
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Exception {
   ///< Payload access out of bounds - Added in OptiX 6.0
   PAYLOAD_ACCESS_OUT_OF_BOUNDS = 1003,
   ///< Exception code of user exception out of bounds - Added in OptiX 6.0
   USER_EXCEPTION_CODE_OUT_OF_BOUNDS = 1004,
   ///< Trace depth exceeded - Added in Optix 6.0
   TRACE_DEPTH_EXCEEDED = 1005,
   ///< Program ID not valid
   PROGRAM_ID_INVALID = 1006,
   ///< Texture ID not valid
   TEXTURE_ID_INVALID = 1007,
   ///< Buffer ID not valid
   BUFFER_ID_INVALID = 1018,
   ///< Index out of bounds
   INDEX_OUT_OF_BOUNDS = 1019,
   ///< Stack overflow
   STACK_OVERFLOW = 1020,
   ///< Buffer index out of bounds
   BUFFER_INDEX_OUT_OF_BOUNDS = 1021,
   ///< Invalid ray
   INVALID_RAY = 1022,
   ///< Internal error
   INTERNAL_ERROR = 1023,
   ///< First user exception code
   USER = 1024,
   ///< Last user exception code
   USER_MAX = 65535,
   ///< All exceptions
   ALL = 2147483647,
}

#[repr(i32)]
/// Result
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RtResult {
   ///< Success
   SUCCESS = 0,
   ///< Timeout callback
   RT_TIMEOUT_CALLBACK = 256,
   ///< Invalid Context
   RT_ERROR_INVALID_CONTEXT = 1280,
   ///< Invalid Value
   RT_ERROR_INVALID_VALUE = 1281,
   ///< Timeout callback
   RT_ERROR_MEMORY_ALLOCATION_FAILED = 1282,
   ///< Type Mismatch
   RT_ERROR_TYPE_MISMATCH = 1283,
   ///< Variable not found
   RT_ERROR_VARIABLE_NOT_FOUND = 1284,
   ///< Variable redeclared
   RT_ERROR_VARIABLE_REDECLARED = 1285,
   ///< Illegal symbol
   RT_ERROR_ILLEGAL_SYMBOL = 1286,
   ///< Invalid source
   RT_ERROR_INVALID_SOURCE = 1287,
   ///< Version mismatch
   RT_ERROR_VERSION_MISMATCH = 1288,
   ///< Object creation failed
   RT_ERROR_OBJECT_CREATION_FAILED = 1536,
   ///< No device
   RT_ERROR_NO_DEVICE = 1537,
   ///< Invalid device
   RT_ERROR_INVALID_DEVICE = 1538,
   ///< Invalid image
   RT_ERROR_INVALID_IMAGE = 1539,
   ///< File not found
   RT_ERROR_FILE_NOT_FOUND = 1540,
   ///< Already mapped
   RT_ERROR_ALREADY_MAPPED = 1541,
   ///< Invalid driver version
   RT_ERROR_INVALID_DRIVER_VERSION = 1542,
   ///< Context creation failed
   RT_ERROR_CONTEXT_CREATION_FAILED = 1543,
   ///< Resource not registered
   RT_ERROR_RESOURCE_NOT_REGISTERED = 1544,
   ///< Resource already registered
   RT_ERROR_RESOURCE_ALREADY_REGISTERED = 1545,
   ///< OptiX DLL failed to load
   RT_ERROR_OPTIX_NOT_LOADED = 1546,
   ///< Denoiser DLL failed to load
   RT_ERROR_DENOISER_NOT_LOADED = 1547,
   ///< SSIM predictor DLL failed to load
   RT_ERROR_SSIM_PREDICTOR_NOT_LOADED = 1548,
   ///< Driver version retrieval failed
   RT_ERROR_DRIVER_VERSION_FAILED = 1549,
   ///< No write permission on disk cache file
   RT_ERROR_DATABASE_FILE_PERMISSIONS = 1550,
   ///< Launch failed
   RT_ERROR_LAUNCH_FAILED = 2304,
   ///< Not supported
   RT_ERROR_NOT_SUPPORTED = 2560,
   ///< Connection failed
   RT_ERROR_CONNECTION_FAILED = 2816,
   ///< Authentication failed
   RT_ERROR_AUTHENTICATION_FAILED = 2817,
   ///< Connection already exists
   RT_ERROR_CONNECTION_ALREADY_EXISTS = 2818,
   ///< Network component failed to load
   RT_ERROR_NETWORK_LOAD_FAILED = 2819,
   ///< Network initialization failed
   RT_ERROR_NETWORK_INIT_FAILED = 2820,
   ///< No cluster is running
   RT_ERROR_CLUSTER_NOT_RUNNING = 2822,
   ///< Cluster is already running
   RT_ERROR_CLUSTER_ALREADY_RUNNING = 2823,
   ///< Not enough free nodes
   RT_ERROR_INSUFFICIENT_FREE_NODES = 2824,
   ///< Invalid global attribute
   RT_ERROR_INVALID_GLOBAL_ATTRIBUTE = 3072,
   ///< Error unknown
   RT_ERROR_UNKNOWN = -1,
}

#[repr(u32)]
/// Device attributes
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceAttribute {
   ///< Max Threads per Block sizeof(int)
   MAX_THREADS_PER_BLOCK = 0,
   ///< Clock rate sizeof(int)
   CLOCK_RATE = 1,
   ///< Multiprocessor count sizeof(int)
   MULTIPROCESSOR_COUNT = 2,
   ///< Execution timeout enabled sizeof(int)
   EXECUTION_TIMEOUT_ENABLED = 3,
   ///< Hardware Texture count sizeof(int)
   MAX_HARDWARE_TEXTURE_COUNT = 4,
   ///< Attribute Name
   NAME = 5,
   ///< Compute Capabilities sizeof(int2)
   COMPUTE_CAPABILITY = 6,
   ///< Total Memory sizeof(RTsize)
   TOTAL_MEMORY = 7,
   ///< TCC driver sizeof(int)
   TCC_DRIVER = 8,
   ///< CUDA device ordinal sizeof(int)
   CUDA_DEVICE_ORDINAL = 9,
   ///< PCI Bus Id
   PCI_BUS_ID = 10,
   ///< Ordinals of compatible devices sizeof(int=N) + N*sizeof(int)
   COMPATIBLE_DEVICES = 11,
}

#[repr(u32)]
/// Global attributes
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GlobalAttribute {
   ///< sizeof(int)
   DISPLAY_DRIVER_VERSION_MAJOR = 1,
   ///< sizeof(int)
   DISPLAY_DRIVER_VERSION_MINOR = 2,
   ///< sizeof(int)
   ENABLE_RTX = 268435456,
   ///< Knobs string
   DEVELOPER_OPTIONS = 268435457,
}

#[repr(u32)]
/// Context attributes
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ContextAttribute {
   ///< sizeof(int)
   MAX_TEXTURE_COUNT = 0,
   ///< sizeof(int)
   CPU_NUM_THREADS = 1,
   ///< sizeof(RTsize)
   USED_HOST_MEMORY = 2,
   ///< sizeof(int)
   GPU_PAGING_ACTIVE = 3,
   ///< sizeof(int)
   GPU_PAGING_FORCED_OFF = 4,
   ///< sizeof(int)
   DISK_CACHE_ENABLED = 5,
   ///< sizeof(int)
   PREFER_FAST_RECOMPILES = 6,
   ///< sizeof(int)
   FORCE_INLINE_USER_FUNCTIONS = 7,
   ///< 32
   OPTIX_SALT = 8,
   ///< 32
   VENDOR_SALT = 9,
   ///< variable
   PUBLIC_VENDOR_KEY = 10,
   ///< sizeof(char*)
   DISK_CACHE_LOCATION = 11,
   ///< sizeof(RTsize[2])
   DISK_CACHE_MEMORY_LIMITS = 12,
   ///< sizeof(RTsize)
   AVAILABLE_DEVICE_MEMORY = 268435456,
}

#[repr(u32)]
/// Buffer attributes
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BufferAttribute {
   ///< Format string
   STREAM_FORMAT = 0,
   ///< sizeof(int)
   STREAM_BITRATE = 1,
   ///< sizeof(int)
   STREAM_FPS = 2,
   ///< sizeof(float)
   STREAM_GAMMA = 3,
}

#[repr(u32)]
/// Motion border modes
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MotionBorderMode {
   ///< Clamp outside of bounds
   CLAMP = 0,
   ///< Vanish outside of bounds
   VANISH = 1,
}

#[repr(u32)]
/// Motion key type
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MotionKeyType {
   ///< No motion keys set
   NONE = 0,
   ///< Affine matrix format - 12 floats
   MATRIX_FLOAT12 = 1,
   ///< SRT format - 16 floats
   SRT_FLOAT16 = 2,
}

#[repr(u32)]
/// GeometryX build flags
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GeometryBuildFlags {
   ///< No special flags set
   NONE = 0,
   ///< User buffers are released after consumption by acceleration structure build
   RELEASE_BUFFERS = 16,
}

#[repr(u32)]
/// Material-dependent flags set on Geometry/GeometryTriangles
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GeometryFlags {
   ///< No special flags set
   NONE = 0,
   ///< Opaque flag, any hit program will be skipped
   DISABLE_ANYHIT = 1,
   ///< Disable primitive splitting to avoid potential duplicate any hit program execution for a single intersection
   NO_SPLITTING = 2,
}

#[repr(u32)]
/// Instance flags which override the behavior of geometry.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum InstanceFlags {
   ///< No special flag set
   NONE = 0,
   ///< Prevent triangles from getting culled
   DISABLE_TRIANGLE_CULLING = 1,
   ///< Flip triangle orientation. This affects front/backface culling.
   FLIP_TRIANGLE_FACING = 2,
   ///< Disable any-hit programs.
   ///This may yield significantly higher performance even in cases
   ///where no any-hit programs are set.
   DISABLE_ANYHIT = 4,
   ///< Override @ref RT_GEOMETRY_FLAG_DISABLE_ANYHIT
   ENFORCE_ANYHIT = 8,
}

#[repr(u32)]
/// Sentinel values
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RTbufferidnull {
   ///< sentinel for describing a non-existent buffer id
   NULL = 0,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RTprogramidnull {
   ///< sentinel for describing a non-existent program id
   NULL = 0,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RTtextureidnull {
   ///< sentinel for describing a non-existent texture id
   NULL = 0,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RTcommandlistidnull {
   ///< sentinel for describing a non-existent command list id
   NULL = 0,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RTpostprocessingstagenull {
   ///< sentinel for describing a non-existent post-processing stage id
   NULL = 0,
}

#[repr(u32)]
/// Ray flags
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum RayFlags {
   NONE = 0,
   ///< Disables any-hit programs for the ray.
   DISABLE_ANYHIT = 1,
   ///< Forces any-hit program execution for the ray.
   ENFORCE_ANYHIT = 2,
   ///< Terminates the ray after the first hit.
   TERMINATE_ON_FIRST_HIT = 4,
   ///< Disables closest-hit programs for the ray.
   DISABLE_CLOSESTHIT = 8,
   ///< Do not intersect triangle back faces.
   CULL_BACK_FACING_TRIANGLES = 16,
   ///< Do not intersect triangle front faces.
   CULL_FRONT_FACING_TRIANGLES = 32,
   ///< Do not intersect geometry which disables any-hit programs.
   CULL_DISABLED_ANYHIT = 64,
   ///< Do not intersect geometry which enforces any-hit programs.
   CULL_ENFORCED_ANYHIT = 128,
}

pub type RTvisibilitymask = ::std::os::raw::c_uint;
pub const RT_VISIBILITY_ALL: _bindgen_ty_1 = _bindgen_ty_1::RT_VISIBILITY_ALL;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum _bindgen_ty_1 {
   ///< Default @ref RTvisibilitymask
   RT_VISIBILITY_ALL = 255,
}

pub type RTsize = ::std::os::raw::c_ulong;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTacceleration_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Acceleration Structures - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTacceleration = *mut RTacceleration_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTbuffer_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Buffers - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTbuffer = *mut RTbuffer_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTcontext_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Contexts - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTcontext = *mut RTcontext_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTgeometry_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Geometry - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTgeometry = *mut RTgeometry_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTgeometrytriangles_api {
   _unused: [u8; 0],
}
/// Opaque type to handle GeometryTriangles - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTgeometrytriangles = *mut RTgeometrytriangles_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTgeometryinstance_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Geometry Instance - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTgeometryinstance = *mut RTgeometryinstance_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTgeometrygroup_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Geometry Group - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTgeometrygroup = *mut RTgeometrygroup_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTgroup_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Group - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTgroup = *mut RTgroup_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTmaterial_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Material - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTmaterial = *mut RTmaterial_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTprogram_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Program - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTprogram = *mut RTprogram_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTselector_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Selector - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTselector = *mut RTselector_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTtexturesampler_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Texture Sampler - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTtexturesampler = *mut RTtexturesampler_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTtransform_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Transform - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTtransform = *mut RTtransform_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTvariable_api {
   _unused: [u8; 0],
}
/// Opaque type to handle Variable - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTvariable = *mut RTvariable_api;

/// Opaque type to handle Object - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTobject = *mut ::std::os::raw::c_void;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTpostprocessingstage_api {
   _unused: [u8; 0],
}
/// Opaque type to handle PostprocessingStage - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTpostprocessingstage = *mut RTpostprocessingstage_api;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RTcommandlist_api {
   _unused: [u8; 0],
}
/// Opaque type to handle CommandList - Note that the *_api type should never be used directly.
///Only the typedef target name will be guaranteed to remain unchanged
pub type RTcommandlist = *mut RTcommandlist_api;

/// Callback signature for use with rtContextSetTimeoutCallback.
/// Deprecated in OptiX 6.0.
pub type RTtimeoutcallback =
   ::std::option::Option<unsafe extern "C" fn() -> ::std::os::raw::c_int>;

/// Callback signature for use with rtContextSetUsageReportCallback.
pub type RTusagereportcallback = ::std::option::Option<
   unsafe extern "C" fn(
      arg1: ::std::os::raw::c_int,
      arg2: *const ::std::os::raw::c_char,
      arg3: *const ::std::os::raw::c_char,
      arg4: *mut ::std::os::raw::c_void,
   ),
>;

extern "C" {
   /// @brief Returns the current OptiX version
   ///
   /// @ingroup ContextFreeFunctions
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGetVersion returns in \a version a numerically comparable
   /// version number of the current OptiX library.
   ///
   /// The encoding for the version number prior to OptiX 4.0.0 is major*1000 + minor*10 + micro.
   /// For versions 4.0.0 and higher, the encoding is major*10000 + minor*100 + micro.
   /// For example, for version 3.5.1 this function would return 3051, and for version 4.5.1 it would return 40501.
   ///
   /// @param[out]  version   OptiX version number
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGetVersion was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtDeviceGetDeviceCount
   ///
   pub fn rtGetVersion(version: *mut ::std::os::raw::c_uint) -> RtResult;
}
extern "C" {
   /// @brief Set a global attribute
   ///
   /// @ingroup ContextFreeFunctions
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGlobalSetAttribute sets \a p as the value of the global attribute
   /// specified by \a attrib.
   ///
   /// Each attribute can have a different size.  The sizes are given in the following list:
   ///
   ///   - @ref RT_GLOBAL_ATTRIBUTE_ENABLE_RTX          sizeof(int)
   ///
   /// @ref RT_GLOBAL_ATTRIBUTE_ENABLE_RTX is an experimental attribute which sets the execution strategy
   /// used by Optix for the next context to be created.  This attribute may be deprecated in a future release.
   /// Possible values: 0 (legacy default), 1 (compile and link programs separately).
   ///
   /// @param[in]   attrib    Attribute to set
   /// @param[in]   size      Size of the attribute being set
   /// @param[in]   p         Pointer to where the value of the attribute will be copied from.  This must point to at least \a size bytes of memory
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_GLOBAL_ATTRIBUTE - Can be returned if an unknown attribute was addressed.
   /// - @ref RT_ERROR_INVALID_VALUE - Can be returned if \a size does not match the proper size of the attribute, or if \a p
   /// is \a NULL
   ///
   /// <B>History</B>
   ///
   /// @ref rtGlobalSetAttribute was introduced in OptiX 5.1.
   ///
   /// <B>See also</B>
   /// @ref rtGlobalGetAttribute
   ///
   pub fn rtGlobalSetAttribute(
      attrib: GlobalAttribute,
      size: RTsize,
      p: *const ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a global attribute
   ///
   /// @ingroup ContextFreeFunctions
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGlobalGetAttribute returns in \a p the value of the global attribute
   /// specified by \a attrib.
   ///
   /// Each attribute can have a different size. The sizes are given in the following list:
   ///
   ///   - @ref RT_GLOBAL_ATTRIBUTE_ENABLE_RTX                            sizeof(int)
   ///   - @ref RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MAJOR           sizeof(unsigned int)
   ///   - @ref RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MINOR           sizeof(unsigend int)
   ///
   /// @ref RT_GLOBAL_ATTRIBUTE_ENABLE_RTX is an experimental setting which sets the execution strategy
   /// used by Optix for the next context to be created.
   ///
   /// @ref RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MAJOR is an attribute to query the major version of the display driver
   /// found on the system. It's the first number in the driver version displayed as xxx.yy.
   ///
   /// @ref RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MINOR is an attribute to query the minor version of the display driver
   /// found on the system. It's the second number in the driver version displayed as xxx.yy.
   ///
   /// @param[in]   attrib    Attribute to query
   /// @param[in]   size      Size of the attribute being queried.  Parameter \a p must have at least this much memory allocated
   /// @param[out]  p         Return pointer where the value of the attribute will be copied into.  This must point to at least \a size bytes of memory
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_GLOBAL_ATTRIBUTE - Can be returned if an unknown attribute was addressed.
   /// - @ref RT_ERROR_INVALID_VALUE - Can be returned if \a size does not match the proper size of the attribute, if \a p is
   /// \a NULL, or if \a attribute+ordinal does not correspond to an OptiX device
   /// - @ref RT_ERROR_DRIVER_VERSION_FAILED - Can be returned if the display driver version could not be obtained.
   ///
   /// <B>History</B>
   ///
   /// @ref rtGlobalGetAttribute was introduced in OptiX 5.1.
   ///
   /// <B>See also</B>
   /// @ref rtGlobalSetAttribute,
   ///
   pub fn rtGlobalGetAttribute(
      attrib: GlobalAttribute,
      size: RTsize,
      p: *mut ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of OptiX capable devices
   ///
   /// @ingroup ContextFreeFunctions
   ///
   /// <B>Description</B>
   ///
   /// @ref rtDeviceGetDeviceCount returns in \a count the number of compute
   /// devices that are available in the host system and will be used by
   /// OptiX.
   ///
   /// @param[out]  count   Number devices available for OptiX
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtDeviceGetDeviceCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGetVersion
   ///
   pub fn rtDeviceGetDeviceCount(
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns an attribute specific to an OptiX device
   ///
   /// @ingroup ContextFreeFunctions
   ///
   /// <B>Description</B>
   ///
   /// @ref rtDeviceGetAttribute returns in \a p the value of the per device attribute
   /// specified by \a attrib for device \a ordinal.
   ///
   /// Each attribute can have a different size.  The sizes are given in the following list:
   ///
   ///   - @ref RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK        sizeof(int)
   ///   - @ref RT_DEVICE_ATTRIBUTE_CLOCK_RATE                   sizeof(int)
   ///   - @ref RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT         sizeof(int)
   ///   - @ref RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED    sizeof(int)
   ///   - @ref RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT   sizeof(int)
   ///   - @ref RT_DEVICE_ATTRIBUTE_NAME                         up to size-1
   ///   - @ref RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY           sizeof(int2)
   ///   - @ref RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY                 sizeof(RTsize)
   ///   - @ref RT_DEVICE_ATTRIBUTE_TCC_DRIVER                   sizeof(int)
   ///   - @ref RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL          sizeof(int)
   ///   - @ref RT_DEVICE_ATTRIBUTE_PCI_BUS_ID                   up to size-1, at most 13 chars
   ///   - @ref RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES           sizeof(int)*(number of devices + 1)
   ///
   /// For \a RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES, the first \a int returned is the number
   /// of compatible device ordinals returned.  A device is always compatible with itself, so
   /// the count will always be at least one.  Size the output buffer based on the number of
   /// devices as returned by \a rtDeviceGetDeviceCount.
   ///
   /// @param[in]   ordinal   OptiX device ordinal
   /// @param[in]   attrib    Attribute to query
   /// @param[in]   size      Size of the attribute being queried.  Parameter \a p must have at least this much memory allocated
   /// @param[out]  p         Return pointer where the value of the attribute will be copied into.  This must point to at least \a size bytes of memory
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE - Can be returned if size does not match the proper size of the attribute, if \a p is
   /// \a NULL, or if \a ordinal does not correspond to an OptiX device
   ///
   /// <B>History</B>
   ///
   /// @ref rtDeviceGetAttribute was introduced in OptiX 2.0.
   /// @ref RT_DEVICE_ATTRIBUTE_TCC_DRIVER was introduced in OptiX 3.0.
   /// @ref RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL was introduced in OptiX 3.0.
   /// @ref RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtDeviceGetDeviceCount,
   /// @ref rtContextGetAttribute
   ///
   pub fn rtDeviceGetAttribute(
      ordinal: ::std::os::raw::c_int,
      attrib: DeviceAttribute,
      size: RTsize,
      p: *mut ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @ingroup rtVariableSet Variable setters
   ///
   /// @brief Functions designed to modify the value of a program variable
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableSet functions modify the value of a program variable or variable array. The
   /// target variable is specificed by \a v, which should be a value returned by
   /// @ref rtContextGetVariable.
   ///
   /// The commands \a rtVariableSet{1-2-3-4}{f-i-ui}v are used to modify the value of a
   /// program variable specified by \a v using the values passed as arguments.
   /// The number specified in the command should match the number of components in
   /// the data type of the specified program variable (e.g., 1 for float, int,
   /// unsigned int; 2 for float2, int2, uint2, etc.). The suffix \a f indicates
   /// that \a v has floating point type, the suffix \a i indicates that
   /// \a v has integral type, and the suffix \a ui indicates that that
   /// \a v has unsigned integral type. The \a v variants of this function
   /// should be used to load the program variable's value from the array specified by
   /// parameter \a v. In this case, the array \a v should contain as many elements as
   /// there are program variable components.
   ///
   /// The commands \a rtVariableSetMatrix{2-3-4}x{2-3-4}fv are used to modify the value
   /// of a program variable whose data type is a matrix. The numbers in the command
   /// names are the number of rows and columns, respectively.
   /// For example, \a 2x4 indicates a matrix with 2 rows and 4 columns (i.e., 8 values).
   /// If \a transpose is \a 0, the matrix is specified in row-major order, otherwise
   /// in column-major order or, equivalently, as a matrix with the number of rows and
   /// columns swapped in row-major order.
   ///
   /// If \a v is not a valid variable, these calls have no effect and return
   /// @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableSet were introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtVariableGet,
   /// @ref rtVariableSet,
   /// @ref rtDeclareVariable
   ///
   /// @{
   ////
   ////**
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   f1         Specifies the new float value of the program variable
   pub fn rtVariableSet1f(v: RTvariable, f1: f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   f1         Specifies the new float value of the program variable
   /// @param[in]   f2         Specifies the new float value of the program variable
   pub fn rtVariableSet2f(v: RTvariable, f1: f32, f2: f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   f1         Specifies the new float value of the program variable
   /// @param[in]   f2         Specifies the new float value of the program variable
   /// @param[in]   f3         Specifies the new float value of the program variable
   pub fn rtVariableSet3f(v: RTvariable, f1: f32, f2: f32, f3: f32)
      -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   f1         Specifies the new float value of the program variable
   /// @param[in]   f2         Specifies the new float value of the program variable
   /// @param[in]   f3         Specifies the new float value of the program variable
   /// @param[in]   f4         Specifies the new float value of the program variable
   pub fn rtVariableSet4f(
      v: RTvariable,
      f1: f32,
      f2: f32,
      f3: f32,
      f4: f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   f          Array of float values to set the variable to
   pub fn rtVariableSet1fv(v: RTvariable, f: *const f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   f          Array of float values to set the variable to
   pub fn rtVariableSet2fv(v: RTvariable, f: *const f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   f          Array of float values to set the variable to
   pub fn rtVariableSet3fv(v: RTvariable, f: *const f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   f          Array of float values to set the variable to
   pub fn rtVariableSet4fv(v: RTvariable, f: *const f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   i1         Specifies the new integer value of the program variable
   pub fn rtVariableSet1i(v: RTvariable, i1: ::std::os::raw::c_int)
      -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   i1         Specifies the new integer value of the program variable
   /// @param[in]   i2         Specifies the new integer value of the program variable
   pub fn rtVariableSet2i(
      v: RTvariable,
      i1: ::std::os::raw::c_int,
      i2: ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   i1         Specifies the new integer value of the program variable
   /// @param[in]   i2         Specifies the new integer value of the program variable
   /// @param[in]   i3         Specifies the new integer value of the program variable
   pub fn rtVariableSet3i(
      v: RTvariable,
      i1: ::std::os::raw::c_int,
      i2: ::std::os::raw::c_int,
      i3: ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   i1         Specifies the new integer value of the program variable
   /// @param[in]   i2         Specifies the new integer value of the program variable
   /// @param[in]   i3         Specifies the new integer value of the program variable
   /// @param[in]   i4         Specifies the new integer value of the program variable
   pub fn rtVariableSet4i(
      v: RTvariable,
      i1: ::std::os::raw::c_int,
      i2: ::std::os::raw::c_int,
      i3: ::std::os::raw::c_int,
      i4: ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   i          Array of integer values to set the variable to
   pub fn rtVariableSet1iv(
      v: RTvariable,
      i: *const ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   i          Array of integer values to set the variable to
   pub fn rtVariableSet2iv(
      v: RTvariable,
      i: *const ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   i          Array of integer values to set the variable to
   pub fn rtVariableSet3iv(
      v: RTvariable,
      i: *const ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   i          Array of integer values to set the variable to
   pub fn rtVariableSet4iv(
      v: RTvariable,
      i: *const ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   u1         Specifies the new unsigned integer value of the program variable
   pub fn rtVariableSet1ui(
      v: RTvariable,
      u1: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   u1         Specifies the new unsigned integer value of the program variable
   /// @param[in]   u2         Specifies the new unsigned integer value of the program variable
   pub fn rtVariableSet2ui(
      v: RTvariable,
      u1: ::std::os::raw::c_uint,
      u2: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   u1         Specifies the new unsigned integer value of the program variable
   /// @param[in]   u2         Specifies the new unsigned integer value of the program variable
   /// @param[in]   u3         Specifies the new unsigned integer value of the program variable
   pub fn rtVariableSet3ui(
      v: RTvariable,
      u1: ::std::os::raw::c_uint,
      u2: ::std::os::raw::c_uint,
      u3: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   u1         Specifies the new unsigned integer value of the program variable
   /// @param[in]   u2         Specifies the new unsigned integer value of the program variable
   /// @param[in]   u3         Specifies the new unsigned integer value of the program variable
   /// @param[in]   u4         Specifies the new unsigned integer value of the program variable
   pub fn rtVariableSet4ui(
      v: RTvariable,
      u1: ::std::os::raw::c_uint,
      u2: ::std::os::raw::c_uint,
      u3: ::std::os::raw::c_uint,
      u4: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   u          Array of unsigned integer values to set the variable to
   pub fn rtVariableSet1uiv(
      v: RTvariable,
      u: *const ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   u          Array of unsigned integer values to set the variable to
   pub fn rtVariableSet2uiv(
      v: RTvariable,
      u: *const ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   u          Array of unsigned integer values to set the variable to
   pub fn rtVariableSet3uiv(
      v: RTvariable,
      u: *const ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   u          Array of unsigned integer values to set the variable to
   pub fn rtVariableSet4uiv(
      v: RTvariable,
      u: *const ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ll1        Specifies the new long long value of the program variable
   pub fn rtVariableSet1ll(
      v: RTvariable,
      ll1: ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ll1        Specifies the new long long value of the program variable
   /// @param[in]   ll2        Specifies the new long long value of the program variable
   pub fn rtVariableSet2ll(
      v: RTvariable,
      ll1: ::std::os::raw::c_longlong,
      ll2: ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ll1        Specifies the new long long value of the program variable
   /// @param[in]   ll2        Specifies the new long long value of the program variable
   /// @param[in]   ll3        Specifies the new long long value of the program variable
   pub fn rtVariableSet3ll(
      v: RTvariable,
      ll1: ::std::os::raw::c_longlong,
      ll2: ::std::os::raw::c_longlong,
      ll3: ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ll1        Specifies the new long long value of the program variable
   /// @param[in]   ll2        Specifies the new long long value of the program variable
   /// @param[in]   ll3        Specifies the new long long value of the program variable
   /// @param[in]   ll4        Specifies the new long long value of the program variable
   pub fn rtVariableSet4ll(
      v: RTvariable,
      ll1: ::std::os::raw::c_longlong,
      ll2: ::std::os::raw::c_longlong,
      ll3: ::std::os::raw::c_longlong,
      ll4: ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ll         Array of long long values to set the variable to
   pub fn rtVariableSet1llv(
      v: RTvariable,
      ll: *const ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ll         Array of long long values to set the variable to
   pub fn rtVariableSet2llv(
      v: RTvariable,
      ll: *const ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ll         Array of long long values to set the variable to
   pub fn rtVariableSet3llv(
      v: RTvariable,
      ll: *const ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ll         Array of long long values to set the variable to
   pub fn rtVariableSet4llv(
      v: RTvariable,
      ll: *const ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ull1       Specifies the new unsigned long long value of the program variable
   pub fn rtVariableSet1ull(
      v: RTvariable,
      ull1: ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ull1       Specifies the new unsigned long long value of the program variable
   /// @param[in]   ull2       Specifies the new unsigned long long value of the program variable
   pub fn rtVariableSet2ull(
      v: RTvariable,
      ull1: ::std::os::raw::c_ulonglong,
      ull2: ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ull1       Specifies the new unsigned long long value of the program variable
   /// @param[in]   ull2       Specifies the new unsigned long long value of the program variable
   /// @param[in]   ull3       Specifies the new unsigned long long value of the program variable
   pub fn rtVariableSet3ull(
      v: RTvariable,
      ull1: ::std::os::raw::c_ulonglong,
      ull2: ::std::os::raw::c_ulonglong,
      ull3: ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ull1       Specifies the new unsigned long long value of the program variable
   /// @param[in]   ull2       Specifies the new unsigned long long value of the program variable
   /// @param[in]   ull3       Specifies the new unsigned long long value of the program variable
   /// @param[in]   ull4       Specifies the new unsigned long long value of the program variable
   pub fn rtVariableSet4ull(
      v: RTvariable,
      ull1: ::std::os::raw::c_ulonglong,
      ull2: ::std::os::raw::c_ulonglong,
      ull3: ::std::os::raw::c_ulonglong,
      ull4: ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ull        Array of unsigned long long values to set the variable to
   pub fn rtVariableSet1ullv(
      v: RTvariable,
      ull: *const ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ull        Array of unsigned long long values to set the variable to
   pub fn rtVariableSet2ullv(
      v: RTvariable,
      ull: *const ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ull        Array of unsigned long long values to set the variable to
   pub fn rtVariableSet3ullv(
      v: RTvariable,
      ull: *const ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   ull        Array of unsigned long long values to set the variable to
   pub fn rtVariableSet4ullv(
      v: RTvariable,
      ull: *const ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   transpose  Specifies row-major or column-major order
   /// @param[in]   m          Array of float values to set the matrix to
   pub fn rtVariableSetMatrix2x2fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   transpose  Specifies row-major or column-major order
   /// @param[in]   m          Array of float values to set the matrix to
   pub fn rtVariableSetMatrix2x3fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   transpose  Specifies row-major or column-major order
   /// @param[in]   m          Array of float values to set the matrix to
   pub fn rtVariableSetMatrix2x4fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   transpose  Specifies row-major or column-major order
   /// @param[in]   m          Array of float values to set the matrix to
   pub fn rtVariableSetMatrix3x2fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   transpose  Specifies row-major or column-major order
   /// @param[in]   m          Array of float values to set the matrix to
   pub fn rtVariableSetMatrix3x3fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   transpose  Specifies row-major or column-major order
   /// @param[in]   m          Array of float values to set the matrix to
   pub fn rtVariableSetMatrix3x4fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   transpose  Specifies row-major or column-major order
   /// @param[in]   m          Array of float values to set the matrix to
   pub fn rtVariableSetMatrix4x2fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   transpose  Specifies row-major or column-major order
   /// @param[in]   m          Array of float values to set the matrix to
   pub fn rtVariableSetMatrix4x3fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   transpose  Specifies row-major or column-major order
   /// @param[in]   m          Array of float values to set the matrix to
   pub fn rtVariableSetMatrix4x4fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets a program variable value to a OptiX object
   ///
   /// @ingroup Variables
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableSetObject sets a program variable to an OptiX object value.  The target
   /// variable is specified by \a v. The new value of the program variable is
   /// specified by \a object. The concrete type of \a object can be one of @ref RTbuffer,
   /// @ref RTtexturesampler, @ref RTgroup, @ref RTprogram, @ref RTselector, @ref
   /// RTgeometrygroup, or @ref RTtransform.  If \a v is not a valid variable or \a
   /// object is not a valid OptiX object, this call has no effect and returns @ref
   /// RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   v          Specifies the program variable to be set
   /// @param[in]   object     Specifies the new value of the program variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableSetObject was introduced in OptiX 1.0.  The ability to bind an @ref
   /// RTprogram to a variable was introduced in OptiX 3.0.
   ///
   /// <B>See also</B>
   /// @ref rtVariableGetObject,
   /// @ref rtContextDeclareVariable
   ///
   pub fn rtVariableSetObject(v: RTvariable, object: RTobject) -> RtResult;
}
extern "C" {
   /// @brief Defined
   ///
   /// @ingroup Variables
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableSetUserData modifies the value of a program variable whose data type is
   /// user-defined. The value copied into the variable is defined by an arbitrary region of
   /// memory, pointed to by \a ptr. The size of the memory region is given by \a size. The
   /// target variable is specified by \a v.  If \a v is not a valid variable,
   /// this call has no effect and returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   v          Specifies the program variable to be modified
   /// @param[in]   size       Specifies the size of the new value, in bytes
   /// @param[in]   ptr        Specifies a pointer to the new value of the program variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableSetUserData was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtVariableGetUserData,
   /// @ref rtContextDeclareVariable
   ///
   pub fn rtVariableSetUserData(
      v: RTvariable,
      size: RTsize,
      ptr: *const ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @ingroup rtVariableGet
   ///
   /// @brief Functions designed to modify the value of a program variable
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableGet functions return the value of a program variable or variable
   /// array. The target variable is specificed by \a v.
   ///
   /// The commands \a rtVariableGet{1-2-3-4}{f-i-ui}v are used to query the value
   /// of a program variable specified by \a v using the pointers passed as arguments
   /// as return locations for each component of the vector-typed variable. The number
   /// specified in the command should match the number of components in the data type
   /// of the specified program variable (e.g., 1 for float, int, unsigned int; 2 for
   /// float2, int2, uint2, etc.). The suffix \a f indicates that floating-point
   /// values are expected to be returned, the suffix \a i indicates that integer
   /// values are expected, and the suffix \a ui indicates that unsigned integer
   /// values are expected, and this type should also match the data type of the
   /// specified program variable. The \a f variants of this function should be used
   /// to query values for program variables defined as float, float2, float3, float4,
   /// or arrays of these. The \a i variants of this function should be used to
   /// query values for program variables defined as int, int2, int3, int4, or
   /// arrays of these. The \a ui variants of this function should be used to query
   /// values for program variables defined as unsigned int, uint2, uint3, uint4,
   /// or arrays of these. The \a v variants of this function should be used to
   /// return the program variable's value to the array specified by parameter
   /// \a v. In this case, the array \a v should be large enough to accommodate all
   /// of the program variable's components.
   ///
   /// The commands \a rtVariableGetMatrix{2-3-4}x{2-3-4}fv are used to query the
   /// value of a program variable whose data type is a matrix. The numbers in the
   /// command names are interpreted as the dimensionality of the matrix. For example,
   /// \a 2x4 indicates a 2 x 4 matrix with 2 columns and 4 rows (i.e., 8
   /// values). If \a transpose is \a 0, the matrix is returned in row major order,
   /// otherwise in column major order.
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableGet were introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtVariableSet,
   /// @ref rtVariableGetType,
   /// @ref rtContextDeclareVariable
   ///
   /// @{
   ////
   ////**
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   f1         Float value to be returned
   pub fn rtVariableGet1f(v: RTvariable, f1: *mut f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   f1         Float value to be returned
   /// @param[in]   f2         Float value to be returned
   pub fn rtVariableGet2f(
      v: RTvariable,
      f1: *mut f32,
      f2: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   f1         Float value to be returned
   /// @param[in]   f2         Float value to be returned
   /// @param[in]   f3         Float value to be returned
   pub fn rtVariableGet3f(
      v: RTvariable,
      f1: *mut f32,
      f2: *mut f32,
      f3: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   f1         Float value to be returned
   /// @param[in]   f2         Float value to be returned
   /// @param[in]   f3         Float value to be returned
   /// @param[in]   f4         Float value to be returned
   pub fn rtVariableGet4f(
      v: RTvariable,
      f1: *mut f32,
      f2: *mut f32,
      f3: *mut f32,
      f4: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   f          Array of float value(s) to be returned
   pub fn rtVariableGet1fv(v: RTvariable, f: *mut f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   f          Array of float value(s) to be returned
   pub fn rtVariableGet2fv(v: RTvariable, f: *mut f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   f          Array of float value(s) to be returned
   pub fn rtVariableGet3fv(v: RTvariable, f: *mut f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   f          Array of float value(s) to be returned
   pub fn rtVariableGet4fv(v: RTvariable, f: *mut f32) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   i1         Integer value to be returned
   pub fn rtVariableGet1i(
      v: RTvariable,
      i1: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   i1         Integer value to be returned
   /// @param[in]   i2         Integer value to be returned
   pub fn rtVariableGet2i(
      v: RTvariable,
      i1: *mut ::std::os::raw::c_int,
      i2: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   i1         Integer value to be returned
   /// @param[in]   i2         Integer value to be returned
   /// @param[in]   i3         Integer value to be returned
   pub fn rtVariableGet3i(
      v: RTvariable,
      i1: *mut ::std::os::raw::c_int,
      i2: *mut ::std::os::raw::c_int,
      i3: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   i1         Integer value to be returned
   /// @param[in]   i2         Integer value to be returned
   /// @param[in]   i3         Integer value to be returned
   /// @param[in]   i4         Integer value to be returned
   pub fn rtVariableGet4i(
      v: RTvariable,
      i1: *mut ::std::os::raw::c_int,
      i2: *mut ::std::os::raw::c_int,
      i3: *mut ::std::os::raw::c_int,
      i4: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   i          Array of integer values to be returned
   pub fn rtVariableGet1iv(
      v: RTvariable,
      i: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   i          Array of integer values to be returned
   pub fn rtVariableGet2iv(
      v: RTvariable,
      i: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   i          Array of integer values to be returned
   pub fn rtVariableGet3iv(
      v: RTvariable,
      i: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   i          Array of integer values to be returned
   pub fn rtVariableGet4iv(
      v: RTvariable,
      i: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u1         Unsigned integer value to be returned
   pub fn rtVariableGet1ui(
      v: RTvariable,
      u1: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u1         Unsigned integer value to be returned
   /// @param[in]   u2         Unsigned integer value to be returned
   pub fn rtVariableGet2ui(
      v: RTvariable,
      u1: *mut ::std::os::raw::c_uint,
      u2: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u1         Unsigned integer value to be returned
   /// @param[in]   u2         Unsigned integer value to be returned
   /// @param[in]   u3         Unsigned integer value to be returned
   pub fn rtVariableGet3ui(
      v: RTvariable,
      u1: *mut ::std::os::raw::c_uint,
      u2: *mut ::std::os::raw::c_uint,
      u3: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u1         Unsigned integer value to be returned
   /// @param[in]   u2         Unsigned integer value to be returned
   /// @param[in]   u3         Unsigned integer value to be returned
   /// @param[in]   u4         Unsigned integer value to be returned
   pub fn rtVariableGet4ui(
      v: RTvariable,
      u1: *mut ::std::os::raw::c_uint,
      u2: *mut ::std::os::raw::c_uint,
      u3: *mut ::std::os::raw::c_uint,
      u4: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u          Array of unsigned integer values to be returned
   pub fn rtVariableGet1uiv(
      v: RTvariable,
      u: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u          Array of unsigned integer values to be returned
   pub fn rtVariableGet2uiv(
      v: RTvariable,
      u: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u          Array of unsigned integer values to be returned
   pub fn rtVariableGet3uiv(
      v: RTvariable,
      u: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u          Array of unsigned integer values to be returned
   pub fn rtVariableGet4uiv(
      v: RTvariable,
      u: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ll1        Integer value to be returned
   pub fn rtVariableGet1ll(
      v: RTvariable,
      ll1: *mut ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ll1        Integer value to be returned
   /// @param[in]   ll2        Integer value to be returned
   pub fn rtVariableGet2ll(
      v: RTvariable,
      ll1: *mut ::std::os::raw::c_longlong,
      ll2: *mut ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ll1        Integer value to be returned
   /// @param[in]   ll2        Integer value to be returned
   /// @param[in]   ll3        Integer value to be returned
   pub fn rtVariableGet3ll(
      v: RTvariable,
      ll1: *mut ::std::os::raw::c_longlong,
      ll2: *mut ::std::os::raw::c_longlong,
      ll3: *mut ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ll1        Integer value to be returned
   /// @param[in]   ll2        Integer value to be returned
   /// @param[in]   ll3        Integer value to be returned
   /// @param[in]   ll4        Integer value to be returned
   pub fn rtVariableGet4ll(
      v: RTvariable,
      ll1: *mut ::std::os::raw::c_longlong,
      ll2: *mut ::std::os::raw::c_longlong,
      ll3: *mut ::std::os::raw::c_longlong,
      ll4: *mut ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ll         Array of integer values to be returned
   pub fn rtVariableGet1llv(
      v: RTvariable,
      ll: *mut ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ll         Array of integer values to be returned
   pub fn rtVariableGet2llv(
      v: RTvariable,
      ll: *mut ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ll         Array of integer values to be returned
   pub fn rtVariableGet3llv(
      v: RTvariable,
      ll: *mut ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ll         Array of integer values to be returned
   pub fn rtVariableGet4llv(
      v: RTvariable,
      ll: *mut ::std::os::raw::c_longlong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u1         Unsigned integer value to be returned
   pub fn rtVariableGet1ull(
      v: RTvariable,
      u1: *mut ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u1         Unsigned integer value to be returned
   /// @param[in]   u2         Unsigned integer value to be returned
   pub fn rtVariableGet2ull(
      v: RTvariable,
      u1: *mut ::std::os::raw::c_ulonglong,
      u2: *mut ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u1         Unsigned integer value to be returned
   /// @param[in]   u2         Unsigned integer value to be returned
   /// @param[in]   u3         Unsigned integer value to be returned
   pub fn rtVariableGet3ull(
      v: RTvariable,
      u1: *mut ::std::os::raw::c_ulonglong,
      u2: *mut ::std::os::raw::c_ulonglong,
      u3: *mut ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   u1         Unsigned integer value to be returned
   /// @param[in]   u2         Unsigned integer value to be returned
   /// @param[in]   u3         Unsigned integer value to be returned
   /// @param[in]   u4         Unsigned integer value to be returned
   pub fn rtVariableGet4ull(
      v: RTvariable,
      u1: *mut ::std::os::raw::c_ulonglong,
      u2: *mut ::std::os::raw::c_ulonglong,
      u3: *mut ::std::os::raw::c_ulonglong,
      u4: *mut ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ull        Array of unsigned integer values to be returned
   pub fn rtVariableGet1ullv(
      v: RTvariable,
      ull: *mut ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ull        Array of unsigned integer values to be returned
   pub fn rtVariableGet2ullv(
      v: RTvariable,
      ull: *mut ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ull        Array of unsigned integer values to be returned
   pub fn rtVariableGet3ullv(
      v: RTvariable,
      ull: *mut ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   ull        Array of unsigned integer values to be returned
   pub fn rtVariableGet4ullv(
      v: RTvariable,
      ull: *mut ::std::os::raw::c_ulonglong,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   transpose  Specify(ies) row-major or column-major order
   /// @param[in]   m          Array of float values to be returned
   pub fn rtVariableGetMatrix2x2fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   transpose  Specify(ies) row-major or column-major order
   /// @param[in]   m          Array of float values to be returned
   pub fn rtVariableGetMatrix2x3fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   transpose  Specify(ies) row-major or column-major order
   /// @param[in]   m          Array of float values to be returned
   pub fn rtVariableGetMatrix2x4fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   transpose  Specify(ies) row-major or column-major order
   /// @param[in]   m          Array of float values to be returned
   pub fn rtVariableGetMatrix3x2fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   transpose  Specify(ies) row-major or column-major order
   /// @param[in]   m          Array of float values to be returned
   pub fn rtVariableGetMatrix3x3fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   transpose  Specify(ies) row-major or column-major order
   /// @param[in]   m          Array of float values to be returned
   pub fn rtVariableGetMatrix3x4fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   transpose  Specify(ies) row-major or column-major order
   /// @param[in]   m          Array of float values to be returned
   pub fn rtVariableGetMatrix4x2fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   transpose  Specify(ies) row-major or column-major order
   /// @param[in]   m          Array of float values to be returned
   pub fn rtVariableGetMatrix4x3fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @param[in]   v          Specifies the program variable whose value is to be returned
   /// @param[in]   transpose  Specify(ies) row-major or column-major order
   /// @param[in]   m          Array of float values to be returned
   pub fn rtVariableGetMatrix4x4fv(
      v: RTvariable,
      transpose: ::std::os::raw::c_int,
      m: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the value of a OptiX object program variable
   ///
   /// @ingroup Variables
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableGetObject queries the value of a program variable whose data type is a
   /// OptiX object.  The target variable is specified by \a v. The value of the
   /// program variable is returned in \a *object. The concrete
   /// type of the program variable can be queried using @ref rtVariableGetType, and the @ref
   /// RTobject handle returned by @ref rtVariableGetObject may safely be cast to an OptiX
   /// handle of corresponding type. If \a v is not a valid variable, this call sets
   /// \a *object to \a NULL and returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   v          Specifies the program variable to be queried
   /// @param[out]  object     Returns the value of the program variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableGetObject was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtVariableSetObject,
   /// @ref rtVariableGetType,
   /// @ref rtContextDeclareVariable
   ///
   pub fn rtVariableGetObject(v: RTvariable, object: *mut RTobject)
      -> RtResult;
}
extern "C" {
   /// @brief Defined
   ///
   /// @ingroup Variables
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableGetUserData queries the value of a program variable whose data type is
   /// user-defined. The variable of interest is specified by \a v.  The size of the
   /// variable's value must match the value given by the parameter \a size.  The value of
   /// the program variable is copied to the memory region pointed to by \a ptr. The storage
   /// at location \a ptr must be large enough to accommodate all of the program variable's
   /// value data. If \a v is not a valid variable, this call has no effect and
   /// returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   v          Specifies the program variable to be queried
   /// @param[in]   size       Specifies the size of the program variable, in bytes
   /// @param[out]  ptr        Location in which to store the value of the variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableGetUserData was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtVariableSetUserData,
   /// @ref rtContextDeclareVariable
   ///
   pub fn rtVariableGetUserData(
      v: RTvariable,
      size: RTsize,
      ptr: *mut ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries the name of a program variable
   ///
   /// @ingroup Variables
   ///
   /// <B>Description</B>
   ///
   /// Queries a program variable's name. The variable of interest is specified by \a
   /// variable, which should be a value returned by @ref rtContextDeclareVariable. A pointer
   /// to the string containing the name of the variable is returned in \a *nameReturn.
   /// If \a v is not a valid variable, this
   /// call sets \a *nameReturn to \a NULL and returns @ref RT_ERROR_INVALID_VALUE.  \a
   /// *nameReturn will point to valid memory until another API function that returns a
   /// string is called.
   ///
   /// @param[in]   v             Specifies the program variable to be queried
   /// @param[out]  nameReturn    Returns the program variable's name
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableGetName was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextDeclareVariable
   ///
   pub fn rtVariableGetName(
      v: RTvariable,
      nameReturn: *mut *const ::std::os::raw::c_char,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries the annotation string of a program variable
   ///
   /// @ingroup Variables
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableGetAnnotation queries a program variable's annotation string. A pointer
   /// to the string containing the annotation is returned in \a *annotationReturn.
   /// If \a v is not a valid variable, this call sets
   /// \a *annotationReturn to \a NULL and returns @ref RT_ERROR_INVALID_VALUE.  \a
   /// *annotationReturn will point to valid memory until another API function that returns
   /// a string is called.
   ///
   /// @param[in]   v                   Specifies the program variable to be queried
   /// @param[out]  annotationReturn    Returns the program variable's annotation string
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableGetAnnotation was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtDeclareVariable,
   /// @ref rtDeclareAnnotation
   ///
   pub fn rtVariableGetAnnotation(
      v: RTvariable,
      annotationReturn: *mut *const ::std::os::raw::c_char,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns type information about a program variable
   ///
   /// @ingroup Variables
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableGetType queries a program variable's type. The variable of interest is
   /// specified by \a v. The program variable's type enumeration is returned in \a *typeReturn,
   /// if it is not \a NULL. It is one of the following:
   ///
   ///   - @ref RT_OBJECTTYPE_UNKNOWN
   ///   - @ref RT_OBJECTTYPE_GROUP
   ///   - @ref RT_OBJECTTYPE_GEOMETRY_GROUP
   ///   - @ref RT_OBJECTTYPE_TRANSFORM
   ///   - @ref RT_OBJECTTYPE_SELECTOR
   ///   - @ref RT_OBJECTTYPE_GEOMETRY_INSTANCE
   ///   - @ref RT_OBJECTTYPE_BUFFER
   ///   - @ref RT_OBJECTTYPE_TEXTURE_SAMPLER
   ///   - @ref RT_OBJECTTYPE_OBJECT
   ///   - @ref RT_OBJECTTYPE_MATRIX_FLOAT2x2
   ///   - @ref RT_OBJECTTYPE_MATRIX_FLOAT2x3
   ///   - @ref RT_OBJECTTYPE_MATRIX_FLOAT2x4
   ///   - @ref RT_OBJECTTYPE_MATRIX_FLOAT3x2
   ///   - @ref RT_OBJECTTYPE_MATRIX_FLOAT3x3
   ///   - @ref RT_OBJECTTYPE_MATRIX_FLOAT3x4
   ///   - @ref RT_OBJECTTYPE_MATRIX_FLOAT4x2
   ///   - @ref RT_OBJECTTYPE_MATRIX_FLOAT4x3
   ///   - @ref RT_OBJECTTYPE_MATRIX_FLOAT4x4
   ///   - @ref RT_OBJECTTYPE_FLOAT
   ///   - @ref RT_OBJECTTYPE_FLOAT2
   ///   - @ref RT_OBJECTTYPE_FLOAT3
   ///   - @ref RT_OBJECTTYPE_FLOAT4
   ///   - @ref RT_OBJECTTYPE_INT
   ///   - @ref RT_OBJECTTYPE_INT2
   ///   - @ref RT_OBJECTTYPE_INT3
   ///   - @ref RT_OBJECTTYPE_INT4
   ///   - @ref RT_OBJECTTYPE_UNSIGNED_INT
   ///   - @ref RT_OBJECTTYPE_UNSIGNED_INT2
   ///   - @ref RT_OBJECTTYPE_UNSIGNED_INT3
   ///   - @ref RT_OBJECTTYPE_UNSIGNED_INT4
   ///   - @ref RT_OBJECTTYPE_USER
   ///
   /// Sets \a *typeReturn to @ref RT_OBJECTTYPE_UNKNOWN if \a v is not a valid variable.
   /// Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
   ///
   /// @param[in]   v             Specifies the program variable to be queried
   /// @param[out]  typeReturn    Returns the type of the program variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableGetType was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextDeclareVariable
   ///
   pub fn rtVariableGetType(
      v: RTvariable,
      typeReturn: *mut ObjectType,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with a program variable
   ///
   /// @ingroup Variables
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableGetContext queries the context associated with a program variable.  The
   /// target variable is specified by \a v. The context of the program variable is
   /// returned to \a *context if the pointer \a context is not \a NULL. If \a v is
   /// not a valid variable, \a *context is set to \a NULL and @ref RT_ERROR_INVALID_VALUE is
   /// returned.
   ///
   /// @param[in]   v          Specifies the program variable to be queried
   /// @param[out]  context    Returns the context associated with the program variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextDeclareVariable
   ///
   pub fn rtVariableGetContext(
      v: RTvariable,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries the size, in bytes, of a variable
   ///
   /// @ingroup Variables
   ///
   /// <B>Description</B>
   ///
   /// @ref rtVariableGetSize queries a declared program variable for its size in bytes.
   /// This is most often used to query the size of a variable that has a user-defined type.
   /// Builtin types (int, float, unsigned int, etc.) may be queried, but object typed
   /// variables, such as buffers, texture samplers and graph nodes, cannot be queried and
   /// will return @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   v          Specifies the program variable to be queried
   /// @param[out]  size       Specifies a pointer where the size of the variable, in bytes, will be returned
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtVariableGetSize was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtVariableGetUserData,
   /// @ref rtContextDeclareVariable
   ///
   pub fn rtVariableGetSize(v: RTvariable, size: *mut RTsize) -> RtResult;
}
extern "C" {
   /// @brief Creates a new context object
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextCreate allocates and returns a handle to a new context object.
   /// Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
   ///
   /// @param[out]  context   Handle to context for return value
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_NO_DEVICE
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   ///
   ///
   pub fn rtContextCreate(context: *mut RTcontext) -> RtResult;
}
extern "C" {
   /// @brief Destroys a context and frees all associated resources
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextDestroy frees all resources, including OptiX objects, associated with
   /// this object.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL context.  @ref
   /// RT_ERROR_LAUNCH_FAILED may be returned if a previous call to @ref rtContextLaunch "rtContextLaunch"
   /// failed.
   ///
   /// @param[in]   context   Handle of the context to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_LAUNCH_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextCreate
   ///
   pub fn rtContextDestroy(context: RTcontext) -> RtResult;
}
extern "C" {
   /// @brief Checks the given context for valid internal state
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextValidate checks the the given context and all of its associated OptiX
   /// objects for a valid state.  These checks include tests for presence of necessary
   /// programs (e.g. an intersection program for a geometry node), invalid internal state
   /// such as \a NULL children in graph nodes, and presence of variables required by all
   /// specified programs. @ref rtContextGetErrorString can be used to retrieve a description
   /// of a validation failure.
   ///
   /// @param[in]   context   The context to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_INVALID_SOURCE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetErrorString
   ///
   pub fn rtContextValidate(context: RTcontext) -> RtResult;
}
extern "C" {
   /// @brief Returns the error string associated with a given
   /// error
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetErrorString return a descriptive string given an error code.  If \a
   /// context is valid and additional information is available from the last OptiX failure,
   /// it will be appended to the generic error code description.  \a stringReturn will be
   /// set to point to this string.  The memory \a stringReturn points to will be valid
   /// until the next API call that returns a string.
   ///
   /// @param[in]   context         The context object to be queried, or \a NULL
   /// @param[in]   code            The error code to be converted to string
   /// @param[out]  stringReturn    The return parameter for the error string
   ///
   /// <B>Return values</B>
   ///
   /// @ref rtContextGetErrorString does not return a value
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetErrorString was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   ///
   ///
   pub fn rtContextGetErrorString(
      context: RTcontext,
      code: RtResult,
      stringReturn: *mut *const ::std::os::raw::c_char,
   );
}
extern "C" {
   /// @brief Set an attribute specific to an OptiX context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetAttribute sets \a p as the value of the per context attribute
   /// specified by \a attrib.
   ///
   /// Each attribute can have a different size.  The sizes are given in the following list:
   ///
   ///   - @ref RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS             sizeof(int)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES      sizeof(int)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS sizeof(int)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION         sizeof(char*)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS    sizeof(RTSize[2])
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS sets the number of host CPU threads OptiX
   /// can use for various tasks.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES is a hint about scene usage.  By
   /// default OptiX produces device kernels that are optimized for the current scene.  Such
   /// kernels generally run faster, but must be recompiled after some types of scene
   /// changes, causing delays.  Setting PREFER_FAST_RECOMPILES to 1 will leave out some
   /// scene-specific optimizations, producing kernels that generally run slower but are less
   /// sensitive to changes in the scene.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS sets whether or not OptiX will
   /// automatically inline user functions, which is the default behavior.  Please see the
   /// Programming Guide for more information about the benefits and limitations of disabling
   /// automatic inlining.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION sets the location where the OptiX disk
   /// cache will be created.  The location must be provided as a \a NULL-terminated
   /// string. OptiX will attempt to create the directory if it does not exist.  An exception
   /// will be thrown if OptiX is unable to create the cache database file at the specified
   /// location for any reason (e.g., the path is invalid or the directory is not writable).
   /// The location of the disk cache can be overridden with the environment variable \a
   /// OPTIX_CAHCE_PATH. This environment variable takes precedence over the RTcontext
   /// attribute.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS sets the low and high watermarks
   /// for disk cache garbage collection.  The limits must be passed in as a two-element
   /// array of \a RTsize values, with the low limit as the first element.  OptiX will throw
   /// an exception if either limit is non-zero and the high limit is not greater than the
   /// low limit.  Setting either limit to zero will disable garbage collection.  Garbage
   /// collection is triggered whenever the cache data size exceeds the high watermark and
   /// proceeds until the size reaches the low watermark.
   ///
   /// @param[in]   context   The context object to be modified
   /// @param[in]   attrib    Attribute to set
   /// @param[in]   size      Size of the attribute being set
   /// @param[in]   p         Pointer to where the value of the attribute will be copied from.  This must point to at least \a size bytes of memory
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE - Can be returned if \a size does not match the proper size of the attribute, or if \a p
   /// is \a NULL
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetAttribute was introduced in OptiX 2.5.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetAttribute
   ///
   pub fn rtContextSetAttribute(
      context: RTcontext,
      attrib: ContextAttribute,
      size: RTsize,
      p: *const ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns an attribute specific to an OptiX context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetAttribute returns in \a p the value of the per context attribute
   /// specified by \a attrib.
   ///
   /// Each attribute can have a different size.  The sizes are given in the following list:
   ///
   ///   - @ref RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT        sizeof(int)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS          sizeof(int)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY         sizeof(RTsize)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY  sizeof(RTsize)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED       sizeof(int)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION      sizeof(char**)
   ///   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS sizeof(RTSize[2])
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT queries the maximum number of textures
   /// handled by OptiX. For OptiX versions below 2.5 this value depends on the number of
   /// textures supported by CUDA.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS queries the number of host CPU threads OptiX
   /// can use for various tasks.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY queries the amount of host memory allocated
   /// by OptiX.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY queries the amount of free device
   /// memory.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED queries whether or not the OptiX disk
   /// cache is enabled.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION queries the file path of the OptiX
   /// disk cache.
   ///
   /// @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS queries the low and high watermark values
   /// for the OptiX disk cache.
   ///
   /// Some attributes are used to get per device information.  In contrast to @ref
   /// rtDeviceGetAttribute, these attributes are determined by the context and are therefore
   /// queried through the context.  This is done by adding the attribute with the OptiX
   /// device ordinal number when querying the attribute.  The following are per device attributes.
   ///
   ///   @ref RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY
   ///
   /// @param[in]   context   The context object to be queried
   /// @param[in]   attrib    Attribute to query
   /// @param[in]   size      Size of the attribute being queried.  Parameter \a p must have at least this much memory allocated
   /// @param[out]  p         Return pointer where the value of the attribute will be copied into.  This must point to at least \a size bytes of memory
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE - Can be returned if \a size does not match the proper size of the attribute, if \a p is
   /// \a NULL, or if \a attribute+ordinal does not correspond to an OptiX device
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetAttribute was introduced in OptiX 2.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetDeviceCount,
   /// @ref rtContextSetAttribute,
   /// @ref rtDeviceGetAttribute
   ///
   pub fn rtContextGetAttribute(
      context: RTcontext,
      attrib: ContextAttribute,
      size: RTsize,
      p: *mut ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Specify a list of hardware devices to be used by the
   /// kernel
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetDevices specifies a list of hardware devices to be used during
   /// execution of the subsequent trace kernels. Note that the device numbers are
   /// OptiX device ordinals, which may not be the same as CUDA device ordinals.
   /// Use @ref rtDeviceGetAttribute with @ref RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL to query the CUDA device
   /// corresponding to a particular OptiX device.
   ///
   /// @param[in]   context   The context to which the hardware list is applied
   /// @param[in]   count     The number of devices in the list
   /// @param[in]   devices   The list of devices
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_NO_DEVICE
   /// - @ref RT_ERROR_INVALID_DEVICE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetDevices was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetDevices,
   /// @ref rtContextGetDeviceCount
   ///
   pub fn rtContextSetDevices(
      context: RTcontext,
      count: ::std::os::raw::c_uint,
      devices: *const ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Retrieve a list of hardware devices being used by the
   /// kernel
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetDevices retrieves a list of hardware devices used by the context.
   /// Note that the device numbers are  OptiX device ordinals, which may not be the same as CUDA device ordinals.
   /// Use @ref rtDeviceGetAttribute with @ref RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL to query the CUDA device
   /// corresponding to a particular OptiX device.
   ///
   /// @param[in]   context   The context to which the hardware list is applied
   /// @param[out]  devices   Return parameter for the list of devices.  The memory must be able to hold entries
   /// numbering least the number of devices as returned by @ref rtContextGetDeviceCount
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetDevices was introduced in OptiX 2.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetDevices,
   /// @ref rtContextGetDeviceCount
   ///
   pub fn rtContextGetDevices(
      context: RTcontext,
      devices: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query the number of devices currently being used
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetDeviceCount - Query the number of devices currently being used.
   ///
   /// @param[in]   context   The context containing the devices
   /// @param[out]  count     Return parameter for the device count
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetDeviceCount was introduced in OptiX 2.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetDevices,
   /// @ref rtContextGetDevices
   ///
   pub fn rtContextGetDeviceCount(
      context: RTcontext,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Set the stack size for a given context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetStackSize sets the stack size for the given context to
   /// \a bytes bytes. Not supported with the RTX execution strategy.
   /// With RTX execution strategy @ref rtContextSetMaxTraceDepth and @ref rtContextSetMaxCallableDepth
   /// should be used to control stack size.
   /// Returns @ref RT_ERROR_INVALID_VALUE if context is not valid.
   ///
   /// @param[in]   context  The context node to be modified
   /// @param[in]   bytes    The desired stack size in bytes
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetStackSize was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetStackSize
   ///
   pub fn rtContextSetStackSize(context: RTcontext, bytes: RTsize) -> RtResult;
}
extern "C" {
   /// @brief Query the stack size for this context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetStackSize passes back the stack size associated with this context in
   /// \a bytes.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
   ///
   /// @param[in]   context The context node to be queried
   /// @param[out]  bytes   Return parameter to store the size of the stack
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetStackSize was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetStackSize
   ///
   pub fn rtContextGetStackSize(
      context: RTcontext,
      bytes: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Set maximum callable program call depth for a given context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetMaxCallableProgramDepth sets the maximum call depth of a chain of callable programs
   /// for the given context to \a maxDepth. This value is only used for stack size computation.
   /// Only supported for RTX execution mode. Default value is 5.
   /// Returns @ref RT_ERROR_INVALID_VALUE if context is not valid.
   ///
   /// @param[in]   context            The context node to be modified
   /// @param[in]   maxDepth           The desired maximum depth
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetMaxCallableProgramDepth was introduced in OptiX 6.0
   ///
   /// <B>See also</B>
   /// @ref rtContextGetMaxCallableProgramDepth
   ///
   pub fn rtContextSetMaxCallableProgramDepth(
      context: RTcontext,
      maxDepth: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query the maximum call depth for callable programs
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetMaxCallableProgramDepth passes back the maximum callable program call depth
   /// associated with this context in \a maxDepth.
   /// Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
   ///
   /// @param[in]   context            The context node to be queried
   /// @param[out]  maxDepth           Return parameter to store the maximum callable program depth
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetMaxCallableProgramDepth was introduced in OptiX 6.0
   ///
   /// <B>See also</B>
   /// @ref rtContextSetMaxCallableProgramDepth
   ///
   pub fn rtContextGetMaxCallableProgramDepth(
      context: RTcontext,
      maxDepth: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Set the maximum trace depth for a given context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetMaxTraceDepth sets the maximum trace depth for the given context to
   /// \a maxDepth. Only supported for RTX execution mode. Default value is 5.
   /// Returns @ref RT_ERROR_INVALID_VALUE if context is not valid.
   ///
   /// @param[in]   context            The context node to be modified
   /// @param[in]   maxDepth           The desired maximum depth
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetMaxTraceDepth was introduced in OptiX 6.0
   ///
   /// <B>See also</B>
   /// @ref rtContextGetMaxTraceDepth
   ///
   pub fn rtContextSetMaxTraceDepth(
      context: RTcontext,
      maxDepth: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query the maximum trace depth for this context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetMaxTraceDepth passes back the maximum trace depth associated with this context in
   /// \a maxDepth.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
   ///
   /// @param[in]   context            The context node to be queried
   /// @param[out]  maxDepth           Return parameter to store the maximum trace depth
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetMaxTraceDepth was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetMaxTraceDepth
   ///
   pub fn rtContextGetMaxTraceDepth(
      context: RTcontext,
      maxDepth: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 6.0. Calling this function has no effect.
   pub fn rtContextSetTimeoutCallback(
      context: RTcontext,
      callback: RTtimeoutcallback,
      minPollingSeconds: f64,
   ) -> RtResult;
}
extern "C" {
   /// @brief Set usage report callback function
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetUsageReportCallback sets an application-side callback
   /// function \a callback and a verbosity level \a verbosity.
   ///
   /// @ref RTusagereportcallback is defined as
   /// \a void (*RTusagereportcallback)(int, const char*, const char*, void*).
   ///
   /// The provided callback will be invoked with the message's verbosity level as
   /// the first parameter.  The second parameter is a descriptive tag string and
   /// the third parameter is the message itself.  The fourth parameter is a pointer
   /// to user-defined data, which may be NULL.  The descriptive tag will give a
   /// terse message category description (eg, 'SCENE STAT').  The messages will
   /// be unstructured and subject to change with subsequent releases.  The
   /// verbosity argument specifies the granularity of these messages.
   ///
   /// \a verbosity of 0 disables reporting.  \a callback is ignored in this case.
   ///
   /// \a verbosity of 1 enables error messages and important warnings.  This
   /// verbosity level can be expected to be efficient and have no significant
   /// overhead.
   ///
   /// \a verbosity of 2 additionally enables minor warnings, performance
   /// recommendations, and scene statistics at startup or recompilation
   /// granularity.  This level may have a performance cost.
   ///
   /// \a verbosity of 3 additionally enables informational messages and per-launch
   /// statistics and messages.
   ///
   /// A NULL \a callback when verbosity is non-zero or a \a verbosity outside of
   /// [0, 3] will result in @ref RT_ERROR_INVALID_VALUE return code.
   ///
   /// Only one report callback function can be specified at any time.
   ///
   /// @param[in]   context               The context node to be modified
   /// @param[in]   callback              The function to be called
   /// @param[in]   verbosity             The verbosity of report messages
   /// @param[in]   cbdata                Pointer to user-defined data that will be sent to the callback.  Can be NULL.
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetUsageReportCallback was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   ///
   pub fn rtContextSetUsageReportCallback(
      context: RTcontext,
      callback: RTusagereportcallback,
      verbosity: ::std::os::raw::c_int,
      cbdata: *mut ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Set the number of entry points for a given context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetEntryPointCount sets the number of entry points associated with
   /// the given context to \a count.
   ///
   /// @param[in]   context The context to be modified
   /// @param[in]   count   The number of entry points to use
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetEntryPointCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetEntryPointCount
   ///
   pub fn rtContextSetEntryPointCount(
      context: RTcontext,
      count: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query the number of entry points for this
   /// context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetEntryPointCount passes back the number of entry points associated
   /// with this context in \a count.  Returns @ref RT_ERROR_INVALID_VALUE if
   /// passed a \a NULL pointer.
   ///
   /// @param[in]   context The context node to be queried
   /// @param[out]  count   Return parameter for passing back the entry point count
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetEntryPointCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetEntryPointCount
   ///
   pub fn rtContextGetEntryPointCount(
      context: RTcontext,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Specifies the ray generation program for
   /// a given context entry point
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetRayGenerationProgram sets \a context's ray generation program at
   /// entry point \a entryPointIndex. @ref RT_ERROR_INVALID_VALUE is returned if \a
   /// entryPointIndex is outside of the range [\a 0, @ref rtContextGetEntryPointCount
   /// \a -1].
   ///
   /// @param[in]   context             The context node to which the exception program will be added
   /// @param[in]   entryPointIndex     The entry point the program will be associated with
   /// @param[in]   program             The ray generation program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetRayGenerationProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetEntryPointCount,
   /// @ref rtContextGetRayGenerationProgram
   ///
   pub fn rtContextSetRayGenerationProgram(
      context: RTcontext,
      entryPointIndex: ::std::os::raw::c_uint,
      program: RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries the ray generation program
   /// associated with the given context and entry point
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetRayGenerationProgram passes back the ray generation program
   /// associated with the given context and entry point.  This program is set via @ref
   /// rtContextSetRayGenerationProgram.  Returns @ref RT_ERROR_INVALID_VALUE if given an
   /// invalid entry point index or \a NULL pointer.
   ///
   /// @param[in]   context             The context node associated with the ray generation program
   /// @param[in]   entryPointIndex     The entry point index for the desired ray generation program
   /// @param[out]  program             Return parameter to store the ray generation program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetRayGenerationProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetRayGenerationProgram
   ///
   pub fn rtContextGetRayGenerationProgram(
      context: RTcontext,
      entryPointIndex: ::std::os::raw::c_uint,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Specifies the exception program for a given context entry point
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetExceptionProgram sets \a context's exception program at entry point
   /// \a entryPointIndex. @ref RT_ERROR_INVALID_VALUE is returned if \a entryPointIndex
   /// is outside of the range [\a 0, @ref rtContextGetEntryPointCount \a -1].
   ///
   /// @param[in]   context             The context node to which the exception program will be added
   /// @param[in]   entryPointIndex     The entry point the program will be associated with
   /// @param[in]   program             The exception program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetExceptionProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetEntryPointCount,
   /// @ref rtContextGetExceptionProgram
   /// @ref rtContextSetExceptionEnabled,
   /// @ref rtContextGetExceptionEnabled,
   /// @ref rtGetExceptionCode,
   /// @ref rtThrow,
   /// @ref rtPrintExceptionDetails
   ///
   pub fn rtContextSetExceptionProgram(
      context: RTcontext,
      entryPointIndex: ::std::os::raw::c_uint,
      program: RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries the exception program associated with
   /// the given context and entry point
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetExceptionProgram passes back the exception program associated with
   /// the given context and entry point.  This program is set via @ref
   /// rtContextSetExceptionProgram.  Returns @ref RT_ERROR_INVALID_VALUE if given an invalid
   /// entry point index or \a NULL pointer.
   ///
   /// @param[in]   context             The context node associated with the exception program
   /// @param[in]   entryPointIndex     The entry point index for the desired exception program
   /// @param[out]  program             Return parameter to store the exception program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetExceptionProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetExceptionProgram,
   /// @ref rtContextSetEntryPointCount,
   /// @ref rtContextSetExceptionEnabled,
   /// @ref rtContextGetExceptionEnabled,
   /// @ref rtGetExceptionCode,
   /// @ref rtThrow,
   /// @ref rtPrintExceptionDetails
   ///
   pub fn rtContextGetExceptionProgram(
      context: RTcontext,
      entryPointIndex: ::std::os::raw::c_uint,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Enable or disable an exception
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetExceptionEnabled is used to enable or disable specific exceptions.
   /// If an exception is enabled, the exception condition is checked for at runtime, and the
   /// exception program is invoked if the condition is met. The exception program can query
   /// the type of the caught exception by calling @ref rtGetExceptionCode.
   /// \a exception may take one of the following values:
   ///
   ///   - @ref RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS
   ///   - @ref RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS
   ///   - @ref RT_EXCEPTION_TRACE_DEPTH_EXCEEDED
   ///   - @ref RT_EXCEPTION_TEXTURE_ID_INVALID
   ///   - @ref RT_EXCEPTION_BUFFER_ID_INVALID
   ///   - @ref RT_EXCEPTION_INDEX_OUT_OF_BOUNDS
   ///   - @ref RT_EXCEPTION_STACK_OVERFLOW
   ///   - @ref RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS
   ///   - @ref RT_EXCEPTION_INVALID_RAY
   ///   - @ref RT_EXCEPTION_INTERNAL_ERROR
   ///   - @ref RT_EXCEPTION_USER
   ///   - @ref RT_EXCEPTION_ALL
   ///
   ///
   /// @ref RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS verifies that accesses to the ray payload are
   /// within valid bounds. This exception is only supported with the RTX execution strategy.
   ///
   /// @ref RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS verifies that the exception code passed
   /// to @ref rtThrow is within the valid range from RT_EXCEPTION_USER to RT_EXCEPTION_USER_MAX.
   ///
   /// @ref RT_EXCEPTION_TRACE_DEPTH_EXCEEDED verifies that the depth of the @ref rtTrace
   /// tree does not exceed the limit of 31. This exception is only supported with the RTX execution
   /// strategy.
   ///
   /// @ref RT_EXCEPTION_TEXTURE_ID_INVALID verifies that every access of a texture id is
   /// valid, including use of RT_TEXTURE_ID_NULL and IDs out of bounds.
   ///
   /// @ref RT_EXCEPTION_BUFFER_ID_INVALID verifies that every access of a buffer id is
   /// valid, including use of RT_BUFFER_ID_NULL and IDs out of bounds.
   ///
   /// @ref RT_EXCEPTION_INDEX_OUT_OF_BOUNDS checks that @ref rtIntersectChild and @ref
   /// rtReportIntersection are called with a valid index.
   ///
   /// @ref RT_EXCEPTION_STACK_OVERFLOW checks the runtime stack against overflow. The most
   /// common cause for an overflow is a too deep @ref rtTrace recursion tree.
   ///
   /// @ref RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS checks every read and write access to
   /// @ref rtBuffer objects to be within valid bounds. This exception is supported with the RTX
   /// execution strategy only.
   ///
   /// @ref RT_EXCEPTION_INVALID_RAY checks the each ray's origin and direction values
   /// against \a NaNs and \a infinity values.
   ///
   /// @ref RT_EXCEPTION_INTERNAL_ERROR indicates an unexpected internal error in the
   /// runtime.
   ///
   /// @ref RT_EXCEPTION_USER is used to enable or disable all user-defined exceptions. See
   /// @ref rtThrow for more information.
   ///
   /// @ref RT_EXCEPTION_ALL is a placeholder value which can be used to enable or disable
   /// all possible exceptions with a single call to @ref rtContextSetExceptionEnabled.
   ///
   /// By default, @ref RT_EXCEPTION_STACK_OVERFLOW is enabled and all other exceptions are
   /// disabled.
   ///
   /// @param[in]   context     The context for which the exception is to be enabled or disabled
   /// @param[in]   exception   The exception which is to be enabled or disabled
   /// @param[in]   enabled     Nonzero to enable the exception, \a 0 to disable the exception
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetExceptionEnabled was introduced in OptiX 1.1.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetExceptionEnabled,
   /// @ref rtContextSetExceptionProgram,
   /// @ref rtContextGetExceptionProgram,
   /// @ref rtGetExceptionCode,
   /// @ref rtThrow,
   /// @ref rtPrintExceptionDetails,
   /// @ref Exception
   ///
   pub fn rtContextSetExceptionEnabled(
      context: RTcontext,
      exception: Exception,
      enabled: ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query whether a specified exception is enabled
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetExceptionEnabled passes back \a 1 in \a *enabled if the given exception is
   /// enabled, \a 0 otherwise. \a exception specifies the type of exception to be queried. For a list
   /// of available types, see @ref rtContextSetExceptionEnabled. If \a exception
   /// is @ref RT_EXCEPTION_ALL, \a enabled is set to \a 1 only if all possible
   /// exceptions are enabled.
   ///
   /// @param[in]   context     The context to be queried
   /// @param[in]   exception   The exception of which to query the state
   /// @param[out]  enabled     Return parameter to store whether the exception is enabled
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetExceptionEnabled was introduced in OptiX 1.1.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetExceptionEnabled,
   /// @ref rtContextSetExceptionProgram,
   /// @ref rtContextGetExceptionProgram,
   /// @ref rtGetExceptionCode,
   /// @ref rtThrow,
   /// @ref rtPrintExceptionDetails,
   /// @ref Exception
   ///
   pub fn rtContextGetExceptionEnabled(
      context: RTcontext,
      exception: Exception,
      enabled: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the number of ray types for a given context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetRayTypeCount Sets the number of ray types associated with the given
   /// context.
   ///
   /// @param[in]   context         The context node
   /// @param[in]   rayTypeCount    The number of ray types to be used
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetRayTypeCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetRayTypeCount
   ///
   pub fn rtContextSetRayTypeCount(
      context: RTcontext,
      rayTypeCount: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query the number of ray types associated with this
   /// context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetRayTypeCount passes back the number of entry points associated with
   /// this context in \a rayTypeCount.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a
   /// NULL pointer.
   ///
   /// @param[in]   context         The context node to be queried
   /// @param[out]  rayTypeCount    Return parameter to store the number of ray types
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetRayTypeCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetRayTypeCount
   ///
   pub fn rtContextGetRayTypeCount(
      context: RTcontext,
      rayTypeCount: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Specifies the miss program for a given context ray type
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetMissProgram sets \a context's miss program associated with ray type
   /// \a rayTypeIndex. @ref RT_ERROR_INVALID_VALUE is returned if \a rayTypeIndex
   /// is outside of the range [\a 0, @ref rtContextGetRayTypeCount \a -1].
   ///
   /// @param[in]   context          The context node to which the miss program will be added
   /// @param[in]   rayTypeIndex     The ray type the program will be associated with
   /// @param[in]   program          The miss program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetMissProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetRayTypeCount,
   /// @ref rtContextGetMissProgram
   ///
   pub fn rtContextSetMissProgram(
      context: RTcontext,
      rayTypeIndex: ::std::os::raw::c_uint,
      program: RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries the miss program associated with the given
   /// context and ray type
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetMissProgram passes back the miss program associated with the
   /// given context and ray type.  This program is set via @ref rtContextSetMissProgram.
   /// Returns @ref RT_ERROR_INVALID_VALUE if given an invalid ray type index or a \a NULL pointer.
   ///
   /// @param[in]   context          The context node associated with the miss program
   /// @param[in]   rayTypeIndex     The ray type index for the desired miss program
   /// @param[out]  program          Return parameter to store the miss program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetMissProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextSetMissProgram,
   /// @ref rtContextGetRayTypeCount
   ///
   pub fn rtContextGetMissProgram(
      context: RTcontext,
      rayTypeIndex: ::std::os::raw::c_uint,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets an RTtexturesampler corresponding to the texture id
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetTextureSamplerFromId returns a handle to the texture sampler in \a *sampler
   /// corresponding to the \a samplerId supplied.  If \a samplerId does not map to a valid
   /// texture handle, \a *sampler is \a NULL or if \a context is invalid, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   context     The context the sampler should be originated from
   /// @param[in]   samplerId   The ID of the sampler to query
   /// @param[out]  sampler     The return handle for the sampler object corresponding to the samplerId
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetTextureSamplerFromId was introduced in OptiX 3.5.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerGetId
   ///
   pub fn rtContextGetTextureSamplerFromId(
      context: RTcontext,
      samplerId: ::std::os::raw::c_int,
      sampler: *mut RTtexturesampler,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 4.0. Calling this function has no effect. The kernel is automatically compiled at launch if needed.
   ///
   pub fn rtContextCompile(context: RTcontext) -> RtResult;
}
extern "C" {
   /// @brief Executes the computation kernel for a given context
   ///
   /// @ingroup rtContextLaunch
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextLaunch "rtContextLaunch" functions execute the computation kernel associated with the
   /// given context.  If the context has not yet been compiled, or if the context has been
   /// modified since the last compile, @ref rtContextLaunch "rtContextLaunch" will recompile the kernel
   /// internally.  Acceleration structures of the context which are marked dirty will be
   /// updated and their dirty flags will be cleared.  Similarly, validation will occur if
   /// necessary.  The ray generation program specified by \a entryPointIndex will be
   /// invoked once for every element (pixel or voxel) of the computation grid specified by
   /// \a width, \a height, and \a depth.
   ///
   /// For 3D launches, the product of \a width and \a depth must be smaller than 4294967296 (2^32).
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_INVALID_SOURCE
   /// - @ref RT_ERROR_LAUNCH_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextLaunch "rtContextLaunch" was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetRunningState,
   /// @ref rtContextValidate
   ///
   ////
   ////**
   /// @ingroup rtContextLaunch
   /// @param[in]   context                                    The context to be executed
   /// @param[in]   entryPointIndex                            The initial entry point into kernel
   /// @param[in]   width                                      Width of the computation grid
   pub fn rtContextLaunch1D(
      context: RTcontext,
      entryPointIndex: ::std::os::raw::c_uint,
      width: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @ingroup rtContextLaunch
   /// @param[in]   context                                    The context to be executed
   /// @param[in]   entryPointIndex                            The initial entry point into kernel
   /// @param[in]   width                                      Width of the computation grid
   /// @param[in]   height                                     Height of the computation grid
   pub fn rtContextLaunch2D(
      context: RTcontext,
      entryPointIndex: ::std::os::raw::c_uint,
      width: RTsize,
      height: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @ingroup rtContextLaunch
   /// @param[in]   context                                    The context to be executed
   /// @param[in]   entryPointIndex                            The initial entry point into kernel
   /// @param[in]   width                                      Width of the computation grid
   /// @param[in]   height                                     Height of the computation grid
   /// @param[in]   depth                                      Depth of the computation grid
   pub fn rtContextLaunch3D(
      context: RTcontext,
      entryPointIndex: ::std::os::raw::c_uint,
      width: RTsize,
      height: RTsize,
      depth: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query whether the given context is currently
   /// running
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// This function is currently unimplemented and it is provided as a placeholder for a future implementation.
   ///
   /// @param[in]   context   The context node to be queried
   /// @param[out]  running   Return parameter to store the running state
   ///
   /// <B>Return values</B>
   ///
   /// Since unimplemented, this function will always throw an assertion failure.
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetRunningState was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextLaunch1D,
   /// @ref rtContextLaunch2D,
   /// @ref rtContextLaunch3D
   ///
   pub fn rtContextGetRunningState(
      context: RTcontext,
      running: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Executes a Progressive Launch for a given context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// Starts the (potentially parallel) generation of subframes for progressive rendering. If
   /// \a maxSubframes is zero, there is no limit on the number of subframes generated. The
   /// generated subframes are automatically composited into a single result and streamed to
   /// the client at regular intervals, where they can be read by mapping an associated stream
   /// buffer. An application can therefore initiate a progressive launch, and then repeatedly
   /// map and display the contents of the stream buffer in order to visualize the progressive
   /// refinement of the image.
   ///
   /// The call is nonblocking. A polling approach should be used to decide when to map and
   /// display the stream buffer contents (see @ref rtBufferGetProgressiveUpdateReady). If a
   /// progressive launch is already in progress at the time of the call and its parameters
   /// match the initial launch, the call has no effect. Otherwise, the accumulated result will be
   /// reset and a new progressive launch will be started.
   ///
   /// If any other OptiX function is called while a progressive launch is in progress, it will
   /// cause the launch to stop generating new subframes (however, subframes that have
   /// already been generated and are currently in flight may still arrive at the client). The only
   /// exceptions to this rule are the operations to map a stream buffer, issuing another
   /// progressive launch with unchanged parameters, and polling for an update. Those
   /// exceptions do not cause the progressive launch to stop generating subframes.
   ///
   /// There is no guarantee that the call actually produces any subframes, especially if
   /// @ref rtContextLaunchProgressive2D and other OptiX commands are called in short
   /// succession. For example, during an animation, @ref rtVariableSet calls may be tightly
   /// interleaved with progressive launches, and when rendering remotely the server may decide to skip some of the
   /// launches in order to avoid a large backlog in the command pipeline.
   ///
   /// @param[in]   context                The context in which the launch is to be executed
   /// @param[in]   entryIndex             The initial entry point into kernel
   /// @param[in]   width                  Width of the computation grid
   /// @param[in]   height                 Height of the computation grid
   /// @param[in]   maxSubframes           The maximum number of subframes to be generated. Set to zero to generate an unlimited number of subframes
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_LAUNCH_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextLaunchProgressive2D was introduced in OptiX 3.8.
   ///
   /// <B>See also</B>
   /// @ref rtContextStopProgressive
   /// @ref rtBufferGetProgressiveUpdateReady
   ///
   pub fn rtContextLaunchProgressive2D(
      context: RTcontext,
      entryIndex: ::std::os::raw::c_uint,
      width: RTsize,
      height: RTsize,
      maxSubframes: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Stops a Progressive Launch
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// If a progressive launch is currently in progress, calling @ref rtContextStopProgressive
   /// terminates it. Otherwise, the call has no effect. If a launch is stopped using this function,
   /// no further subframes will arrive at the client, even if they have already been generated
   /// by the server and are currently in flight.
   ///
   /// This call should only be used if the application must guarantee that frames generated by
   /// previous progressive launches won't be accessed. Do not call @ref rtContextStopProgressive in
   /// the main rendering loop if the goal is only to change OptiX state (e.g. rtVariable values).
   /// The call is unnecessary in that case and will degrade performance.
   ///
   /// @param[in]   context                The context associated with the progressive launch
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_INVALID_CONTEXT
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextStopProgressive was introduced in OptiX 3.8.
   ///
   /// <B>See also</B>
   /// @ref rtContextLaunchProgressive2D
   ///
   pub fn rtContextStopProgressive(context: RTcontext) -> RtResult;
}
extern "C" {
   /// @brief Enable or disable text printing from programs
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetPrintEnabled is used to control whether text printing in programs
   /// through @ref rtPrintf is currently enabled for this context.
   ///
   /// @param[in]   context   The context for which printing is to be enabled or disabled
   /// @param[in]   enabled   Setting this parameter to a nonzero value enables printing, \a 0 disables printing
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetPrintEnabled was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtPrintf,
   /// @ref rtContextGetPrintEnabled,
   /// @ref rtContextSetPrintBufferSize,
   /// @ref rtContextGetPrintBufferSize,
   /// @ref rtContextSetPrintLaunchIndex,
   /// @ref rtContextGetPrintLaunchIndex
   ///
   pub fn rtContextSetPrintEnabled(
      context: RTcontext,
      enabled: ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query whether text printing from programs
   /// is enabled
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetPrintEnabled passes back \a 1 if text printing from programs through
   /// @ref rtPrintf is currently enabled for this context; \a 0 otherwise.  Returns @ref
   /// RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
   ///
   /// @param[in]   context   The context to be queried
   /// @param[out]  enabled   Return parameter to store whether printing is enabled
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetPrintEnabled was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtPrintf,
   /// @ref rtContextSetPrintEnabled,
   /// @ref rtContextSetPrintBufferSize,
   /// @ref rtContextGetPrintBufferSize,
   /// @ref rtContextSetPrintLaunchIndex,
   /// @ref rtContextGetPrintLaunchIndex
   ///
   pub fn rtContextGetPrintEnabled(
      context: RTcontext,
      enabled: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Set the size of the print buffer
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetPrintBufferSize is used to set the buffer size available to hold
   /// data generated by @ref rtPrintf.
   /// Returns @ref RT_ERROR_INVALID_VALUE if it is called after the first invocation of rtContextLaunch.
   ///
   ///
   /// @param[in]   context             The context for which to set the print buffer size
   /// @param[in]   bufferSizeBytes     The print buffer size in bytes
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetPrintBufferSize was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtPrintf,
   /// @ref rtContextSetPrintEnabled,
   /// @ref rtContextGetPrintEnabled,
   /// @ref rtContextGetPrintBufferSize,
   /// @ref rtContextSetPrintLaunchIndex,
   /// @ref rtContextGetPrintLaunchIndex
   ///
   pub fn rtContextSetPrintBufferSize(
      context: RTcontext,
      bufferSizeBytes: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Get the current size of the print buffer
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetPrintBufferSize is used to query the buffer size available to hold
   /// data generated by @ref rtPrintf. Returns @ref RT_ERROR_INVALID_VALUE if passed a \a
   /// NULL pointer.
   ///
   /// @param[in]   context             The context from which to query the print buffer size
   /// @param[out]  bufferSizeBytes     The returned print buffer size in bytes
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetPrintBufferSize was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtPrintf,
   /// @ref rtContextSetPrintEnabled,
   /// @ref rtContextGetPrintEnabled,
   /// @ref rtContextSetPrintBufferSize,
   /// @ref rtContextSetPrintLaunchIndex,
   /// @ref rtContextGetPrintLaunchIndex
   ///
   pub fn rtContextGetPrintBufferSize(
      context: RTcontext,
      bufferSizeBytes: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the active launch index to limit text output
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextSetPrintLaunchIndex is used to control for which launch indices @ref
   /// rtPrintf generates output. The initial value of (x,y,z) is (\a -1,\a -1,\a -1), which
   /// generates output for all indices.
   ///
   /// @param[in]   context   The context for which to set the print launch index
   /// @param[in]   x         The launch index in the x dimension to which to limit the output of @ref rtPrintf invocations.
   /// If set to \a -1, output is generated for all launch indices in the x dimension
   /// @param[in]   y         The launch index in the y dimension to which to limit the output of @ref rtPrintf invocations.
   /// If set to \a -1, output is generated for all launch indices in the y dimension
   /// @param[in]   z         The launch index in the z dimension to which to limit the output of @ref rtPrintf invocations.
   /// If set to \a -1, output is generated for all launch indices in the z dimension
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextSetPrintLaunchIndex was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtPrintf,
   /// @ref rtContextGetPrintEnabled,
   /// @ref rtContextSetPrintEnabled,
   /// @ref rtContextSetPrintBufferSize,
   /// @ref rtContextGetPrintBufferSize,
   /// @ref rtContextGetPrintLaunchIndex
   ///
   pub fn rtContextSetPrintLaunchIndex(
      context: RTcontext,
      x: ::std::os::raw::c_int,
      y: ::std::os::raw::c_int,
      z: ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the active print launch index
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetPrintLaunchIndex is used to query for which launch indices @ref
   /// rtPrintf generates output. The initial value of (x,y,z) is (\a -1,\a -1,\a -1), which
   /// generates output for all indices.
   ///
   /// @param[in]   context   The context from which to query the print launch index
   /// @param[out]  x         Returns the launch index in the x dimension to which the output of @ref rtPrintf invocations
   /// is limited. Will not be written to if a \a NULL pointer is passed
   /// @param[out]  y         Returns the launch index in the y dimension to which the output of @ref rtPrintf invocations
   /// is limited. Will not be written to if a \a NULL pointer is passed
   /// @param[out]  z         Returns the launch index in the z dimension to which the output of @ref rtPrintf invocations
   /// is limited. Will not be written to if a \a NULL pointer is passed
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetPrintLaunchIndex was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtPrintf,
   /// @ref rtContextGetPrintEnabled,
   /// @ref rtContextSetPrintEnabled,
   /// @ref rtContextSetPrintBufferSize,
   /// @ref rtContextGetPrintBufferSize,
   /// @ref rtContextSetPrintLaunchIndex
   ///
   pub fn rtContextGetPrintLaunchIndex(
      context: RTcontext,
      x: *mut ::std::os::raw::c_int,
      y: *mut ::std::os::raw::c_int,
      z: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Declares a new named variable associated with this
   /// context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextDeclareVariable - Declares a new variable named \a name and associated
   /// with this context.  Only a single variable of a given name can exist for a given
   /// context and any attempt to create multiple variables with the same name will cause a
   /// failure with a return value of @ref RT_ERROR_VARIABLE_REDECLARED.  Returns @ref
   /// RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.  Return @ref
   /// RT_ERROR_ILLEGAL_SYMBOL if \a name is not syntactically valid.
   ///
   /// @param[in]   context   The context node to which the variable will be attached
   /// @param[in]   name      The name that identifies the variable to be queried
   /// @param[out]  v         Pointer to variable handle used to return the new object
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_VARIABLE_REDECLARED
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextDeclareVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryDeclareVariable,
   /// @ref rtGeometryInstanceDeclareVariable,
   /// @ref rtMaterialDeclareVariable,
   /// @ref rtProgramDeclareVariable,
   /// @ref rtSelectorDeclareVariable,
   /// @ref rtContextGetVariable,
   /// @ref rtContextGetVariableCount,
   /// @ref rtContextQueryVariable,
   /// @ref rtContextRemoveVariable
   ///
   pub fn rtContextDeclareVariable(
      context: RTcontext,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a named variable associated with this context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextQueryVariable queries a variable identified by the string \a name
   /// from \a context and stores the result in \a *v. A variable must
   /// be declared with @ref rtContextDeclareVariable before it can be queried, otherwise \a *v will be set to \a NULL.
   /// @ref RT_ERROR_INVALID_VALUE will be returned if \a name or \a v is \a NULL.
   ///
   /// @param[in]   context   The context node to query a variable from
   /// @param[in]   name      The name that identifies the variable to be queried
   /// @param[out]  v         Return value to store the queried variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextQueryVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryQueryVariable,
   /// @ref rtGeometryInstanceQueryVariable,
   /// @ref rtMaterialQueryVariable,
   /// @ref rtProgramQueryVariable,
   /// @ref rtSelectorQueryVariable,
   /// @ref rtContextDeclareVariable,
   /// @ref rtContextGetVariableCount,
   /// @ref rtContextGetVariable,
   /// @ref rtContextRemoveVariable
   ///
   pub fn rtContextQueryVariable(
      context: RTcontext,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Removes a variable from the given context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextRemoveVariable removes variable \a v from \a context if present.
   /// Returns @ref RT_ERROR_VARIABLE_NOT_FOUND if the variable is not attached to this
   /// context. Returns @ref RT_ERROR_INVALID_VALUE if passed an invalid variable.
   ///
   /// @param[in]   context   The context node from which to remove a variable
   /// @param[in]   v         The variable to be removed
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextRemoveVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryRemoveVariable,
   /// @ref rtGeometryInstanceRemoveVariable,
   /// @ref rtMaterialRemoveVariable,
   /// @ref rtProgramRemoveVariable,
   /// @ref rtSelectorRemoveVariable,
   /// @ref rtContextDeclareVariable,
   /// @ref rtContextGetVariable,
   /// @ref rtContextGetVariableCount,
   /// @ref rtContextQueryVariable,
   ///
   pub fn rtContextRemoveVariable(
      context: RTcontext,
      v: RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of variables associated
   /// with this context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetVariableCount returns the number of variables that are currently
   /// attached to \a context.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
   ///
   /// @param[in]   context   The context to be queried for number of attached variables
   /// @param[out]  count     Return parameter to store the number of variables
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetVariableCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetVariableCount,
   /// @ref rtGeometryInstanceGetVariableCount,
   /// @ref rtMaterialGetVariableCount,
   /// @ref rtProgramGetVariableCount,
   /// @ref rtSelectorGetVariable,
   /// @ref rtContextDeclareVariable,
   /// @ref rtContextGetVariable,
   /// @ref rtContextQueryVariable,
   /// @ref rtContextRemoveVariable
   ///
   pub fn rtContextGetVariableCount(
      context: RTcontext,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries an indexed variable associated with this context
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetVariable queries the variable at position \a index in the
   /// variable array from \a context and stores the result in the parameter \a v.
   /// A variable must be declared first with @ref rtContextDeclareVariable and
   /// \a index must be in the range [\a 0, @ref rtContextGetVariableCount \a -1].
   ///
   /// @param[in]   context   The context node to be queried for an indexed variable
   /// @param[in]   index     The index that identifies the variable to be queried
   /// @param[out]  v         Return value to store the queried variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetVariable,
   /// @ref rtGeometryInstanceGetVariable,
   /// @ref rtMaterialGetVariable,
   /// @ref rtProgramGetVariable,
   /// @ref rtSelectorGetVariable,
   /// @ref rtContextDeclareVariable,
   /// @ref rtContextGetVariableCount,
   /// @ref rtContextQueryVariable,
   /// @ref rtContextRemoveVariable
   ///
   pub fn rtContextGetVariable(
      context: RTcontext,
      index: ::std::os::raw::c_uint,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new program object
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramCreateFromPTXString allocates and returns a handle to a new program
   /// object.  The program is created from PTX code held in the \a NULL-terminated string \a
   /// ptx from function \a programName.
   ///
   /// @param[in]   context        The context to create the program in
   /// @param[in]   ptx            The string containing the PTX code
   /// @param[in]   programName    The name of the PTX function to create the program from
   /// @param[in]   program        Handle to the program to be created
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_INVALID_SOURCE
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramCreateFromPTXString was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref RT_PROGRAM,
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramCreateFromPTXFiles,
   /// @ref rtProgramCreateFromPTXStrings,
   /// @ref rtProgramDestroy
   ///
   pub fn rtProgramCreateFromPTXString(
      context: RTcontext,
      ptx: *const ::std::os::raw::c_char,
      programName: *const ::std::os::raw::c_char,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new program object
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramCreateFromPTXStrings allocates and returns a handle to a new program
   /// object.  The program is created by linking PTX code held in one or more \a NULL-terminated strings.
   /// C-style linking rules apply: global functions and variables are visible across input strings and must
   /// be defined uniquely.  There must be a visible function for \a programName.
   ///
   /// @param[in]   context        The context to create the program in
   /// @param[in]   n              Number of ptx strings
   /// @param[in]   ptxStrings     Array of strings containing PTX code
   /// @param[in]   programName    The name of the PTX function to create the program from
   /// @param[in]   program        Handle to the program to be created
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_INVALID_SOURCE
   ///
   /// <B>History</B>
   ///
   /// <B>See also</B>
   /// @ref RT_PROGRAM,
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramCreateFromPTXFiles,
   /// @ref rtProgramCreateFromPTXString,
   /// @ref rtProgramDestroy
   ///
   pub fn rtProgramCreateFromPTXStrings(
      context: RTcontext,
      n: ::std::os::raw::c_uint,
      ptxStrings: *mut *const ::std::os::raw::c_char,
      programName: *const ::std::os::raw::c_char,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new program object
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramCreateFromPTXFile allocates and returns a handle to a new program object.
   /// The program is created from PTX code held in \a filename from function \a programName.
   ///
   /// @param[in]   context        The context to create the program in
   /// @param[in]   filename       Path to the file containing the PTX code
   /// @param[in]   programName    The name of the PTX function to create the program from
   /// @param[in]   program        Handle to the program to be created
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_INVALID_SOURCE
   /// - @ref RT_ERROR_FILE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramCreateFromPTXFile was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref RT_PROGRAM,
   /// @ref rtProgramCreateFromPTXString,
   /// @ref rtProgramDestroy
   ///
   pub fn rtProgramCreateFromPTXFile(
      context: RTcontext,
      filename: *const ::std::os::raw::c_char,
      programName: *const ::std::os::raw::c_char,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new program object
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramCreateFromPTXFiles allocates and returns a handle to a new program object.
   /// The program is created by linking PTX code held in one or more files.
   /// C-style linking rules apply: global functions and variables are visible across input files and must
   /// be defined uniquely.  There must be a visible function for \a programName.
   ///
   /// @param[in]   context        The context to create the program in
   /// @param[in]   n              Number of filenames
   /// @param[in]   filenames      Array of one or more paths to files containing PTX code
   /// @param[in]   programName    The name of the PTX function to create the program from
   /// @param[in]   program        Handle to the program to be created
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_INVALID_SOURCE
   /// - @ref RT_ERROR_FILE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// <B>See also</B>
   /// @ref RT_PROGRAM,
   /// @ref rtProgramCreateFromPTXString,
   /// @ref rtProgramCreateFromPTXStrings,
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramDestroy
   ///
   pub fn rtProgramCreateFromPTXFiles(
      context: RTcontext,
      n: ::std::os::raw::c_uint,
      filenames: *mut *const ::std::os::raw::c_char,
      programName: *const ::std::os::raw::c_char,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a program object
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramDestroy removes \a program from its context and deletes it.
   /// \a program should be a value returned by \a rtProgramCreate*.
   /// Associated variables declared via @ref rtProgramDeclareVariable are destroyed.
   /// After the call, \a program is no longer a valid handle.
   ///
   /// @param[in]   program   Handle of the program to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramCreateFromPTXString
   ///
   pub fn rtProgramDestroy(program: RTprogram) -> RtResult;
}
extern "C" {
   /// @brief Validates the state of a program
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramValidate checks \a program for completeness.  If \a program or any of
   /// the objects attached to \a program are not valid, returns @ref
   /// RT_ERROR_INVALID_CONTEXT.
   ///
   /// @param[in]   program   The program to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramCreateFromPTXString
   ///
   pub fn rtProgramValidate(program: RTprogram) -> RtResult;
}
extern "C" {
   /// @brief Gets the context object that created a program
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramGetContext returns a handle to the context object that was used to
   /// create \a program. Returns @ref RT_ERROR_INVALID_VALUE if \a context is \a NULL.
   ///
   /// @param[in]   program   The program to be queried for its context object
   /// @param[out]  context   The return handle for the requested context object
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextCreate
   ///
   pub fn rtProgramGetContext(
      program: RTprogram,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Declares a new named variable associated with a program
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramDeclareVariable declares a new variable, \a name, and associates it with
   /// the program.  A variable can only be declared with the same name once on the program.
   /// Any attempt to declare multiple variables with the same name will cause the call to
   /// fail and return @ref RT_ERROR_VARIABLE_REDECLARED.  If \a name or\a v is \a NULL
   /// returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   program   The program the declared variable will be attached to
   /// @param[in]   name      The name of the variable to be created
   /// @param[out]  v         Return handle to the variable to be created
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_REDECLARED
   /// - @ref RT_ERROR_ILLEGAL_SYMBOL
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramDeclareVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtProgramRemoveVariable,
   /// @ref rtProgramGetVariable,
   /// @ref rtProgramGetVariableCount,
   /// @ref rtProgramQueryVariable
   ///
   pub fn rtProgramDeclareVariable(
      program: RTprogram,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a handle to the named variable attached to a program
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramQueryVariable returns a handle to a variable object, in \a *v, attached
   /// to \a program referenced by the \a NULL-terminated string \a name.  If \a name is not
   /// the name of a variable attached to \a program, \a *v will be \a NULL after the call.
   ///
   /// @param[in]   program   The program to be queried for the named variable
   /// @param[in]   name      The name of the program to be queried for
   /// @param[out]  v         The return handle to the variable object
   /// @param  program   Handle to the program to be created
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramQueryVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtProgramDeclareVariable,
   /// @ref rtProgramRemoveVariable,
   /// @ref rtProgramGetVariable,
   /// @ref rtProgramGetVariableCount
   ///
   pub fn rtProgramQueryVariable(
      program: RTprogram,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Removes the named variable from a program
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramRemoveVariable removes variable \a v from the \a program object.  Once a
   /// variable has been removed from this program, another variable with the same name as
   /// the removed variable may be declared.
   ///
   /// @param[in]   program   The program to remove the variable from
   /// @param[in]   v         The variable to remove
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramRemoveVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtProgramDeclareVariable,
   /// @ref rtProgramGetVariable,
   /// @ref rtProgramGetVariableCount,
   /// @ref rtProgramQueryVariable
   ///
   pub fn rtProgramRemoveVariable(
      program: RTprogram,
      v: RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of variables attached to a program
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramGetVariableCount returns, in \a *count, the number of variable objects that
   /// have been attached to \a program.
   ///
   /// @param[in]   program   The program to be queried for its variable count
   /// @param[out]  count     The return handle for the number of variables attached to this program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramGetVariableCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtProgramDeclareVariable,
   /// @ref rtProgramRemoveVariable,
   /// @ref rtProgramGetVariable,
   /// @ref rtProgramQueryVariable
   ///
   pub fn rtProgramGetVariableCount(
      program: RTprogram,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a handle to a variable attached to a program by index
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramGetVariable returns a handle to a variable in \a *v attached to \a
   /// program with @ref rtProgramDeclareVariable by \a index.  \a index must be between
   /// 0 and one less than the value returned by @ref rtProgramGetVariableCount.  The order
   /// in which variables are enumerated is not constant and may change as variables are
   /// attached and removed from the program object.
   ///
   /// @param[in]   program   The program to be queried for the indexed variable object
   /// @param[in]   index     The index of the variable to return
   /// @param[out]  v         Return handle to the variable object specified by the index
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramGetVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtProgramDeclareVariable,
   /// @ref rtProgramRemoveVariable,
   /// @ref rtProgramGetVariableCount,
   /// @ref rtProgramQueryVariable
   ///
   pub fn rtProgramGetVariable(
      program: RTprogram,
      index: ::std::os::raw::c_uint,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the ID for the Program object
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramGetId returns an ID for the provided program.  The returned ID is used
   /// to reference \a program from device code.  If \a programId is \a NULL or the \a
   /// program is not a valid \a RTprogram, returns @ref RT_ERROR_INVALID_VALUE.
   /// @ref RT_PROGRAM_ID_NULL can be used as a sentinel for a non-existent program, since
   /// this value will never be returned as a valid program id.
   ///
   /// @param[in]   program      The program to be queried for its id
   /// @param[out]  programId    The returned ID of the program.
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramGetId was introduced in OptiX 3.6.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetProgramFromId
   ///
   pub fn rtProgramGetId(
      program: RTprogram,
      programId: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the program ids that may potentially be called at a call site
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtProgramCallsiteSetPotentialCallees specifies the program IDs of potential
   /// callees at the call site in the \a program identified by \a name to the list
   /// provided in \a ids. If \a program is bit a valid \a RTprogram or the \a program
   /// does not contain a call site with the identifier \a name or \a ids contains
   /// invalid program ids, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in] program        The program that includes the call site.
   /// @param[in] name           The string identifier for the call site to modify.
   /// @param[in] ids            The program IDs of the programs that may potentially be called at the call site
   /// @param[in] numIds         The size of the array passed in for \a ids.
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtProgramCallsiteSetPotentialCallees was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtProgramGetId
   ///
   pub fn rtProgramCallsiteSetPotentialCallees(
      program: RTprogram,
      name: *const ::std::os::raw::c_char,
      ids: *const ::std::os::raw::c_int,
      numIds: ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets an RTprogram corresponding to the program id
   ///
   /// @ingroup Program
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetProgramFromId returns a handle to the program in \a *program
   /// corresponding to the \a programId supplied.  If \a programId is not a valid
   /// program handle, \a *program is set to \a NULL. Returns @ref RT_ERROR_INVALID_VALUE
   /// if \a context is invalid or \a programId is not a valid program handle.
   ///
   /// @param[in]   context     The context the program should be originated from
   /// @param[in]   programId   The ID of the program to query
   /// @param[out]  program     The return handle for the program object corresponding to the programId
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetProgramFromId was introduced in OptiX 3.6.
   ///
   /// <B>See also</B>
   /// @ref rtProgramGetId
   ///
   pub fn rtContextGetProgramFromId(
      context: RTcontext,
      programId: ::std::os::raw::c_int,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new group
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupCreate creates a new group within a context. \a context
   /// specifies the target context, and should be a value returned by
   /// @ref rtContextCreate.  Sets \a *group to the handle of a newly created group
   /// within \a context. Returns @ref RT_ERROR_INVALID_VALUE if \a group is \a NULL.
   ///
   /// @param[in]   context   Specifies a context within which to create a new group
   /// @param[out]  group     Returns a newly created group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupDestroy,
   /// @ref rtContextCreate
   ///
   pub fn rtGroupCreate(context: RTcontext, group: *mut RTgroup) -> RtResult;
}
extern "C" {
   /// @brief Destroys a group node
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupDestroy removes \a group from its context and deletes it.
   /// \a group should be a value returned by @ref rtGroupCreate.
   /// No child graph nodes are destroyed.
   /// After the call, \a group is no longer a valid handle.
   ///
   /// @param[in]   group   Handle of the group node to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupCreate
   ///
   pub fn rtGroupDestroy(group: RTgroup) -> RtResult;
}
extern "C" {
   /// @brief Verifies the state of the group
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupValidate checks \a group for completeness. If \a group or
   /// any of the objects attached to \a group are not valid, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   group   Specifies the group to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupCreate
   ///
   pub fn rtGroupValidate(group: RTgroup) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with a group
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupGetContext queries a group for its associated context.
   /// \a group specifies the group to query, and must be a value returned by
   /// @ref rtGroupCreate. Sets \a *context to the context
   /// associated with \a group.
   ///
   /// @param[in]   group     Specifies the group to query
   /// @param[out]  context   Returns the context associated with the group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextCreate,
   /// @ref rtGroupCreate
   ///
   pub fn rtGroupGetContext(
      group: RTgroup,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Set the acceleration structure for a group
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupSetAcceleration attaches an acceleration structure to a group. The acceleration
   /// structure must have been previously created using @ref rtAccelerationCreate. Every group is
   /// required to have an acceleration structure assigned in order to pass validation. The acceleration
   /// structure will be built over the children of the group. For example, if an acceleration structure
   /// is attached to a group that has a selector, a geometry group, and a transform child,
   /// the acceleration structure will be built over the bounding volumes of these three objects.
   ///
   /// Note that it is legal to attach a single RTacceleration object to multiple groups, as long as
   /// the underlying bounds of the children are the same. For example, if another group has three
   /// children which are known to have the same bounding volumes as the ones in the example
   /// above, the two groups can share an acceleration structure, thus saving build time. This is
   /// true even if the details of the children, such as the actual type of a node or its geometry
   /// content, differ from the first set of group children. All that is required is for a child
   /// node at a given index to have the same bounds as the other group's child node at the same index.
   ///
   /// Sharing an acceleration structure this way corresponds to attaching an acceleration structure
   /// to multiple geometry groups at lower graph levels using @ref rtGeometryGroupSetAcceleration.
   ///
   /// @param[in]   group          The group handle
   /// @param[in]   acceleration   The acceleration structure to attach to the group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupSetAcceleration was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupGetAcceleration,
   /// @ref rtAccelerationCreate,
   /// @ref rtGeometryGroupSetAcceleration
   ///
   pub fn rtGroupSetAcceleration(
      group: RTgroup,
      acceleration: RTacceleration,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the visibility mask for a group.
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   /// Geometry is intersected by rays if the ray's @ref RTvisibilitymask shares at
   /// least one bit with the geometry's mask. This mechanism allows for a number of
   /// user-defined visibility groups that can be excluded from certain types of rays
   /// as needed.
   ///
   /// Note that the @pre mask is currently limited to 8 bits.
   ///
   /// @param[in] group   The group handle
   /// @param[in] mask    A set of bits for which rays will intersect the group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupSetVisibilityMask was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupSetVisibilityMask,
   /// @ref rtGroupGetVisibilityMask,
   /// @ref rtTrace
   pub fn rtGroupSetVisibilityMask(
      group: RTgroup,
      mask: RTvisibilitymask,
   ) -> RtResult;
}
extern "C" {
   /// @brief Retrieves the visibility mask of a group.
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   /// See @ref rtGroupSetVisibilityMask for details.
   ///
   /// @param[in] group   The group handle
   /// @param[out] mask   A set of bits for which rays will intersect the group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupGetVisibilityMask was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupGetVisibilityMask,
   /// @ref rtGroupSetVisibilityMask,
   /// @ref rtTrace
   pub fn rtGroupGetVisibilityMask(
      group: RTgroup,
      mask: *mut RTvisibilitymask,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the acceleration structure attached to a group
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupGetAcceleration returns the acceleration structure attached to a group using @ref rtGroupSetAcceleration.
   /// If no acceleration structure has previously been set, \a *acceleration is set to \a NULL.
   ///
   /// @param[in]   group          The group handle
   /// @param[out]  acceleration   The returned acceleration structure object
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupGetAcceleration was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupSetAcceleration,
   /// @ref rtAccelerationCreate
   ///
   pub fn rtGroupGetAcceleration(
      group: RTgroup,
      acceleration: *mut RTacceleration,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the number of child nodes to be attached to the group
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupSetChildCount specifies the number of child slots in this group. Potentially existing links to children
   /// at indices greater than \a count-1 are removed. If the call increases the number of slots, the newly
   /// created slots are empty and need to be filled using @ref rtGroupSetChild before validation.
   ///
   /// @param[in]   group   The parent group handle
   /// @param[in]   count   Number of child slots to allocate for the group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupSetChildCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupGetChild,
   /// @ref rtGroupGetChildCount,
   /// @ref rtGroupGetChildType,
   /// @ref rtGroupSetChild
   ///
   pub fn rtGroupSetChildCount(
      group: RTgroup,
      count: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of child slots for a group
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupGetChildCount returns the number of child slots allocated using @ref
   /// rtGroupSetChildCount.  This includes empty slots which may not yet have actual children assigned
   /// by @ref rtGroupSetChild.  Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
   ///
   /// @param[in]   group   The parent group handle
   /// @param[out]  count   Returned number of child slots
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupGetChildCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupSetChild,
   /// @ref rtGroupGetChild,
   /// @ref rtGroupSetChildCount,
   /// @ref rtGroupGetChildType
   ///
   pub fn rtGroupGetChildCount(
      group: RTgroup,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Attaches a child node to a group
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// Attaches a new child node \a child to the parent node
   /// \a group. \a index specifies the number of the slot where the child
   /// node gets attached. A sufficient number of slots must be allocated
   /// using @ref rtGroupSetChildCount.
   /// Legal child node types are @ref RTgroup, @ref RTselector, @ref RTgeometrygroup, and
   /// @ref RTtransform.
   ///
   /// @param[in]   group   The parent group handle
   /// @param[in]   index   The index in the parent's child slot array
   /// @param[in]   child   The child node to be attached. Can be of type {@ref RTgroup, @ref RTselector,
   /// @ref RTgeometrygroup, @ref RTtransform}
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupSetChild was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupSetChildCount,
   /// @ref rtGroupGetChildCount,
   /// @ref rtGroupGetChild,
   /// @ref rtGroupGetChildType
   ///
   pub fn rtGroupSetChild(
      group: RTgroup,
      index: ::std::os::raw::c_uint,
      child: RTobject,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a child node of a group
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupGetChild returns the child object at slot \a index of the parent \a group.
   /// If no child has been assigned to the given slot, \a *child is set to \a NULL.
   /// Returns @ref RT_ERROR_INVALID_VALUE if given an invalid child index or \a NULL pointer.
   ///
   /// @param[in]   group   The parent group handle
   /// @param[in]   index   The index of the child slot to query
   /// @param[out]  child   The returned child object
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupGetChild was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupSetChild,
   /// @ref rtGroupSetChildCount,
   /// @ref rtGroupGetChildCount,
   /// @ref rtGroupGetChildType
   ///
   pub fn rtGroupGetChild(
      group: RTgroup,
      index: ::std::os::raw::c_uint,
      child: *mut RTobject,
   ) -> RtResult;
}
extern "C" {
   /// @brief Get the type of a group child
   ///
   /// @ingroup GroupNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGroupGetChildType returns the type of the group child at slot \a index.
   /// If no child is associated with the given index, \a *type is set to
   /// @ref RT_OBJECTTYPE_UNKNOWN and @ref RT_ERROR_INVALID_VALUE is returned.
   /// Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
   ///
   /// @param[in]   group   The parent group handle
   /// @param[in]   index   The index of the child slot to query
   /// @param[out]  type    The returned child type
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGroupGetChildType was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupSetChild,
   /// @ref rtGroupGetChild,
   /// @ref rtGroupSetChildCount,
   /// @ref rtGroupGetChildCount
   ///
   pub fn rtGroupGetChildType(
      group: RTgroup,
      index: ::std::os::raw::c_uint,
      type_: *mut ObjectType,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// Creates a new Selector node within \a context. After calling
   /// @ref rtSelectorCreate the new node is in an invalid state.  For the node
   /// to be valid, a visit program must be assigned using
   /// @ref rtSelectorSetVisitProgram. Furthermore, a number of (zero or
   /// more) children can be attached by using @ref rtSelectorSetChildCount and
   /// @ref rtSelectorSetChild. Sets \a *selector to the handle of a newly
   /// created selector within \a context. Returns @ref RT_ERROR_INVALID_VALUE if \a selector is \a NULL.
   ///
   /// @param[in]   context    Specifies the rendering context of the Selector node
   /// @param[out]  selector   New Selector node handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorDestroy,
   /// @ref rtSelectorValidate,
   /// @ref rtSelectorGetContext,
   /// @ref rtSelectorSetVisitProgram,
   /// @ref rtSelectorSetChildCount,
   /// @ref rtSelectorSetChild
   ///
   pub fn rtSelectorCreate(
      context: RTcontext,
      selector: *mut RTselector,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorDestroy removes \a selector from its context and deletes it.  \a selector should
   /// be a value returned by @ref rtSelectorCreate.  Associated variables declared via @ref
   /// rtSelectorDeclareVariable are destroyed, but no child graph nodes are destroyed.  After the
   /// call, \a selector is no longer a valid handle.
   ///
   /// @param[in]   selector   Handle of the selector node to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorCreate,
   /// @ref rtSelectorValidate,
   /// @ref rtSelectorGetContext
   ///
   pub fn rtSelectorDestroy(selector: RTselector) -> RtResult;
}
extern "C" {
   /// @brief Checks a Selector node for internal consistency
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorValidate recursively checks consistency of the Selector
   /// node \a selector and its children, i.e., it tries to validate the
   /// whole model sub-tree with \a selector as root. For a Selector node to
   /// be valid, it must be assigned a visit program, and the number of its
   /// children must match the number specified by
   /// @ref rtSelectorSetChildCount.
   ///
   /// @param[in]   selector   Selector root node of a model sub-tree to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorCreate,
   /// @ref rtSelectorDestroy,
   /// @ref rtSelectorGetContext,
   /// @ref rtSelectorSetVisitProgram,
   /// @ref rtSelectorSetChildCount,
   /// @ref rtSelectorSetChild
   ///
   pub fn rtSelectorValidate(selector: RTselector) -> RtResult;
}
extern "C" {
   /// @brief Returns the context of a Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorGetContext returns in \a context the rendering context
   /// in which the Selector node \a selector has been created.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[out]  context    The context, \a selector belongs to
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorCreate,
   /// @ref rtSelectorDestroy,
   /// @ref rtSelectorValidate
   ///
   pub fn rtSelectorGetContext(
      selector: RTselector,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Assigns a visit program to a Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorSetVisitProgram specifies a visit program that is
   /// executed when the Selector node \a selector gets visited by a ray
   /// during traversal of the model graph. A visit program steers how
   /// traversal of the Selectors's children is performed.  It usually
   /// chooses only a single child to continue traversal, but is also allowed
   /// to process zero or multiple children. Programs can be created from PTX
   /// files using @ref rtProgramCreateFromPTXFile.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[in]   program    Program handle associated with a visit program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorSetVisitProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorGetVisitProgram,
   /// @ref rtProgramCreateFromPTXFile
   ///
   pub fn rtSelectorSetVisitProgram(
      selector: RTselector,
      program: RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the currently assigned visit program
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorGetVisitProgram returns in \a program a handle of the
   /// visit program curently bound to \a selector.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[out]  program    Current visit progam assigned to \a selector
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorGetVisitProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorSetVisitProgram
   ///
   pub fn rtSelectorGetVisitProgram(
      selector: RTselector,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Specifies the number of child nodes to be
   /// attached to a Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorSetChildCount allocates a number of children slots,
   /// i.e., it pre-defines the exact number of child nodes the parent
   /// Selector node \a selector will have.  Child nodes have to be attached
   /// to the Selector node using @ref rtSelectorSetChild. Empty slots will
   /// cause a validation error.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[in]   count      Number of child nodes to be attached to \a selector
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorSetChildCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorValidate,
   /// @ref rtSelectorGetChildCount,
   /// @ref rtSelectorSetChild,
   /// @ref rtSelectorGetChild,
   /// @ref rtSelectorGetChildType
   ///
   pub fn rtSelectorSetChildCount(
      selector: RTselector,
      count: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of child node slots of
   /// a Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorGetChildCount returns in \a count the number of child
   /// node slots that have been previously reserved for the Selector node
   /// \a selector by @ref rtSelectorSetChildCount. The value of \a count
   /// does not reflect the actual number of child nodes that have so far
   /// been attached to the Selector node using @ref rtSelectorSetChild.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[out]  count      Number of child node slots reserved for \a selector
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorGetChildCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorSetChildCount,
   /// @ref rtSelectorSetChild,
   /// @ref rtSelectorGetChild,
   /// @ref rtSelectorGetChildType
   ///
   pub fn rtSelectorGetChildCount(
      selector: RTselector,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Attaches a child node to a Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// Attaches a new child node \a child to the parent node
   /// \a selector. \a index specifies the number of the slot where the child
   /// node gets attached.  The index value must be lower than the number
   /// previously set by @ref rtSelectorSetChildCount, thus it must be in
   /// the range from \a 0 to @ref rtSelectorGetChildCount \a -1.  Legal child
   /// node types are @ref RTgroup, @ref RTselector, @ref RTgeometrygroup, and
   /// @ref RTtransform.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[in]   index      Index of the parent slot the node \a child gets attached to
   /// @param[in]   child      Child node to be attached. Can be {@ref RTgroup, @ref RTselector,
   /// @ref RTgeometrygroup, @ref RTtransform}
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorSetChild was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorSetChildCount,
   /// @ref rtSelectorGetChildCount,
   /// @ref rtSelectorGetChild,
   /// @ref rtSelectorGetChildType
   ///
   pub fn rtSelectorSetChild(
      selector: RTselector,
      index: ::std::os::raw::c_uint,
      child: RTobject,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a child node that is attached to a
   /// Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorGetChild returns in \a child a handle of the child node
   /// currently attached to \a selector at slot \a index. The index value
   /// must be lower than the number previously set by
   /// @ref rtSelectorSetChildCount, thus it must be in the range from \a 0
   /// to @ref rtSelectorGetChildCount \a - 1. The returned pointer is of generic
   /// type @ref RTobject and needs to be cast to the actual child type, which
   /// can be @ref RTgroup, @ref RTselector, @ref RTgeometrygroup, or
   /// @ref RTtransform. The actual type of \a child can be queried using
   /// @ref rtSelectorGetChildType;
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[in]   index      Child node index
   /// @param[out]  child      Child node handle. Can be {@ref RTgroup, @ref RTselector,
   /// @ref RTgeometrygroup, @ref RTtransform}
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorGetChild was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorSetChildCount,
   /// @ref rtSelectorGetChildCount,
   /// @ref rtSelectorSetChild,
   /// @ref rtSelectorGetChildType
   ///
   pub fn rtSelectorGetChild(
      selector: RTselector,
      index: ::std::os::raw::c_uint,
      child: *mut RTobject,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns type information about a Selector
   /// child node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorGetChildType queries the type of the child node
   /// attached to \a selector at slot \a index.
   /// If no child is associated with the given index, \a *type is set to
   /// @ref RT_OBJECTTYPE_UNKNOWN and @ref RT_ERROR_INVALID_VALUE is returned.
   /// Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
   /// The returned type is one of:
   ///
   ///   @ref RT_OBJECTTYPE_GROUP
   ///   @ref RT_OBJECTTYPE_GEOMETRY_GROUP
   ///   @ref RT_OBJECTTYPE_TRANSFORM
   ///   @ref RT_OBJECTTYPE_SELECTOR
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[in]   index      Child node index
   /// @param[out]  type       Type of the child node
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorGetChildType was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorSetChildCount,
   /// @ref rtSelectorGetChildCount,
   /// @ref rtSelectorSetChild,
   /// @ref rtSelectorGetChild
   ///
   pub fn rtSelectorGetChildType(
      selector: RTselector,
      index: ::std::os::raw::c_uint,
      type_: *mut ObjectType,
   ) -> RtResult;
}
extern "C" {
   /// @brief Declares a variable associated with a
   /// Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// Declares a new variable identified by \a name, and associates it with
   /// the Selector node \a selector. The new variable handle is returned in
   /// \a v. After declaration, a variable does not have a type until its
   /// value is set by an \a rtVariableSet{...} function. Once a variable
   /// type has been set, it cannot be changed, i.e., only
   /// \a rtVariableSet{...} functions of the same type can be used to
   /// change the value of the variable.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[in]   name       Variable identifier
   /// @param[out]  v          New variable handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_REDECLARED
   /// - @ref RT_ERROR_ILLEGAL_SYMBOL
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorDeclareVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorQueryVariable,
   /// @ref rtSelectorRemoveVariable,
   /// @ref rtSelectorGetVariableCount,
   /// @ref rtSelectorGetVariable,
   /// @ref rtVariableSet{...}
   ///
   pub fn rtSelectorDeclareVariable(
      selector: RTselector,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a variable associated with a
   /// Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// Returns in \a v a handle to the variable identified by \a name, which
   /// is associated with the Selector node \a selector. The current value of
   /// a variable can be retrieved from its handle by using an appropriate
   /// \a rtVariableGet{...} function matching the variable's type.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[in]   name       Variable identifier
   /// @param[out]  v          Variable handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorQueryVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorDeclareVariable,
   /// @ref rtSelectorRemoveVariable,
   /// @ref rtSelectorGetVariableCount,
   /// @ref rtSelectorGetVariable,
   /// \a rtVariableGet{...}
   ///
   pub fn rtSelectorQueryVariable(
      selector: RTselector,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Removes a variable from a Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorRemoveVariable removes the variable \a v from the
   /// Selector node \a selector and deletes it. The handle \a v must be
   /// considered invalid afterwards.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[in]   v          Variable handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorRemoveVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorDeclareVariable,
   /// @ref rtSelectorQueryVariable,
   /// @ref rtSelectorGetVariableCount,
   /// @ref rtSelectorGetVariable
   ///
   pub fn rtSelectorRemoveVariable(
      selector: RTselector,
      v: RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of variables
   /// attached to a Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtSelectorGetVariableCount returns in \a count the number of
   /// variables that are currently attached to the Selector node
   /// \a selector.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[out]  count      Number of variables associated with \a selector
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorGetVariableCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorDeclareVariable,
   /// @ref rtSelectorQueryVariable,
   /// @ref rtSelectorRemoveVariable,
   /// @ref rtSelectorGetVariable
   ///
   pub fn rtSelectorGetVariableCount(
      selector: RTselector,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a variable associated with a
   /// Selector node
   ///
   /// @ingroup SelectorNode
   ///
   /// <B>Description</B>
   ///
   /// Returns in \a v a handle to the variable located at position \a index
   /// in the Selectors's variable array. \a index is a sequential number
   /// depending on the order of variable declarations. The index must be
   /// in the range from \a 0 to @ref rtSelectorGetVariableCount \a - 1.  The
   /// current value of a variable can be retrieved from its handle by using
   /// an appropriate \a rtVariableGet{...} function matching the
   /// variable's type.
   ///
   /// @param[in]   selector   Selector node handle
   /// @param[in]   index      Variable index
   /// @param[out]  v          Variable handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtSelectorGetVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtSelectorDeclareVariable,
   /// @ref rtSelectorQueryVariable,
   /// @ref rtSelectorRemoveVariable,
   /// @ref rtSelectorGetVariableCount,
   /// \a rtVariableGet{...}
   ///
   pub fn rtSelectorGetVariable(
      selector: RTselector,
      index: ::std::os::raw::c_uint,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   ///
   /// Creates a new Transform node within the given context. For the node to be functional, a child
   /// node must be attached using @ref rtTransformSetChild.  A transformation matrix can be associated
   /// with the transform node with @ref rtTransformSetMatrix. Sets \a *transform to the handle of a
   /// newly created transform within \a context. Returns @ref RT_ERROR_INVALID_VALUE if \a transform
   /// is \a NULL.
   ///
   /// @param[in]   context    Specifies the rendering context of the Transform node
   /// @param[out]  transform  New Transform node handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformDestroy,
   /// @ref rtTransformValidate,
   /// @ref rtTransformGetContext,
   /// @ref rtTransformSetMatrix,
   /// @ref rtTransformGetMatrix,
   /// @ref rtTransformSetChild,
   /// @ref rtTransformGetChild,
   /// @ref rtTransformGetChildType
   ///
   pub fn rtTransformCreate(
      context: RTcontext,
      transform: *mut RTtransform,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTransformDestroy removes \a transform from its context and deletes it.
   /// \a transform should be a value returned by @ref rtTransformCreate.
   /// No child graph nodes are destroyed.
   /// After the call, \a transform is no longer a valid handle.
   ///
   /// @param[in]   transform   Handle of the transform node to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformCreate,
   /// @ref rtTransformValidate,
   /// @ref rtTransformGetContext
   ///
   pub fn rtTransformDestroy(transform: RTtransform) -> RtResult;
}
extern "C" {
   /// @brief Checks a Transform node for internal
   /// consistency
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTransformValidate recursively checks consistency of the
   /// Transform node \a transform and its child, i.e., it tries to validate
   /// the whole model sub-tree with \a transform as root. For a Transform
   /// node to be valid, it must have a child node attached. It is, however,
   /// not required to explicitly set a transformation matrix. Without a specified
   /// transformation matrix, the identity matrix is applied.
   ///
   /// @param[in]   transform   Transform root node of a model sub-tree to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformCreate,
   /// @ref rtTransformDestroy,
   /// @ref rtTransformGetContext,
   /// @ref rtTransformSetMatrix,
   /// @ref rtTransformSetChild
   ///
   pub fn rtTransformValidate(transform: RTtransform) -> RtResult;
}
extern "C" {
   /// @brief Returns the context of a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTransformGetContext queries a transform node for its associated context.  \a transform
   /// specifies the transform node to query, and should be a value returned by @ref
   /// rtTransformCreate. Sets \a *context to the context associated with \a transform.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[out]  context     The context associated with \a transform
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformCreate,
   /// @ref rtTransformDestroy,
   /// @ref rtTransformValidate
   ///
   pub fn rtTransformGetContext(
      transform: RTtransform,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Associates an affine transformation matrix
   /// with a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTransformSetMatrix associates a 4x4 matrix with the Transform
   /// node \a transform. The provided transformation matrix results in a
   /// corresponding affine transformation of all geometry contained in the
   /// sub-tree with \a transform as root. At least one of the pointers
   /// \a matrix and \a inverseMatrix must be non-\a NULL. If exactly one
   /// pointer is valid, the other matrix will be computed. If both are
   /// valid, the matrices will be used as-is. If \a transpose is \a 0,
   /// source matrices are expected to be in row-major format, i.e., matrix
   /// rows are contiguously laid out in memory:
   ///
   ///   float matrix[4*4] = { a11,  a12,  a13,  a14,
   ///                         a21,  a22,  a23,  a24,
   ///                         a31,  a32,  a33,  a34,
   ///                         a41,  a42,  a43,  a44 };
   ///
   /// Here, the translational elements \a a14, \a a24, and \a a34 are at the
   /// 4th, 8th, and 12th position the matrix array.  If the supplied
   /// matrices are in column-major format, a non-0 \a transpose flag
   /// can be used to trigger an automatic transpose of the input matrices.
   ///
   /// Calling this function clears any motion keys previously set for the Transform.
   ///
   /// @param[in]   transform        Transform node handle
   /// @param[in]   transpose        Flag indicating whether \a matrix and \a inverseMatrix should be
   /// transposed
   /// @param[in]   matrix           Affine matrix (4x4 float array)
   /// @param[in]   inverseMatrix    Inverted form of \a matrix
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformSetMatrix was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformGetMatrix
   ///
   pub fn rtTransformSetMatrix(
      transform: RTtransform,
      transpose: ::std::os::raw::c_int,
      matrix: *const f32,
      inverseMatrix: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the affine matrix and its inverse associated with a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTransformGetMatrix returns in \a matrix the affine matrix that
   /// is currently used to perform a transformation of the geometry
   /// contained in the sub-tree with \a transform as root. The corresponding
   /// inverse matrix will be returned in \a inverseMatrix. One or both
   /// pointers are allowed to be \a NULL. If \a transpose is \a 0, matrices
   /// are returned in row-major format, i.e., matrix rows are contiguously
   /// laid out in memory. If \a transpose is non-zero, matrices are returned
   /// in column-major format. If non-\a NULL, matrix pointers must point to a
   /// float array of at least 16 elements.
   ///
   /// @param[in]   transform        Transform node handle
   /// @param[in]   transpose        Flag indicating whether \a matrix and \a inverseMatrix should be
   /// transposed
   /// @param[out]  matrix           Affine matrix (4x4 float array)
   /// @param[out]  inverseMatrix    Inverted form of \a matrix
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformGetMatrix was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformSetMatrix
   ///
   pub fn rtTransformGetMatrix(
      transform: RTtransform,
      transpose: ::std::os::raw::c_int,
      matrix: *mut f32,
      inverseMatrix: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the motion time range for a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   /// Sets the inclusive motion time range [timeBegin, timeEnd] for \a transform, where timeBegin <= timeEnd.
   /// The default time range is [0.0, 1.0].  Has no effect unless @ref rtTransformSetMotionKeys
   /// is also called, in which case the left endpoint of the time range, \a timeBegin, is associated with
   /// the first motion key, and the right endpoint, \a timeEnd, with the last motion key.  The keys uniformly
   /// divide the time range.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[in]   timeBegin   Beginning time value of range
   /// @param[in]   timeEnd     Ending time value of range
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformSetMotionRange was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformGetMotionRange,
   /// @ref rtTransformSetMotionBorderMode,
   /// @ref rtTransformSetMotionKeys,
   ///
   pub fn rtTransformSetMotionRange(
      transform: RTtransform,
      timeBegin: f32,
      timeEnd: f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the motion time range associated with a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   /// @ref rtTransformGetMotionRange returns the motion time range set for the Transform.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[out]  timeBegin   Beginning time value of range
   /// @param[out]  timeEnd     Ending time value of range
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformGetMotionRange was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformSetMotionRange,
   /// @ref rtTransformGetMotionBorderMode,
   /// @ref rtTransformGetMotionKeyCount,
   /// @ref rtTransformGetMotionKeyType,
   /// @ref rtTransformGetMotionKeys,
   ///
   pub fn rtTransformGetMotionRange(
      transform: RTtransform,
      timeBegin: *mut f32,
      timeEnd: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the motion border modes of a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   /// @ref rtTransformSetMotionBorderMode sets the behavior of \a transform
   /// outside its motion time range. The \a beginMode and \a endMode arguments
   /// correspond to timeBegin and timeEnd set with @ref rtTransformSetMotionRange.
   /// The arguments are independent, and each has one of the following values:
   ///
   /// - @ref RT_MOTIONBORDERMODE_CLAMP :
   ///   The transform and the scene under it still exist at times less than timeBegin
   ///   or greater than timeEnd, with the transform clamped to its values at timeBegin
   ///   or timeEnd, respectively.
   ///
   /// - @ref RT_MOTIONBORDERMODE_VANISH :
   ///   The transform and the scene under it vanish for times less than timeBegin
   ///   or greater than timeEnd.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[in]   beginMode   Motion border mode at motion range begin
   /// @param[in]   endMode     Motion border mode at motion range end
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformSetMotionBorderMode was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformGetMotionBorderMode,
   /// @ref rtTransformSetMotionRange,
   /// @ref rtTransformSetMotionKeys,
   ///
   pub fn rtTransformSetMotionBorderMode(
      transform: RTtransform,
      beginMode: MotionBorderMode,
      endMode: MotionBorderMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the motion border modes of a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   /// @ref rtTransformGetMotionBorderMode returns the motion border modes
   /// for the time range associated with \a transform.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[out]  beginMode   Motion border mode at motion time range begin
   /// @param[out]  endMode     Motion border mode at motion time range end
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformGetMotionBorderMode was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformSetMotionBorderMode,
   /// @ref rtTransformGetMotionRange,
   /// @ref rtTransformGetMotionKeyCount,
   /// @ref rtTransformGetMotionKeyType,
   /// @ref rtTransformGetMotionKeys,
   ///
   pub fn rtTransformGetMotionBorderMode(
      transform: RTtransform,
      beginMode: *mut MotionBorderMode,
      endMode: *mut MotionBorderMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the motion keys associated with a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   /// @ref rtTransformSetMotionKeys sets a series of key values defining how
   /// \a transform varies with time.  The float values in \a keys are one of the
   /// following types:
   ///
   /// - @ref RT_MOTIONKEYTYPE_MATRIX_FLOAT12
   ///   Each key is a 12-float 3x4 matrix in row major order (3 rows, 4 columns).
   ///   The length of \a keys is 12*n.
   ///
   /// - @ref RT_MOTIONKEYTYPE_SRT_FLOAT16
   ///   Each key is a packed 16-float array in this order:
   ///     [sx, a, b, pvx, sy, c, pvy, sz, pvz, qx, qy, qz, qw, tx, ty, tz]
   ///   The length of \a keys is 16*n.
   ///
   ///   These are packed components of a scale/shear S, a quaternion R, and a translation T.
   ///
   ///   S = [ sx  a  b  pvx ]
   ///       [  * sy  c  pvy ]
   ///       [  *  * sz  pvz ]
   ///
   ///   R = [ qx, qy, qz, qw ]
   ///     where qw = cos(theta/2) and [qx, qy, qz] = sin(theta/2)*normalized_axis.
   ///
   ///   T = [ tx, ty, tz ]
   ///
   /// Removing motion keys:
   ///
   /// Passing a single key with \a n == 1, or calling @ref rtTransformSetMatrix, removes any
   /// motion data from \a transform, and sets its matrix to values derived from the single key.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[in]   n           Number of motion keys >= 1
   /// @param[in]   type        Type of motion keys
   /// @param[in]   keys        \a n Motion keys associated with this Transform
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformSetMotionKeys was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformGetMotionKeyCount,
   /// @ref rtTransformGetMotionKeyType,
   /// @ref rtTransformGetMotionKeys,
   /// @ref rtTransformSetMotionBorderMode,
   /// @ref rtTransformSetMotionRange,
   ///
   pub fn rtTransformSetMotionKeys(
      transform: RTtransform,
      n: ::std::os::raw::c_uint,
      type_: MotionKeyType,
      keys: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the motion key type associated with a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   /// @ref rtTransformGetMotionKeyType returns the key type from the most recent
   /// call to @ref rtTransformSetMotionKeys, or @ref RT_MOTIONKEYTYPE_NONE if no
   /// keys have been set.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[out]  type        Motion key type associated with this Transform
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformGetMotionKeyType was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformSetMotionKeys,
   /// @ref rtTransformGetMotionBorderMode,
   /// @ref rtTransformGetMotionRange,
   /// @ref rtTransformGetMotionKeyCount,
   /// @ref rtTransformGetMotionKeys
   ///
   pub fn rtTransformGetMotionKeyType(
      transform: RTtransform,
      type_: *mut MotionKeyType,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of motion keys associated with a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   /// @ref rtTransformGetMotionKeyCount returns in \a n the number of motion keys associated
   /// with \a transform using @ref rtTransformSetMotionKeys.  Note that the default value
   /// is 1, not 0, for a transform without motion.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[out]  n           Number of motion steps n >= 1
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformGetMotionKeyCount was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformSetMotionKeys,
   /// @ref rtTransformGetMotionBorderMode,
   /// @ref rtTransformGetMotionRange,
   /// @ref rtTransformGetMotionKeyType
   /// @ref rtTransformGetMotionKeys
   ///
   pub fn rtTransformGetMotionKeyCount(
      transform: RTtransform,
      n: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the motion keys associated with a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   /// @ref rtTransformGetMotionKeys returns in \a keys packed float values for
   /// all motion keys.  The \a keys array must be large enough to hold all the keys,
   /// based on the key type returned by @ref rtTransformGetMotionKeyType and the
   /// number of keys returned by @ref rtTransformGetMotionKeyCount.  A single key
   /// consists of either 12 floats (type RT_MOTIONKEYTYPE_MATRIX_FLOAT12) or
   /// 16 floats (type RT_MOTIONKEYTYPE_SRT_FLOAT16).
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[out]  keys        Motion keys associated with this Transform
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformGetMotionKeys was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformSetMotionKeys,
   /// @ref rtTransformGetMotionBorderMode,
   /// @ref rtTransformGetMotionRange,
   /// @ref rtTransformGetMotionKeyCount,
   /// @ref rtTransformGetMotionKeyType
   ///
   pub fn rtTransformGetMotionKeys(
      transform: RTtransform,
      keys: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Attaches a child node to a Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   ///
   /// Attaches a child node \a child to the parent node \a transform. Legal
   /// child node types are @ref RTgroup, @ref RTselector, @ref RTgeometrygroup,
   /// and @ref RTtransform. A transform node must have exactly one child.  If
   /// a transformation matrix has been attached to \a transform with
   /// @ref rtTransformSetMatrix, it is effective on the model sub-tree with
   /// \a child as root node.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[in]   child       Child node to be attached. Can be {@ref RTgroup, @ref RTselector,
   /// @ref RTgeometrygroup, @ref RTtransform}
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformSetChild was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformSetMatrix,
   /// @ref rtTransformGetChild,
   /// @ref rtTransformGetChildType
   ///
   pub fn rtTransformSetChild(
      transform: RTtransform,
      child: RTobject,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the child node that is attached to a
   /// Transform node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTransformGetChild returns in \a child a handle of the child
   /// node currently attached to \a transform. The returned pointer is of
   /// generic type @ref RTobject and needs to be cast to the actual child
   /// type, which can be @ref RTgroup, @ref RTselector, @ref RTgeometrygroup, or
   /// @ref RTtransform. The actual type of \a child can be queried using
   /// @ref rtTransformGetChildType.
   /// Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[out]  child       Child node handle. Can be {@ref RTgroup, @ref RTselector,
   /// @ref RTgeometrygroup, @ref RTtransform}
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformGetChild was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformSetChild,
   /// @ref rtTransformGetChildType
   ///
   pub fn rtTransformGetChild(
      transform: RTtransform,
      child: *mut RTobject,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns type information about a
   /// Transform child node
   ///
   /// @ingroup TransformNode
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTransformGetChildType queries the type of the child node
   /// attached to \a transform. If no child is attached, \a *type is set to
   /// @ref RT_OBJECTTYPE_UNKNOWN and @ref RT_ERROR_INVALID_VALUE is returned.
   /// Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
   /// The returned type is one of:
   ///
   ///  - @ref RT_OBJECTTYPE_GROUP
   ///  - @ref RT_OBJECTTYPE_GEOMETRY_GROUP
   ///  - @ref RT_OBJECTTYPE_TRANSFORM
   ///  - @ref RT_OBJECTTYPE_SELECTOR
   ///
   /// @param[in]   transform   Transform node handle
   /// @param[out]  type        Type of the child node
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTransformGetChildType was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTransformSetChild,
   /// @ref rtTransformGetChild
   ///
   pub fn rtTransformGetChildType(
      transform: RTtransform,
      type_: *mut ObjectType,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new geometry group
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupCreate creates a new geometry group within a context. \a context
   /// specifies the target context, and should be a value returned by @ref rtContextCreate.
   /// Sets \a *geometrygroup to the handle of a newly created geometry group within \a context.
   /// Returns @ref RT_ERROR_INVALID_VALUE if \a geometrygroup is \a NULL.
   ///
   /// @param[in]   context         Specifies a context within which to create a new geometry group
   /// @param[out]  geometrygroup   Returns a newly created geometry group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupDestroy,
   /// @ref rtContextCreate
   ///
   pub fn rtGeometryGroupCreate(
      context: RTcontext,
      geometrygroup: *mut RTgeometrygroup,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a geometry group node
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupDestroy removes \a geometrygroup from its context and deletes it.
   /// \a geometrygroup should be a value returned by @ref rtGeometryGroupCreate.
   /// No child graph nodes are destroyed.
   /// After the call, \a geometrygroup is no longer a valid handle.
   ///
   /// @param[in]   geometrygroup   Handle of the geometry group node to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupCreate
   ///
   pub fn rtGeometryGroupDestroy(geometrygroup: RTgeometrygroup) -> RtResult;
}
extern "C" {
   /// @brief Validates the state of the geometry group
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupValidate checks \a geometrygroup for completeness. If \a geometrygroup or
   /// any of the objects attached to \a geometrygroup are not valid, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   geometrygroup   Specifies the geometry group to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupCreate
   ///
   pub fn rtGeometryGroupValidate(geometrygroup: RTgeometrygroup) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with a geometry group
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupGetContext queries a geometry group for its associated context.
   /// \a geometrygroup specifies the geometry group to query, and must be a value returned by
   /// @ref rtGeometryGroupCreate. Sets \a *context to the context
   /// associated with \a geometrygroup.
   ///
   /// @param[in]   geometrygroup   Specifies the geometry group to query
   /// @param[out]  context         Returns the context associated with the geometry group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextCreate,
   /// @ref rtGeometryGroupCreate
   ///
   pub fn rtGeometryGroupGetContext(
      geometrygroup: RTgeometrygroup,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Set the acceleration structure for a group
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupSetAcceleration attaches an acceleration structure to a geometry group. The
   /// acceleration structure must have been previously created using @ref rtAccelerationCreate. Every
   /// geometry group is required to have an acceleration structure assigned in order to pass
   /// validation. The acceleration structure will be built over the primitives contained in all
   /// children of the geometry group. This enables a single acceleration structure to be built over
   /// primitives of multiple geometry instances.  Note that it is legal to attach a single
   /// RTacceleration object to multiple geometry groups, as long as the underlying geometry of all
   /// children is the same. This corresponds to attaching an acceleration structure to multiple groups
   /// at higher graph levels using @ref rtGroupSetAcceleration.
   ///
   /// @param[in]   geometrygroup   The geometry group handle
   /// @param[in]   acceleration    The acceleration structure to attach to the geometry group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupSetAcceleration was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupGetAcceleration,
   /// @ref rtAccelerationCreate,
   /// @ref rtGroupSetAcceleration
   ///
   pub fn rtGeometryGroupSetAcceleration(
      geometrygroup: RTgeometrygroup,
      acceleration: RTacceleration,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the acceleration structure attached to a geometry group
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupGetAcceleration returns the acceleration structure attached to a geometry
   /// group using @ref rtGeometryGroupSetAcceleration.  If no acceleration structure has previously
   /// been set, \a *acceleration is set to \a NULL.
   ///
   /// @param[in]   geometrygroup   The geometry group handle
   /// @param[out]  acceleration    The returned acceleration structure object
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupGetAcceleration was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupSetAcceleration,
   /// @ref rtAccelerationCreate
   ///
   pub fn rtGeometryGroupGetAcceleration(
      geometrygroup: RTgeometrygroup,
      acceleration: *mut RTacceleration,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets instance flags for a geometry group.
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// This function controls the @ref InstanceFlags of the given geometry group.
   /// The flags override the @ref GeometryFlags of the underlying geometry where appropriate.
   ///
   /// @param[in] group   The group handle
   /// @param[in] flags   Instance flags for the given geometry group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupSetFlags was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetFlagsPerMaterial,
   /// @ref rtGeometrySetFlags,
   /// @ref rtGeometryGroupGetFlags,
   /// @ref rtTrace
   pub fn rtGeometryGroupSetFlags(
      group: RTgeometrygroup,
      flags: InstanceFlags,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets instance flags of a geometry group.
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// See @ref rtGeometryGroupSetFlags for details.
   ///
   /// @param[in] group   The group handle
   /// @param[out] flags  Instance flags for the given geometry group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupGetFlags was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupSetFlags,
   /// @ref rtTrace
   pub fn rtGeometryGroupGetFlags(
      group: RTgeometrygroup,
      flags: *mut InstanceFlags,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the visibility mask of a geometry group.
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   /// Geometry is intersected by rays if the ray's @ref RTvisibilitymask shares at
   /// least one bit with the geometry's mask. This mechanism allows for a number of
   /// user-defined visibility groups that can be excluded from certain types of rays
   /// as needed.
   ///
   /// Note that the @pre mask is currently limited to 8 bits.
   ///
   /// @param[in] group   The group handle
   /// @param[in] mask    A set of bits for which rays will intersect the group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupSetVisibilityMask was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupSetVisibilityMask
   /// @ref rtGeometryGroupGetVisibilityMask,
   /// @ref rtTrace
   pub fn rtGeometryGroupSetVisibilityMask(
      group: RTgeometrygroup,
      mask: RTvisibilitymask,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the visibility mask of a geometry group.
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   /// See @ref rtGeometryGroupSetVisibilityMask for details/
   ///
   /// @param[in] group   The group handle
   /// @param[out] mask   A set of bits for which rays will intersect the group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupGetVisibilityMask was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGroupGetVisibilityMask
   /// @ref rtGeometryGroupSetVisibilityMask,
   /// @ref rtTrace
   pub fn rtGeometryGroupGetVisibilityMask(
      group: RTgeometrygroup,
      mask: *mut RTvisibilitymask,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the number of child nodes to be attached to the group
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupSetChildCount specifies the number of child slots in this geometry
   /// group. Potentially existing links to children at indices greater than \a count-1 are removed. If
   /// the call increases the number of slots, the newly created slots are empty and need to be filled
   /// using @ref rtGeometryGroupSetChild before validation.
   ///
   /// @param[in]   geometrygroup   The parent geometry group handle
   /// @param[in]   count           Number of child slots to allocate for the geometry group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupSetChildCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupGetChild,
   /// @ref rtGeometryGroupGetChildCount
   /// @ref rtGeometryGroupSetChild
   ///
   pub fn rtGeometryGroupSetChildCount(
      geometrygroup: RTgeometrygroup,
      count: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of child slots for a group
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupGetChildCount returns the number of child slots allocated using @ref
   /// rtGeometryGroupSetChildCount.  This includes empty slots which may not yet have actual children
   /// assigned by @ref rtGeometryGroupSetChild.
   ///
   /// @param[in]   geometrygroup   The parent geometry group handle
   /// @param[out]  count           Returned number of child slots
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupGetChildCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupSetChild,
   /// @ref rtGeometryGroupGetChild,
   /// @ref rtGeometryGroupSetChildCount
   ///
   pub fn rtGeometryGroupGetChildCount(
      geometrygroup: RTgeometrygroup,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Attaches a child node to a geometry group
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupSetChild attaches a new child node \a geometryinstance to the parent node
   /// \a geometrygroup. \a index specifies the number of the slot where the child
   /// node gets attached.  The index value must be lower than the number
   /// previously set by @ref rtGeometryGroupSetChildCount.
   ///
   /// @param[in]   geometrygroup      The parent geometry group handle
   /// @param[in]   index              The index in the parent's child slot array
   /// @param[in]   geometryinstance   The child node to be attached
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupSetChild was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupSetChildCount,
   /// @ref rtGeometryGroupGetChildCount,
   /// @ref rtGeometryGroupGetChild
   ///
   pub fn rtGeometryGroupSetChild(
      geometrygroup: RTgeometrygroup,
      index: ::std::os::raw::c_uint,
      geometryinstance: RTgeometryinstance,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a child node of a geometry group
   ///
   /// @ingroup GeometryGroup
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGroupGetChild returns the child geometry instance at slot \a index of the parent
   /// \a geometrygroup.  If no child has been assigned to the given slot, \a *geometryinstance is set
   /// to \a NULL.  Returns @ref RT_ERROR_INVALID_VALUE if given an invalid child index or \a NULL
   /// pointer.
   ///
   /// @param[in]   geometrygroup      The parent geometry group handle
   /// @param[in]   index              The index of the child slot to query
   /// @param[out]  geometryinstance   The returned child geometry instance
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGroupGetChild was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGroupSetChild,
   /// @ref rtGeometryGroupSetChildCount,
   /// @ref rtGeometryGroupGetChildCount,
   ///
   pub fn rtGeometryGroupGetChild(
      geometrygroup: RTgeometrygroup,
      index: ::std::os::raw::c_uint,
      geometryinstance: *mut RTgeometryinstance,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new acceleration structure
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationCreate creates a new ray tracing acceleration structure within a context.  An
   /// acceleration structure is used by attaching it to a group or geometry group by calling @ref
   /// rtGroupSetAcceleration or @ref rtGeometryGroupSetAcceleration. Note that an acceleration
   /// structure can be shared by attaching it to multiple groups or geometry groups if the underlying
   /// geometric structures are the same, see @ref rtGroupSetAcceleration and @ref
   /// rtGeometryGroupSetAcceleration for more details. A newly created acceleration structure is
   /// initially in dirty state.  Sets \a *acceleration to the handle of a newly created acceleration
   /// structure within \a context.  Returns @ref RT_ERROR_INVALID_VALUE if \a acceleration is \a NULL.
   ///
   /// @param[in]   context        Specifies a context within which to create a new acceleration structure
   /// @param[out]  acceleration   Returns the newly created acceleration structure
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationDestroy,
   /// @ref rtContextCreate,
   /// @ref rtAccelerationMarkDirty,
   /// @ref rtAccelerationIsDirty,
   /// @ref rtGroupSetAcceleration,
   /// @ref rtGeometryGroupSetAcceleration
   ///
   pub fn rtAccelerationCreate(
      context: RTcontext,
      acceleration: *mut RTacceleration,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys an acceleration structure object
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationDestroy removes \a acceleration from its context and deletes it.
   /// \a acceleration should be a value returned by @ref rtAccelerationCreate.
   /// After the call, \a acceleration is no longer a valid handle.
   ///
   /// @param[in]   acceleration   Handle of the acceleration structure to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationCreate
   ///
   pub fn rtAccelerationDestroy(acceleration: RTacceleration) -> RtResult;
}
extern "C" {
   /// @brief Validates the state of an acceleration structure
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationValidate checks \a acceleration for completeness. If \a acceleration is
   /// not valid, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   acceleration   The acceleration structure handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationCreate
   ///
   pub fn rtAccelerationValidate(acceleration: RTacceleration) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with an acceleration structure
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationGetContext queries an acceleration structure for its associated context.
   /// The context handle is returned in \a *context.
   ///
   /// @param[in]   acceleration   The acceleration structure handle
   /// @param[out]  context        Returns the context associated with the acceleration structure
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationCreate
   ///
   pub fn rtAccelerationGetContext(
      acceleration: RTacceleration,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Specifies the builder to be used for an acceleration structure
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationSetBuilder specifies the method used to construct the ray tracing
   /// acceleration structure represented by \a acceleration. A builder must be set for the
   /// acceleration structure to pass validation.  The current builder can be changed at any time,
   /// including after a call to @ref rtContextLaunch "rtContextLaunch".  In this case, data previously
   /// computed for the acceleration structure is invalidated and the acceleration will be marked
   /// dirty.
   ///
   /// \a builder can take one of the following values:
   ///
   /// - "NoAccel": Specifies that no acceleration structure is explicitly built. Traversal linearly loops through the
   /// list of primitives to intersect. This can be useful e.g. for higher level groups with only few children, where managing a more complex structure introduces unnecessary overhead.
   ///
   /// - "Bvh": A standard bounding volume hierarchy, useful for most types of graph levels and geometry. Medium build speed, good ray tracing performance.
   ///
   /// - "Sbvh": A high quality BVH variant for maximum ray tracing performance. Slower build speed and slightly higher memory footprint than "Bvh".
   ///
   /// - "Trbvh": High quality similar to Sbvh but with fast build performance. The Trbvh builder uses about 2.5 times the size of the final BVH for scratch space. A CPU-based Trbvh builder that does not have the memory constraints is available. OptiX includes an optional automatic fallback to the CPU version when out of GPU memory. Please refer to the Programming Guide for more details.  Supports motion blur.
   ///
   /// - "MedianBvh": Deprecated in OptiX 4.0. This builder is now internally remapped to Trbvh.
   ///
   /// - "Lbvh": Deprecated in OptiX 4.0. This builder is now internally remapped to Trbvh.
   ///
   /// - "TriangleKdTree": Deprecated in OptiX 4.0. This builder is now internally remapped to Trbvh.
   ///
   /// @param[in]   acceleration   The acceleration structure handle
   /// @param[in]   builder        String value specifying the builder type
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationSetBuilder was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationGetBuilder,
   /// @ref rtAccelerationSetProperty
   ///
   pub fn rtAccelerationSetBuilder(
      acceleration: RTacceleration,
      builder: *const ::std::os::raw::c_char,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query the current builder from an acceleration structure
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationGetBuilder returns the name of the builder currently
   /// used in the acceleration structure \a acceleration. If no builder has
   /// been set for \a acceleration, an empty string is returned.
   /// \a stringReturn will be set to point to the returned string. The
   /// memory \a stringReturn points to will be valid until the next API
   /// call that returns a string.
   ///
   /// @param[in]   acceleration    The acceleration structure handle
   /// @param[out]  stringReturn    Return string buffer
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationGetBuilder was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationSetBuilder
   ///
   pub fn rtAccelerationGetBuilder(
      acceleration: RTacceleration,
      stringReturn: *mut *const ::std::os::raw::c_char,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 4.0. Setting a traverser is no longer necessary and will be ignored.
   ///
   pub fn rtAccelerationSetTraverser(
      acceleration: RTacceleration,
      traverser: *const ::std::os::raw::c_char,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 4.0.
   ///
   pub fn rtAccelerationGetTraverser(
      acceleration: RTacceleration,
      stringReturn: *mut *const ::std::os::raw::c_char,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets an acceleration structure property
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationSetProperty sets a named property value for an
   /// acceleration structure. Properties can be used to fine tune the way an
   /// acceleration structure is built, in order to achieve faster build
   /// times or better ray tracing performance.  Properties are evaluated and
   /// applied by the acceleration structure during build time, and
   /// different builders recognize different properties. Setting a property
   /// will never fail as long as \a acceleration is a valid
   /// handle. Properties that are not recognized by an acceleration
   /// structure will be ignored.
   ///
   /// The following is a list of the properties used by the individual builders:
   ///
   /// - "refit":
   /// Available in: Trbvh, Bvh
   /// If set to "1", the builder will only readjust the node bounds of the bounding
   /// volume hierarchy instead of constructing it from scratch. Refit is only
   /// effective if there is an initial BVH already in place, and the underlying
   /// geometry has undergone relatively modest deformation.  In this case, the
   /// builder delivers a very fast BVH update without sacrificing too much ray
   /// tracing performance.  The default is "0".
   ///
   /// - "vertex_buffer_name":
   /// Available in: Trbvh, Sbvh
   /// The name of the buffer variable holding triangle vertex data.  Each vertex
   /// consists of 3 floats.  The default is "vertex_buffer".
   ///
   /// - "vertex_buffer_stride":
   /// Available in: Trbvh, Sbvh
   /// The offset between two vertices in the vertex buffer, given in bytes.  The
   /// default value is "0", which assumes the vertices are tightly packed.
   ///
   /// - "index_buffer_name":
   /// Available in: Trbvh, Sbvh
   /// The name of the buffer variable holding vertex index data. The entries in
   /// this buffer are indices of type int, where each index refers to one entry in
   /// the vertex buffer. A sequence of three indices represents one triangle. If no
   /// index buffer is given, the vertices in the vertex buffer are assumed to be a
   /// list of triangles, i.e. every 3 vertices in a row form a triangle.  The
   /// default is "index_buffer".
   ///
   /// - "index_buffer_stride":
   /// Available in: Trbvh, Sbvh
   /// The offset between two indices in the index buffer, given in bytes.  The
   /// default value is "0", which assumes the indices are tightly packed.
   ///
   /// - "chunk_size":
   /// Available in: Trbvh
   /// Number of bytes to be used for a partitioned acceleration structure build. If
   /// no chunk size is set, or set to "0", the chunk size is chosen automatically.
   /// If set to "-1", the chunk size is unlimited. The minimum chunk size is 64MB.
   /// Please note that specifying a small chunk size reduces the peak-memory
   /// footprint of the Trbvh but can result in slower rendering performance.
   ///
   /// - " motion_steps"
   /// Available in: Trbvh
   /// Number of motion steps to build into an acceleration structure that contains
   /// motion geometry or motion transforms. Ignored for acceleration structures
   /// built over static nodes. Gives a tradeoff between device memory
   /// and time: if the input geometry or transforms have many motion steps,
   /// then increasing the motion steps in the acceleration structure may result in
   /// faster traversal, at the cost of linear increase in memory usage.
   /// Default 2, and clamped >=1.
   ///
   /// @param[in]   acceleration   The acceleration structure handle
   /// @param[in]   name           String value specifying the name of the property
   /// @param[in]   value          String value specifying the value of the property
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationSetProperty was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationGetProperty,
   /// @ref rtAccelerationSetBuilder,
   ///
   pub fn rtAccelerationSetProperty(
      acceleration: RTacceleration,
      name: *const ::std::os::raw::c_char,
      value: *const ::std::os::raw::c_char,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries an acceleration structure property
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationGetProperty returns the value of the acceleration
   /// structure property \a name.  See @ref rtAccelerationSetProperty for a
   /// list of supported properties.  If the property name is not found, an
   /// empty string is returned.  \a stringReturn will be set to point to
   /// the returned string. The memory \a stringReturn points to will be
   /// valid until the next API call that returns a string.
   ///
   /// @param[in]   acceleration    The acceleration structure handle
   /// @param[in]   name            The name of the property to be queried
   /// @param[out]  stringReturn    Return string buffer
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationGetProperty was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationSetProperty,
   /// @ref rtAccelerationSetBuilder,
   ///
   pub fn rtAccelerationGetProperty(
      acceleration: RTacceleration,
      name: *const ::std::os::raw::c_char,
      stringReturn: *mut *const ::std::os::raw::c_char,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 4.0. Should not be called.
   ///
   pub fn rtAccelerationGetDataSize(
      acceleration: RTacceleration,
      size: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 4.0. Should not be called.
   ///
   pub fn rtAccelerationGetData(
      acceleration: RTacceleration,
      data: *mut ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 4.0. Should not be called.
   ///
   pub fn rtAccelerationSetData(
      acceleration: RTacceleration,
      data: *const ::std::os::raw::c_void,
      size: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Marks an acceleration structure as dirty
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationMarkDirty sets the dirty flag for \a acceleration.
   ///
   /// Any acceleration structure which is marked dirty will be rebuilt on a call to one of the @ref
   /// rtContextLaunch "rtContextLaunch" functions, and its dirty flag will be reset.
   ///
   /// An acceleration structure which is not marked dirty will never be rebuilt, even if associated
   /// groups, geometry, properties, or any other values have changed.
   ///
   /// Initially after creation, acceleration structures are marked dirty.
   ///
   /// @param[in]   acceleration   The acceleration structure handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationMarkDirty was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationIsDirty,
   /// @ref rtContextLaunch
   ///
   pub fn rtAccelerationMarkDirty(acceleration: RTacceleration) -> RtResult;
}
extern "C" {
   /// @brief Returns the dirty flag of an acceleration structure
   ///
   /// @ingroup AccelerationStructure
   ///
   /// <B>Description</B>
   ///
   /// @ref rtAccelerationIsDirty returns whether the acceleration structure is currently marked dirty.
   /// If the flag is set, a nonzero value will be returned in \a *dirty. Otherwise, zero is returned.
   ///
   /// Any acceleration structure which is marked dirty will be rebuilt on a call to one of the @ref
   /// rtContextLaunch "rtContextLaunch" functions, and its dirty flag will be reset.
   ///
   /// An acceleration structure which is not marked dirty will never be rebuilt, even if associated
   /// groups, geometry, properties, or any other values have changed.
   ///
   /// Initially after creation, acceleration structures are marked dirty.
   ///
   /// @param[in]   acceleration   The acceleration structure handle
   /// @param[out]  dirty          Returned dirty flag
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtAccelerationIsDirty was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtAccelerationMarkDirty,
   /// @ref rtContextLaunch
   ///
   pub fn rtAccelerationIsDirty(
      acceleration: RTacceleration,
      dirty: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new geometry instance node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceCreate creates a new geometry instance node within a context. \a context
   /// specifies the target context, and should be a value returned by @ref rtContextCreate.
   /// Sets \a *geometryinstance to the handle of a newly created geometry instance within \a context.
   /// Returns @ref RT_ERROR_INVALID_VALUE if \a geometryinstance is \a NULL.
   ///
   /// @param[in]   context            Specifies the rendering context of the GeometryInstance node
   /// @param[out]  geometryinstance   New GeometryInstance node handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceDestroy,
   /// @ref rtGeometryInstanceDestroy,
   /// @ref rtGeometryInstanceGetContext
   ///
   pub fn rtGeometryInstanceCreate(
      context: RTcontext,
      geometryinstance: *mut RTgeometryinstance,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a geometry instance node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceDestroy removes \a geometryinstance from its context and deletes it.  \a
   /// geometryinstance should be a value returned by @ref rtGeometryInstanceCreate.  Associated
   /// variables declared via @ref rtGeometryInstanceDeclareVariable are destroyed, but no child graph
   /// nodes are destroyed.  After the call, \a geometryinstance is no longer a valid handle.
   ///
   /// @param[in]   geometryinstance   Handle of the geometry instance node to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceCreate
   ///
   pub fn rtGeometryInstanceDestroy(
      geometryinstance: RTgeometryinstance,
   ) -> RtResult;
}
extern "C" {
   /// @brief Checks a GeometryInstance node for internal consistency
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceValidate checks \a geometryinstance for completeness. If \a geomertryinstance or
   /// any of the objects attached to \a geometry are not valid, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   geometryinstance   GeometryInstance node of a model sub-tree to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceCreate
   ///
   pub fn rtGeometryInstanceValidate(
      geometryinstance: RTgeometryinstance,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with a geometry instance node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceGetContext queries a geometry instance node for its associated context.
   /// \a geometryinstance specifies the geometry node to query, and should be a value returned by
   /// @ref rtGeometryInstanceCreate. Sets \a *context to the context
   /// associated with \a geometryinstance.
   ///
   /// @param[in]   geometryinstance   Specifies the geometry instance
   /// @param[out]  context            Handle for queried context
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceGetContext
   ///
   pub fn rtGeometryInstanceGetContext(
      geometryinstance: RTgeometryinstance,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Attaches a Geometry node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceSetGeometry attaches a Geometry node to a GeometryInstance.
   /// Only one GeometryTriangles or Geometry node can be attached to a GeometryInstance at a time.
   /// However, it is possible at any time to attach a different GeometryTriangles or Geometry via
   /// rtGeometryInstanceSetGeometryTriangles or rtGeometryInstanceSetGeometry respectively.
   ///
   /// @param[in]   geometryinstance   GeometryInstance node handle to attach \a geometry to
   /// @param[in]   geometry           Geometry handle to attach to \a geometryinstance
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceSetGeometry was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceGetGeometry
   /// @ref rtGeometryInstanceGetGeometryTriangles
   /// @ref rtGeometryInstanceSetGeometryTriangles
   ///
   pub fn rtGeometryInstanceSetGeometry(
      geometryinstance: RTgeometryinstance,
      geometry: RTgeometry,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the attached Geometry node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceGetGeometry sets \a geometry to the handle of the attached Geometry node.
   /// Only one GeometryTriangles or Geometry node can be attached to a GeometryInstance at a time.
   ///
   /// @param[in]   geometryinstance   GeometryInstance node handle to query geometry
   /// @param[out]  geometry           Handle to attached Geometry node
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceGetGeometry was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceCreate,
   /// @ref rtGeometryInstanceDestroy,
   /// @ref rtGeometryInstanceValidate,
   /// @ref rtGeometryInstanceSetGeometry
   /// @ref rtGeometryInstanceSetGeometryTriangles
   /// @ref rtGeometryInstanceGetGeometryTriangles
   ///
   pub fn rtGeometryInstanceGetGeometry(
      geometryinstance: RTgeometryinstance,
      geometry: *mut RTgeometry,
   ) -> RtResult;
}
extern "C" {
   /// @brief Attaches a Geometry node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceSetGeometryTriangles attaches a GeometryTriangles node to a GeometryInstance.
   /// Only one GeometryTriangles or Geometry node can be attached to a GeometryInstance at a time.
   /// However, it is possible at any time to attach a different GeometryTriangles or Geometry via
   /// rtGeometryInstanceSetGeometryTriangles or rtGeometryInstanceSetGeometry respectively.
   ///
   /// @param[in]   geometryinstance   GeometryInstance node handle to attach \a geometrytriangles to
   /// @param[in]   geometrytriangles  GeometryTriangles handle to attach to \a geometryinstance
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceSetGeometryTriangles was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceGetGeometryTriangles
   /// @ref rtGeometryInstanceSetGeometry
   /// @ref rtGeometryInstanceGetGeometry
   ///
   pub fn rtGeometryInstanceSetGeometryTriangles(
      geometryinstance: RTgeometryinstance,
      geometrytriangles: RTgeometrytriangles,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the attached Geometry node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceGetGeometryTriangles sets \a geometrytriangles to the handle of the attached GeometryTriangles node.
   /// If no GeometryTriangles node is attached or a Geometry node is attached, @ref RT_ERROR_INVALID_VALUE is returned, else @ref RT_SUCCESS.
   ///
   /// @param[in]   geometryinstance   GeometryInstance node handle to query geometrytriangles
   /// @param[out]  geometrytriangles  Handle to attached GeometryTriangles node
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceGetGeometryTriangles was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceCreate,
   /// @ref rtGeometryInstanceDestroy,
   /// @ref rtGeometryInstanceValidate,
   /// @ref rtGeometryInstanceSetGeometryTriangles
   /// @ref rtGeometryInstanceSetGeometry
   /// @ref rtGeometryInstanceGetGeometry
   ///
   pub fn rtGeometryInstanceGetGeometryTriangles(
      geometryinstance: RTgeometryinstance,
      geometrytriangles: *mut RTgeometrytriangles,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the number of materials
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceSetMaterialCount sets the number of materials \a count that will be
   /// attached to \a geometryinstance. The number of attached materials can be changed at any
   /// time.  Increasing the number of materials will not modify already assigned materials.
   /// Decreasing the number of materials will not modify the remaining already assigned
   /// materials.
   ///
   /// @param[in]   geometryinstance   GeometryInstance node to set number of materials
   /// @param[in]   count              Number of materials to be set
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceSetMaterialCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceGetMaterialCount
   ///
   pub fn rtGeometryInstanceSetMaterialCount(
      geometryinstance: RTgeometryinstance,
      count: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of attached materials
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceGetMaterialCount returns for \a geometryinstance the number of attached
   /// Material nodes \a count. The number of materials can be set with @ref
   /// rtGeometryInstanceSetMaterialCount.
   ///
   /// @param[in]   geometryinstance   GeometryInstance node to query from the number of materials
   /// @param[out]  count              Number of attached materials
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceGetMaterialCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceSetMaterialCount
   ///
   pub fn rtGeometryInstanceGetMaterialCount(
      geometryinstance: RTgeometryinstance,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets a material
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceSetMaterial attaches \a material to \a geometryinstance at position \a index
   /// in its internal Material node list.  \a index must be in the range \a 0 to @ref
   /// rtGeometryInstanceGetMaterialCount \a - 1.
   ///
   /// @param[in]   geometryinstance   GeometryInstance node for which to set a material
   /// @param[in]   index              Index into the material list
   /// @param[in]   material           Material handle to attach to \a geometryinstance
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceSetMaterial was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceGetMaterialCount,
   /// @ref rtGeometryInstanceSetMaterialCount
   ///
   pub fn rtGeometryInstanceSetMaterial(
      geometryinstance: RTgeometryinstance,
      index: ::std::os::raw::c_uint,
      material: RTmaterial,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a material handle
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceGetMaterial returns handle \a material for the Material node at position
   /// \a index in the material list of \a geometryinstance. Returns @ref RT_ERROR_INVALID_VALUE if \a
   /// index is invalid.
   ///
   /// @param[in]   geometryinstance   GeometryInstance node handle to query material
   /// @param[in]   index              Index of material
   /// @param[out]  material           Handle to material
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceGetMaterial was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceGetMaterialCount,
   /// @ref rtGeometryInstanceSetMaterial
   ///
   pub fn rtGeometryInstanceGetMaterial(
      geometryinstance: RTgeometryinstance,
      index: ::std::os::raw::c_uint,
      material: *mut RTmaterial,
   ) -> RtResult;
}
extern "C" {
   /// @brief Declares a new named variable associated with a geometry node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceDeclareVariable declares a new variable associated with a geometry
   /// instance node. \a geometryinstance specifies the target geometry node, and should be a value
   /// returned by @ref rtGeometryInstanceCreate. \a name specifies the name of the variable, and
   /// should be a \a NULL-terminated string. If there is currently no variable associated with \a
   /// geometryinstance named \a name, a new variable named \a name will be created and associated with
   /// \a geometryinstance.  After the call, \a *v will be set to the handle of the newly-created
   /// variable.  Otherwise, \a *v will be set to \a NULL. After declaration, the variable can be
   /// queried with @ref rtGeometryInstanceQueryVariable or @ref rtGeometryInstanceGetVariable. A
   /// declared variable does not have a type until its value is set with one of the @ref rtVariableSet
   /// functions. Once a variable is set, its type cannot be changed anymore.
   ///
   /// @param[in]   geometryinstance   Specifies the associated GeometryInstance node
   /// @param[in]   name               The name that identifies the variable
   /// @param[out]  v                  Returns a handle to a newly declared variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceDeclareVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref Variables,
   /// @ref rtGeometryInstanceQueryVariable,
   /// @ref rtGeometryInstanceGetVariable,
   /// @ref rtGeometryInstanceRemoveVariable
   ///
   pub fn rtGeometryInstanceDeclareVariable(
      geometryinstance: RTgeometryinstance,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a handle to a named variable of a geometry node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceQueryVariable queries the handle of a geometry instance node's named
   /// variable.  \a geometryinstance specifies the target geometry instance node, as returned by
   /// @ref rtGeometryInstanceCreate. \a name specifies the name of the variable, and should be a \a
   /// \a NULL -terminated string. If \a name is the name of a variable attached to \a geometryinstance,
   /// returns a handle to that variable in \a *v, otherwise \a NULL.  Geometry instance variables have
   /// to be declared with @ref rtGeometryInstanceDeclareVariable before they can be queried.
   ///
   /// @param[in]   geometryinstance   The GeometryInstance node to query from a variable
   /// @param[in]   name               The name that identifies the variable to be queried
   /// @param[out]  v                  Returns the named variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceQueryVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceDeclareVariable,
   /// @ref rtGeometryInstanceRemoveVariable,
   /// @ref rtGeometryInstanceGetVariableCount,
   /// @ref rtGeometryInstanceGetVariable
   ///
   pub fn rtGeometryInstanceQueryVariable(
      geometryinstance: RTgeometryinstance,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Removes a named variable from a geometry instance node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceRemoveVariable removes a named variable from a geometry instance. The
   /// target geometry instance is specified by \a geometryinstance, which should be a value returned
   /// by @ref rtGeometryInstanceCreate. The variable to be removed is specified by \a v, which should
   /// be a value returned by @ref rtGeometryInstanceDeclareVariable. Once a variable has been removed
   /// from this geometry instance, another variable with the same name as the removed variable may be
   /// declared.
   ///
   /// @param[in]   geometryinstance   The GeometryInstance node from which to remove a variable
   /// @param[in]   v                  The variable to be removed
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceRemoveVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextRemoveVariable,
   /// @ref rtGeometryInstanceDeclareVariable
   ///
   pub fn rtGeometryInstanceRemoveVariable(
      geometryinstance: RTgeometryinstance,
      v: RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of attached variables
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceGetVariableCount queries the number of variables attached to a geometry instance.
   /// \a geometryinstance specifies the geometry instance, and should be a value returned by @ref rtGeometryInstanceCreate.
   /// After the call, the number of variables attached to \a geometryinstance is returned to \a *count.
   ///
   /// @param[in]   geometryinstance   The GeometryInstance node to query from the number of attached variables
   /// @param[out]  count              Returns the number of attached variables
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceGetVariableCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryInstanceGetVariableCount,
   /// @ref rtGeometryInstanceDeclareVariable,
   /// @ref rtGeometryInstanceRemoveVariable
   ///
   pub fn rtGeometryInstanceGetVariableCount(
      geometryinstance: RTgeometryinstance,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a handle to an indexed variable of a geometry instance node
   ///
   /// @ingroup GeometryInstance
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryInstanceGetVariable queries the handle of a geometry instance's indexed variable.
   /// \a geometryinstance specifies the target geometry instance and should be a value returned by
   /// @ref rtGeometryInstanceCreate. \a index specifies the index of the variable, and should be a
   /// value less than @ref rtGeometryInstanceGetVariableCount. If \a index is the index of a variable
   /// attached to \a geometryinstance, returns a handle to that variable in \a *v, and \a NULL
   /// otherwise. \a *v must be declared first with @ref rtGeometryInstanceDeclareVariable before it
   /// can be queried.
   ///
   /// @param[in]   geometryinstance   The GeometryInstance node from which to query a variable
   /// @param[in]   index              The index that identifies the variable to be queried
   /// @param[out]  v                  Returns handle to indexed variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryInstanceGetVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryDeclareVariable,
   /// @ref rtGeometryGetVariableCount,
   /// @ref rtGeometryRemoveVariable,
   /// @ref rtGeometryQueryVariable
   ///
   pub fn rtGeometryInstanceGetVariable(
      geometryinstance: RTgeometryinstance,
      index: ::std::os::raw::c_uint,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new geometry node
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryCreate creates a new geometry node within a context. \a context
   /// specifies the target context, and should be a value returned by @ref rtContextCreate.
   /// Sets \a *geometry to the handle of a newly created geometry within \a context.
   /// Returns @ref RT_ERROR_INVALID_VALUE if \a geometry is \a NULL.
   ///
   /// @param[in]   context    Specifies the rendering context of the Geometry node
   /// @param[out]  geometry   New Geometry node handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryDestroy,
   /// @ref rtGeometrySetBoundingBoxProgram,
   /// @ref rtGeometrySetIntersectionProgram
   ///
   pub fn rtGeometryCreate(
      context: RTcontext,
      geometry: *mut RTgeometry,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a geometry node
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryDestroy removes \a geometry from its context and deletes it.  \a geometry should
   /// be a value returned by @ref rtGeometryCreate.  Associated variables declared via
   /// @ref rtGeometryDeclareVariable are destroyed, but no child graph nodes are destroyed.  After the
   /// call, \a geometry is no longer a valid handle.
   ///
   /// @param[in]   geometry   Handle of the geometry node to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryCreate,
   /// @ref rtGeometrySetPrimitiveCount,
   /// @ref rtGeometryGetPrimitiveCount
   ///
   pub fn rtGeometryDestroy(geometry: RTgeometry) -> RtResult;
}
extern "C" {
   /// @brief Validates the geometry nodes integrity
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryValidate checks \a geometry for completeness. If \a geometry or any of the
   /// objects attached to \a geometry are not valid, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   geometry   The geometry node to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextValidate
   ///
   pub fn rtGeometryValidate(geometry: RTgeometry) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with a geometry node
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGetContext queries a geometry node for its associated context.  \a geometry
   /// specifies the geometry node to query, and should be a value returned by @ref
   /// rtGeometryCreate. Sets \a *context to the context associated with \a geometry.
   ///
   /// @param[in]   geometry   Specifies the geometry to query
   /// @param[out]  context    The context associated with \a geometry
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryCreate
   ///
   pub fn rtGeometryGetContext(
      geometry: RTgeometry,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the number of primitives
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometrySetPrimitiveCount sets the number of primitives \a primitiveCount in \a geometry.
   ///
   /// @param[in]   geometry         The geometry node for which to set the number of primitives
   /// @param[in]   primitiveCount   The number of primitives
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometrySetPrimitiveCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetPrimitiveCount
   ///
   pub fn rtGeometrySetPrimitiveCount(
      geometry: RTgeometry,
      primitiveCount: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of primitives
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGetPrimitiveCount returns for \a geometry the number of set primitives. The
   /// number of primitvies can be set with @ref rtGeometryGetPrimitiveCount.
   ///
   /// @param[in]   geometry         Geometry node to query from the number of primitives
   /// @param[out]  primitiveCount   Number of primitives
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetPrimitiveCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometrySetPrimitiveCount
   ///
   pub fn rtGeometryGetPrimitiveCount(
      geometry: RTgeometry,
      primitiveCount: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the primitive index offset
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometrySetPrimitiveIndexOffset sets the primitive index offset
   /// \a indexOffset in \a geometry.  In the past, a @ref Geometry object's primitive
   /// index range always started at zero (i.e., a Geometry with \a N primitives would
   /// have a primitive index range of [0,N-1]).  The index offset is used to allow
   /// @ref Geometry objects to have primitive index ranges starting at non-zero
   /// positions (i.e., a Geometry with \a N primitives and an index offset of \a M
   /// would have a primitive index range of [M,M+N-1]).  This feature enables the
   /// sharing of vertex index buffers between multiple @ref Geometry objects.
   ///
   /// @param[in]   geometry       The geometry node for which to set the primitive index offset
   /// @param[in]   indexOffset    The primitive index offset
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometrySetPrimitiveIndexOffset was introduced in OptiX 3.5.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetPrimitiveIndexOffset
   ///
   pub fn rtGeometrySetPrimitiveIndexOffset(
      geometry: RTgeometry,
      indexOffset: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the current primitive index offset
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGetPrimitiveIndexOffset returns for \a geometry the primitive index offset. The
   /// primitive index offset can be set with @ref rtGeometrySetPrimitiveIndexOffset.
   ///
   /// @param[in]   geometry       Geometry node to query for the primitive index offset
   /// @param[out]  indexOffset    Primitive index offset
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetPrimitiveIndexOffset was introduced in OptiX 3.5.
   ///
   /// <B>See also</B>
   /// @ref rtGeometrySetPrimitiveIndexOffset
   ///
   pub fn rtGeometryGetPrimitiveIndexOffset(
      geometry: RTgeometry,
      indexOffset: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the motion time range for a Geometry node.
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   /// Sets the inclusive motion time range [timeBegin, timeEnd] for \a geometry,
   /// where timeBegin <= timeEnd.  The default time range is [0.0, 1.0].  The
   /// time range has no effect unless @ref rtGeometrySetMotionSteps is
   /// called, in which case the time steps uniformly divide the time range.  See
   /// @ref rtGeometrySetMotionSteps for additional requirements on the bounds
   /// program.
   ///
   /// @param[in]   geometry    Geometry node handle
   /// @param[out]  timeBegin   Beginning time value of range
   /// @param[out]  timeEnd     Ending time value of range
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometrySetMotionRange was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetMotionRange
   /// @ref rtGeometrySetMotionBorderMode
   /// @ref rtGeometrySetMotionSteps
   ///
   pub fn rtGeometrySetMotionRange(
      geometry: RTgeometry,
      timeBegin: f32,
      timeEnd: f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the motion time range associated with a Geometry node.
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   /// @ref rtGeometryGetMotionRange returns the motion time range associated with
   /// \a geometry from a previous call to @ref rtGeometrySetMotionRange, or the
   /// default values of [0.0, 1.0].
   ///
   ///
   /// @param[in]   geometry    Geometry node handle
   /// @param[out]  timeBegin   Beginning time value of range
   /// @param[out]  timeEnd     Ending time value of range
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetMotionRange was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometrySetMotionRange
   /// @ref rtGeometryGetMotionBorderMode
   /// @ref rtGeometryGetMotionSteps
   ///
   pub fn rtGeometryGetMotionRange(
      geometry: RTgeometry,
      timeBegin: *mut f32,
      timeEnd: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the motion border modes of a Geometry node
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   /// @ref rtGeometrySetMotionBorderMode sets the behavior of \a geometry
   /// outside its motion time range. Options are @ref RT_MOTIONBORDERMODE_CLAMP
   /// or @ref RT_MOTIONBORDERMODE_VANISH.  See @ref rtTransformSetMotionBorderMode
   /// for details.
   ///
   /// @param[in]   geometry    Geometry node handle
   /// @param[in]   beginMode   Motion border mode at motion range begin
   /// @param[in]   endMode     Motion border mode at motion range end
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometrySetMotionBorderMode was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetMotionBorderMode
   /// @ref rtGeometrySetMotionRange
   /// @ref rtGeometrySetMotionSteps
   ///
   pub fn rtGeometrySetMotionBorderMode(
      geometry: RTgeometry,
      beginMode: MotionBorderMode,
      endMode: MotionBorderMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the motion border modes of a Geometry node
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   /// @ref rtGeometryGetMotionBorderMode returns the motion border modes
   /// for the time range associated with \a geometry.
   ///
   /// @param[in]   geometry    Geometry node handle
   /// @param[out]  beginMode   Motion border mode at motion range begin
   /// @param[out]  endMode     Motion border mode at motion range end
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetMotionBorderMode was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometrySetMotionBorderMode
   /// @ref rtGeometryGetMotionRange
   /// @ref rtGeometryGetMotionSteps
   ///
   pub fn rtGeometryGetMotionBorderMode(
      geometry: RTgeometry,
      beginMode: *mut MotionBorderMode,
      endMode: *mut MotionBorderMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Specifies the number of motion steps associated with a Geometry
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   /// @ref rtGeometrySetMotionSteps sets the number of motion steps associated
   /// with \a geometry.  If the value of \a n is greater than 1, then \a geometry
   /// must have an associated bounding box program that takes both a primitive index
   /// and a motion index as arguments, and computes an aabb at the motion index.
   /// See @ref rtGeometrySetBoundingBoxProgram.
   ///
   /// Note that all Geometry has at least one 1 motion step (the default), and
   /// Geometry that linearly moves has 2 motion steps.
   ///
   /// @param[in]   geometry    Geometry node handle
   /// @param[in]   n           Number of motion steps >= 1
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometrySetMotionSteps was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetMotionSteps
   /// @ref rtGeometrySetMotionBorderMode
   /// @ref rtGeometrySetMotionRange
   ///
   pub fn rtGeometrySetMotionSteps(
      geometry: RTgeometry,
      n: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of motion steps associated with a Geometry node
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   /// @ref rtGeometryGetMotionSteps returns in \a n the number of motion steps
   /// associated with \a geometry.  Note that the default value is 1, not 0,
   /// for geometry without motion.
   ///
   /// @param[in]   geometry    Geometry node handle
   /// @param[out]  n           Number of motion steps n >= 1
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetMotionSteps was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetMotionSteps
   /// @ref rtGeometrySetMotionBorderMode
   /// @ref rtGeometrySetMotionRange
   ///
   pub fn rtGeometryGetMotionSteps(
      geometry: RTgeometry,
      n: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the bounding box program
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometrySetBoundingBoxProgram sets for \a geometry the \a program that computes an axis aligned bounding box
   /// for each attached primitive to \a geometry. RTprogram's can be either generated with @ref rtProgramCreateFromPTXFile or
   /// @ref rtProgramCreateFromPTXString. A bounding box program is mandatory for every geometry node.
   ///
   /// If \a geometry has more than one motion step, set using @ref rtGeometrySetMotionSteps, then the bounding
   /// box program must compute a bounding box per primitive and per motion step.
   ///
   /// @param[in]   geometry   The geometry node for which to set the bounding box program
   /// @param[in]   program    Handle to the bounding box program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometrySetBoundingBoxProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetBoundingBoxProgram,
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramCreateFromPTXString
   ///
   pub fn rtGeometrySetBoundingBoxProgram(
      geometry: RTgeometry,
      program: RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the attached bounding box program
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGetBoundingBoxProgram returns the handle \a program for
   /// the attached bounding box program of \a geometry.
   ///
   /// @param[in]   geometry   Geometry node handle from which to query program
   /// @param[out]  program    Handle to attached bounding box program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetBoundingBoxProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometrySetBoundingBoxProgram
   ///
   pub fn rtGeometryGetBoundingBoxProgram(
      geometry: RTgeometry,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the intersection program
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometrySetIntersectionProgram sets for \a geometry the \a program that performs ray primitive intersections.
   /// RTprogram's can be either generated with @ref rtProgramCreateFromPTXFile or @ref rtProgramCreateFromPTXString. An intersection
   /// program is mandatory for every geometry node.
   ///
   /// @param[in]   geometry   The geometry node for which to set the intersection program
   /// @param[in]   program    A handle to the ray primitive intersection program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometrySetIntersectionProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetIntersectionProgram,
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramCreateFromPTXString
   ///
   pub fn rtGeometrySetIntersectionProgram(
      geometry: RTgeometry,
      program: RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the attached intersection program
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGetIntersectionProgram returns in \a program a handle of the attached intersection program.
   ///
   /// @param[in]   geometry   Geometry node handle to query program
   /// @param[out]  program    Handle to attached intersection program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetIntersectionProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometrySetIntersectionProgram,
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramCreateFromPTXString
   ///
   pub fn rtGeometryGetIntersectionProgram(
      geometry: RTgeometry,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets geometry flags
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// See @ref rtGeometryTrianglesSetFlagsPerMaterial for a description of the behavior of the
   /// various flags.
   ///
   /// @param[in] geometry        The group handle
   /// @param[out] flags          Flags for the given geometry group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometrySetFlags was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetFlagsPerMaterial,
   /// @ref rtTrace
   pub fn rtGeometrySetFlags(
      geometry: RTgeometry,
      flags: GeometryFlags,
   ) -> RtResult;
}
extern "C" {
   /// @brief Retrieves geometry flags
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// See @ref rtGeometrySetFlags for details.
   ///
   /// @param[in] geometry        The group handle
   /// @param[out] flags          Flags for the given geometry group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetFlags was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetFlagsPerMaterial,
   /// @ref rtTrace
   pub fn rtGeometryGetFlags(
      geometry: RTgeometry,
      flags: *mut GeometryFlags,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 4.0. Calling this function has no effect.
   ///
   pub fn rtGeometryMarkDirty(geometry: RTgeometry) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 4.0. Calling this function has no effect.
   ///
   pub fn rtGeometryIsDirty(
      geometry: RTgeometry,
      dirty: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Declares a new named variable associated with a geometry instance
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryDeclareVariable declares a new variable associated with a geometry node. \a
   /// geometry specifies the target geometry node, and should be a value returned by @ref
   /// rtGeometryCreate. \a name specifies the name of the variable, and should be a \a NULL-terminated
   /// string. If there is currently no variable associated with \a geometry named \a name, a new
   /// variable named \a name will be created and associated with \a geometry.  Returns the handle of
   /// the newly-created variable in \a *v or \a NULL otherwise.  After declaration, the variable can
   /// be queried with @ref rtGeometryQueryVariable or @ref rtGeometryGetVariable. A declared variable
   /// does not have a type until its value is set with one of the @ref rtVariableSet functions. Once a
   /// variable is set, its type cannot be changed anymore.
   ///
   /// @param[in]   geometry   Specifies the associated Geometry node
   /// @param[in]   name       The name that identifies the variable
   /// @param[out]  v          Returns a handle to a newly declared variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_REDECLARED
   /// - @ref RT_ERROR_ILLEGAL_SYMBOL
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryDeclareVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref Variables,
   /// @ref rtGeometryQueryVariable,
   /// @ref rtGeometryGetVariable,
   /// @ref rtGeometryRemoveVariable
   ///
   pub fn rtGeometryDeclareVariable(
      geometry: RTgeometry,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a handle to a named variable of a geometry node
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryQueryVariable queries the handle of a geometry node's named variable.
   /// \a geometry specifies the target geometry node and should be a value returned
   /// by @ref rtGeometryCreate. \a name specifies the name of the variable, and should
   /// be a \a NULL-terminated string. If \a name is the name of a variable attached to
   /// \a geometry, returns a handle to that variable in \a *v or \a NULL otherwise. Geometry
   /// variables must be declared with @ref rtGeometryDeclareVariable before they can be queried.
   ///
   /// @param[in]   geometry   The geometry node to query from a variable
   /// @param[in]   name       The name that identifies the variable to be queried
   /// @param[out]  v          Returns the named variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryQueryVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryDeclareVariable,
   /// @ref rtGeometryRemoveVariable,
   /// @ref rtGeometryGetVariableCount,
   /// @ref rtGeometryGetVariable
   ///
   pub fn rtGeometryQueryVariable(
      geometry: RTgeometry,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Removes a named variable from a geometry node
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryRemoveVariable removes a named variable from a geometry node. The
   /// target geometry is specified by \a geometry, which should be a value
   /// returned by @ref rtGeometryCreate. The variable to remove is specified by
   /// \a v, which should be a value returned by @ref rtGeometryDeclareVariable.
   /// Once a variable has been removed from this geometry node, another variable with the
   /// same name as the removed variable may be declared.
   ///
   /// @param[in]   geometry   The geometry node from which to remove a variable
   /// @param[in]   v          The variable to be removed
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryRemoveVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextRemoveVariable
   ///
   pub fn rtGeometryRemoveVariable(
      geometry: RTgeometry,
      v: RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of attached variables
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGetVariableCount queries the number of variables attached to a geometry node.
   /// \a geometry specifies the geometry node, and should be a value returned by @ref rtGeometryCreate.
   /// After the call, the number of variables attached to \a geometry is returned to \a *count.
   ///
   /// @param[in]   geometry   The Geometry node to query from the number of attached variables
   /// @param[out]  count      Returns the number of attached variables
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetVariableCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryGetVariableCount,
   /// @ref rtGeometryDeclareVariable,
   /// @ref rtGeometryRemoveVariable
   ///
   pub fn rtGeometryGetVariableCount(
      geometry: RTgeometry,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a handle to an indexed variable of a geometry node
   ///
   /// @ingroup Geometry
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryGetVariable queries the handle of a geometry node's indexed variable.
   /// \a geometry specifies the target geometry and should be a value returned
   /// by @ref rtGeometryCreate. \a index specifies the index of the variable, and
   /// should be a value less than @ref rtGeometryGetVariableCount. If \a index is the
   /// index of a variable attached to \a geometry, returns its handle in \a *v or \a NULL otherwise.
   /// \a *v must be declared first with @ref rtGeometryDeclareVariable before it can be queried.
   ///
   /// @param[in]   geometry   The geometry node from which to query a variable
   /// @param[in]   index      The index that identifies the variable to be queried
   /// @param[out]  v          Returns handle to indexed variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryGetVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryDeclareVariable,
   /// @ref rtGeometryGetVariableCount,
   /// @ref rtGeometryRemoveVariable,
   /// @ref rtGeometryQueryVariable
   ///
   pub fn rtGeometryGetVariable(
      geometry: RTgeometry,
      index: ::std::os::raw::c_uint,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new GeometryTriangles node
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesCreate creates a new GeometryTriangles node within a context. \a context
   /// specifies the target context, and should be a value returned by @ref rtContextCreate.
   /// Sets \a *geometrytriangles to the handle of a newly created GeometryTriangles node within \a context.
   /// Returns @ref RT_ERROR_INVALID_VALUE if \a geometrytriangles is \a NULL.
   ///
   /// @param[in]   context            Specifies the rendering context of the GeometryTriangles node
   /// @param[out]  geometrytriangles  New GeometryTriangles node handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesCreate was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesDestroy,
   ///
   pub fn rtGeometryTrianglesCreate(
      context: RTcontext,
      geometrytriangles: *mut RTgeometrytriangles,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a GeometryTriangles node
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesDestroy removes \a geometrytriangles from its context and deletes it.  \a geometrytriangles should
   /// be a value returned by @ref rtGeometryTrianglesCreate.  After the call, \a geometrytriangles is no longer a valid handle.
   ///
   /// @param[in]   geometrytriangles   Handle of the GeometryTriangles node to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesDestroy was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesCreate,
   /// @ref rtGeometryTrianglesSetPrimitiveCount,
   /// @ref rtGeometryTrianglesGetPrimitiveCount
   ///
   pub fn rtGeometryTrianglesDestroy(
      geometrytriangles: RTgeometrytriangles,
   ) -> RtResult;
}
extern "C" {
   /// @brief Validates the GeometryTriangles nodes integrity
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesValidate checks \a geometrytriangles for completeness. If \a geometrytriangles or any of the
   /// objects attached to \a geometrytriangles are not valid, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   geometrytriangles   The GeometryTriangles node to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesValidate was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextValidate
   ///
   pub fn rtGeometryTrianglesValidate(
      geometrytriangles: RTgeometrytriangles,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with a GeometryTriangles node
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesGetContext queries a GeometryTriangles node for its associated context.  \a geometrytriangles
   /// specifies the GeometryTriangles node to query, and should be a value returned by @ref
   /// rtGeometryTrianglesCreate. Sets \a *context to the context associated with \a geometrytriangles.
   ///
   /// @param[in]   geometrytriangles   Specifies the GeometryTriangles to query
   /// @param[out]  context             The context associated with \a geometrytriangles
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetContext was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesCreate
   ///
   pub fn rtGeometryTrianglesGetContext(
      geometrytriangles: RTgeometrytriangles,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the primitive index offset
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesSetPrimitiveIndexOffset sets the primitive index offset
   /// \a indexOffset in \a geometrytriangles.
   /// With an offset of zero, a GeometryTriangles with \a N triangles has a primitive index range of [0,N-1].
   /// The index offset is used to allow GeometryTriangles objects to have primitive index ranges starting at non-zero
   /// positions (i.e., a GeometryTriangles with \a N triangles and an index offset of \a M
   /// has a primitive index range of [M,M+N-1]).
   /// Note that this offset only affects the primitive index that is reported in case of an intersection and does not
   /// affect the input data that is specified via @ref rtGeometryTrianglesSetVertices or @ref
   /// rtGeometryTrianglesSetTriangleIndices.
   /// This feature enables the packing of multiple Geometries or GeometryTriangles into a single buffer.
   /// While the same effect could be reached via a user variable, it is recommended to specify the offset via
   /// @ref rtGeometryTrianglesSetPrimitiveIndexOffset.
   ///
   /// @param[in]   geometrytriangles  The GeometryTriangles node for which to set the primitive index offset
   /// @param[in]   indexOffset        The primitive index offset
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetPrimitiveIndexOffset was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometrySetPrimitiveIndexOffset
   /// @ref rtGeometryTrianglesGetPrimitiveIndexOffset
   ///
   pub fn rtGeometryTrianglesSetPrimitiveIndexOffset(
      geometrytriangles: RTgeometrytriangles,
      indexOffset: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the current primitive index offset
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesGetPrimitiveIndexOffset returns for \a geometrytriangles the primitive index offset. The
   /// primitive index offset can be set with @ref rtGeometryTrianglesSetPrimitiveIndexOffset.
   ///
   /// @param[in]   geometrytriangles  GeometryTriangles node to query for the primitive index offset
   /// @param[out]  indexOffset        Primitive index offset
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetPrimitiveIndexOffset was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetPrimitiveIndexOffset
   ///
   pub fn rtGeometryTrianglesGetPrimitiveIndexOffset(
      geometrytriangles: RTgeometrytriangles,
      indexOffset: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets a pre-transform matrix
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesSetPreTransformMatrix can be used to bake a transformation for a mesh.
   /// Vertices of triangles are multiplied by the user-specified 3x4 matrix before the acceleration build.
   /// Note that the input triangle data stays untouched (set via @ref rtGeometryTrianglesSetVertices).
   /// Triangle intersection uses transformed triangles.
   /// The 3x4 matrix is expected to be in a row-major data layout, use the transpose option if \a matrix is in a column-major data layout.
   /// Use rtGeometryTrianglesSetPreTransformMatrix(geometrytriangles, false, 0); to unset a previously set matrix.
   ///
   /// @param[in]   geometrytriangles  Geometry node to query from the number of primitives
   /// @param[in]   transpose          If the input matrix is column-major and needs to be transposed before usage
   /// @param[in]   matrix             The 3x4 matrix that is used to transform the vertices
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetPreTransformMatrix was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesGetPreTransformMatrix
   ///
   pub fn rtGeometryTrianglesSetPreTransformMatrix(
      geometrytriangles: RTgeometrytriangles,
      transpose: ::std::os::raw::c_int,
      matrix: *const f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets a pre-transform matrix
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesGetPreTransformMatrix returns a previously set 3x4 matrix or the 'identity' matrix (with ones in the main diagonal of the 3x3 submatrix) if no matrix is set.
   ///
   /// @param[in]   geometrytriangles  Geometry node to query from the number of primitives
   /// @param[in]   transpose          Set to true if the output matrix is expected to be column-major rather than row-major
   /// @param[out]  matrix             The 3x4 matrix that is used to transform the vertices
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetPreTransformMatrix was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetPreTransformMatrix
   ///
   pub fn rtGeometryTrianglesGetPreTransformMatrix(
      geometrytriangles: RTgeometrytriangles,
      transpose: ::std::os::raw::c_int,
      matrix: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the number of triangles
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesSetPrimitiveCount sets the number of triangles \a triangleCount in \a geometrytriangles.
   /// A triangle geometry is either a triangle soup for which every three vertices stored in the vertex buffer form a triangle,
   /// or indexed triangles are used for which three indices reference different vertices.
   /// In the latter case, an index buffer must be set (@ref rtGeometryTrianglesSetTriangleIndices).
   /// The vertices of the triangles are specified via one of the SetVertices functions.
   ///
   /// @param[in]   geometrytriangles  GeometryTriangles node for which to set the number of triangles
   /// @param[in]   triangleCount      Number of triangles
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetPrimitiveCount was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesGetPrimitiveCount
   /// @ref rtGeometrySetPrimitiveCount
   ///
   pub fn rtGeometryTrianglesSetPrimitiveCount(
      geometrytriangles: RTgeometrytriangles,
      triangleCount: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of triangles
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesGetPrimitiveCount returns the number of set triangles for \a geometrytriangles. The
   /// number of primitives can be set with @ref rtGeometryTrianglesSetPrimitiveCount.
   ///
   /// @param[in]   geometrytriangles  GeometryTriangles node to query from the number of primitives
   /// @param[out]  triangleCount      Number of triangles
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetPrimitiveCount was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetPrimitiveCount
   /// @ref rtGeometryGetPrimitiveCount
   ///
   pub fn rtGeometryTrianglesGetPrimitiveCount(
      geometrytriangles: RTgeometrytriangles,
      triangleCount: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the index buffer of indexed triangles
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesSetTriangleIndices is used to set the index buffer for indexed triangles.
   /// Triplets of indices from buffer \a indexBuffer index vertices to form triangles.
   /// If the buffer is set, it is assumed that the geometry is given as indexed triangles.
   /// If the index buffer is not set, it is assumed that the geometry is given as a triangle soup.
   /// A previously set index buffer can be unset by passing NULL as \a indexBuffer parameter, e.g., rtGeometryTrianglesSetTriangleIndices( geometrytriangles, NULL, 0, 0, RT_FORMAT_UNSIGNED_INT3);
   /// Buffer \a indexBuffer is expected to hold 3 times \a triangleCount indices (see @ref rtGeometryTrianglesSetPrimitiveCount).
   /// Parameter \a indexBufferByteOffset can be used to specify a byte offset to the first index in buffer \a indexBuffer.
   /// Parameter \a triIndicesByteStride sets the stride in bytes between triplets of indices. There mustn't be any spacing between indices within a triplet, spacing is only supported between triplets.
   /// Parameter \a triIndicesFormat must be one of the following: RT_FORMAT_UNSIGNED_INT3, RT_FORMAT_UNSIGNED_SHORT3.
   ///
   /// @param[in]   geometrytriangles               GeometryTriangles node to query for the primitive index offset
   /// @param[in]   indexBuffer                     Buffer that holds the indices into the vertex buffer of the triangles
   /// @param[in]   indexBufferByteOffset           Offset in bytes to the first index in buffer indexBuffer
   /// @param[in]   triIndicesByteStride            Stride in bytes between triplets of indices
   /// @param[in]   triIndicesFormat                Format of the triplet of indices to index the vertices of a triangle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetTriangleIndices was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetVertices
   ///
   pub fn rtGeometryTrianglesSetTriangleIndices(
      geometrytriangles: RTgeometrytriangles,
      indexBuffer: RTbuffer,
      indexBufferByteOffset: RTsize,
      triIndicesByteStride: RTsize,
      triIndicesFormat: Format,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the vertex buffer of a triangle soup
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesSetVertices interprets the buffer \a vertexBuffer as the vertices of triangles of the GeometryTriangles \a geometrytriangles.
   /// The number of vertices is set as \a vertexCount.
   /// If an index buffer is set, it is assumed that the geometry is given as indexed triangles.
   /// If the index buffer is not set, it is assumed that the geometry is given as a triangle soup and \a vertexCount must be 3 times triangleCount (see @ref rtGeometryTrianglesSetPrimitiveCount).
   /// Buffer \a vertexBuffer is expected to hold \a vertexCount vertices.
   /// Parameter \a vertexBufferByteOffset can be used to specify a byte offset to the position of the first vertex in buffer \a vertexBuffer.
   /// Parameter \a vertexByteStride sets the stride in bytes between vertices.
   /// Parameter \a positionFormat must be one of the following: RT_FORMAT_FLOAT3, RT_FORMAT_HALF3, RT_FORMAT_FLOAT2, RT_FORMAT_HALF2.
   /// In case of formats RT_FORMAT_FLOAT2 or RT_FORMAT_HALF2 the third component is assumed to be zero, which can be useful for planar geometry.
   /// Calling this function overrides any previous call to anyone of the set(Motion)Vertices functions.
   ///
   /// @param[in]   geometrytriangles            GeometryTriangles node to query for the primitive index offset
   /// @param[in]   vertexCount                  Number of vertices of the geometry
   /// @param[in]   vertexBuffer                 Buffer that holds the vertices of the triangles
   /// @param[in]   vertexByteStride             Stride in bytes between vertices
   /// @param[in]   vertexBufferByteOffset       Offset in bytes to the first vertex in buffer vertexBuffer
   /// @param[in]   positionFormat               Format of the position attribute of a vertex
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetVertices was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetTriangleIndices
   /// @ref rtGeometryTrianglesSetMotionVertices
   ///
   pub fn rtGeometryTrianglesSetVertices(
      geometrytriangles: RTgeometrytriangles,
      vertexCount: ::std::os::raw::c_uint,
      vertexBuffer: RTbuffer,
      vertexBufferByteOffset: RTsize,
      vertexByteStride: RTsize,
      positionFormat: Format,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the vertex buffer of motion triangles
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesSetMotionVertices interprets the buffer \a vertexBuffer as the vertices of triangles of the GeometryTriangles \a geometrytriangles.
   /// The number of triangles for one motion step is set as \a vertexCount.
   /// Similar to it's non-motion counterpart, \a vertexCount must be 3 times \a triangleCount if no index buffer is set.
   /// The total number of vertices stored in \a vertexBuffer is \a vertexCount times \a motionStepCount (see @ref rtGeometryTrianglesSetMotionSteps).
   /// Triangles are linearly interpolated between motion steps.
   /// Parameter \a vertexBufferByteOffset can be used to specify a byte offset to the position of the first vertex of the first motion step in buffer \a vertexBuffer.
   /// Parameter \a vertexByteStride sets the stride in bytes between vertices within a motion step.
   /// Parameter \a vertexMotionStepByteStride sets the stride in bytes between motion steps for a single vertex.
   /// The stride parameters allow for two types of layouts of the motion data:
   /// a) serialized: vertexByteStride = sizeof(Vertex), vertexMotionStepByteStride = vertexCount * vertexByteStride
   /// b) interleaved: motion_step_byte_stride = sizeof(Vertex), vertexByteStride = sizeof(Vertex) * motion_steps
   /// Vertex N at time step i is at: vertexBuffer[N * vertexByteStride + i * vertexMotionStepByteStride + vertexBufferByteOffset]
   /// Parameter \a positionFormat must be one of the following: RT_FORMAT_FLOAT3, RT_FORMAT_HALF3, RT_FORMAT_FLOAT2, RT_FORMAT_HALF2.
   /// In case of formats RT_FORMAT_FLOAT2 or RT_FORMAT_HALF2 the third component is assumed to be zero, which can be useful for planar geometry.
   /// Calling this function overrides any previous call to anyone of the set(Motion)Vertices functions.
   ///
   /// @param[in]   geometrytriangles               GeometryTriangles node to query for the primitive index offset
   /// @param[in]   vertexCount                     Number of vertices for one motion step
   /// @param[in]   vertexBuffer                    Buffer that holds the vertices of the triangles for all motion steps
   /// @param[in]   vertexBufferByteOffset          Offset in bytes to the first vertex of the first motion step in buffer vertexBuffer
   /// @param[in]   vertexByteStride                Stride in bytes between vertices, belonging to the same motion step
   /// @param[in]   vertexMotionStepByteStride      Stride in bytes between vertices of the same triangle, but neighboring motion step
   /// @param[in]   positionFormat                  Format of the position attribute of a vertex
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetMotionVertices was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetVertices
   /// @ref rtGeometryTrianglesSetMotionVerticesMultiBuffer
   ///
   pub fn rtGeometryTrianglesSetMotionVertices(
      geometrytriangles: RTgeometrytriangles,
      vertexCount: ::std::os::raw::c_uint,
      vertexBuffer: RTbuffer,
      vertexBufferByteOffset: RTsize,
      vertexByteStride: RTsize,
      vertexMotionStepByteStride: RTsize,
      positionFormat: Format,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the vertex buffer of motion triangles
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesSetMotionVerticesMultiBuffer can be used instead of @ref rtGeometryTrianglesSetMotionVertices if the vertices for the different motion steps are stored in separate buffers.
   /// Parameter \a vertexBuffers must point to an array of buffers of minimal size \a motionStepCount (see @ref rtGeometryTrianglesSetMotionSteps).
   /// All buffers must, however, share the same byte offset as well as vertex stride and position format.
   /// Calling this function overrides any previous call to any of the set(Motion)Vertices functions.
   ///
   /// @param[in]   geometrytriangles               GeometryTriangles node to query for the primitive index offset
   /// @param[in]   vertexCount                     Number of vertices for one motion step
   /// @param[in]   vertexBuffers                   Buffers that hold the vertices of the triangles per motion step
   /// @param[in]   vertexBufferCount               Number of buffers passed, must match the number of motion steps before a launch call
   /// @param[in]   vertexBufferByteOffset          Offset in bytes to the first vertex in every buffer vertexBuffers
   /// @param[in]   vertexByteStride                Stride in bytes between vertices, belonging to the same motion step
   /// @param[in]   positionFormat                  Format of the position attribute of a vertex
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetMotionVertices was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetVertices
   /// @ref rtGeometryTrianglesSetMotionVertices
   ///
   pub fn rtGeometryTrianglesSetMotionVerticesMultiBuffer(
      geometrytriangles: RTgeometrytriangles,
      vertexCount: ::std::os::raw::c_uint,
      vertexBuffers: *const RTbuffer,
      vertexBufferCount: ::std::os::raw::c_uint,
      vertexBufferByteOffset: RTsize,
      vertexByteStride: RTsize,
      positionFormat: Format,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the number of motion steps associated with a GeometryTriangles node
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// @ref rtGeometryTrianglesSetMotionSteps sets the number of motion steps as specified in \a motionStepCount
   /// associated with \a geometrytriangles.  Note that the default value is 1, not 0,
   /// for geometry without motion.
   ///
   /// @param[in]   geometrytriangles    GeometryTriangles node handle
   /// @param[in]   motionStepCount      Number of motion steps, motionStepCount >= 1
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetMotionSteps was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetMotionVertices
   /// @ref rtGeometryTrianglesSetMotionVerticesMultiBuffer
   /// @ref rtGeometryTrianglesGetMotionSteps
   /// @ref rtGeometryTrianglesSetMotionBorderMode
   /// @ref rtGeometryTrianglesSetMotionRange
   ///
   pub fn rtGeometryTrianglesSetMotionSteps(
      geometrytriangles: RTgeometrytriangles,
      motionStepCount: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of motion steps associated with a GeometryTriangles node
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// @ref rtGeometryTrianglesGetMotionSteps returns in \a motionStepCount the number of motion steps
   /// associated with \a geometrytriangles.  Note that the default value is 1, not 0,
   /// for geometry without motion.
   ///
   /// @param[in]   geometrytriangles    GeometryTriangles node handle
   /// @param[out]  motionStepCount      Number of motion steps motionStepCount >= 1
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetMotionSteps was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetMotionSteps
   /// @ref rtGeometryTrianglesGetMotionBorderMode
   /// @ref rtGeometryTrianglesGetMotionRange
   ///
   pub fn rtGeometryTrianglesGetMotionSteps(
      geometrytriangles: RTgeometrytriangles,
      motionStepCount: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the motion time range for a GeometryTriangles node.
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// Sets the inclusive motion time range [timeBegin, timeEnd] for \a geometrytriangles,
   /// where timeBegin <= timeEnd.  The default time range is [0.0, 1.0].  The
   /// time range has no effect unless @ref rtGeometryTrianglesSetMotionVertices or
   /// @ref rtGeometryTrianglesSetMotionVerticesMultiBuffer with motionStepCount > 1 is
   /// called, in which case the time steps uniformly divide the time range.
   ///
   /// @param[in]   geometrytriangles    GeometryTriangles node handle
   /// @param[out]  timeBegin            Beginning time value of range
   /// @param[out]  timeEnd              Ending time value of range
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetMotionRange was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesGetMotionRange
   /// @ref rtGeometryTrianglesSetMotionBorderMode
   /// @ref rtGeometryTrianglesGetMotionSteps
   ///
   pub fn rtGeometryTrianglesSetMotionRange(
      geometrytriangles: RTgeometrytriangles,
      timeBegin: f32,
      timeEnd: f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the motion time range associated with a GeometryTriangles node.
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// @ref rtGeometryTrianglesGetMotionRange returns the motion time range associated with
   /// \a geometrytriangles from a previous call to @ref rtGeometryTrianglesSetMotionRange, or the
   /// default values of [0.0, 1.0].
   ///
   ///
   /// @param[in]   geometrytriangles    GeometryTriangles node handle
   /// @param[out]  timeBegin            Beginning time value of range
   /// @param[out]  timeEnd              Ending time value of range
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetMotionRange was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetMotionRange
   /// @ref rtGeometryTrianglesGetMotionBorderMode
   /// @ref rtGeometryTrianglesGetMotionSteps
   ///
   pub fn rtGeometryTrianglesGetMotionRange(
      geometrytriangles: RTgeometrytriangles,
      timeBegin: *mut f32,
      timeEnd: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the motion border modes of a GeometryTriangles node
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// @ref rtGeometryTrianglesSetMotionBorderMode sets the behavior of \a geometrytriangles
   /// outside its motion time range. Options are @ref RT_MOTIONBORDERMODE_CLAMP
   /// or @ref RT_MOTIONBORDERMODE_VANISH.  See @ref rtTransformSetMotionBorderMode
   /// for details.
   ///
   /// @param[in]   geometrytriangles    GeometryTriangles node handle
   /// @param[in]   beginMode            Motion border mode at motion range begin
   /// @param[in]   endMode              Motion border mode at motion range end
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetMotionBorderMode was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesGetMotionBorderMode
   /// @ref rtGeometryTrianglesSetMotionRange
   /// @ref rtGeometryTrianglesGetMotionSteps
   ///
   pub fn rtGeometryTrianglesSetMotionBorderMode(
      geometrytriangles: RTgeometrytriangles,
      beginMode: MotionBorderMode,
      endMode: MotionBorderMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the motion border modes of a GeometryTriangles node
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// @ref rtGeometryTrianglesGetMotionBorderMode returns the motion border modes
   /// for the time range associated with \a geometrytriangles.
   ///
   /// @param[in]   geometrytriangles   GeometryTriangles node handle
   /// @param[out]  beginMode           Motion border mode at motion range begin
   /// @param[out]  endMode             Motion border mode at motion range end
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetMotionBorderMode was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetMotionBorderMode
   /// @ref rtGeometryTrianglesGetMotionRange
   /// @ref rtGeometryTrianglesGetMotionSteps
   ///
   pub fn rtGeometryTrianglesGetMotionBorderMode(
      geometrytriangles: RTgeometrytriangles,
      beginMode: *mut MotionBorderMode,
      endMode: *mut MotionBorderMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets flags that influence the behavior of traversal
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// @ref rtGeometryTrianglesSetBuildFlags can be used to set object-specific flags that affect the acceleration-structure-build behavior.
   /// If parameter \a buildFlags contains the RT_GEOMETRY_BUILD_FLAG_RELEASE_BUFFERS flag, all buffers (including the vertex, index, and materialIndex buffer) holding
   /// information that is evaluated at acceleration-structure-build time will be released after the build.
   /// OptiX does not take ownership over the buffers, but simply frees the corresponding device memory.
   /// Sharing buffers with other GeometryTriangles nodes is possible if all of them are built within one OptiX launch.
   /// Note that it is the users responsibility that the buffers hold data for the next acceleration structure build if the acceleration structure is marked dirty.
   /// E.g., if the flag is set, an OptiX launch will cause the acceleration structure build and release the memory afterwards.
   /// If the acceleration structure is marked dirty before the next launch (e.g., due to refitting), the user needs to map the buffers before the launch to fill them with data.
   /// Further, there are certain configurations with motion when the buffers cannot be released in which case the flag is ignored and the data is not freed.
   /// The buffers can only be released if all GeometryTriangles belonging to a GeometryGroup have the same number of motion steps and equal motion begin / end times.
   ///
   /// @param[in]   geometrytriangles    GeometryTriangles node handle
   /// @param[in]   buildFlags           The flags to set
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetBuildFlags was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetBuildFlags
   ///
   pub fn rtGeometryTrianglesSetBuildFlags(
      geometrytriangles: RTgeometrytriangles,
      buildFlags: GeometryBuildFlags,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the number of materials used for the GeometryTriangles
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// @ref rtGeometryTrianglesGetMaterialCount returns the number of materials that are used with \a geometrytriangles.
   /// As default there is one material slot.
   ///
   ///
   /// @param[in]   geometrytriangles    GeometryTriangles node handle
   /// @param[out]  numMaterials         Number of materials used with this GeometryTriangles node
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetMaterialCount was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetMaterialCount
   ///
   pub fn rtGeometryTrianglesGetMaterialCount(
      geometrytriangles: RTgeometrytriangles,
      numMaterials: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the number of materials used for the GeometryTriangles
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// @ref rtGeometryTrianglesSetMaterialCount sets the number of materials that are used with \a geometrytriangles.
   /// As default, there is one material slot.
   /// This number must be equal to the number of materials that is set at the GeometryInstance where \a geometrytriangles is attached to.
   /// Multi-material support for GeometryTriangles is limited to a fixed partition of the geometry into sets of triangles.
   /// Each triangle set maps to one material slot (within range [0;numMaterials]).
   /// The mapping is set via @ref rtGeometryTrianglesSetMaterialIndices.
   /// The actual materials are set at the GeometryInstance.
   /// The geometry can be instanced when attached to multiple GeometryInstances.
   /// In that case, the materials attached to each GeometryInstance can differ (effectively causing different materials per instance of the geometry).
   /// \a numMaterials must be >=1 and <= 2^16.
   ///
   /// @param[in]   geometrytriangles    GeometryTriangles node handle
   /// @param[in]   numMaterials         Number of materials used with this geometry
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetMaterialCount was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesGetMaterialCount
   /// @ref rtGeometryTrianglesSetMaterialIndices
   /// @ref rtGeometryTrianglesSetFlagsPerMaterial
   ///
   pub fn rtGeometryTrianglesSetMaterialCount(
      geometrytriangles: RTgeometrytriangles,
      numMaterials: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the index buffer of indexed triangles
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesSetMaterialIndices set the material slot per triangle of \a geometrytriangles.
   /// Hence, buffer \a materialIndexBuffer must hold triangleCount entries.
   /// Every material index must be in range [0; numMaterials-1] (see @ref rtGeometryTrianglesSetMaterialCount).
   /// Parameter \a materialIndexBufferByteOffset can be used to specify a byte offset to the first index in buffer \a materialIndexBuffer.
   /// Parameter \a materialIndexByteStride sets the stride in bytes between indices.
   /// Parameter \a materialIndexFormat must be one of the following: RT_FORMAT_UNSIGNED_INT, RT_FORMAT_UNSIGNED_SHORT, RT_FORMAT_UNSIGNED_BYTE.
   /// The buffer is only used if the number of materials as set via @ref rtGeometryTrianglesSetMaterialCount is larger than one.
   ///
   /// @param[in]   geometrytriangles                   GeometryTriangles node to query for the primitive index offset
   /// @param[in]   materialIndexBuffer                 Buffer that holds the indices into the vertex buffer of the triangles
   /// @param[in]   materialIndexBufferByteOffset       Offset to first index in buffer indexBuffer
   /// @param[in]   materialIndexByteStride             Stride in bytes between triplets of indices
   /// @param[in]   materialIndexFormat                 Format of the triplet of indices to index the vertices of a triangle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetMaterialIndices was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetMaterialCount
   /// @ref rtGeometryTrianglesSetFlagsPerMaterial
   ///
   pub fn rtGeometryTrianglesSetMaterialIndices(
      geometrytriangles: RTgeometrytriangles,
      materialIndexBuffer: RTbuffer,
      materialIndexBufferByteOffset: RTsize,
      materialIndexByteStride: RTsize,
      materialIndexFormat: Format,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets geometry-specific flags that influence the behavior of traversal
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   /// @ref rtGeometryTrianglesSetFlagsPerMaterial can be used to set geometry-specific flags that will eventually
   /// change the behavior of traversal when intersecting the geometry.
   /// Note that the flags are evaluated at acceleration-structure-build time.
   /// An acceleration must be marked dirty for changes to the flags to take effect.
   /// Setting the flags RT_GEOMETRY_FLAG_NO_SPLITTING and/or RT_GEOMETRY_FLAG_DISABLE_ANYHIT should be dependent on the
   /// material that is used for the intersection.
   /// Therefore, the flags are set per material slot (with the actual material binding begin set at the GeomteryInstance).
   /// If the geometry is instanced and different instances apply different materials to the geometry, the per-material geometry-specific flags need to apply to the materials of all instances.
   /// Example with two instances with each having two materials, node graph:
   ///        G
   ///       / \
   ///      /   \
   ///     T0    T1
   ///     |     |
   ///    GG0-A-GG1
   ///     |     |
   /// M0-GI0   GI1-M2
   ///    /  \ /  \
   ///  M1    GT   M3
   /// with: G-Group, GG-GeometryGroup, T-Transform, A-Acceleration, GI-GeometryInstance, M-Material, GT-GeometryTriangles
   /// RT_GEOMETRY_FLAG_NO_SPLITTING needs to be set for material index 0, if M0 or M2 require it.
   /// RT_GEOMETRY_FLAG_DISABLE_ANYHIT should be set for material index 0, if M0 and M2 allow it.
   /// RT_GEOMETRY_FLAG_NO_SPLITTING needs to be set for material index 1, if M1 or M3 require it.
   /// RT_GEOMETRY_FLAG_DISABLE_ANYHIT should be set for material index 1, if M1 and M3 allow it.
   ///
   /// Setting RT_GEOMETRY_FLAG_NO_SPLITTING prevents splitting the primitive during the bvh build.
   /// Splitting is done to increase performance, but as a side-effect may result in multiple executions of the any hit program for a single intersection.
   /// To avoid further side effects (e.g., multiple accumulations of a value) that may result of a multiple execution, RT_GEOMETRY_FLAG_NO_SPLITTING needs to be set.
   /// RT_GEOMETRY_FLAG_DISABLE_ANYHIT is an optimization due to which the execution of the any hit program is skipped.
   /// If possible, the flag should be set.
   /// Note that even if no any hit program is set on a material, this flag needs to be set to skip the any hit program.
   /// This requirement is because the information whether or not to skip the any hit program needs to be available at bvh build time (while materials can change afterwards without a bvh rebuild).
   /// Note that the final decision whether or not to execute the any hit program at run time also depends on the flags set on the ray as well as the geometry group that this geometry is part of.
   ///
   /// @param[in]   geometrytriangles    GeometryTriangles node handle
   /// @param[in]   materialIndex        The material index for which to set the flags
   /// @param[in]   flags                The flags to set.
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetFlagsPerMaterial was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetMaterialCount
   /// @ref rtGeometryTrianglesSetMaterialIndices
   /// @ref rtGeometryTrianglesSetBuildFlags
   ///
   pub fn rtGeometryTrianglesSetFlagsPerMaterial(
      geometrytriangles: RTgeometrytriangles,
      materialIndex: ::std::os::raw::c_uint,
      flags: GeometryFlags,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets geometry flags for triangles.
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// See @ref rtGeometryTrianglesSetFlagsPerMaterial for details.
   ///
   /// @param[in] triangles       The triangles handle
   /// @param[in] materialIndex   The index of the material for which to retrieve the flags
   /// @param[out] flags          Flags for the given geometry group
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetFlagsPerMaterial was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesSetFlagsPerMaterial,
   /// @ref rtGeometryTrianglesSetMaterialIndices
   /// @ref rtTrace
   pub fn rtGeometryTrianglesGetFlagsPerMaterial(
      triangles: RTgeometrytriangles,
      materialIndex: ::std::os::raw::c_uint,
      flags: *mut GeometryFlags,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new material
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialCreate creates a new material within a context. \a context specifies the target
   /// context, as returned by @ref rtContextCreate. Sets \a *material to the handle of a newly
   /// created material within \a context. Returns @ref RT_ERROR_INVALID_VALUE if \a material is \a NULL.
   ///
   /// @param[in]   context    Specifies a context within which to create a new material
   /// @param[out]  material   Returns a newly created material
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialDestroy,
   /// @ref rtContextCreate
   ///
   pub fn rtMaterialCreate(
      context: RTcontext,
      material: *mut RTmaterial,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a material object
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialDestroy removes \a material from its context and deletes it.  \a material should
   /// be a value returned by @ref rtMaterialCreate.  Associated variables declared via @ref
   /// rtMaterialDeclareVariable are destroyed, but no child graph nodes are destroyed.  After the
   /// call, \a material is no longer a valid handle.
   ///
   /// @param[in]   material   Handle of the material node to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialCreate
   ///
   pub fn rtMaterialDestroy(material: RTmaterial) -> RtResult;
}
extern "C" {
   /// @brief Verifies the state of a material
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialValidate checks \a material for completeness. If \a material or
   /// any of the objects attached to \a material are not valid, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   material   Specifies the material to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialCreate
   ///
   pub fn rtMaterialValidate(material: RTmaterial) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with a material
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialGetContext queries a material for its associated context.
   /// \a material specifies the material to query, and should be a value returned by
   /// @ref rtMaterialCreate. If both parameters are valid, \a *context
   /// sets to the context associated with \a material. Otherwise, the call
   /// has no effect and returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   material   Specifies the material to query
   /// @param[out]  context    Returns the context associated with the material
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialCreate
   ///
   pub fn rtMaterialGetContext(
      material: RTmaterial,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the closest hit program associated with a (material, ray type) tuple
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialSetClosestHitProgram specifies a closest hit program to associate
   /// with a (material, ray type) tuple. \a material specifies the material of
   /// interest and should be a value returned by @ref rtMaterialCreate.
   /// \a rayTypeIndex specifies the type of ray to which the program applies and
   /// should be a value less than the value returned by @ref rtContextGetRayTypeCount.
   /// \a program specifies the target closest hit program which applies to
   /// the tuple (\a material, \a rayTypeIndex) and should be a value returned by
   /// either @ref rtProgramCreateFromPTXString or @ref rtProgramCreateFromPTXFile.
   ///
   /// @param[in]   material         Specifies the material of the (material, ray type) tuple to modify
   /// @param[in]   rayTypeIndex     Specifies the ray type of the (material, ray type) tuple to modify
   /// @param[in]   program          Specifies the closest hit program to associate with the (material, ray type) tuple
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialSetClosestHitProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialGetClosestHitProgram,
   /// @ref rtMaterialCreate,
   /// @ref rtContextGetRayTypeCount,
   /// @ref rtProgramCreateFromPTXString,
   /// @ref rtProgramCreateFromPTXFile
   ///
   pub fn rtMaterialSetClosestHitProgram(
      material: RTmaterial,
      rayTypeIndex: ::std::os::raw::c_uint,
      program: RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the closest hit program associated with a (material, ray type) tuple
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialGetClosestHitProgram queries the closest hit program associated
   /// with a (material, ray type) tuple. \a material specifies the material of
   /// interest and should be a value returned by @ref rtMaterialCreate.
   /// \a rayTypeIndex specifies the target ray type and should be a value
   /// less than the value returned by @ref rtContextGetRayTypeCount.
   /// If all parameters are valid, \a *program sets to the handle of the
   /// any hit program associated with the tuple (\a material, \a rayTypeIndex).
   /// Otherwise, the call has no effect and returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   material         Specifies the material of the (material, ray type) tuple to query
   /// @param[in]   rayTypeIndex     Specifies the type of ray of the (material, ray type) tuple to query
   /// @param[out]  program          Returns the closest hit program associated with the (material, ray type) tuple
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialGetClosestHitProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialSetClosestHitProgram,
   /// @ref rtMaterialCreate,
   /// @ref rtContextGetRayTypeCount
   ///
   pub fn rtMaterialGetClosestHitProgram(
      material: RTmaterial,
      rayTypeIndex: ::std::os::raw::c_uint,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the any hit program associated with a (material, ray type) tuple
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialSetAnyHitProgram specifies an any hit program to associate with a
   /// (material, ray type) tuple. \a material specifies the target material and
   /// should be a value returned by @ref rtMaterialCreate. \a rayTypeIndex specifies
   /// the type of ray to which the program applies and should be a value less than
   /// the value returned by @ref rtContextGetRayTypeCount. \a program specifies the
   /// target any hit program which applies to the tuple (\a material,
   /// \a rayTypeIndex) and should be a value returned by either
   /// @ref rtProgramCreateFromPTXString or @ref rtProgramCreateFromPTXFile.
   ///
   /// @param[in]   material         Specifies the material of the (material, ray type) tuple to modify
   /// @param[in]   rayTypeIndex     Specifies the type of ray of the (material, ray type) tuple to modify
   /// @param[in]   program          Specifies the any hit program to associate with the (material, ray type) tuple
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialSetAnyHitProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialGetAnyHitProgram,
   /// @ref rtMaterialCreate,
   /// @ref rtContextGetRayTypeCount,
   /// @ref rtProgramCreateFromPTXString,
   /// @ref rtProgramCreateFromPTXFile
   ///
   pub fn rtMaterialSetAnyHitProgram(
      material: RTmaterial,
      rayTypeIndex: ::std::os::raw::c_uint,
      program: RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the any hit program associated with a (material, ray type) tuple
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialGetAnyHitProgram queries the any hit program associated
   /// with a (material, ray type) tuple. \a material specifies the material of
   /// interest and should be a value returned by @ref rtMaterialCreate.
   /// \a rayTypeIndex specifies the target ray type and should be a value
   /// less than the value returned by @ref rtContextGetRayTypeCount.
   /// if all parameters are valid, \a *program sets to the handle of the
   /// any hit program associated with the tuple (\a material, \a rayTypeIndex).
   /// Otherwise, the call has no effect and returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   material         Specifies the material of the (material, ray type) tuple to query
   /// @param[in]   rayTypeIndex     Specifies the type of ray of the (material, ray type) tuple to query
   /// @param[out]  program          Returns the any hit program associated with the (material, ray type) tuple
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialGetAnyHitProgram was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialSetAnyHitProgram,
   /// @ref rtMaterialCreate,
   /// @ref rtContextGetRayTypeCount
   ///
   pub fn rtMaterialGetAnyHitProgram(
      material: RTmaterial,
      rayTypeIndex: ::std::os::raw::c_uint,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Declares a new named variable to be associated with a material
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialDeclareVariable declares a new variable to be associated with a material.
   /// \a material specifies the target material, and should be a value returned by @ref
   /// rtMaterialCreate. \a name specifies the name of the variable, and should be a \a NULL-terminated
   /// string. If there is currently no variable associated with \a material named \a name, and \a v is
   /// not \a NULL, a new variable named \a name will be created and associated with \a material and \a
   /// *v will be set to the handle of the newly-created variable. Otherwise, this call has no effect
   /// and returns either @ref RT_ERROR_INVALID_VALUE if either \a name or \a v is \a NULL or @ref
   /// RT_ERROR_VARIABLE_REDECLARED if \a name is the name of an existing variable associated with the
   /// material.
   ///
   /// @param[in]   material   Specifies the material to modify
   /// @param[in]   name       Specifies the name of the variable
   /// @param[out]  v          Returns a handle to a newly declared variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_REDECLARED
   /// - @ref RT_ERROR_ILLEGAL_SYMBOL
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialDeclareVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialGetVariable,
   /// @ref rtMaterialQueryVariable,
   /// @ref rtMaterialCreate
   ///
   pub fn rtMaterialDeclareVariable(
      material: RTmaterial,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries for the existence of a named variable of a material
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialQueryVariable queries for the existence of a material's named variable. \a
   /// material specifies the target material and should be a value returned by @ref rtMaterialCreate.
   /// \a name specifies the name of the variable, and should be a \a NULL-terminated
   /// string. If \a material is a valid material and \a name is the name of a variable attached to \a
   /// material, \a *v is set to a handle to that variable after the call. Otherwise, \a *v is set to
   /// \a NULL. If \a material is not a valid material, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   material   Specifies the material to query
   /// @param[in]   name       Specifies the name of the variable to query
   /// @param[out]  v          Returns a the named variable, if it exists
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialQueryVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialGetVariable,
   /// @ref rtMaterialCreate
   ///
   pub fn rtMaterialQueryVariable(
      material: RTmaterial,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Removes a variable from a material
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialRemoveVariable removes a variable from a material. The material of
   /// interest is specified by \a material, which should be a value returned by
   /// @ref rtMaterialCreate. The variable to remove is specified by \a v, which
   /// should be a value returned by @ref rtMaterialDeclareVariable. Once a variable
   /// has been removed from this material, another variable with the same name as the
   /// removed variable may be declared. If \a material does not refer to a valid material,
   /// this call has no effect and returns @ref RT_ERROR_INVALID_VALUE. If \a v is not
   /// a valid variable or does not belong to \a material, this call has no effect and
   /// returns @ref RT_ERROR_INVALID_VALUE or @ref RT_ERROR_VARIABLE_NOT_FOUND, respectively.
   ///
   /// @param[in]   material   Specifies the material to modify
   /// @param[in]   v          Specifies the variable to remove
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialRemoveVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialDeclareVariable,
   /// @ref rtMaterialCreate
   ///
   pub fn rtMaterialRemoveVariable(
      material: RTmaterial,
      v: RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of variables attached to a material
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialGetVariableCount queries the number of variables attached to a
   /// material. \a material specifies the material, and should be a value returned by
   /// @ref rtMaterialCreate. After the call, if both parameters are valid, the number
   /// of variables attached to \a material is returned to \a *count. Otherwise, the
   /// call has no effect and returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   material   Specifies the material to query
   /// @param[out]  count      Returns the number of variables
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialGetVariableCount was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialCreate
   ///
   pub fn rtMaterialGetVariableCount(
      material: RTmaterial,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a handle to an indexed variable of a material
   ///
   /// @ingroup Material
   ///
   /// <B>Description</B>
   ///
   /// @ref rtMaterialGetVariable queries the handle of a material's indexed variable.  \a material
   /// specifies the target material and should be a value returned by @ref rtMaterialCreate. \a index
   /// specifies the index of the variable, and should be a value less than
   /// @ref rtMaterialGetVariableCount. If \a material is a valid material and \a index is the index of a
   /// variable attached to \a material, \a *v is set to a handle to that variable. Otherwise, \a *v is
   /// set to \a NULL and either @ref RT_ERROR_INVALID_VALUE or @ref RT_ERROR_VARIABLE_NOT_FOUND is
   /// returned depending on the validity of \a material, or \a index, respectively.
   ///
   /// @param[in]   material   Specifies the material to query
   /// @param[in]   index      Specifies the index of the variable to query
   /// @param[out]  v          Returns the indexed variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_VARIABLE_NOT_FOUND
   ///
   /// <B>History</B>
   ///
   /// @ref rtMaterialGetVariable was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtMaterialQueryVariable,
   /// @ref rtMaterialGetVariableCount,
   /// @ref rtMaterialCreate
   ///
   pub fn rtMaterialGetVariable(
      material: RTmaterial,
      index: ::std::os::raw::c_uint,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new texture sampler object
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerCreate allocates a texture sampler object.
   /// Sets \a *texturesampler to the handle of a newly created texture sampler within \a context.
   /// Returns @ref RT_ERROR_INVALID_VALUE if \a texturesampler is \a NULL.
   ///
   /// @param[in]   context          The context the texture sampler object will be created in
   /// @param[out]  texturesampler   The return handle to the new texture sampler object
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerCreate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerDestroy
   ///
   pub fn rtTextureSamplerCreate(
      context: RTcontext,
      texturesampler: *mut RTtexturesampler,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a texture sampler object
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerDestroy removes \a texturesampler from its context and deletes it.
   /// \a texturesampler should be a value returned by @ref rtTextureSamplerCreate.
   /// After the call, \a texturesampler is no longer a valid handle.
   /// Any API object that referenced \a texturesampler will have its reference invalidated.
   ///
   /// @param[in]   texturesampler   Handle of the texture sampler to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerCreate
   ///
   pub fn rtTextureSamplerDestroy(texturesampler: RTtexturesampler)
      -> RtResult;
}
extern "C" {
   /// @brief Validates the state of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerValidate checks \a texturesampler for completeness.  If \a texturesampler does not have buffers
   /// attached to all of its MIP levels and array slices or if the filtering modes are incompatible with the current
   /// MIP level and array slice configuration then returns @ref RT_ERROR_INVALID_CONTEXT.
   ///
   /// @param[in]   texturesampler   The texture sampler to be validated
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextValidate
   ///
   pub fn rtTextureSamplerValidate(
      texturesampler: RTtexturesampler,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the context object that created this texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetContext returns a handle to the context object that was used to create
   /// \a texturesampler.  If \a context is \a NULL, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be queried for its context
   /// @param[out]  context          The return handle for the context object of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextCreate
   ///
   pub fn rtTextureSamplerGetContext(
      texturesampler: RTtexturesampler,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 3.9. Use @ref rtBufferSetMipLevelCount instead.
   ///
   pub fn rtTextureSamplerSetMipLevelCount(
      texturesampler: RTtexturesampler,
      mipLevelCount: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 3.9. Use @ref rtBufferGetMipLevelCount instead.
   ///
   pub fn rtTextureSamplerGetMipLevelCount(
      texturesampler: RTtexturesampler,
      mipLevelCount: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 3.9. Use texture samplers with layered buffers instead. See @ref rtBufferCreate.
   ///
   pub fn rtTextureSamplerSetArraySize(
      texturesampler: RTtexturesampler,
      textureCount: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// Deprecated in OptiX 3.9. Use texture samplers with layered buffers instead. See @ref rtBufferCreate.
   ///
   pub fn rtTextureSamplerGetArraySize(
      texturesampler: RTtexturesampler,
      textureCount: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the wrapping mode of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerSetWrapMode sets the wrapping mode of
   /// \a texturesampler to \a wrapmode for the texture dimension specified
   /// by \a dimension.  \a wrapmode can take one of the following values:
   ///
   ///  - @ref RT_WRAP_REPEAT
   ///  - @ref RT_WRAP_CLAMP_TO_EDGE
   ///  - @ref RT_WRAP_MIRROR
   ///  - @ref RT_WRAP_CLAMP_TO_BORDER
   ///
   /// The wrapping mode controls the behavior of the texture sampler as
   /// texture coordinates wrap around the range specified by the indexing
   /// mode.  These values mirror the CUDA behavior of textures.
   /// See CUDA programming guide for details.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be changed
   /// @param[in]   dimension        Dimension of the texture
   /// @param[in]   wrapmode         The new wrap mode of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerSetWrapMode was introduced in OptiX 1.0.
   /// @ref RT_WRAP_MIRROR and @ref RT_WRAP_CLAMP_TO_BORDER were introduced in OptiX 3.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerGetWrapMode
   ///
   pub fn rtTextureSamplerSetWrapMode(
      texturesampler: RTtexturesampler,
      dimension: ::std::os::raw::c_uint,
      wrapmode: WrapMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the wrap mode of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetWrapMode gets the texture wrapping mode of \a texturesampler and stores it in \a *wrapmode.
   /// See @ref rtTextureSamplerSetWrapMode for a list of values @ref WrapMode can take.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be queried
   /// @param[in]   dimension        Dimension for the wrapping
   /// @param[out]  wrapmode         The return handle for the wrap mode of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetWrapMode was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerSetWrapMode
   ///
   pub fn rtTextureSamplerGetWrapMode(
      texturesampler: RTtexturesampler,
      dimension: ::std::os::raw::c_uint,
      wrapmode: *mut WrapMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the filtering modes of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerSetFilteringModes sets the minification, magnification and MIP mapping filter modes for \a texturesampler.
   /// FilterMode must be one of the following values:
   ///
   ///  - @ref RT_FILTER_NEAREST
   ///  - @ref RT_FILTER_LINEAR
   ///  - @ref RT_FILTER_NONE
   ///
   /// These filter modes specify how the texture sampler will interpolate
   /// buffer data that has been attached to it.  \a minification and
   /// \a magnification must be one of @ref RT_FILTER_NEAREST or
   /// @ref RT_FILTER_LINEAR.  \a mipmapping may be any of the three values but
   /// must be @ref RT_FILTER_NONE if the texture sampler contains only a
   /// single MIP level or one of @ref RT_FILTER_NEAREST or @ref RT_FILTER_LINEAR
   /// if the texture sampler contains more than one MIP level.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be changed
   /// @param[in]   minification     The new minification filter mode of the texture sampler
   /// @param[in]   magnification    The new magnification filter mode of the texture sampler
   /// @param[in]   mipmapping       The new MIP mapping filter mode of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerSetFilteringModes was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerGetFilteringModes
   ///
   pub fn rtTextureSamplerSetFilteringModes(
      texturesampler: RTtexturesampler,
      minification: FilterMode,
      magnification: FilterMode,
      mipmapping: FilterMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the filtering modes of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetFilteringModes gets the minification, magnification and MIP mapping filtering modes from
   /// \a texturesampler and stores them in \a *minification, \a *magnification and \a *mipmapping, respectively.  See
   /// @ref rtTextureSamplerSetFilteringModes for the values @ref FilterMode may take.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be queried
   /// @param[out]  minification     The return handle for the minification filtering mode of the texture sampler
   /// @param[out]  magnification    The return handle for the magnification filtering mode of the texture sampler
   /// @param[out]  mipmapping       The return handle for the MIP mapping filtering mode of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetFilteringModes was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerSetFilteringModes
   ///
   pub fn rtTextureSamplerGetFilteringModes(
      texturesampler: RTtexturesampler,
      minification: *mut FilterMode,
      magnification: *mut FilterMode,
      mipmapping: *mut FilterMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the maximum anisotropy of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerSetMaxAnisotropy sets the maximum anisotropy of \a texturesampler to \a value.  A float
   /// value specifies the maximum anisotropy ratio to be used when doing anisotropic filtering. This value will be clamped to the range [1,16]
   ///
   /// @param[in]   texturesampler   The texture sampler object to be changed
   /// @param[in]   value            The new maximum anisotropy level of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerSetMaxAnisotropy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerGetMaxAnisotropy
   ///
   pub fn rtTextureSamplerSetMaxAnisotropy(
      texturesampler: RTtexturesampler,
      value: f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the maximum anisotropy level for a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetMaxAnisotropy gets the maximum anisotropy level for \a texturesampler and stores
   /// it in \a *value.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be queried
   /// @param[out]  value            The return handle for the maximum anisotropy level of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetMaxAnisotropy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerSetMaxAnisotropy
   ///
   pub fn rtTextureSamplerGetMaxAnisotropy(
      texturesampler: RTtexturesampler,
      value: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the minimum and the maximum MIP level access range of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerSetMipLevelClamp sets lower end and the upper end of the MIP level range to clamp access to.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be changed
   /// @param[in]   minLevel         The new minimum mipmap level of the texture sampler
   /// @param[in]   maxLevel         The new maximum mipmap level of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerSetMipLevelClamp was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerGetMipLevelClamp
   ///
   pub fn rtTextureSamplerSetMipLevelClamp(
      texturesampler: RTtexturesampler,
      minLevel: f32,
      maxLevel: f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the minimum and the maximum MIP level access range for a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetMipLevelClamp gets the minimum and the maximum MIP level access range for \a texturesampler and stores
   /// it in \a *minLevel and \a maxLevel.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be queried
   /// @param[out]  minLevel         The return handle for the minimum mipmap level of the texture sampler
   /// @param[out]  maxLevel         The return handle for the maximum mipmap level of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetMipLevelClamp was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerSetMipLevelClamp
   ///
   pub fn rtTextureSamplerGetMipLevelClamp(
      texturesampler: RTtexturesampler,
      minLevel: *mut f32,
      maxLevel: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the mipmap offset of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerSetMipLevelBias sets the offset to be applied to the calculated mipmap level.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be changed
   /// @param[in]   value            The new mipmap offset of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerSetMipLevelBias was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerGetMipLevelBias
   ///
   pub fn rtTextureSamplerSetMipLevelBias(
      texturesampler: RTtexturesampler,
      value: f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the mipmap offset for a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetMipLevelBias gets the mipmap offset for \a texturesampler and stores
   /// it in \a *value.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be queried
   /// @param[out]  value            The return handle for the mipmap offset of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetMipLevelBias was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerSetMipLevelBias
   ///
   pub fn rtTextureSamplerGetMipLevelBias(
      texturesampler: RTtexturesampler,
      value: *mut f32,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the read mode of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerSetReadMode sets the data read mode of \a texturesampler to \a readmode.
   /// \a readmode can take one of the following values:
   ///
   ///  - @ref RT_TEXTURE_READ_ELEMENT_TYPE
   ///  - @ref RT_TEXTURE_READ_NORMALIZED_FLOAT
   ///  - @ref RT_TEXTURE_READ_ELEMENT_TYPE_SRGB
   ///  - @ref RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB
   ///
   /// @ref RT_TEXTURE_READ_ELEMENT_TYPE_SRGB and @ref RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB were introduced in OptiX 3.9
   /// and apply sRGB to linear conversion during texture read for 8-bit integer buffer formats.
   /// \a readmode controls the returned value of the texture sampler when it is used to sample
   /// textures.  @ref RT_TEXTURE_READ_ELEMENT_TYPE will return data of the type of the underlying
   /// buffer objects.  @ref RT_TEXTURE_READ_NORMALIZED_FLOAT will return floating point values
   /// normalized by the range of the underlying type.  If the underlying type is floating point,
   /// @ref RT_TEXTURE_READ_NORMALIZED_FLOAT and @ref RT_TEXTURE_READ_ELEMENT_TYPE are equivalent,
   /// always returning the unmodified floating point value.
   ///
   /// For example, a texture sampler that samples a buffer of type @ref RT_FORMAT_UNSIGNED_BYTE with
   /// a read mode of @ref RT_TEXTURE_READ_NORMALIZED_FLOAT will convert integral values from the
   /// range [0,255] to floating point values in the range [0,1] automatically as the buffer is
   /// sampled from.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be changed
   /// @param[in]   readmode         The new read mode of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerSetReadMode was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerGetReadMode
   ///
   pub fn rtTextureSamplerSetReadMode(
      texturesampler: RTtexturesampler,
      readmode: TextureReadMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the read mode of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetReadMode gets the read mode of \a texturesampler and stores it in \a *readmode.
   /// See @ref rtTextureSamplerSetReadMode for a list of values @ref TextureReadMode can take.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be queried
   /// @param[out]  readmode         The return handle for the read mode of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetReadMode was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerSetReadMode
   ///
   pub fn rtTextureSamplerGetReadMode(
      texturesampler: RTtexturesampler,
      readmode: *mut TextureReadMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets whether texture coordinates for this texture sampler are normalized
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerSetIndexingMode sets the indexing mode of \a texturesampler to \a indexmode.  \a indexmode
   /// can take on one of the following values:
   ///
   ///  - @ref RT_TEXTURE_INDEX_NORMALIZED_COORDINATES,
   ///  - @ref RT_TEXTURE_INDEX_ARRAY_INDEX
   ///
   /// These values are used to control the interpretation of texture coordinates.  If the index mode is set to
   /// @ref RT_TEXTURE_INDEX_NORMALIZED_COORDINATES, the texture is parameterized over [0,1].  If the index
   /// mode is set to @ref RT_TEXTURE_INDEX_ARRAY_INDEX then texture coordinates are interpreted as array indices
   /// into the contents of the underlying buffer objects.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be changed
   /// @param[in]   indexmode        The new indexing mode of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerSetIndexingMode was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerGetIndexingMode
   ///
   pub fn rtTextureSamplerSetIndexingMode(
      texturesampler: RTtexturesampler,
      indexmode: TextureIndexMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the indexing mode of a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetIndexingMode gets the indexing mode of \a texturesampler and stores it in \a *indexmode.
   /// See @ref rtTextureSamplerSetIndexingMode for the values @ref TextureIndexMode may take.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be queried
   /// @param[out]  indexmode        The return handle for the indexing mode of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetIndexingMode was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerSetIndexingMode
   ///
   pub fn rtTextureSamplerGetIndexingMode(
      texturesampler: RTtexturesampler,
      indexmode: *mut TextureIndexMode,
   ) -> RtResult;
}
extern "C" {
   /// @brief Attaches a buffer object to a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerSetBuffer attaches \a buffer to \a texturesampler.
   ///
   /// @param[in]   texturesampler      The texture sampler object that will contain the buffer
   /// @param[in]   deprecated0         Deprecated in OptiX 3.9, must be 0
   /// @param[in]   deprecated1         Deprecated in OptiX 3.9, must be 0
   /// @param[in]   buffer              The buffer to be attached to the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerSetBuffer was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerGetBuffer
   ///
   pub fn rtTextureSamplerSetBuffer(
      texturesampler: RTtexturesampler,
      deprecated0: ::std::os::raw::c_uint,
      deprecated1: ::std::os::raw::c_uint,
      buffer: RTbuffer,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets a buffer object handle from a texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetBuffer gets a buffer object from
   /// \a texturesampler and
   /// stores it in \a *buffer.
   ///
   /// @param[in]   texturesampler      The texture sampler object to be queried for the buffer
   /// @param[in]   deprecated0         Deprecated in OptiX 3.9, must be 0
   /// @param[in]   deprecated1         Deprecated in OptiX 3.9, must be 0
   /// @param[out]  buffer              The return handle to the buffer attached to the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetBuffer was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerSetBuffer
   ///
   pub fn rtTextureSamplerGetBuffer(
      texturesampler: RTtexturesampler,
      deprecated0: ::std::os::raw::c_uint,
      deprecated1: ::std::os::raw::c_uint,
      buffer: *mut RTbuffer,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the texture ID of this texture sampler
   ///
   /// @ingroup TextureSampler
   ///
   /// <B>Description</B>
   ///
   /// @ref rtTextureSamplerGetId returns a handle to the texture sampler
   /// \a texturesampler to be used in OptiX programs on the device to
   /// reference the associated texture. The returned ID cannot be used on
   /// the host side. If \a textureId is \a NULL, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   texturesampler   The texture sampler object to be queried for its ID
   /// @param[out]  textureId        The returned device-side texture ID of the texture sampler
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtTextureSamplerGetId was introduced in OptiX 3.0.
   ///
   /// <B>See also</B>
   /// @ref rtTextureSamplerCreate
   ///
   pub fn rtTextureSamplerGetId(
      texturesampler: RTtexturesampler,
      textureId: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new buffer object
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferCreate allocates and returns a new handle to a new buffer object in \a *buffer associated
   /// with \a context. The backing storage of the buffer is managed by OptiX. A buffer is specified by a bitwise
   /// \a or combination of a \a type and \a flags in \a bufferdesc. The supported types are:
   ///
   /// -  @ref RT_BUFFER_INPUT
   /// -  @ref RT_BUFFER_OUTPUT
   /// -  @ref RT_BUFFER_INPUT_OUTPUT
   /// -  @ref RT_BUFFER_PROGRESSIVE_STREAM
   ///
   /// The type values are used to specify the direction of data flow from the host to the OptiX devices.
   /// @ref RT_BUFFER_INPUT specifies that the host may only write to the buffer and the device may only read from the buffer.
   /// @ref RT_BUFFER_OUTPUT specifies the opposite, read only access on the host and write only access on the device.
   /// Devices and the host may read and write from buffers of type @ref RT_BUFFER_INPUT_OUTPUT.  Reading or writing to
   /// a buffer of the incorrect type (e.g., the host writing to a buffer of type @ref RT_BUFFER_OUTPUT) is undefined.
   /// @ref RT_BUFFER_PROGRESSIVE_STREAM is used to receive stream updates generated by progressive launches (see @ref rtContextLaunchProgressive2D).
   ///
   /// The supported flags are:
   ///
   /// -  @ref RT_BUFFER_GPU_LOCAL
   /// -  @ref RT_BUFFER_COPY_ON_DIRTY
   /// -  @ref RT_BUFFER_LAYERED
   /// -  @ref RT_BUFFER_CUBEMAP
   /// -  @ref RT_BUFFER_DISCARD_HOST_MEMORY
   ///
   /// If RT_BUFFER_LAYERED flag is set, buffer depth specifies the number of layers, not the depth of a 3D buffer.
   /// If RT_BUFFER_CUBEMAP flag is set, buffer depth specifies the number of cube faces, not the depth of a 3D buffer.
   /// See details in @ref rtBufferSetSize3D
   ///
   /// Flags can be used to optimize data transfers between the host and its devices. The flag @ref RT_BUFFER_GPU_LOCAL can only be
   /// used in combination with @ref RT_BUFFER_INPUT_OUTPUT. @ref RT_BUFFER_INPUT_OUTPUT and @ref RT_BUFFER_GPU_LOCAL used together specify a buffer
   /// that allows the host to \a only write, and the device to read \a and write data. The written data will never be visible
   /// on the host side and will generally not be visible on other devices.
   ///
   /// If @ref rtBufferGetDevicePointer has been called for a single device for a given buffer,
   /// the user can change the buffer's content on that device through the pointer. OptiX must then synchronize the new buffer contents to all devices.
   /// These synchronization copies occur at every @ref rtContextLaunch "rtContextLaunch", unless the buffer is created with @ref RT_BUFFER_COPY_ON_DIRTY.
   /// In this case, @ref rtBufferMarkDirty can be used to notify OptiX that the buffer has been dirtied and must be synchronized.
   ///
   /// The flag @ref RT_BUFFER_DISCARD_HOST_MEMORY can only be used in combination with @ref RT_BUFFER_INPUT. The data will be
   /// synchronized to the devices as soon as the buffer is unmapped from the host using @ref rtBufferUnmap or
   /// @ref rtBufferUnmapEx and the memory allocated on the host will be deallocated.
   /// It is preferred to map buffers created with the @ref RT_BUFFER_DISCARD_HOST_MEMORY using @ref rtBufferMapEx with the
   /// @ref RT_BUFFER_MAP_WRITE_DISCARD option enabled. If it is mapped using @ref rtBufferMap or the @ref RT_BUFFER_MAP_WRITE
   /// option instead, the data needs to be synchronized to the host during mapping.
   /// Note that the data that is allocated on the devices will not be deallocated until the buffer is destroyed.
   ///
   /// Returns @ref RT_ERROR_INVALID_VALUE if \a buffer is \a NULL.
   ///
   /// @param[in]   context      The context to create the buffer in
   /// @param[in]   bufferdesc   Bitwise \a or combination of the \a type and \a flags of the new buffer
   /// @param[out]  buffer       The return handle for the buffer object
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferCreate was introduced in OptiX 1.0.
   ///
   /// @ref RT_BUFFER_GPU_LOCAL was introduced in OptiX 2.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferCreateFromGLBO,
   /// @ref rtBufferDestroy,
   /// @ref rtBufferMarkDirty
   /// @ref rtBufferBindProgressiveStream
   ///
   pub fn rtBufferCreate(
      context: RTcontext,
      bufferdesc: ::std::os::raw::c_uint,
      buffer: *mut RTbuffer,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroys a buffer object
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferDestroy removes \a buffer from its context and deletes it.
   /// \a buffer should be a value returned by @ref rtBufferCreate.
   /// After the call, \a buffer is no longer a valid handle.
   /// Any API object that referenced \a buffer will have its reference invalidated.
   ///
   /// @param[in]   buffer   Handle of the buffer to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferDestroy was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferCreate,
   /// @ref rtBufferCreateFromGLBO
   ///
   pub fn rtBufferDestroy(buffer: RTbuffer) -> RtResult;
}
extern "C" {
   /// @brief Validates the state of a buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferValidate checks \a buffer for completeness.  If \a buffer has not had its dimensionality, size or format
   /// set, this call will return @ref RT_ERROR_INVALID_CONTEXT.
   ///
   /// @param[in]   buffer   The buffer to validate
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferValidate was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferCreate,
   /// @ref rtBufferCreateFromGLBO
   /// @ref rtContextValidate
   ///
   pub fn rtBufferValidate(buffer: RTbuffer) -> RtResult;
}
extern "C" {
   /// @brief Returns the context object that created this buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetContext returns a handle to the context that created \a buffer in \a *context.
   /// If \a *context is \a NULL, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   buffer    The buffer to be queried for its context
   /// @param[out]  context   The return handle for the buffer's context
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetContext was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextCreate
   ///
   pub fn rtBufferGetContext(
      buffer: RTbuffer,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the format of this buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferSetFormat changes the \a format of \a buffer to the specified value.
   /// The data elements of the buffer will have the specified type and can either be
   /// vector formats, or a user-defined type whose size is specified with
   /// @ref rtBufferSetElementSize. Possible values for \a format are:
   ///
   ///   - @ref RT_FORMAT_HALF
   ///   - @ref RT_FORMAT_HALF2
   ///   - @ref RT_FORMAT_HALF3
   ///   - @ref RT_FORMAT_HALF4
   ///   - @ref RT_FORMAT_FLOAT
   ///   - @ref RT_FORMAT_FLOAT2
   ///   - @ref RT_FORMAT_FLOAT3
   ///   - @ref RT_FORMAT_FLOAT4
   ///   - @ref RT_FORMAT_BYTE
   ///   - @ref RT_FORMAT_BYTE2
   ///   - @ref RT_FORMAT_BYTE3
   ///   - @ref RT_FORMAT_BYTE4
   ///   - @ref RT_FORMAT_UNSIGNED_BYTE
   ///   - @ref RT_FORMAT_UNSIGNED_BYTE2
   ///   - @ref RT_FORMAT_UNSIGNED_BYTE3
   ///   - @ref RT_FORMAT_UNSIGNED_BYTE4
   ///   - @ref RT_FORMAT_SHORT
   ///   - @ref RT_FORMAT_SHORT2
   ///   - @ref RT_FORMAT_SHORT3
   ///   - @ref RT_FORMAT_SHORT4
   ///   - @ref RT_FORMAT_UNSIGNED_SHORT
   ///   - @ref RT_FORMAT_UNSIGNED_SHORT2
   ///   - @ref RT_FORMAT_UNSIGNED_SHORT3
   ///   - @ref RT_FORMAT_UNSIGNED_SHORT4
   ///   - @ref RT_FORMAT_INT
   ///   - @ref RT_FORMAT_INT2
   ///   - @ref RT_FORMAT_INT3
   ///   - @ref RT_FORMAT_INT4
   ///   - @ref RT_FORMAT_UNSIGNED_INT
   ///   - @ref RT_FORMAT_UNSIGNED_INT2
   ///   - @ref RT_FORMAT_UNSIGNED_INT3
   ///   - @ref RT_FORMAT_UNSIGNED_INT4
   ///   - @ref RT_FORMAT_LONG_LONG
   ///   - @ref RT_FORMAT_LONG_LONG2
   ///   - @ref RT_FORMAT_LONG_LONG3
   ///   - @ref RT_FORMAT_LONG_LONG4
   ///   - @ref RT_FORMAT_UNSIGNED_LONG_LONG
   ///   - @ref RT_FORMAT_UNSIGNED_LONG_LONG2
   ///   - @ref RT_FORMAT_UNSIGNED_LONG_LONG3
   ///   - @ref RT_FORMAT_UNSIGNED_LONG_LONG4
   ///   - @ref RT_FORMAT_UNSIGNED_BC1
   ///   - @ref RT_FORMAT_UNSIGNED_BC2
   ///   - @ref RT_FORMAT_UNSIGNED_BC3
   ///   - @ref RT_FORMAT_UNSIGNED_BC4
   ///   - @ref RT_FORMAT_BC4
   ///   - @ref RT_FORMAT_UNSIGNED_BC5
   ///   - @ref RT_FORMAT_BC5
   ///   - @ref RT_FORMAT_UNSIGNED_BC6H
   ///   - @ref RT_FORMAT_BC6H
   ///   - @ref RT_FORMAT_UNSIGNED_BC7
   ///   - @ref RT_FORMAT_USER
   ///
   /// Buffers of block-compressed formats like @ref RT_FORMAT_BC6H must be sized
   /// to a quarter of the uncompressed view resolution in each dimension, i.e.
   /// @code rtBufferSetSize2D( buffer, width/4, height/4 ); @endcode
   /// The base type of the internal buffer will then correspond to @ref RT_FORMAT_UNSIGNED_INT2
   /// for BC1 and BC4 formats and @ref RT_FORMAT_UNSIGNED_INT4 for all other BC formats.
   ///
   /// @param[in]   buffer   The buffer to have its format set
   /// @param[in]   format   The target format of the buffer
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferSetFormat was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetFormat,
   /// @ref rtBufferGetFormat,
   /// @ref rtBufferGetFormat,
   /// @ref rtBufferGetElementSize,
   /// @ref rtBufferSetElementSize
   ///
   pub fn rtBufferSetFormat(buffer: RTbuffer, format: Format) -> RtResult;
}
extern "C" {
   /// @brief Gets the format of this buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetFormat returns, in \a *format, the format of \a buffer.  See @ref rtBufferSetFormat for a listing
   /// of @ref RTbuffer values.
   ///
   /// @param[in]   buffer   The buffer to be queried for its format
   /// @param[out]  format   The return handle for the buffer's format
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetFormat was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetFormat,
   /// @ref rtBufferGetFormat
   ///
   pub fn rtBufferGetFormat(buffer: RTbuffer, format: *mut Format) -> RtResult;
}
extern "C" {
   /// @brief Modifies the size in bytes of a buffer's individual elements
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferSetElementSize modifies the size in bytes of a buffer's user-formatted
   /// elements. The target buffer is specified by \a buffer, which should be a
   /// value returned by @ref rtBufferCreate and should have format @ref RT_FORMAT_USER.
   /// The new size of the buffer's individual elements is specified by
   /// \a elementSize and should not be 0. If the buffer has
   /// format @ref RT_FORMAT_USER, and \a elementSize is not 0, then the buffer's individual
   /// element size is set to \a elemenSize and all storage associated with the buffer is reset.
   /// Otherwise, this call has no effect and returns either @ref RT_ERROR_TYPE_MISMATCH if
   /// the buffer does not have format @ref RT_FORMAT_USER or @ref RT_ERROR_INVALID_VALUE if the
   /// buffer has format @ref RT_FORMAT_USER but \a elemenSize is 0.
   ///
   /// @param[in]   buffer            Specifies the buffer to be modified
   /// @param[in]   elementSize       Specifies the new size in bytes of the buffer's individual elements
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_TYPE_MISMATCH
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferSetElementSize was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferGetElementSize,
   /// @ref rtBufferCreate
   ///
   pub fn rtBufferSetElementSize(
      buffer: RTbuffer,
      elementSize: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the size of a buffer's individual elements
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetElementSize queries the size of a buffer's elements. The target buffer
   /// is specified by \a buffer, which should be a value returned by
   /// @ref rtBufferCreate. The size, in bytes, of the buffer's
   /// individual elements is returned in \a *elementSize.
   /// Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
   ///
   /// @param[in]   buffer                Specifies the buffer to be queried
   /// @param[out]  elementSize           Returns the size of the buffer's individual elements
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_UNKNOWN
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetElementSize was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetElementSize,
   /// @ref rtBufferCreate
   ///
   pub fn rtBufferGetElementSize(
      buffer: RTbuffer,
      elementSize: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the width and dimensionality of this buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferSetSize1D sets the dimensionality of \a buffer to 1 and sets its width to
   /// \a width.
   /// Fails with @ref RT_ERROR_ALREADY_MAPPED if called on a buffer that is mapped.
   ///
   /// @param[in]   buffer   The buffer to be resized
   /// @param[in]   width    The width of the resized buffer
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_ALREADY_MAPPED
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferSetSize1D was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferSetSize1D(buffer: RTbuffer, width: RTsize) -> RtResult;
}
extern "C" {
   /// @brief Get the width of this buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetSize1D stores the width of \a buffer in \a *width.
   ///
   /// @param[in]   buffer   The buffer to be queried for its dimensions
   /// @param[out]  width    The return handle for the buffer's width
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetSize1D was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferGetSize1D(buffer: RTbuffer, width: *mut RTsize) -> RtResult;
}
extern "C" {
   /// @brief Sets the width, height and dimensionality of this buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferSetSize2D sets the dimensionality of \a buffer to 2 and sets its width
   /// and height to \a width and \a height, respectively.  If \a width or \a height is
   /// zero, they both must be zero.
   /// Fails with @ref RT_ERROR_ALREADY_MAPPED if called on a buffer that is mapped.
   ///
   /// @param[in]   buffer   The buffer to be resized
   /// @param[in]   width    The width of the resized buffer
   /// @param[in]   height   The height of the resized buffer
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_ALREADY_MAPPED
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferSetSize2D was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferSetSize2D(
      buffer: RTbuffer,
      width: RTsize,
      height: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the width and height of this buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetSize2D stores the width and height of \a buffer in \a *width and
   /// \a *height, respectively.
   ///
   /// @param[in]   buffer   The buffer to be queried for its dimensions
   /// @param[out]  width    The return handle for the buffer's width
   /// @param[out]  height   The return handle for the buffer's height
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetSize2D was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferGetSize2D(
      buffer: RTbuffer,
      width: *mut RTsize,
      height: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the width, height, depth and dimensionality of a buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferSetSize3D sets the dimensionality of \a buffer to 3 and sets its width,
   /// height and depth to \a width, \a height and \a depth, respectively.  If \a width,
   /// \a height or \a depth is zero, they all must be zero.
   ///
   /// A 1D layered mipmapped buffer is allocated if \a height is 1 and the @ref RT_BUFFER_LAYERED flag was set at buffer creating. The number of layers is determined by the \a depth.
   /// A 2D layered mipmapped buffer is allocated if the @ref RT_BUFFER_LAYERED flag was set at buffer creating. The number of layers is determined by the \a depth.
   /// A cubemap mipmapped buffer is allocated if the @ref RT_BUFFER_CUBEMAP flag was set at buffer creating. \a width must be equal to \a height and the number of cube faces is determined by the \a depth,
   /// it must be six or a multiple of six, if the @ref RT_BUFFER_LAYERED flag was also set.
   /// Layered, mipmapped and cubemap buffers are supported only as texture buffers.
   ///
   /// Fails with @ref RT_ERROR_ALREADY_MAPPED if called on a buffer that is mapped.
   ///
   /// @param[in]   buffer   The buffer to be resized
   /// @param[in]   width    The width of the resized buffer
   /// @param[in]   height   The height of the resized buffer
   /// @param[in]   depth    The depth of the resized buffer
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_ALREADY_MAPPED
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferSetSize3D was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferSetSize3D(
      buffer: RTbuffer,
      width: RTsize,
      height: RTsize,
      depth: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the MIP level count of a buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferSetMipLevelCount sets the number of MIP levels to \a levels. The default number of MIP levels is 1.
   /// Fails with @ref RT_ERROR_ALREADY_MAPPED if called on a buffer that is mapped.
   ///
   /// @param[in]   buffer   The buffer to be resized
   /// @param[in]   levels   Number of mip levels
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_ALREADY_MAPPED
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferSetMipLevelCount was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferSetMipLevelCount(
      buffer: RTbuffer,
      levels: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the width, height and depth of this buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetSize3D stores the width, height and depth of \a buffer in \a *width,
   /// \a *height and \a *depth, respectively.
   ///
   /// @param[in]   buffer   The buffer to be queried for its dimensions
   /// @param[out]  width    The return handle for the buffer's width
   /// @param[out]  height   The return handle for the buffer's height
   /// @param[out]  depth    The return handle for the buffer's depth
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetSize3D was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferGetSize3D(
      buffer: RTbuffer,
      width: *mut RTsize,
      height: *mut RTsize,
      depth: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the width of buffer specific MIP level
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetMipLevelSize1D stores the width of \a buffer in \a *width.
   ///
   /// @param[in]   buffer   The buffer to be queried for its dimensions
   /// @param[in]   level    The buffer MIP level index to be queried for its dimensions
   /// @param[out]  width    The return handle for the buffer's width
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetMipLevelSize1D was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferGetMipLevelSize1D(
      buffer: RTbuffer,
      level: ::std::os::raw::c_uint,
      width: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the width, height of buffer specific MIP level
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetMipLevelSize2D stores the width, height of \a buffer in \a *width and
   /// \a *height respectively.
   ///
   /// @param[in]   buffer   The buffer to be queried for its dimensions
   /// @param[in]   level    The buffer MIP level index to be queried for its dimensions
   /// @param[out]  width    The return handle for the buffer's width
   /// @param[out]  height   The return handle for the buffer's height
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetMipLevelSize2D was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferGetMipLevelSize2D(
      buffer: RTbuffer,
      level: ::std::os::raw::c_uint,
      width: *mut RTsize,
      height: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the width, height and depth of buffer specific MIP level
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetMipLevelSize3D stores the width, height and depth of \a buffer in \a *width,
   /// \a *height and \a *depth, respectively.
   ///
   /// @param[in]   buffer   The buffer to be queried for its dimensions
   /// @param[in]   level    The buffer MIP level index to be queried for its dimensions
   /// @param[out]  width    The return handle for the buffer's width
   /// @param[out]  height   The return handle for the buffer's height
   /// @param[out]  depth    The return handle for the buffer's depth
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetMipLevelSize3D was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferGetMipLevelSize3D(
      buffer: RTbuffer,
      level: ::std::os::raw::c_uint,
      width: *mut RTsize,
      height: *mut RTsize,
      depth: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the dimensionality and dimensions of a buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferSetSizev sets the dimensionality of \a buffer to \a dimensionality and
   /// sets the dimensions of the buffer to the values stored at *\a dims, which must contain
   /// a number of values equal to \a dimensionality.  If any of values of \a dims is zero
   /// they must all be zero.
   ///
   /// @param[in]   buffer           The buffer to be resized
   /// @param[in]   dimensionality   The dimensionality the buffer will be resized to
   /// @param[in]   dims             The array of sizes for the dimension of the resize
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_ALREADY_MAPPED
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferSetSizev was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferSetSizev(
      buffer: RTbuffer,
      dimensionality: ::std::os::raw::c_uint,
      dims: *const RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the dimensions of this buffer
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetSizev stores the dimensions of \a buffer in \a *dims.  The number of
   /// dimensions returned is specified by \a dimensionality.  The storage at \a dims must be
   /// large enough to hold the number of requested buffer dimensions.
   ///
   /// @param[in]   buffer           The buffer to be queried for its dimensions
   /// @param[in]   dimensionality   The number of requested dimensions
   /// @param[out]  dims             The array of dimensions to store to
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetSizev was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetMipLevelCount,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D
   ///
   pub fn rtBufferGetSizev(
      buffer: RTbuffer,
      dimensionality: ::std::os::raw::c_uint,
      dims: *mut RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the dimensionality of this buffer object
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetDimensionality returns the dimensionality of \a buffer in \a
   /// *dimensionality.  The value returned will be one of 1, 2 or 3, corresponding to 1D, 2D
   /// and 3D buffers, respectively.
   ///
   /// @param[in]   buffer           The buffer to be queried for its dimensionality
   /// @param[out]  dimensionality   The return handle for the buffer's dimensionality
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetDimensionality was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// \a rtBufferSetSize{1-2-3}D
   ///
   pub fn rtBufferGetDimensionality(
      buffer: RTbuffer,
      dimensionality: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the number of mipmap levels of this buffer object
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetMipLevelCount returns the number of mipmap levels. Default number of MIP levels is 1.
   ///
   /// @param[in]   buffer           The buffer to be queried for its number of mipmap levels
   /// @param[out]  level            The return number of mipmap levels
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetMipLevelCount was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetMipLevelCount,
   /// @ref rtBufferSetSize1D,
   /// @ref rtBufferSetSize2D,
   /// @ref rtBufferSetSize3D,
   /// @ref rtBufferSetSizev,
   /// @ref rtBufferGetMipLevelSize1D,
   /// @ref rtBufferGetMipLevelSize2D,
   /// @ref rtBufferGetMipLevelSize3D,
   /// @ref rtBufferGetSize1D,
   /// @ref rtBufferGetSize2D,
   /// @ref rtBufferGetSize3D,
   /// @ref rtBufferGetSizev
   ///
   pub fn rtBufferGetMipLevelCount(
      buffer: RTbuffer,
      level: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Maps a buffer object to the host
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferMap returns a pointer, accessible by the host, in \a *userPointer that
   /// contains a mapped copy of the contents of \a buffer.  The memory pointed to by \a *userPointer
   /// can be written to or read from, depending on the type of \a buffer.  For
   /// example, this code snippet demonstrates creating and filling an input buffer with
   /// floats.
   ///
   ///@code
   ///  RTbuffer buffer;
   ///  float* data;
   ///  rtBufferCreate(context, RT_BUFFER_INPUT, &buffer);
   ///  rtBufferSetFormat(buffer, RT_FORMAT_FLOAT);
   ///  rtBufferSetSize1D(buffer, 10);
   ///  rtBufferMap(buffer, (void*)&data);
   ///  for(int i = 0; i < 10; ++i)
   ///    data[i] = 4.f * i;
   ///  rtBufferUnmap(buffer);
   ///@endcode
   /// If \a buffer has already been mapped, returns @ref RT_ERROR_ALREADY_MAPPED.
   /// If \a buffer has size zero, the returned pointer is undefined.
   ///
   /// Note that this call does not stop a progressive render if called on a stream buffer.
   ///
   /// @param[in]   buffer         The buffer to be mapped
   /// @param[out]  userPointer    Return handle to a user pointer where the buffer will be mapped to
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_ALREADY_MAPPED
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferMap was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferUnmap,
   /// @ref rtBufferMapEx,
   /// @ref rtBufferUnmapEx
   ///
   pub fn rtBufferMap(
      buffer: RTbuffer,
      userPointer: *mut *mut ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Unmaps a buffer's storage from the host
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferUnmap unmaps a buffer from the host after a call to @ref rtBufferMap.  @ref rtContextLaunch "rtContextLaunch" cannot be called
   /// while buffers are still mapped to the host.  A call to @ref rtBufferUnmap that does not follow a matching @ref rtBufferMap
   /// call will return @ref RT_ERROR_INVALID_VALUE.
   ///
   /// Note that this call does not stop a progressive render if called with a stream buffer.
   ///
   /// @param[in]   buffer   The buffer to unmap
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferUnmap was introduced in OptiX 1.0.
   ///
   /// <B>See also</B>
   /// @ref rtBufferMap,
   /// @ref rtBufferMapEx,
   /// @ref rtBufferUnmapEx
   ///
   pub fn rtBufferUnmap(buffer: RTbuffer) -> RtResult;
}
extern "C" {
   /// @brief Maps mipmap level of buffer object to the host
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferMapEx makes the buffer contents available on the host, either by returning a pointer in \a *optixOwned, or by copying the contents
   /// to a memory location pointed to by \a userOwned. Calling @ref rtBufferMapEx with proper map flags can result in better performance than using @ref rtBufferMap, because
   /// fewer synchronization copies are required in certain situations.
   /// @ref rtBufferMapEx with \a mapFlags = @ref RT_BUFFER_MAP_READ_WRITE and \a level = 0 is equivalent to @ref rtBufferMap.
   ///
   /// Note that this call does not stop a progressive render if called on a stream buffer.
   ///
   /// @param[in]   buffer         The buffer to be mapped
   /// @param[in]   mapFlags       Map flags, see below
   /// @param[in]   level          The mipmap level to be mapped
   /// @param[in]   userOwned      Not yet supported. Must be NULL
   /// @param[out]  optixOwned     Return handle to a user pointer where the buffer will be mapped to
   ///
   /// The following flags are supported for mapFlags. They are mutually exclusive:
   ///
   /// -  @ref RT_BUFFER_MAP_READ
   /// -  @ref RT_BUFFER_MAP_WRITE
   /// -  @ref RT_BUFFER_MAP_READ_WRITE
   /// -  @ref RT_BUFFER_MAP_WRITE_DISCARD
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_ALREADY_MAPPED
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferMapEx was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtBufferMap,
   /// @ref rtBufferUnmap,
   /// @ref rtBufferUnmapEx
   ///
   pub fn rtBufferMapEx(
      buffer: RTbuffer,
      mapFlags: ::std::os::raw::c_uint,
      level: ::std::os::raw::c_uint,
      userOwned: *mut ::std::os::raw::c_void,
      optixOwned: *mut *mut ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Unmaps mipmap level storage from the host
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferUnmapEx unmaps buffer level from the host after a call to @ref rtBufferMapEx.  @ref rtContextLaunch "rtContextLaunch" cannot be called
   /// while buffers are still mapped to the host.  A call to @ref rtBufferUnmapEx that does not follow a matching @ref rtBufferMapEx
   /// call will return @ref RT_ERROR_INVALID_VALUE. @ref rtBufferUnmap is equivalent to @ref rtBufferUnmapEx with \a level = 0.
   ///
   /// Note that this call does not stop a progressive render if called with a stream buffer.
   ///
   /// @param[in]   buffer   The buffer to unmap
   /// @param[in]   level    The mipmap level to unmap
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferUnmapEx was introduced in OptiX 3.9.
   ///
   /// <B>See also</B>
   /// @ref rtBufferMap,
   /// @ref rtBufferUnmap,
   /// @ref rtBufferMapEx
   ///
   pub fn rtBufferUnmapEx(
      buffer: RTbuffer,
      level: ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets an id suitable for use with buffers of buffers
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetId returns an ID for the provided buffer.  The returned ID is used on
   /// the device to reference the buffer.  It needs to be copied into a buffer of type @ref
   /// RT_FORMAT_BUFFER_ID or used in a @ref rtBufferId object.. If \a *bufferId is \a NULL
   /// or the \a buffer is not a valid RTbuffer, returns @ref
   /// RT_ERROR_INVALID_VALUE.  @ref RT_BUFFER_ID_NULL can be used as a sentinel for a
   /// non-existent buffer, since this value will never be returned as a valid buffer id.
   ///
   /// @param[in]   buffer      The buffer to be queried for its id
   /// @param[out]  bufferId    The returned ID of the buffer
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetId was introduced in OptiX 3.5.
   ///
   /// <B>See also</B>
   /// @ref rtContextGetBufferFromId
   ///
   pub fn rtBufferGetId(
      buffer: RTbuffer,
      bufferId: *mut ::std::os::raw::c_int,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets an RTbuffer corresponding to the buffer id
   ///
   /// @ingroup Context
   ///
   /// <B>Description</B>
   ///
   /// @ref rtContextGetBufferFromId returns a handle to the buffer in \a *buffer corresponding to
   /// the \a bufferId supplied.  If \a bufferId does not map to a valid buffer handle,
   /// \a *buffer is \a NULL or if \a context is invalid, returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   context     The context the buffer should be originated from
   /// @param[in]   bufferId    The ID of the buffer to query
   /// @param[out]  buffer      The return handle for the buffer object corresponding to the bufferId
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtContextGetBufferFromId was introduced in OptiX 3.5.
   ///
   /// <B>See also</B>
   /// @ref rtBufferGetId
   ///
   pub fn rtContextGetBufferFromId(
      context: RTcontext,
      bufferId: ::std::os::raw::c_int,
      buffer: *mut RTbuffer,
   ) -> RtResult;
}
extern "C" {
   /// @brief Check whether stream buffer content has been updated by a Progressive Launch
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// Returns whether or not the result of a progressive launch in \a buffer has been updated
   /// since the last time this function was called. A client application should use this call in its
   /// main render/display loop to poll for frame refreshes after initiating a progressive launch. If \a subframeCount and
   /// \a maxSubframes are non-null, they will be filled with the corresponding counters if and
   /// only if \a ready returns 1.
   ///
   /// Note that this call does not stop a progressive render.
   ///
   /// @param[in]   buffer             The stream buffer to be queried
   /// @param[out]  ready              Ready flag. Will be set to 1 if an update is available, or 0 if no update is available.
   /// @param[out]  subframeCount      The number of subframes accumulated in the latest result
   /// @param[out]  maxSubframes       The \a maxSubframes parameter as specified in the call to @ref rtContextLaunchProgressive2D
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetProgressiveUpdateReady was introduced in OptiX 3.8.
   ///
   /// <B>See also</B>
   /// @ref rtContextLaunchProgressive2D
   ///
   pub fn rtBufferGetProgressiveUpdateReady(
      buffer: RTbuffer,
      ready: *mut ::std::os::raw::c_int,
      subframeCount: *mut ::std::os::raw::c_uint,
      maxSubframes: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Bind a stream buffer to an output buffer source
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// Binds an output buffer to a progressive stream. The output buffer thereby becomes the
   /// data source for the stream. To form a valid output/stream pair, the stream buffer must be
   /// of format @ref RT_FORMAT_UNSIGNED_BYTE4, and the output buffer must be of format @ref RT_FORMAT_FLOAT3 or @ref RT_FORMAT_FLOAT4.
   /// The use of @ref RT_FORMAT_FLOAT4 is recommended for performance reasons, even if the fourth component is unused.
   /// The output buffer must be of type @ref RT_BUFFER_OUTPUT; it may not be of type @ref RT_BUFFER_INPUT_OUTPUT.
   ///
   /// @param[in]   stream             The stream buffer for which the source is to be specified
   /// @param[in]   source             The output buffer to function as the stream's source
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferBindProgressiveStream was introduced in OptiX 3.8.
   ///
   /// <B>See also</B>
   /// @ref rtBufferCreate
   /// @ref rtBufferSetAttribute
   /// @ref rtBufferGetAttribute
   ///
   pub fn rtBufferBindProgressiveStream(
      stream: RTbuffer,
      source: RTbuffer,
   ) -> RtResult;
}
extern "C" {
   /// @brief Set a buffer attribute
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// Sets a buffer attribute. Currently, all available attributes refer to stream buffers only,
   /// and attempting to set them on a non-stream buffer will generate an error.
   ///
   /// Each attribute can have a different size.  The sizes are given in the following list:
   ///
   ///   - @ref RT_BUFFER_ATTRIBUTE_STREAM_FORMAT      strlen(input_string)
   ///   - @ref RT_BUFFER_ATTRIBUTE_STREAM_BITRATE     sizeof(int)
   ///   - @ref RT_BUFFER_ATTRIBUTE_STREAM_FPS         sizeof(int)
   ///   - @ref RT_BUFFER_ATTRIBUTE_STREAM_GAMMA       sizeof(float)
   ///
   /// @ref RT_BUFFER_ATTRIBUTE_STREAM_FORMAT sets the encoding format used for streams sent over the network, specified as a string.
   /// The default is "auto". Various other common stream and image formats are available (e.g. "h264", "png"). This
   /// attribute has no effect if the progressive API is used locally.
   ///
   /// @ref RT_BUFFER_ATTRIBUTE_STREAM_BITRATE sets the target bitrate for streams sent over the network, if the stream format supports
   /// it. The data is specified as a 32-bit integer. The default is 5000000. This attribute has no
   /// effect if the progressive API is used locally or if the stream format does not support
   /// variable bitrates.
   ///
   /// @ref RT_BUFFER_ATTRIBUTE_STREAM_FPS sets the target update rate per second for streams sent over the network, if the stream
   /// format supports it. The data is specified as a 32-bit integer. The default is 30. This
   /// attribute has no effect if the progressive API is used locally or if the stream format does
   /// not support variable framerates.
   ///
   /// @ref RT_BUFFER_ATTRIBUTE_STREAM_GAMMA sets the gamma value for the built-in tonemapping operator. The data is specified as a
   /// 32-bit float, the default is 1.0. Tonemapping is executed before encoding the
   /// accumulated output into the stream, i.e. on the server side if remote rendering is used.
   /// See the section on Buffers below for more details.
   ///
   /// @param[in]   buffer             The buffer on which to set the attribute
   /// @param[in]   attrib             The attribute to set
   /// @param[in]   size               The size of the attribute value, in bytes
   /// @param[in]   p                  Pointer to the attribute value
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferSetAttribute was introduced in OptiX 3.8.
   ///
   /// <B>See also</B>
   /// @ref rtBufferGetAttribute
   ///
   pub fn rtBufferSetAttribute(
      buffer: RTbuffer,
      attrib: BufferAttribute,
      size: RTsize,
      p: *const ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Query a buffer attribute
   ///
   /// @ingroup Buffer
   ///
   /// <B>Description</B>
   ///
   /// @ref rtBufferGetAttribute is used to query buffer attributes. For a list of available attributes, please refer to @ref rtBufferSetAttribute.
   ///
   /// @param[in]   buffer             The buffer to query the attribute from
   /// @param[in]   attrib             The attribute to query
   /// @param[in]   size               The size of the attribute value, in bytes. For string attributes, this is the maximum buffer size the returned string will use (including a terminating null character).
   /// @param[out]  p                  Pointer to the attribute value to be filled in. Must point to valid memory of at least \a size bytes.
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtBufferGetAttribute was introduced in OptiX 3.8.
   ///
   /// <B>See also</B>
   /// @ref rtBufferSetAttribute
   ///
   pub fn rtBufferGetAttribute(
      buffer: RTbuffer,
      attrib: BufferAttribute,
      size: RTsize,
      p: *mut ::std::os::raw::c_void,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new post-processing stage
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtPostProcessingStageCreateBuiltin creates a new post-processing stage selected from a list of
   /// pre-defined post-processing stages. The \a context specifies the target context, and should be
   /// a value returned by @ref rtContextCreate.
   /// Sets \a *stage to the handle of a newly created stage within \a context.
   ///
   /// @param[in]   context      Specifies the rendering context to which the post-processing stage belongs
   /// @param[in]   builtinName  The name of the built-in stage to instantiate
   /// @param[out]  stage        New post-processing stage handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtPostProcessingStageCreateBuiltin was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtPostProcessingStageDestroy,
   /// @ref rtPostProcessingStageGetContext,
   /// @ref rtPostProcessingStageQueryVariable,
   /// @ref rtPostProcessingStageGetVariableCount
   /// @ref rtPostProcessingStageGetVariable
   ///
   pub fn rtPostProcessingStageCreateBuiltin(
      context: RTcontext,
      builtinName: *const ::std::os::raw::c_char,
      stage: *mut RTpostprocessingstage,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroy a post-processing stage
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtPostProcessingStageDestroy destroys a post-processing stage from its context and deletes
   /// it. The variables built into the stage are destroyed. After the call, \a stage is no longer a valid handle.
   /// After a post-processing stage was destroyed all command lists containing that stage are invalidated and
   /// can no longer be used.
   ///
   /// @param[in]  stage        Handle of the post-processing stage to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtPostProcessingStageDestroy was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtPostProcessingStageCreateBuiltin,
   /// @ref rtPostProcessingStageGetContext,
   /// @ref rtPostProcessingStageQueryVariable,
   /// @ref rtPostProcessingStageGetVariableCount
   /// @ref rtPostProcessingStageGetVariable
   ///
   pub fn rtPostProcessingStageDestroy(
      stage: RTpostprocessingstage,
   ) -> RtResult;
}
extern "C" {
   /// @brief Declares a new named variable associated with a PostprocessingStage
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtPostProcessingStageDeclareVariable declares a new variable associated with a
   /// postprocessing stage. \a stage specifies the post-processing stage, and should be a value
   /// returned by @ref rtPostProcessingStageCreateBuiltin. \a name specifies the name of the variable, and
   /// should be a \a NULL-terminated string. If there is currently no variable associated with \a
   /// stage named \a name, a new variable named \a name will be created and associated with
   /// \a stage.  After the call, \a *v will be set to the handle of the newly-created
   /// variable.  Otherwise, \a *v will be set to \a NULL. After declaration, the variable can be
   /// queried with @ref rtPostProcessingStageQueryVariable or @ref rtPostProcessingStageGetVariable. A
   /// declared variable does not have a type until its value is set with one of the @ref rtVariableSet
   /// functions. Once a variable is set, its type cannot be changed anymore.
   ///
   /// @param[in]   stage   Specifies the associated postprocessing stage
   /// @param[in]   name               The name that identifies the variable
   /// @param[out]  v                  Returns a handle to a newly declared variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtPostProcessingStageDeclareVariable was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref Variables,
   /// @ref rtPostProcessingStageQueryVariable,
   /// @ref rtPostProcessingStageGetVariable
   ///
   pub fn rtPostProcessingStageDeclareVariable(
      stage: RTpostprocessingstage,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with a post-processing stage.
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtPostProcessingStageGetContext queries a stage for its associated context.
   /// \a stage specifies the post-processing stage to query, and should be a value
   /// returned by @ref rtPostProcessingStageCreateBuiltin. If both parameters are valid,
   /// \a *context is set to the context associated with \a stage. Otherwise, the call
   /// has no effect and returns @ref RT_ERROR_INVALID_VALUE.
   ///
   /// @param[in]   stage      Specifies the post-processing stage to query
   /// @param[out]  context    Returns the context associated with the material
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtPostProcessingStageGetContext was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtPostProcessingStageCreateBuiltin,
   /// @ref rtPostProcessingStageDestroy,
   /// @ref rtPostProcessingStageQueryVariable,
   /// @ref rtPostProcessingStageGetVariableCount
   /// @ref rtPostProcessingStageGetVariable
   ///
   pub fn rtPostProcessingStageGetContext(
      stage: RTpostprocessingstage,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a handle to a named variable of a post-processing stage
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtPostProcessingStageQueryVariable queries the handle of a post-processing stage's named
   /// variable. \a stage specifies the source post-processing stage, as returned by
   /// @ref rtPostProcessingStageCreateBuiltin. \a name specifies the name of the variable, and should be a
   /// \a NULL -terminated string. If \a name is the name of a variable attached to \a stage, the call
   /// returns a handle to that variable in \a *variable, otherwise \a NULL. Only pre-defined variables of that
   /// built-in stage type can be queried. It is not possible to add or remove variables.
   ///
   /// @param[in]   stage              The post-processing stage to query the variable from
   /// @param[in]   name               The name that identifies the variable to be queried
   /// @param[out]  variable           Returns the named variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtPostProcessingStageQueryVariable was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtPostProcessingStageCreateBuiltin,
   /// @ref rtPostProcessingStageDestroy,
   /// @ref rtPostProcessingStageGetContext,
   /// @ref rtPostProcessingStageGetVariableCount
   /// @ref rtPostProcessingStageGetVariable
   ///
   pub fn rtPostProcessingStageQueryVariable(
      stage: RTpostprocessingstage,
      name: *const ::std::os::raw::c_char,
      variable: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns the number of variables pre-defined in a post-processing stage.
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtPostProcessingStageGetVariableCount returns the number of variables which are pre-defined
   /// in a post-processing stage. This can be used to iterate over the variables. Sets \a *count to the
   /// number.
   ///
   /// @param[in]   stage              The post-processing stage to query the number of variables from
   /// @param[out]  count              Returns the number of pre-defined variables
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtPostProcessingStageGetVariableCount was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtPostProcessingStageCreateBuiltin,
   /// @ref rtPostProcessingStageDestroy,
   /// @ref rtPostProcessingStageGetContext,
   /// @ref rtPostProcessingStageQueryVariable,
   /// @ref rtPostProcessingStageGetVariable
   ///
   pub fn rtPostProcessingStageGetVariableCount(
      stage: RTpostprocessingstage,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Returns a handle to a variable of a post-processing stage. The variable is defined by index.
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtPostProcessingStageGetVariable queries the handle of a post-processing stage's variable which
   /// is identified by its index . \a stage specifies the source post-processing stage, as returned by
   /// @ref rtPostProcessingStageCreateBuiltin. \a index specifies the index of the variable, and should be a
   /// less than the value return by @ref rtPostProcessingStageGetVariableCount. If \a index is in the valid
   /// range, the call returns a handle to that variable in \a *variable, otherwise \a NULL.
   ///
   /// @param[in]   stage              The post-processing stage to query the variable from
   /// @param[in]   index              The index identifying the variable to be returned
   /// @param[out]  variable           Returns the variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtPostProcessingStageGetVariable was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtPostProcessingStageCreateBuiltin,
   /// @ref rtPostProcessingStageDestroy,
   /// @ref rtPostProcessingStageGetContext,
   /// @ref rtPostProcessingStageQueryVariable,
   /// @ref rtPostProcessingStageGetVariableCount
   ///
   pub fn rtPostProcessingStageGetVariable(
      stage: RTpostprocessingstage,
      index: ::std::os::raw::c_uint,
      variable: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Creates a new command list
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtCommandListCreate creates a new command list. The \a context specifies the target
   /// context, and should be a value returned by @ref rtContextCreate. The call
   /// sets \a *list to the handle of a newly created list within \a context.
   /// Returns @ref RT_ERROR_INVALID_VALUE if \a list is \a NULL.
   ///
   /// A command list can be used to assemble a list of different types of commands and execute them
   /// later. At this point, commands can be built-in post-processing stages or context launches. Those
   /// are appended to the list using @ref rtCommandListAppendPostprocessingStage, and @ref
   /// rtCommandListAppendLaunch2D, respectively. Commands will be executed in the order they have been
   /// appended to the list. Thus later commands can use the results of earlier commands. Note that
   /// all commands added to the created list must be associated with the same \a context. It is
   /// invalid to mix commands from  different contexts.
   ///
   /// @param[in]   context     Specifies the rendering context of the command list
   /// @param[out]  list        New command list handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   /// - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
   ///
   /// <B>History</B>
   ///
   /// @ref rtCommandListCreate was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtCommandListDestroy,
   /// @ref rtCommandListAppendPostprocessingStage,
   /// @ref rtCommandListAppendLaunch2D,
   /// @ref rtCommandListFinalize,
   /// @ref rtCommandListExecute
   ///
   pub fn rtCommandListCreate(
      context: RTcontext,
      list: *mut RTcommandlist,
   ) -> RtResult;
}
extern "C" {
   /// @brief Destroy a command list
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtCommandListDestroy destroys a command list from its context and deletes it. After the
   /// call, \a list is no longer a valid handle. Any stages associated with the command list are not destroyed.
   ///
   /// @param[in]  list        Handle of the command list to destroy
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtCommandListDestroy was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtCommandListCreate,
   /// @ref rtCommandListAppendPostprocessingStage,
   /// @ref rtCommandListAppendLaunch2D,
   /// @ref rtCommandListFinalize,
   /// @ref rtCommandListExecute
   ///
   pub fn rtCommandListDestroy(list: RTcommandlist) -> RtResult;
}
extern "C" {
   /// @brief Append a post-processing stage to the command list \a list
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtCommandListAppendPostprocessingStage appends a post-processing stage to the command list
   /// \a list. The command list must have been created from the same context as the the post-processing
   /// stage.
   /// The launchWidth and launchHeight specify the launch dimensions and may be different than the
   /// input or output buffers associated with each post-processing stage depending on the requirements
   /// of the post-processing stage appended.
   /// It is invalid to call @ref rtCommandListAppendPostprocessingStage after calling @ref
   /// rtCommandListFinalize.
   ///
   /// NOTE: A post-processing stage can be added to multiple command lists or added to the same command
   /// list multiple times.  Also note that destroying a post-processing stage will invalidate all command
   /// lists it was added to.
   ///
   /// @param[in]  list          Handle of the command list to append to
   /// @param[in]  stage         The post-processing stage to append to the command list
   /// @param[in]  launchWidth   This is a hint for the width of the launch dimensions to use for this stage.
   ///                           The stage can ignore this and use a suitable launch width instead.
   /// @param[in]  launchWidth   This is a hint for the height of the launch dimensions to use for this stage.
   ///                           The stage can ignore this and use a suitable launch height instead.
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtCommandListAppendPostprocessingStage was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtCommandListCreate,
   /// @ref rtCommandListDestroy,
   /// @ref rtCommandListAppendLaunch2D,
   /// @ref rtCommandListFinalize,
   /// @ref rtCommandListExecute
   /// @ref rtPostProcessingStageCreateBuiltin,
   ///
   pub fn rtCommandListAppendPostprocessingStage(
      list: RTcommandlist,
      stage: RTpostprocessingstage,
      launchWidth: RTsize,
      launchHeight: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Append a launch to the command list \a list
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtCommandListAppendLaunch2D appends a context launch to the command list \a list. It is
   /// invalid to call @ref rtCommandListAppendLaunch2D after calling @ref rtCommandListFinalize.
   ///
   /// @param[in]  list              Handle of the command list to append to
   /// @param[in]  entryPointIndex   The initial entry point into the kernel
   /// @param[in]  launchWidth       Width of the computation grid
   /// @param[in]  launchHeight      Height of the computation grid
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtCommandListAppendLaunch2D was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtCommandListCreate,
   /// @ref rtCommandListDestroy,
   /// @ref rtCommandListAppendPostprocessingStage,
   /// @ref rtCommandListFinalize,
   /// @ref rtCommandListExecute
   ///
   pub fn rtCommandListAppendLaunch2D(
      list: RTcommandlist,
      entryPointIndex: ::std::os::raw::c_uint,
      launchWidth: RTsize,
      launchHeight: RTsize,
   ) -> RtResult;
}
extern "C" {
   /// @brief Finalize the command list. This must be done before executing the command list.
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtCommandListFinalize finalizes the command list. This will do all work necessary to
   /// prepare the command list for execution. Specifically it will do all work which can be shared
   /// between subsequent calls to @ref rtCommandListExecute.
   /// It is invalid to call @ref rtCommandListExecute before calling @ref rtCommandListFinalize. It is
   /// invalid to call @ref rtCommandListAppendPostprocessingStage or
   /// @ref rtCommandListAppendLaunch2D after calling finalize and will result in an error. Also
   /// @ref rtCommandListFinalize can only be called once on each command list.
   ///
   /// @param[in]  list              Handle of the command list to finalize
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtCommandListFinalize was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtCommandListCreate,
   /// @ref rtCommandListDestroy,
   /// @ref rtCommandListAppendPostprocessingStage,
   /// @ref rtCommandListAppendLaunch2D,
   /// @ref rtCommandListExecute
   ///
   pub fn rtCommandListFinalize(list: RTcommandlist) -> RtResult;
}
extern "C" {
   /// @brief Execute the command list.
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtCommandListExecute executes the command list. All added commands will be executed in the
   /// order in which they were added. Commands can access the results of earlier executed commands.
   /// This must be called after calling @ref rtCommandListFinalize, otherwise an error will be returned
   /// and the command list is not executed.
   /// @ref rtCommandListExecute can be called multiple times, but only one call may be active at the
   /// same time. Overlapping calls from multiple threads will result in undefined behavior.
   ///
   /// @param[in]  list              Handle of the command list to execute
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtCommandListExecute was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtCommandListCreate,
   /// @ref rtCommandListDestroy,
   /// @ref rtCommandListAppendPostprocessingStage,
   /// @ref rtCommandListAppendLaunch2D,
   /// @ref rtCommandListFinalize,
   ///
   pub fn rtCommandListExecute(list: RTcommandlist) -> RtResult;
}
extern "C" {
   /// @brief Returns the context associated with a command list
   ///
   /// @ingroup CommandList
   ///
   /// <B>Description</B>
   ///
   /// @ref rtCommandListGetContext queries the context associated with a command list. The
   /// target command list is specified by \a list. The context of the command list is
   /// returned to \a *context if the pointer \a context is not \a NULL. If \a list is
   /// not a valid command list, \a *context is set to \a NULL and @ref RT_ERROR_INVALID_VALUE is
   /// returned.
   ///
   /// @param[in]   list       Specifies the command list to be queried
   /// @param[out]  context    Returns the context associated with the command list
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtCommandListGetContext was introduced in OptiX 5.0.
   ///
   /// <B>See also</B>
   /// @ref rtContextDeclareVariable
   ///
   pub fn rtCommandListGetContext(
      list: RTcommandlist,
      context: *mut RTcontext,
   ) -> RtResult;
}
extern "C" {
   /// @brief Sets the attribute program on a GeometryTriangles object
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesSetAttributeProgram sets for \a geometrytriangles the \a
   /// program that performs attribute computation.  RTprograms can be either generated with
   /// @ref rtProgramCreateFromPTXFile or @ref rtProgramCreateFromPTXString. An attribute
   /// program is optional.  If no attribute program is specified, a default attribute
   /// program will be provided.  Attributes are computed after intersection and before any
   /// hit or closest hit programs that require those attributes.  No assumptions about the
   /// precise invocation time should be made.
   ///
   /// The default attribute program will provide the following attributes:
   ///   float2 barycentrics;
   ///   unsigned int instanceid;
   ///
   /// Names are case sensitive and types must match.  To use the attributes, declare the following
   ///    rtDeclareVariable( float2, barycentrics, attribute barycentrics, );
   ///    rtDeclareVariable( unsigned int, instanceid, attribute instanceid, );
   ///
   /// If you provide an attribute program, the following device side functions will be available.
   ///    float2 rtGetTriangleBarycentrics();
   ///    unsigned int rtGetInstanceId();
   ///
   /// These device functions are only available in attribute programs.
   ///
   /// @param[in]   geometrytriangles  The geometrytriangles node for which to set the attribute program
   /// @param[in]   program            A handle to the attribute program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesSetAttributeProgram was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesGetAttributeProgram,
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramCreateFromPTXString,
   /// @ref rtGetTriangleBarycentrics,
   /// @ref rtGetInstanceId
   ///
   pub fn rtGeometryTrianglesSetAttributeProgram(
      geometrytriangles: RTgeometrytriangles,
      program: RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Gets the attribute program of a GeometryTriangles object
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesGetAttributeProgram gets the attribute \a program of a given
   /// \a geometrytriangles object.  If no program has been set, 0 is returned.
   ///
   /// @param[in]   geometrytriangles  The geometrytriangles node for which to set the attribute program
   /// @param[out]  program            A pointer to a handle to the attribute program
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetAttributeProgram was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesDeclareVariable,
   /// @ref rtGeometryTrianglesSetAttributeProgram,
   /// @ref rtProgramCreateFromPTXFile,
   /// @ref rtProgramCreateFromPTXString
   ///
   pub fn rtGeometryTrianglesGetAttributeProgram(
      geometrytriangles: RTgeometrytriangles,
      program: *mut RTprogram,
   ) -> RtResult;
}
extern "C" {
   /// @brief Declares a geometry variable for a GeometryTriangles object
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesDeclareVariable declares a \a variable attribute of a \a geometrytriangles object with
   /// a specified \a name.
   ///
   /// @param[in]   geometrytriangles     A geometry node
   /// @param[in]   name                  The name of the variable
   /// @param[out]  v                     A pointer to a handle to the variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesDeclareVariable was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesGetVariable,
   /// @ref rtGeometryTrianglesGetVariableCount,
   /// @ref rtGeometryTrianglesQueryVariable,
   /// @ref rtGeometryTrianglesRemoveVariable
   ///
   pub fn rtGeometryTrianglesDeclareVariable(
      geometrytriangles: RTgeometrytriangles,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Queries a variable attached to a GeometryTriangles object
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesQueryVariable gets a variable with a given \a name from
   /// a \a geometrytriangles object.
   ///
   /// @param[in]   geometrytriangles    A geometrytriangles object
   /// @param[in]   name                 Thee name of the variable
   /// @param[out]  v                    A pointer to a handle to the variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesQueryVariable was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesGetVariable,
   /// @ref rtGeometryTrianglesGetVariableCount,
   /// @ref rtGeometryTrianglesQueryVariable,
   /// @ref rtGeometryTrianglesRemoveVariable
   ///
   pub fn rtGeometryTrianglesQueryVariable(
      geometrytriangles: RTgeometrytriangles,
      name: *const ::std::os::raw::c_char,
      v: *mut RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Removes a variable from GeometryTriangles object
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesRemoveVariable removes a variable from
   /// a \a geometrytriangles object.
   ///
   /// @param[in]   geometrytriangles     A geometrytriangles object
   /// @param[in]   v                     A pointer to a handle to the variable
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesRemoveVariable was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesDeclareVariable,
   /// @ref rtGeometryTrianglesGetVariable,
   /// @ref rtGeometryTrianglesGetVariableCount,
   /// @ref rtGeometryTrianglesQueryVariable
   ///
   pub fn rtGeometryTrianglesRemoveVariable(
      geometrytriangles: RTgeometrytriangles,
      v: RTvariable,
   ) -> RtResult;
}
extern "C" {
   /// @brief Get the number of variables attached to a GeometryTriangles object
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesGetVariableCount returns a \a count of the number
   /// of variables attached to a \a geometrytriangles object.
   ///
   /// @param[in]   geometrytriangles   A geometrytriangles node
   /// @param[out]  v                   A pointer to an unsigned int
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetVariableCount was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesDeclareVariable,
   /// @ref rtGeometryTrianglesGetVariable,
   /// @ref rtGeometryTrianglesQueryVariable,
   /// @ref rtGeometryTrianglesRemoveVariable
   ///
   pub fn rtGeometryTrianglesGetVariableCount(
      geometrytriangles: RTgeometrytriangles,
      count: *mut ::std::os::raw::c_uint,
   ) -> RtResult;
}
extern "C" {
   /// @brief Get a variable attached to a GeometryTriangles object at a specified index.
   ///
   /// @ingroup GeometryTriangles
   ///
   /// <B>Description</B>
   ///
   /// @ref rtGeometryTrianglesGetVariable returns the variable attached at a given
   /// index to the specified GeometryTriangles object.
   ///
   /// @param[in]   geometrytriangles   A geometry node
   /// @param[in]   index               The index of the variable
   /// @param[out]  v                   A pointer to a variable handle
   ///
   /// <B>Return values</B>
   ///
   /// Relevant return values:
   /// - @ref RT_SUCCESS
   /// - @ref RT_ERROR_INVALID_CONTEXT
   /// - @ref RT_ERROR_INVALID_VALUE
   ///
   /// <B>History</B>
   ///
   /// @ref rtGeometryTrianglesGetVariable was introduced in OptiX 6.0.
   ///
   /// <B>See also</B>
   /// @ref rtGeometryTrianglesDeclareVariable,
   /// @ref rtGeometryTrianglesGetVariableCount,
   /// @ref rtGeometryTrianglesQueryVariable,
   /// @ref rtGeometryTrianglesRemoveVariable
   ///
   pub fn rtGeometryTrianglesGetVariable(
      geometrytriangles: RTgeometrytriangles,
      index: ::std::os::raw::c_uint,
      v: *mut RTvariable,
   ) -> RtResult;
}
