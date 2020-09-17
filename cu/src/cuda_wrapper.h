#include <cuda.h>
#include <nvrtc.h>

enum TextureReadFlags {
    None = 0,
    ReadAsInteger = CU_TRSF_READ_AS_INTEGER,
    NormalizedCoordinates = CU_TRSF_NORMALIZED_COORDINATES,
    Srgb = CU_TRSF_SRGB,
    DisableTrilinearOptimization = CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION
};