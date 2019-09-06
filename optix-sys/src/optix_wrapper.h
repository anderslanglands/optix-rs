#include <optix_host.h>

static const size_t OptixSbtRecordHeaderSize = OPTIX_SBT_RECORD_HEADER_SIZE;
static const size_t OptixSbtRecordAlignment = OPTIX_SBT_RECORD_ALIGNMENT;

/**
 * <div rustbindgen replaces="OptixGeometryFlags"></div>
 */
enum GeometryFlags {
    None = OPTIX_GEOMETRY_FLAG_NONE,
    DisableAnyHit = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
    RequireSingleAnyHitCall = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
};