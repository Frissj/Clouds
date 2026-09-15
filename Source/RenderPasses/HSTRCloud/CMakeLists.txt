add_plugin(HSTRCloud)

target_sources(HSTRCloud PRIVATE
    CloudFormat.cpp
    CloudFormat.h
    CloudLibrary.cpp
    CloudLibrary.h
    CloudPayloadPool.cpp
    CloudPayloadPool.h
    CloudResidency.cpp
    CloudResidency.h
    CloudSea.cpp
    CloudSea.h
    HSTRCloud.cpp
    HSTRCloud.cs.slang
    HSTRCloud.h
    HSTRCloudTypes.slang
)

# Cloud library packages are GDeflate pages that DirectStorage loads (GPU decompression, RTX IO, where supported);
# HSTRCloudCompiler builds them offline, so the renderer never reads OpenVDB.
if(NOT FALCOR_HAS_DIRECTSTORAGE)
    message(FATAL_ERROR "HSTRCloud needs DirectStorage (downloaded into external/ when configuring on Windows).")
endif()
target_link_libraries(HSTRCloud PRIVATE zlib directstorage)
add_dependencies(HSTRCloud deploy_directstorage)

target_copy_shaders(HSTRCloud RenderPasses/HSTRCloud)
target_source_group(HSTRCloud "RenderPasses")
