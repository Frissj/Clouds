add_plugin(HSTRCloud)

target_sources(HSTRCloud PRIVATE
    CloudFormat.cpp
    CloudFormat.h
    CloudLibrary.cpp
    CloudLibrary.h
    CloudResidency.cpp
    CloudResidency.h
    CloudSea.cpp
    CloudSea.h
    HSTRCloud.cpp
    HSTRCloud.cs.slang
    HSTRCloud.h
    HSTRCloudTypes.slang
)

# Cloud library packages are deflated; HSTRCloudCompiler builds them offline, so the renderer never reads OpenVDB.
target_link_libraries(HSTRCloud PRIVATE zlib)

target_copy_shaders(HSTRCloud RenderPasses/HSTRCloud)
target_source_group(HSTRCloud "RenderPasses")
