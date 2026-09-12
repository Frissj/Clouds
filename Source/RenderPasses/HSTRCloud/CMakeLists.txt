add_plugin(HSTRCloud)

target_sources(HSTRCloud PRIVATE
    HSTRCloud.cpp
    HSTRCloud.cs.slang
    HSTRCloud.h
    HSTRCloudTypes.slang
)

target_copy_shaders(HSTRCloud RenderPasses/HSTRCloud)
target_source_group(HSTRCloud "RenderPasses")
