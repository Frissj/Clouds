add_falcor_executable(HSTRCloudCompiler)

target_sources(HSTRCloudCompiler PRIVATE
    HSTRCloudCompiler.cpp
    ${CMAKE_SOURCE_DIR}/Source/RenderPasses/HSTRCloud/CloudFormat.cpp
    ${CMAKE_SOURCE_DIR}/Source/RenderPasses/HSTRCloud/CloudFormat.h
)

target_include_directories(HSTRCloudCompiler PRIVATE ${CMAKE_SOURCE_DIR}/Source)

# OpenVDB is only needed offline: the renderer reads the compiled package. OpenVDB names blosc without a path.
target_link_libraries(HSTRCloudCompiler PRIVATE OpenVDB OpenEXR zlib directstorage)
add_dependencies(HSTRCloudCompiler deploy_directstorage)
target_link_directories(HSTRCloudCompiler PRIVATE $<IF:$<CONFIG:Debug>,${CMAKE_SOURCE_DIR}/external/packman/deps/debug/lib,${CMAKE_SOURCE_DIR}/external/packman/deps/lib>)
