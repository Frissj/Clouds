add_falcor_executable(Mogwai)

target_sources(Mogwai PRIVATE
    AppData.cpp
    AppData.h
    Mogwai.cpp
    Mogwai.h
    MogwaiScripting.cpp
    MogwaiSettings.cpp
    MogwaiSettings.h

    Extensions/Capture/CaptureTrigger.cpp
    Extensions/Capture/CaptureTrigger.h
    Extensions/Capture/FrameCapture.cpp
    Extensions/Capture/FrameCapture.h
    Extensions/Profiler/TimingCapture.cpp
    Extensions/Profiler/TimingCapture.h
    Extensions/Profiler/GpuTrace.cpp
    Extensions/Profiler/GpuTrace.h
)

# Nsight Graphics SDK (header only): lets scripts start/stop a GPU Trace at chosen frames (see GpuTrace.h).
set(NGFX_SDK_INCLUDE "C:/Program Files/NVIDIA Corporation/Nsight Graphics 2026.3.1/SDKs/NsightGraphicsSDK/0.9.2/include" CACHE PATH "Nsight Graphics SDK include directory")
target_include_directories(Mogwai PRIVATE ${NGFX_SDK_INCLUDE})

target_link_libraries(Mogwai PRIVATE args)

target_source_group(Mogwai "/")
