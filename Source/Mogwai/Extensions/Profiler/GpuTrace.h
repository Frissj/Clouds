#pragma once
#include <string>

namespace Mogwai
{
// Nsight Graphics GPU Trace bracketed by a script (ngfx --activity "GPU Trace Profiler" --start-with-ngfx-sdk --stop-with-ngfx-sdk).
// Headless Mogwai never presents, so ngfx's own frame triggers never fire; scripts start and stop around the frames they want.
// In-process injection (NGFX_GPUTrace_Inject_D3D12, no ngfx host) was tried and hangs right after device creation: it needs the
// host. commandQueue is the ID3D12CommandQueue the frames are submitted on.
void gpuTraceStart(void* commandQueue);
// Stops once the queue's work is done and waits for the trace file; returns its path.
std::string gpuTraceStop(void* commandQueue);
} // namespace Mogwai
