#include "GpuTrace.h"
#include "Core/Error.h"
#include "Utils/StringUtils.h"
#include <d3d12.h>
#include <NGFX_GPUTrace_D3D12.h>

namespace Mogwai
{
namespace
{
uint32_t sTraces = 0; // Trace files this process has produced; the next one's artifact index.

void check(NGFX_Result result, const char* call)
{
    if (result != NGFX_Result_Success)
        FALCOR_THROW("{} failed ({}). Is Mogwai running under ngfx --activity \"GPU Trace Profiler\" --start-with-ngfx-sdk?", call, (int)result);
}
} // namespace

void gpuTraceStart(void* commandQueue)
{
    bool injected = false;
    check(NGFX_GPUTrace_IsInjected(&injected), "NGFX_GPUTrace_IsInjected");
    if (!injected)
        FALCOR_THROW("gpuTraceStart: GPU Trace is not injected; launch Mogwai through ngfx --activity \"GPU Trace Profiler\"");
    NGFX_GPUTrace_InitializeActivity_D3D12_Params init = {NGFX_GPUTrace_InitializeActivity_D3D12_Params_VER};
    check(NGFX_GPUTrace_InitializeActivity_D3D12(&init), "NGFX_GPUTrace_InitializeActivity_D3D12");
    NGFX_GPUTrace_GetStatus_Params status = {NGFX_GPUTrace_GetStatus_Params_VER};
    check(NGFX_GPUTrace_GetStatus(&status), "NGFX_GPUTrace_GetStatus");
    if (status.status == NGFX_GPUTrace_Status_Inactive)
    {
        NGFX_GPUTrace_ActivateTrace_D3D12_Params activate = {NGFX_GPUTrace_ActivateTrace_D3D12_Params_VER};
        activate.commandQueue = static_cast<ID3D12CommandQueue*>(commandQueue);
        check(NGFX_GPUTrace_ActivateTrace_D3D12(&activate), "NGFX_GPUTrace_ActivateTrace_D3D12");
    }
    NGFX_GPUTrace_StartTrace_D3D12_Params start = {NGFX_GPUTrace_StartTrace_D3D12_Params_VER};
    check(NGFX_GPUTrace_StartTrace_D3D12(&start), "NGFX_GPUTrace_StartTrace_D3D12");
}

std::string gpuTraceStop(void* commandQueue)
{
    NGFX_GPUTrace_StopTrace_D3D12_Params stop = {NGFX_GPUTrace_StopTrace_D3D12_Params_VER};
    // ImmediateCollection: with None the trace drains "as future work is submitted", and this function blocks below without
    // submitting any, so it never completed (ngfx4: host stuck at "Waiting for generated GPU Trace report", wait timed out).
    // The caller has already drained the queue (submit(true)), so no queue waits on unscheduled work.
    stop.flags = NGFX_GPUTrace_StopTraceFlag_ImmediateCollection;
    stop.commandQueue = static_cast<ID3D12CommandQueue*>(commandQueue);
    check(NGFX_GPUTrace_StopTrace_D3D12(&stop), "NGFX_GPUTrace_StopTrace_D3D12");
    wchar_t path[1024] = {};
    NGFX_WaitForArtifactFilePath_Params wait = {NGFX_WaitForArtifactFilePath_Params_VER};
    wait.artifactIndex = sTraces++;
    wait.timeoutMs = 60000; // A one-frame trace completes in seconds; anything longer is a stall, not a slow host.
    wait.filePath = path;
    wait.filePathCapacity = 1024;
    check(NGFX_GPUTrace_WaitForTraceFilePath(&wait), "NGFX_GPUTrace_WaitForTraceFilePath");
    return Falcor::wstring_2_string(path);
}
} // namespace Mogwai
