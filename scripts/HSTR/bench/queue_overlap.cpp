// Standalone D3D12 test: do a DIRECT queue's and a COMPUTE queue's dispatches run at the same time on this GPU / driver?
// Outside Falcor and slang-gfx, so a "no" here is the driver / hardware, a "yes" is something in how Falcor drives the GPU.
// Build (VS developer prompt): cl /O2 /EHsc queue_overlap.cpp d3d12.lib dxgi.lib d3dcompiler.lib
// Run: queue_overlap.exe [groups] [iters]  - prints A alone, B alone, A + B together (GPU ms from each queue's timestamps).
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;

#ifdef AGILITY
// /DAGILITY: Falcor's bundled D3D12 runtime (FALCOR_D3D12_AGILITY_SDK_VERSION 4, D3D12Core.dll in .\D3D12\ beside the exe) instead
// of the OS's - Mogwai loads it, the plain build does not.
extern "C" __declspec(dllexport) extern const unsigned int D3D12SDKVersion = 4;
extern "C" __declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
#endif
#ifdef WITH_DSTORAGE
// /DWITH_DSTORAGE: a DirectStorage queue on the device first, as HSTRCloud's CloudPayloadPool creates one (GPU decompression).
#include <dstorage.h>
#pragma comment(lib, "dstorage.lib")
#endif
#ifdef WITH_CUDA
// /DWITH_CUDA: a CUDA context on the same GPU first, as Falcor's contexts create one (Device::initCudaDevice).
#include <cuda.h>
#pragma comment(lib, "cuda.lib")
#endif

static const char kSource[] = R"hlsl(
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0)"
cbuffer C : register(b0) { uint gIters; };
RWStructuredBuffer<uint> gOut : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float x = 1.f + float(id.x) * 1e-7f;
    [loop]
    for (uint i = 0; i < gIters; ++i)
        x = x * 0.99999f + 1e-5f;
    if (x == 1234.5f)
        gOut[0] = 1u;
}
)hlsl";

#define CHECK(x)                                                                 \
    do                                                                           \
    {                                                                            \
        HRESULT hr_ = (x);                                                       \
        if (FAILED(hr_))                                                         \
        {                                                                        \
            std::printf("FAILED 0x%08lx: %s\n", (unsigned long)hr_, #x);         \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

struct Queue
{
    D3D12_COMMAND_LIST_TYPE type;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12QueryHeap> queries;
    ComPtr<ID3D12Resource> readback;
    ComPtr<ID3D12Fence> fence;
    UINT64 fenceValue = 0;
    UINT64 frequency = 0;
};

// A CUDA context on the GPU first, through the driver's nvcuda.dll (no toolkit needed): Mogwai has cudart64_110.dll and
// nvcuda64.dll loaded (qoverlap3), a plain D3D12 program does not. Argument 5 = 1.
static bool createCudaContext()
{
    HMODULE cuda = LoadLibraryA("nvcuda.dll");
    if (!cuda)
        return false;
    typedef int (*CuInit)(unsigned int);
    typedef int (*CuDeviceGet)(int*, int);
    typedef int (*CuPrimaryCtxRetain)(void**, int);
    typedef int (*CuCtxSetCurrent)(void*);
    auto cuInit = (CuInit)GetProcAddress(cuda, "cuInit");
    auto cuDeviceGet = (CuDeviceGet)GetProcAddress(cuda, "cuDeviceGet");
    auto cuDevicePrimaryCtxRetain = (CuPrimaryCtxRetain)GetProcAddress(cuda, "cuDevicePrimaryCtxRetain");
    auto cuCtxSetCurrent = (CuCtxSetCurrent)GetProcAddress(cuda, "cuCtxSetCurrent");
    int device = 0;
    void* context = nullptr;
    return cuInit && cuDeviceGet && cuDevicePrimaryCtxRetain && cuCtxSetCurrent && cuInit(0) == 0 && cuDeviceGet(&device, 0) == 0 &&
           cuDevicePrimaryCtxRetain(&context, device) == 0 && cuCtxSetCurrent(context) == 0;
}

int main(int argc, char** argv)
{
    if (argc > 5 && std::atoi(argv[5]) != 0)
        std::printf("CUDA context: %s\n", createCudaContext() ? "created" : "FAILED");
    const UINT groups = argc > 1 ? UINT(std::atoi(argv[1])) : 232; // 58 SMs x 4 groups of 2 warps: 8 warps an SM, room for both.
    const UINT iters = argc > 2 ? UINT(std::atoi(argv[2])) : 400000;
    // The DIRECT queue's work split into this many dispatches with a UAV barrier between them (Falcor's frame: ~250 dispatches,
    // most behind a barrier). 1: one dispatch.
    const UINT chain = argc > 3 ? UINT(std::atoi(argv[3])) : 1;
    // 1: a 256-byte CopyBufferRegion between the chained dispatches as well (Falcor's frame mixes copies / uploads in).
    const bool copies = argc > 4 && std::atoi(argv[4]) != 0;

#ifdef WITH_CUDA
    CUdevice cudaDevice;
    CUcontext cudaContext;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&cudaDevice, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice) != CUDA_SUCCESS || cuCtxSetCurrent(cudaContext) != CUDA_SUCCESS)
    {
        std::printf("CUDA init failed\n");
        return 1;
    }
    std::printf("CUDA context created\n");
#endif
    ComPtr<IDXGIFactory6> factory;
    CHECK(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter;
    CHECK(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
    DXGI_ADAPTER_DESC1 adapterDesc;
    adapter->GetDesc1(&adapterDesc);
    std::printf("adapter: %ls\n", adapterDesc.Description);
    ComPtr<ID3D12Device> device;
    CHECK(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
    {
        // Which D3D12 runtime this device runs on (the Agility SDK's .\D3D12\D3D12Core.dll or the OS's).
        char path[MAX_PATH] = "not loaded";
        if (HMODULE core = GetModuleHandleA("D3D12Core.dll"))
            GetModuleFileNameA(core, path, MAX_PATH);
        std::printf("D3D12Core: %s\n", path);
    }
#ifdef VRAM_GB
    // /DVRAM_GB=n: n GB of committed default-heap buffers made resident first (Mogwai holds GBs of atlases and pages; residency
    // management under pressure can serialize submissions).
    std::vector<ComPtr<ID3D12Resource>> ballast;
    {
        D3D12_HEAP_PROPERTIES heap = {D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = 256ull << 20;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        for (int i = 0; i < VRAM_GB * 4; ++i)
        {
            ComPtr<ID3D12Resource> buffer;
            if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                       IID_PPV_ARGS(&buffer))))
                break;
            ballast.push_back(buffer);
        }
        std::printf("ballast: %zu MB\n", ballast.size() * 256);
    }
#endif
#ifdef STABLE
    // /DSTABLE: clocks locked for profiling (needs Windows developer mode) - in Mogwai each kernel alone runs 7% slower.
    std::printf("SetStablePowerState: 0x%08lx\n", (unsigned long)device->SetStablePowerState(TRUE));
#endif
#ifdef WITH_DSTORAGE
    // As CloudPayloadPool: a file-source queue with GPU decompression on this device.
    ComPtr<IDStorageFactory> storage;
    CHECK(DStorageGetFactory(IID_PPV_ARGS(&storage)));
    DSTORAGE_QUEUE_DESC storageDesc = {};
    storageDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    storageDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
    storageDesc.Priority = DSTORAGE_PRIORITY_NORMAL;
    storageDesc.Name = "overlap test";
    storageDesc.Device = device.Get();
    ComPtr<IDStorageQueue> storageQueue;
    CHECK(storage->CreateQueue(&storageDesc, IID_PPV_ARGS(&storageQueue)));
    std::printf("DirectStorage queue created\n");
#endif

    ComPtr<ID3DBlob> code, errors;
    if (FAILED(D3DCompile(kSource, sizeof(kSource) - 1, "probe", nullptr, nullptr, "main", "cs_5_1", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
                          &code, &errors)))
    {
        std::printf("%s\n", errors ? (const char*)errors->GetBufferPointer() : "compile failed");
        return 1;
    }
    ComPtr<ID3D12RootSignature> root;
    CHECK(device->CreateRootSignature(0, code->GetBufferPointer(), code->GetBufferSize(), IID_PPV_ARGS(&root)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = root.Get();
    psoDesc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    ComPtr<ID3D12PipelineState> pso;
    CHECK(device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)));

    D3D12_HEAP_PROPERTIES defaultHeap = {D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Width = 256;
    bufferDesc.Height = bufferDesc.DepthOrArraySize = bufferDesc.MipLevels = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> out[3]; // [2]: the copies' destination.
    for (auto& o : out)
        CHECK(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              IID_PPV_ARGS(&o)));

    // Argument 6: idle DIRECT queues, as Mogwai's process already owns slang-gfx's queue(s) before the test's. Argument 7 = 1:
    // create them after the test's queues instead of before.
    const int idleCount = argc > 6 ? std::atoi(argv[6]) : 0;
    const bool idleAfter = argc > 7 && std::atoi(argv[7]) != 0;
    std::vector<ComPtr<ID3D12CommandQueue>> idle;
    auto createIdle = [&]()
    {
        for (int i = 0; i < idleCount; ++i)
        {
            D3D12_COMMAND_QUEUE_DESC idleDesc = {D3D12_COMMAND_LIST_TYPE_DIRECT};
            ComPtr<ID3D12CommandQueue> q;
            CHECK(device->CreateCommandQueue(&idleDesc, IID_PPV_ARGS(&q)));
            idle.push_back(q);
        }
        if (idleCount)
            std::printf("idle DIRECT queues created %s the test's: %d\n", idleAfter ? "after" : "before", idleCount);
    };
    if (!idleAfter)
        createIdle();
    // Argument 8 = 1: release the idle queues again before the test's are created (does a destroyed first queue keep the slot?).
    if (argc > 8 && std::atoi(argv[8]) != 0)
    {
        idle.clear();
        std::printf("idle queues released\n");
    }
    // Argument 9 = 1: the second test queue is a DIRECT queue too (does the first DIRECT queue overlap another DIRECT queue?).
    const bool twoDirect = argc > 9 && std::atoi(argv[9]) != 0;
    Queue queues[2] = {{D3D12_COMMAND_LIST_TYPE_DIRECT}, {twoDirect ? D3D12_COMMAND_LIST_TYPE_DIRECT : D3D12_COMMAND_LIST_TYPE_COMPUTE}};
    if (twoDirect)
        std::printf("second test queue: DIRECT\n");
    D3D12_HEAP_PROPERTIES readbackHeap = {D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC readbackDesc = bufferDesc;
    readbackDesc.Width = 2 * sizeof(UINT64);
    readbackDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    for (Queue& q : queues)
    {
        D3D12_COMMAND_QUEUE_DESC queueDesc = {q.type};
        CHECK(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&q.queue)));
        CHECK(device->CreateCommandAllocator(q.type, IID_PPV_ARGS(&q.allocator)));
        CHECK(device->CreateCommandList(0, q.type, q.allocator.Get(), pso.Get(), IID_PPV_ARGS(&q.list)));
        q.list->Close();
        D3D12_QUERY_HEAP_DESC queryDesc = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2};
        CHECK(device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&q.queries)));
        CHECK(device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &readbackDesc, D3D12_RESOURCE_STATE_COPY_DEST,
                                              nullptr, IID_PPV_ARGS(&q.readback)));
        CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&q.fence)));
        CHECK(q.queue->GetTimestampFrequency(&q.frequency));
    }
    if (idleAfter)
        createIdle();

    auto record = [&](Queue& q, ID3D12Resource* target)
    {
        q.allocator->Reset();
        q.list->Reset(q.allocator.Get(), pso.Get());
        q.list->EndQuery(q.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        q.list->SetComputeRootSignature(root.Get());
        q.list->SetComputeRootUnorderedAccessView(1, target->GetGPUVirtualAddress());
        const UINT links = q.type == D3D12_COMMAND_LIST_TYPE_DIRECT ? chain : 1u;
        q.list->SetComputeRoot32BitConstant(0, iters / links, 0);
        for (UINT k = 0; k < links; ++k)
        {
            if (k > 0)
            {
                D3D12_RESOURCE_BARRIER barrier = {D3D12_RESOURCE_BARRIER_TYPE_UAV};
                barrier.UAV.pResource = target;
                q.list->ResourceBarrier(1, &barrier);
                if (copies)
                {
                    // Buffers promote implicitly from COMMON; the copy reads target, writes out[2].
                    q.list->CopyBufferRegion(out[2].Get(), 0, target, 0, 256);
                }
            }
            q.list->Dispatch(groups, 1, 1);
        }
        q.list->EndQuery(q.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        q.list->ResolveQueryData(q.queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, q.readback.Get(), 0);
        q.list->Close();
    };
    auto wait = [](Queue& q)
    {
        q.queue->Signal(q.fence.Get(), ++q.fenceValue);
        while (q.fence->GetCompletedValue() < q.fenceValue)
            ;
    };
    auto stamps = [](Queue& q, UINT64& begin, UINT64& end)
    {
        UINT64* p = nullptr;
        D3D12_RANGE range = {0, 2 * sizeof(UINT64)};
        q.readback->Map(0, &range, (void**)&p);
        begin = p[0];
        end = p[1];
        q.readback->Unmap(0, nullptr);
    };
    auto ms = [](const Queue& q, UINT64 ticks) { return 1000.0 * double(ticks) / double(q.frequency); };

    // Gate both queues on one fence so they start together; each queue's timestamps then say when it ran.
    ComPtr<ID3D12Fence> gate;
    CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate)));
    UINT64 gateValue = 0;
    for (int run = 0; run < 4; ++run) // Run 0 warms up clocks and caches.
    {
        double alone[2];
        for (int i = 0; i < 2; ++i)
        {
            record(queues[i], out[i].Get());
            ID3D12CommandList* lists[] = {queues[i].list.Get()};
            queues[i].queue->ExecuteCommandLists(1, lists);
            wait(queues[i]);
            UINT64 b, e;
            stamps(queues[i], b, e);
            alone[i] = ms(queues[i], e - b);
        }
        ++gateValue;
        for (int i = 0; i < 2; ++i)
        {
            record(queues[i], out[i].Get());
            queues[i].queue->Wait(gate.Get(), gateValue);
            ID3D12CommandList* lists[] = {queues[i].list.Get()};
            queues[i].queue->ExecuteCommandLists(1, lists);
        }
        gate->Signal(gateValue);
        wait(queues[0]);
        wait(queues[1]);
        UINT64 b0, e0, b1, e1;
        stamps(queues[0], b0, e0);
        stamps(queues[1], b1, e1);
        // Both queues' timestamps share the GPU clock on one adapter (same frequency); the union is the together time.
        const UINT64 begin = b0 < b1 ? b0 : b1, end = e0 > e1 ? e0 : e1;
        std::printf("run %d: direct alone %.3f ms, compute alone %.3f ms | together: direct %.3f, compute %.3f, union %.3f ms "
                    "(sum of alone %.3f, max %.3f)\n",
                    run, alone[0], alone[1], ms(queues[0], e0 - b0), ms(queues[1], e1 - b1), ms(queues[0], end - begin),
                    alone[0] + alone[1], alone[0] > alone[1] ? alone[0] : alone[1]);
    }
    // NVIDIA / capture modules in this process (Mogwai has nvspcap64.dll, nvcuda64.dll).
    HMODULE modules[1024];
    DWORD bytes = 0;
    if (EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &bytes))
        for (DWORD i = 0; i < bytes / sizeof(HMODULE) && i < 1024; ++i)
        {
            char path[MAX_PATH] = "";
            GetModuleFileNameA(modules[i], path, MAX_PATH);
            if (std::strstr(path, "nvspcap") || std::strstr(path, "nvcuda") || std::strstr(path, "cudart"))
                std::printf("module: %s\n", path);
        }
    return 0;
}
