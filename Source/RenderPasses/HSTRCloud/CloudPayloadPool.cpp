/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "CloudPayloadPool.h"
#include "Core/API/NativeHandleTraits.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <dstorage.h>

namespace hstrcloud
{
namespace
{
constexpr uint32_t kTickets = 1024; ///< Outstanding DirectStorage loads (the residency queues at most 256 pages).

template<typename T>
void release(T*& p)
{
    if (p)
        p->Release();
    p = nullptr;
}
} // namespace

CloudPayloadPool::CloudPayloadPool(ref<Device> pDevice, const std::vector<std::filesystem::path>& files, uint32_t poolMB, bool useDirectStorage)
    : mpDevice(pDevice)
{
    const uint32_t words = std::max(1u, poolMB) * (1024 * 1024 / 4);
    mpBuffer = mpDevice->createStructuredBuffer(sizeof(uint32_t), words, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, nullptr, false);
    mFreeRanges[0] = words;

    if (!useDirectStorage)
    {
        logInfo("HSTRCloud: cloud page payloads decompress on the CPU (GDeflate) and upload: DirectStorage is disabled.");
        return;
    }
#if FALCOR_HAS_D3D12
    if (mpDevice->getType() != Device::Type::D3D12)
    {
        logInfo("HSTRCloud: cloud page payloads decompress on the CPU (GDeflate) and upload: DirectStorage needs D3D12.");
        return;
    }
    auto fail = [&](const char* what, HRESULT result)
    {
        logWarning("HSTRCloud: DirectStorage {} failed (0x{:08X}); cloud page payloads decompress on the CPU (GDeflate) and upload.", what, uint32_t(result));
        release(mpStatus);
        release(mpQueue);
        for (IDStorageFile*& file : mFiles)
            release(file);
        mFiles.clear();
        release(mpFactory);
    };
    HRESULT result = DStorageGetFactory(IID_PPV_ARGS(&mpFactory));
    if (FAILED(result))
    {
        fail("factory", result);
        return;
    }
    for (const auto& path : files)
    {
        IDStorageFile* file = nullptr;
        result = mpFactory->OpenFile(path.c_str(), IID_PPV_ARGS(&file));
        if (FAILED(result))
        {
            fail("opening a package", result);
            return;
        }
        mFiles.push_back(file);
    }
    DSTORAGE_QUEUE_DESC desc = {};
    desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    desc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
    desc.Priority = DSTORAGE_PRIORITY_NORMAL;
    desc.Name = "HSTRCloud pages";
    desc.Device = mpDevice->getNativeHandle(0).as<ID3D12Device*>();
    result = mpFactory->CreateQueue(&desc, IID_PPV_ARGS(&mpQueue));
    if (FAILED(result))
    {
        fail("queue", result);
        return;
    }
    result = mpFactory->CreateStatusArray(kTickets, "HSTRCloud page status", IID_PPV_ARGS(&mpStatus));
    if (FAILED(result))
    {
        fail("status array", result);
        return;
    }
    for (uint32_t t = kTickets; t-- > 0;)
        mFreeTickets.push_back(t);

    // The path the queue's runtime chose for GDeflate on this device.
    DSTORAGE_COMPRESSION_SUPPORT support = DSTORAGE_COMPRESSION_SUPPORT_NONE;
    IDStorageQueue2* queue2 = nullptr;
    if (SUCCEEDED(mpQueue->QueryInterface(IID_PPV_ARGS(&queue2))))
    {
        support = queue2->GetCompressionSupport(DSTORAGE_COMPRESSION_FORMAT_GDEFLATE);
        queue2->Release();
    }
    const char* path = (support & DSTORAGE_COMPRESSION_SUPPORT_GPU_OPTIMIZED) ? "on the GPU with the vendor-optimized path (RTX IO on NVIDIA GPUs)"
                       : (support & DSTORAGE_COMPRESSION_SUPPORT_GPU_FALLBACK) ? "on the GPU with DirectStorage's generic shader (no vendor-optimized path)"
                       : (support & DSTORAGE_COMPRESSION_SUPPORT_CPU_FALLBACK) ? "on the CPU inside DirectStorage (no GPU decompression on this device)"
                                                                               : "by a path DirectStorage did not report";
    logInfo("HSTRCloud: cloud page payloads load with DirectStorage; GDeflate decompresses {} (support flags 0x{:X}).", path, uint32_t(support));
#else
    logInfo("HSTRCloud: cloud page payloads decompress on the CPU (GDeflate) and upload: DirectStorage needs D3D12.");
#endif
}

CloudPayloadPool::~CloudPayloadPool()
{
    if (mpQueue)
        mpQueue->Close();
    release(mpStatus);
    release(mpQueue);
    for (IDStorageFile*& file : mFiles)
        release(file);
    release(mpFactory);
}

uint32_t CloudPayloadPool::allocate(uint32_t bytes)
{
    const uint32_t words = (bytes + 3) / 4;
    for (auto it = mFreeRanges.begin(); it != mFreeRanges.end(); ++it)
    {
        if (it->second < words)
            continue;
        const uint32_t word = it->first;
        const uint32_t rest = it->second - words;
        mFreeRanges.erase(it);
        if (rest > 0)
            mFreeRanges[word + words] = rest;
        mUsedWords += words;
        return word;
    }
    return kNone;
}

void CloudPayloadPool::free(uint32_t word, uint32_t bytes)
{
    if (word == kNone || bytes == 0)
        return;
    uint32_t words = (bytes + 3) / 4;
    mUsedWords -= words;
    auto next = mFreeRanges.lower_bound(word);
    if (next != mFreeRanges.end() && word + words == next->first)
    {
        words += next->second;
        next = mFreeRanges.erase(next);
    }
    if (next != mFreeRanges.begin())
    {
        auto previous = std::prev(next);
        if (previous->first + previous->second == word)
        {
            previous->second += words;
            return;
        }
    }
    mFreeRanges[word] = words;
}

uint32_t CloudPayloadPool::beginLoad(uint32_t file, const BlobRef& blob, uint32_t word)
{
    if (!mpQueue || file >= mFiles.size())
        return kNone;
    uint32_t ticket;
    {
        std::lock_guard lock(mTicketMutex);
        if (mFreeTickets.empty())
            return kNone;
        ticket = mFreeTickets.back();
        mFreeTickets.pop_back();
    }
    DSTORAGE_REQUEST request = {};
    request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
    request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_GDEFLATE;
    request.Source.File.Source = mFiles[file];
    request.Source.File.Offset = blob.offset;
    request.Source.File.Size = blob.compressed;
    request.UncompressedSize = blob.raw;
    request.Destination.Buffer.Resource = mpBuffer->getNativeHandle().as<ID3D12Resource*>();
    request.Destination.Buffer.Offset = uint64_t(word) * 4;
    request.Destination.Buffer.Size = blob.raw;
    mpQueue->EnqueueRequest(&request);
    mpQueue->EnqueueStatus(mpStatus, ticket);
    mpQueue->Submit();
    return ticket;
}

int CloudPayloadPool::pollLoad(uint32_t ticket)
{
    if (!mpStatus->IsComplete(ticket))
        return 0;
    const bool ok = SUCCEEDED(mpStatus->GetHResult(ticket));
    std::lock_guard lock(mTicketMutex);
    mFreeTickets.push_back(ticket);
    return ok ? 1 : -1;
}

void CloudPayloadPool::upload(uint32_t word, const std::vector<uint8_t>& bytes)
{
    if (!bytes.empty())
        mpBuffer->setBlob(bytes.data(), uint64_t(word) * 4, bytes.size());
}
} // namespace hstrcloud
