/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "CloudFormat.h"

#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct IDStorageFactory;
struct IDStorageFile;
struct IDStorageQueue;
struct IDStorageStatusArray;

namespace hstrcloud
{
/** GPU memory of the packed coefficients of resident pages, and their loading.
 *
 * With DirectStorage (D3D12), a page payload's GDeflate blob goes from the package file straight into its range of the pool
 * buffer; DirectStorage decompresses it on the GPU where supported (RTX IO on NVIDIA GPUs), else on the CPU. Without it (another
 * API, or DirectStorage unavailable), the caller decompresses on its I/O threads and uploads the bytes here.
 */
class CloudPayloadPool
{
public:
    static constexpr uint32_t kNone = 0xFFFFFFFF;

    CloudPayloadPool(ref<Device> pDevice, const std::vector<std::filesystem::path>& files, uint32_t poolMB, bool useDirectStorage);
    ~CloudPayloadPool();

    const ref<Buffer>& getBuffer() const { return mpBuffer; }
    /// Whether payloads load through DirectStorage (beginLoad), rather than upload.
    bool isDirect() const { return mpQueue != nullptr; }
    uint32_t getUsedBytes() const { return mUsedWords * 4; }

    /// First word of a range for bytes (a multiple of 4), or kNone when the pool is full. Main thread.
    uint32_t allocate(uint32_t bytes);
    void free(uint32_t word, uint32_t bytes);

    /// DirectStorage: enqueues loading a payload blob of a package file into the range from word. Thread-safe. Returns a ticket for
    /// pollLoad, or kNone when too many loads are outstanding.
    uint32_t beginLoad(uint32_t file, const BlobRef& blob, uint32_t word);
    /// 0 while loading, 1 once the payload is in GPU memory, -1 when it failed. A finished ticket is released.
    int pollLoad(uint32_t ticket);

    /// Without DirectStorage: uploads decompressed bytes into the range from word. Main thread.
    void upload(uint32_t word, const std::vector<uint8_t>& bytes);

private:
    ref<Device> mpDevice;
    ref<Buffer> mpBuffer;
    std::map<uint32_t, uint32_t> mFreeRanges; ///< First word to length in words.
    uint32_t mUsedWords = 0;

    IDStorageFactory* mpFactory = nullptr;
    std::vector<IDStorageFile*> mFiles;
    IDStorageQueue* mpQueue = nullptr;
    IDStorageStatusArray* mpStatus = nullptr;
    std::mutex mTicketMutex;
    std::vector<uint32_t> mFreeTickets;
};
} // namespace hstrcloud
