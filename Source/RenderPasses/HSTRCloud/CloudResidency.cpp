/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "CloudResidency.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <numeric>
#include <optional>
#include <thread>
#include <unordered_map>

namespace hstrcloud
{
namespace
{
constexpr uint32_t kAtlasBricksXY = 64; ///< Atlas bricks per X and Y axis (getAtlasShift).
constexpr uint32_t kNodeBlock = 256;
constexpr uint32_t kBrickBlock = 4096;

uint64_t recordKey(uint32_t level, uint32_t coord)
{
    return (uint64_t(level) << 32) | coord;
}

uint32_t levelOf(const HSTRCloudBrick& brick)
{
    return brick.info & 15u;
}
} // namespace

CloudResidency::CloudResidency(ref<Device> pDevice, const CloudSea& sea, const CloudResidencyDesc& desc) : mpDevice(pDevice), mDesc(desc)
{
    const auto& assets = sea.getAssets();
    uint32_t slots = uint32_t(std::clamp<uint64_t>(uint64_t(mDesc.poolMB) * 1024 * 1024 / kBrickValues, 4096, 4096 * 204));
    const uint32_t atlasDepth = (slots + kAtlasBricksXY * kAtlasBricksXY - 1) / (kAtlasBricksXY * kAtlasBricksXY);
    slots = atlasDepth * kAtlasBricksXY * kAtlasBricksXY;
    // Every GPU brick owns a slot; page nodes are shared by up to 64 bricks.
    const uint32_t brickCapacity = slots;
    const uint32_t nodeCapacity = std::max(4096u, slots / 8);

    for (uint32_t s = slots; s-- > 0;)
        mFreeSlots.push_back(s);
    mBricks.resize(brickCapacity);
    mBrickOwner.assign(brickCapacity, kNoHandle);
    mMappedState.assign(brickCapacity, 0);
    mMappedSnapshot.assign(brickCapacity, 0);
    for (uint32_t b = brickCapacity; b-- > 0;)
        mFreeBricks.push_back(b);
    mNodes.assign(size_t(nodeCapacity) * 64, kCloudRefNone);
    for (uint32_t n = nodeCapacity; n-- > 0;)
        mFreeNodes.push_back(n);
    mNodeBlocksDirty.assign((nodeCapacity + kNodeBlock - 1) / kNodeBlock, 0);
    mBrickBlocksDirty.assign((brickCapacity + kBrickBlock - 1) / kBrickBlock, 0);

    mpPayload = std::make_unique<CloudPayloadPool>(mpDevice, mDesc.files, mDesc.payloadPoolMB, mDesc.directStorage);
    std::vector<HSTRCloudAsset> gpuAssets;
    for (uint32_t a = 0; a < assets.size(); ++a)
    {
        const CloudAsset& asset = assets[a];
        mAssetRecords.push_back(&asset);
        AssetState state;
        state.directoryOffset = uint32_t(mDirectory.size());
        const uint32_t chunkCount = asset.chunkDims.x * asset.chunkDims.y * asset.chunkDims.z;
        mDirectory.resize(mDirectory.size() + chunkCount, kCloudRefNone);
        state.chunkStores.assign(chunkCount, kNone);
        state.chunkBrick.assign(chunkCount, kNoHandle);
        state.changeDims = (asset.dims + 31u) / 32u;
        if (!mDesc.gpuSun)
            state.changeFrame.assign(size_t(state.changeDims.x) * state.changeDims.y * state.changeDims.z, 0);
        mAssets.push_back(std::move(state));
        // The coarse pages are always resident; their payloads upload once.
        DecodedPage coarse;
        if (!decodeMeta(asset.coarseMeta, uint32_t(asset.coarsePayload.size()), coarse))
            FALCOR_THROW("HSTRCloud: corrupt coarse page of cloud '{}'.", asset.name);
        const uint32_t payloadBytes = uint32_t(asset.coarsePayload.size());
        const uint32_t payloadWord = payloadBytes > 0 ? mpPayload->allocate(payloadBytes) : CloudPayloadPool::kNone;
        if (payloadBytes > 0 && payloadWord == CloudPayloadPool::kNone)
            FALCOR_THROW("HSTRCloud: the {} MB cloud payload pool cannot hold the coarse pages.", mDesc.payloadPoolMB);
        mpPayload->upload(payloadWord, asset.coarsePayload);
        const uint32_t store = createStore(StoreKind::Coarse, a, kNone, std::move(coarse), kNoHandle, payloadWord, payloadBytes);
        mAssets[a].coarseStore = store;
        for (uint32_t i = 0; i < mStores[store]->bricks.size(); ++i)
        {
            const BrickHeader& record = mStores[store]->bricks[i].record;
            if (record.level == asset.topLevel)
                mAssets[a].top = makeHandle(store, i);
            if (record.level == kChunkLevel && all(record.brick() < asset.chunkDims))
                mAssets[a].chunkBrick[asset.chunkIndex(record.brick())] = makeHandle(store, i);
        }
        HSTRCloudAsset gpu;
        gpu.dims = asset.dims;
        gpu.directoryOffset = mAssets[a].directoryOffset;
        gpu.chunkDims = asset.chunkDims;
        gpu.topLevel = asset.topLevel;
        gpu.sourceVoxelWorld = asset.voxelWorld * sea.getFitScale();
        gpuAssets.push_back(gpu);
    }
    if (mDirectory.empty())
        mDirectory.push_back(kCloudRefNone);
    mInstances.resize(sea.getTiles().size());
    mTileForward.resize(sea.getTiles().size(), float3x3::identity());

    const auto shaderResource = ResourceBindFlags::ShaderResource;
    mpAssets = mpDevice->createStructuredBuffer(
        sizeof(HSTRCloudAsset), uint32_t(gpuAssets.size()), shaderResource, MemoryType::DeviceLocal, gpuAssets.data(), false
    );
    mpInstances = mpDevice->createStructuredBuffer(
        sizeof(HSTRCloudInstance), uint32_t(mInstances.size()), shaderResource, MemoryType::DeviceLocal, mInstances.data(), false
    );
    mpDirectory = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), uint32_t(mDirectory.size()), shaderResource, MemoryType::DeviceLocal, mDirectory.data(), false
    );
    mpNodes =
        mpDevice->createStructuredBuffer(sizeof(uint32_t), uint32_t(mNodes.size()), shaderResource, MemoryType::DeviceLocal, mNodes.data(), false);
    // Read-write: advanceCloudFades runs the fades in it.
    mpBricks = mpDevice->createStructuredBuffer(
        sizeof(HSTRCloudBrick), uint32_t(mBricks.size()), shaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal,
        mBricks.data(), false
    );
    mFadeStep = 1.f / float(std::max(1u, mDesc.fadeFrames));
    mFadeEnds.resize(std::max(1u, mDesc.fadeFrames) + 1);
    mFadeProcessed = mFrame;
    mpFadeFrame = mpDevice->createStructuredBuffer(sizeof(HSTRCloudFadeFrame), 1, shaderResource, MemoryType::DeviceLocal, nullptr, false);
    mpAtlas = mpDevice->createTexture3D(
        kAtlasBricksXY * 10u,
        kAtlasBricksXY * 10u,
        atlasDepth * 10u,
        ResourceFormat::R8Unorm,
        1,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    mpOccupancy = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), brickCapacity * kCloudCellWords, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false
    );
    mpSunAtlas = mpDevice->createTexture3D(
        mpAtlas->getWidth(),
        mpAtlas->getHeight(),
        mpAtlas->getDepth(),
        ResourceFormat::R16Float,
        1,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    const auto readWrite = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    mpSunBakes = mpDevice->createStructuredBuffer(
        sizeof(HSTRCloudSunBake), std::max(1u, mDesc.sunBakesPerFrame), readWrite, MemoryType::DeviceLocal, nullptr, false
    );
    // As many sun slots as density slots: bricks baked for several orientations are few, and a brick short of a slot keeps the live
    // march for that orientation.
    for (uint32_t s = slots; s-- > 0;)
        mFreeSunSlots.push_back(s);
    mSunSlotTable.resize(size_t(brickCapacity) * 2 * kCloudSunBakesPerBrick);
    for (size_t k = 0; k < mSunSlotTable.size(); k += 2)
    {
        mSunSlotTable[k] = kCloudRefNone;
        mSunSlotTable[k + 1] = kCloudRefNone;
    }
    mSunBakeFrames.assign(size_t(brickCapacity) * kCloudSunBakesPerBrick, 0);
    mSunTableBlocksDirty.assign((brickCapacity + kBrickBlock - 1) / kBrickBlock, 0);
    mpSunSlotTable = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), uint32_t(mSunSlotTable.size()), readWrite, MemoryType::DeviceLocal, mSunSlotTable.data(), false
    );
    if (mDesc.gpuSun)
    {
        const uint32_t bakeMax = std::max(1u, mDesc.sunBakesPerFrame);
        auto create = [&](uint32_t elementSize, uint32_t count, const void* data)
        { return mpDevice->createStructuredBuffer(elementSize, std::max(1u, count), readWrite, MemoryType::DeviceLocal, data, false); };
        mSunSched.assign(brickCapacity, HSTRCloudSunSched{});
        mSunSchedBlocksDirty.assign(mBrickBlocksDirty.size(), 0);
        mpSunSched = create(sizeof(HSTRCloudSunSched), brickCapacity, mSunSched.data());
        for (auto& state : mSunState)
            state.assign(brickCapacity, 0u);
        mSunStateBlocksDirty.assign(mBrickBlocksDirty.size(), 0);
        mpSunState = create(sizeof(uint32_t), brickCapacity, mSunState[0].data());
        mpSunFrames = create(sizeof(uint32_t), uint32_t(mSunBakeFrames.size()), mSunBakeFrames.data());
        mpSunFree = create(sizeof(uint32_t), uint32_t(mFreeSunSlots.size()), mFreeSunSlots.data());
        const uint32_t freeTop = uint32_t(mFreeSunSlots.size());
        mpSunFreeTop = create(sizeof(uint32_t), 1, &freeTop);
        mpSunCounters = create(sizeof(uint32_t), kCloudSunCounters, nullptr);
        mpSunHistogram = create(sizeof(uint32_t), 2 * kCloudSunBins, nullptr);
        mpSunSelect = create(sizeof(uint32_t), 5, nullptr);
        mpSunNeeds = create(sizeof(uint32_t), brickCapacity, nullptr);
        mpSunCandidates = create(sizeof(float4), bakeMax, nullptr);
        mpSunHolders = create(sizeof(uint32_t), bakeMax, nullptr);
        mpSunRelease = create(sizeof(uint32_t), brickCapacity, nullptr);
        mpSunDirections = create(sizeof(float4), 16, nullptr);
        mSunDirections.assign(16, float4(0.f));
        // The invalidation field: a level >= 2 block per (asset, class) pair some sea slot uses (so at most one per slot), then a
        // row per directory node for levels 0-1.
        mSunFieldLevels.assign(mAssets.size() * kCloudSunFieldLevels, 0);
        for (size_t a = 0; a < mAssets.size(); ++a)
        {
            const CloudAsset& asset = *mAssetRecords[a];
            uint32_t size = 0;
            for (uint32_t level = 2; level <= std::min(asset.topLevel, kCloudSunFieldLevels - 1); ++level)
            {
                const uint3 dims = (asset.dims + (8u << level) - 1u) / (8u << level);
                mSunFieldLevels[a * kCloudSunFieldLevels + level] = size;
                size += dims.x * dims.y * dims.z;
            }
            mSunFieldBlockSize = std::max(mSunFieldBlockSize, size);
        }
        const uint32_t blocks = uint32_t(std::min(sea.getTiles().size(), mAssets.size() * kCloudSunClasses));
        for (uint32_t block = blocks; block-- > 0;)
            mFreeSunFieldBlocks.push_back(block);
        mSunFieldBlocks.assign(mAssets.size() * kCloudSunClasses, kNone);
        mSunFieldUsers.assign(mSunFieldBlocks.size(), 0);
        mSlotSunPair.assign(sea.getTiles().size(), kNone);
        mSunChangeCapacity = brickCapacity;
        mSunFineBase = mSunFieldBlockSize * blocks;
        const uint64_t entries = uint64_t(mSunFineBase) + uint64_t(nodeCapacity) * kCloudSunFineEntries;
        FALCOR_CHECK(entries < (1ull << 32), "HSTRCloud: the sun bake invalidation field is too large.");
        mSunFieldRows = uint32_t((entries + kCloudSunFieldWidth - 1) / kCloudSunFieldWidth);
        mpSunField = mpDevice->createTexture2D(kCloudSunFieldWidth, mSunFieldRows, ResourceFormat::R16Uint, 1, 1, nullptr, readWrite);
        mpSunResets = create(sizeof(uint32_t), blocks + nodeCapacity, nullptr);
        mpSunFrameInfo = create(sizeof(HSTRCloudSunFrame), 1, nullptr);
        mpSunFieldBlocks = create(sizeof(uint32_t), uint32_t(mSunFieldBlocks.size()), mSunFieldBlocks.data());
        mpSunFieldLevels = create(sizeof(uint32_t), uint32_t(mSunFieldLevels.size()), mSunFieldLevels.data());
        mpSunChanges = create(sizeof(HSTRCloudSunChange), mSunChangeCapacity, nullptr);
        logInfo(
            "HSTRCloud: sun bake invalidation field of {} blocks of {} entries and {} node rows ({:.1f} MB).",
            blocks,
            mSunFieldBlockSize,
            nodeCapacity,
            double(entries) * sizeof(uint16_t) / (1024 * 1024)
        );
    }
    mStagingInfo.resize(mDesc.loadsPerFrame);
    mpResiduals = mpDevice->createStructuredBuffer(
        sizeof(float), mDesc.loadsPerFrame * kCoreValues, ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal, nullptr, false
    );
    mpStagingInfo = mpDevice->createStructuredBuffer(
        sizeof(HSTRCloudStaging), uint32_t(mStagingInfo.size()), shaderResource, MemoryType::DeviceLocal, nullptr, false
    );

    for (uint32_t i = 0; i < std::max(1u, mDesc.ioThreads); ++i)
        mIoThreads.emplace_back([this] { ioWorker(); });
    logInfo(
        "HSTRCloud: cloud residency with {} atlas bricks ({} MB), {} page nodes, {} directory entries.",
        slots,
        slots * kBrickValues / (1024 * 1024),
        nodeCapacity,
        mDirectory.size()
    );
}

CloudResidency::~CloudResidency()
{
    waitForCut();
    if (mCutThread.joinable())
    {
        {
            std::lock_guard lock(mCutMutex);
            mCutStop = true;
        }
        mCutWake.notify_all();
        mCutThread.join();
    }
    {
        std::lock_guard lock(mIoMutex);
        mStop = true;
    }
    mIoWake.notify_all();
    for (auto& thread : mIoThreads)
        thread.join();
}

uint32_t CloudResidency::createStore(
    StoreKind kind,
    uint32_t asset,
    uint32_t chunk,
    DecodedPage page,
    uint64_t parentHandle,
    uint32_t payloadWord,
    uint32_t payloadBytes
)
{
    uint32_t index;
    if (!mFreeStores.empty())
    {
        index = mFreeStores.back();
        mFreeStores.pop_back();
    }
    else
    {
        index = uint32_t(mStores.size());
        mStores.emplace_back();
    }
    const uint32_t generation = mStores[index] ? mStores[index]->generation + 1 : 1;
    mStores[index] = std::make_unique<Store>();
    Store& store = *mStores[index];
    store.kind = kind;
    store.asset = asset;
    store.chunk = chunk;
    store.generation = generation;
    store.lastUsedFrame = mFrame;
    store.payloadWord = payloadWord;
    store.payloadBytes = payloadBytes;
    store.pages = std::move(page.pages);
    store.pageStores.assign(store.pages.size(), kNone);
    std::unordered_map<uint64_t, uint32_t> lookup;
    store.bricks.resize(page.bricks.size());
    if (kind != StoreKind::Coarse && !store.bricks.empty())
        ++mPageStores;
    for (uint32_t i = 0; i < page.bricks.size(); ++i)
    {
        store.bricks[i].record = page.bricks[i];
        lookup[recordKey(page.bricks[i].level, page.bricks[i].coord)] = i;
    }
    // The top brick of a chunk store (level 3) or page store (level 2) hangs from the parent store.
    const uint32_t firstLevel = kind == StoreKind::Chunk ? kChunkLevel - 1 : kPageLevel;
    for (uint32_t i = 0; i < store.bricks.size(); ++i)
    {
        Brick& b = store.bricks[i];
        if (kind != StoreKind::Coarse && b.record.level == firstLevel)
            b.parent = parentHandle;
        if (kind == StoreKind::Page && b.record.level == kPageLevel)
            store.heads.push_back(i);
        b.children = uint32_t(store.children.size());
        if (kind == StoreKind::Chunk)
        {
            // Level-3 bricks: their children are the level-2 pages below them.
            for (uint32_t p = 0; p < store.pages.size(); ++p)
                if (all(BrickHeader{store.pages[p].coord}.brick() / 2u == b.record.brick()))
                {
                    store.children.push_back(p);
                    ++b.childCount;
                }
            continue;
        }
        if (b.record.level == 0 || (kind == StoreKind::Coarse && b.record.level == kChunkLevel))
            continue; // Level-4 children live in the chunk stores.
        const uint3 base = b.record.brick() * 2u;
        for (uint32_t bit = 0; bit < 8; ++bit)
        {
            if ((b.record.childMask & (1u << bit)) == 0)
                continue;
            const uint3 c = base + uint3(bit & 1u, (bit >> 1) & 1u, bit >> 2);
            const auto it = lookup.find(recordKey(b.record.level - 1u, BrickHeader::pack(c)));
            if (it == lookup.end())
                continue; // Predicted by this brick: not in the package.
            store.children.push_back(it->second);
            store.bricks[it->second].parent = makeHandle(index, i);
            ++b.childCount;
        }
    }
    ++mCutCommits;
    return index;
}

void CloudResidency::releaseStore(uint32_t index)
{
    Store& store = *mStores[index];
    FALCOR_ASSERT(store.loaded == 0);
    if (store.kind != StoreKind::Coarse && !store.bricks.empty())
        --mPageStores;
    if (store.kind == StoreKind::Chunk)
        mAssets[store.asset].chunkStores[store.chunk] = kNone;
    else if (store.kind == StoreKind::Page && store.parentStore != kNone)
        mStores[store.parentStore]->pageStores[store.parentPage] = kNone;
    mpPayload->free(store.payloadWord, store.payloadBytes);
    const uint32_t generation = store.generation;
    mStores[index] = std::make_unique<Store>();
    mStores[index]->generation = generation;
    mFreeStores.push_back(index);
}

void CloudResidency::ioWorker()
{
    std::vector<std::ifstream> files(mDesc.files.size());
    std::vector<uint8_t> raw;
    for (;;)
    {
        Request request;
        {
            std::unique_lock lock(mIoMutex);
            mIoWake.wait(lock, [&] { return mStop || !mQueue.empty(); });
            if (mStop)
                return;
            std::pop_heap(mQueue.begin(), mQueue.end());
            request = mQueue.back();
            mQueue.pop_back();
            ++mInFlight;
        }
        Completion completion;
        completion.request = request;
        // The meta decompresses here; the payload loads with DirectStorage into its pool range, or decompresses here to upload.
        const BlobRef& payload = request.blobs.payload;
        const uint32_t fileIndex = mAssetRecords[request.asset]->file;
        std::ifstream& file = files[fileIndex];
        if (!file.is_open())
            file.open(mDesc.files[fileIndex], std::ios::binary);
        completion.ok = readBlob(file, request.blobs.meta, raw) && decodeMeta(raw, payload.raw, completion.page);
        if (completion.ok && payload.raw > 0)
        {
            if (mpPayload->isDirect())
            {
                completion.ticket = mpPayload->beginLoad(fileIndex, payload, request.payloadWord);
                completion.ok = completion.ticket != CloudPayloadPool::kNone;
            }
            else
                completion.ok = readBlob(file, payload, completion.payload);
        }
        std::lock_guard lock(mIoMutex);
        mCompletions.push_back(std::move(completion));
        --mInFlight;
    }
}

void CloudResidency::resetPending(const Request& request)
{
    mpPayload->free(request.payloadWord, request.blobs.payload.raw);
    if (request.page == kNone)
    {
        uint32_t& store = mAssets[request.asset].chunkStores[request.chunk];
        if (store == kPendingStore)
            store = kNone;
    }
    else if (request.store < mStores.size() && mStores[request.store]->generation == request.generation)
    {
        uint32_t& store = mStores[request.store]->pageStores[request.page];
        if (store == kPendingStore)
            store = kNone;
    }
}

void CloudResidency::enqueue(std::vector<Request> requests)
{
    std::lock_guard lock(mIoMutex);
    // Requests not yet started are replaced by this frame's priorities.
    for (const Request& old : mQueue)
        resetPending(old);
    // A few frames of pages at most wait in memory.
    const size_t outstanding = mCompletions.size() + mInFlight;
    const size_t capacity = 256 > outstanding ? 256 - outstanding : 0;
    // The cut sorted them, most important first.
    if (requests.size() > capacity)
        requests.resize(capacity);
    std::vector<Request> accepted;
    accepted.reserve(requests.size());
    for (Request& request : requests)
    {
        // The payload's pool range is reserved now, so loading can write it; a full pool releases unused stores.
        if (request.blobs.payload.raw > 0)
        {
            request.payloadWord = mpPayload->allocate(request.blobs.payload.raw);
            if (request.payloadWord == CloudPayloadPool::kNone)
            {
                mReleaseStores = true;
                continue;
            }
        }
        if (request.page == kNone)
            mAssets[request.asset].chunkStores[request.chunk] = kPendingStore;
        else
            mStores[request.store]->pageStores[request.page] = kPendingStore;
        accepted.push_back(request);
    }
    mQueue = std::move(accepted);
    std::make_heap(mQueue.begin(), mQueue.end());
    mIoWake.notify_all();
}

bool CloudResidency::inFrustum(const CloudView& view, float3 lo, float3 hi) const
{
    const float4x4& m = view.viewProjection;
    const float4 r0 = m.getRow(0);
    const float4 r1 = m.getRow(1);
    const float4 r2 = m.getRow(2);
    const float4 r3 = m.getRow(3);
    // A guard band keeps bricks just outside the view resident for turning.
    const float4 guard = 1.3f * r3;
    const float4 planes[5] = {guard + r0, guard - r0, guard + r1, guard - r1, r2};
    for (const float4& plane : planes)
    {
        const float3 p(plane.x >= 0.f ? hi.x : lo.x, plane.y >= 0.f ? hi.y : lo.y, plane.z >= 0.f ? hi.z : lo.z);
        if (dot(plane.xyz(), p) + plane.w < 0.f)
            return false;
    }
    return true;
}

float CloudResidency::transmittance(const CloudSea& sea, const CloudView& view, float3 target) const
{
    const CloudSeaDesc& desc = sea.getDesc();
    const uint3 dims = sea.getDims();
    const float voxel = sea.getVoxelWorld();
    const float3 delta = target - view.position;
    const float length = math::length(delta);
    if (length <= voxel)
        return 1.f;
    const uint32_t steps = std::min(128u, uint32_t(std::ceil(length / voxel)));
    const float dt = length / float(steps);
    float depth = 0.f;
    for (uint32_t i = 0; i < steps && depth < 6.f; ++i)
    {
        const float3 p = view.position + delta * ((float(i) + 0.5f) / float(steps));
        const int3 v = int3(floor((p - desc.origin) / voxel));
        if (v.y < 0 || v.y >= int32_t(dims.y))
            continue;
        const int32_t x = (v.x % int32_t(dims.x) + int32_t(dims.x)) % int32_t(dims.x);
        const int32_t z = (v.z % int32_t(dims.z) + int32_t(dims.z)) % int32_t(dims.z);
        depth += sea.meanAt(uint3(uint32_t(x), uint32_t(v.y), uint32_t(z))) * view.densityScale * dt;
    }
    return std::exp(-depth);
}

float CloudResidency::brickPriority(const CloudSea& sea, uint32_t slot, uint64_t handle, const CloudView& view, float& visibility)
{
    const Brick& b = brick(handle);
    const CloudSea::Tile& tile = sea.getTiles()[slot];
    const CloudSeaDesc& desc = sea.getDesc();
    const HSTRCloudInstance& instance = tile.instance;
    const uint32_t level = b.record.level;
    const float scale = float(1u << level);
    // Source voxels the brick's values cover (with the apron), to world bounds.
    const float3 sourceLo = float3(b.record.brick() * 8u) * scale - scale;
    const float3 sourceHi = float3((b.record.brick() + 1u) * 8u) * scale + scale;
    const float3 offset(instance.row0.w, instance.row1.w, instance.row2.w);
    const float3x3& forward = mTileForward[slot];
    const float3 p0 = mul(forward, sourceLo - offset);
    const float3 p1 = mul(forward, sourceHi - offset);
    const float3 tileCorner = desc.origin + float3(float(tile.world.x), 0.f, float(tile.world.y)) * desc.tileWorld;
    const float voxel = sea.getVoxelWorld();
    const float3 lo = tileCorner + (min(p0, p1) + 0.5f) * voxel;
    const float3 hi = tileCorner + (max(p0, p1) + 0.5f) * voxel;

    // Within the motion envelope the camera may come closer by its margin, and see the brick from anywhere in it.
    const float3 nearest = clamp(view.position, lo, hi);
    const float distance = std::max(math::length(nearest - view.position) - mCutMarginWorld, 0.f);
    if (distance > view.maxDistance)
        return -1.f;
    const float voxelWorld = scale * instance.sourceVoxelWorld;
    const float pixels = voxelWorld / (std::max(distance, 0.5f * voxelWorld) * view.pixelAngle);
    // Hysteresis: a brick whose children are mapped keeps them until its error falls a quarter level below the threshold.
    const bool mappedChildren = b.gpu != kNone && (mMappedSnapshot[b.gpu] & kSnapChildren);
    const float hysteresis = mappedChildren ? 0.25f : 0.f;
    if (std::log2(std::max(pixels / mDesc.lodPixels, 1e-6f)) - view.lodBias + hysteresis <= 0.f)
        return -1.f;
    // The image error a brick's missing detail causes is its voxels' screen size scaled by how much of it reaches the camera: the
    // transmittance from the camera through the always-resident proxy (measured for coarse bricks, inherited by finer ones, and
    // re-measured once the camera moved by a tenth of the distance), and a small share outside the frustum swept towards the sun.
    // The most visible of the brick's nearest point and corners counts: a silhouette brick's nearest point is often behind the
    // cloud's surface while its outer part is seen against the sky, and left coarse it rendered as spikes of clipped coarse density.
    const float3 sweep = view.sunDirection * view.sunReach;
    const bool inside = inFrustum(view, min(lo, lo + sweep) - mCutMarginWorld, max(hi, hi + sweep) + mCutMarginWorld);
    if (inside && level >= 3)
    {
        const uint32_t generation = storeOf(handle).generation;
        auto [it, inserted] = mVisibility[slot].try_emplace(handle);
        VisibilityEntry& entry = it->second;
        const float3 moved = view.position - entry.camera;
        if (inserted || entry.generation != generation || dot(moved, moved) > 0.01f * distance * distance + 1.f)
        {
            float best = transmittance(sea, view, nearest);
            for (uint32_t corner = 0; corner < 8 && best < 0.99f; ++corner)
            {
                const float3 p((corner & 1) ? hi.x : lo.x, (corner & 2) ? hi.y : lo.y, (corner & 4) ? hi.z : lo.z);
                best = std::max(best, transmittance(sea, view, p));
            }
            entry.visibility = best;
            entry.camera = view.position;
            entry.generation = generation;
        }
        visibility = entry.visibility;
    }
    const float importance = (inside ? 1.f : 0.125f) * std::max(visibility, 0.01f);
    return std::log2(std::max(pixels * importance / mDesc.lodPixels, 1e-6f)) - view.lodBias + hysteresis;
}

bool CloudResidency::cutSeed(const CloudSea& sea, const CloudView& view, uint32_t slot, CutEntry& entry)
{
    const CloudSea::Tile& tile = sea.getTiles()[slot];
    if (!tile.occupied || mAssets[tile.instance.asset].top == kNoHandle)
        return false;
    // Instances whose proxy already projects below the pixel threshold need no fine bricks.
    const float distance = std::max(math::length(clamp(view.position, tile.worldMin, tile.worldMax) - view.position) - mCutMarginWorld, 0.f);
    const float proxyPixels = sea.getVoxelWorld() / (std::max(distance, 1e-3f) * view.pixelAngle);
    if (std::log2(proxyPixels / mDesc.lodPixels) - view.lodBias <= 0.f || distance > view.maxDistance)
        return false;
    const uint64_t top = mAssets[tile.instance.asset].top;
    float visibility = 1.f;
    const float priority = std::max(brickPriority(sea, slot, top, view, visibility), 1e-3f);
    entry = {priority, top, slot, visibility};
    return true;
}

void CloudResidency::cutChildren(const CutEntry& entry, std::vector<uint64_t>& children, std::vector<Request>& requests) const
{
    children.clear();
    const uint32_t storeIndex = uint32_t(entry.handle >> 32);
    const Store& store = *mStores[storeIndex];
    const Brick& b = store.bricks[uint32_t(entry.handle)];
    const CloudAsset& asset = *mAssetRecords[store.asset];
    if (store.kind == StoreKind::Coarse && b.record.level == kChunkLevel)
    {
        // Children: the level-3 bricks of the chunk page.
        if (b.record.childMask == 0 || !all(b.record.brick() < asset.chunkDims))
            return;
        const uint32_t chunk = asset.chunkIndex(b.record.brick());
        const uint32_t chunkStore = mAssets[store.asset].chunkStores[chunk];
        if (asset.chunks[chunk].meta.raw == 0)
            return;
        if (chunkStore == kNone || chunkStore == kPendingStore)
        {
            Request request;
            request.priority = entry.priority + 1.f; // Pages precede the bricks they unlock.
            request.asset = store.asset;
            request.chunk = chunk;
            request.blobs = asset.chunks[chunk];
            requests.push_back(request);
            return;
        }
        for (uint32_t i = 0; i < mStores[chunkStore]->bricks.size(); ++i)
            children.push_back(makeHandle(chunkStore, i));
    }
    else if (store.kind == StoreKind::Chunk)
    {
        // Children: the level-2 bricks heading the level-2 pages below this level-3 brick.
        for (uint32_t c = 0; c < b.childCount; ++c)
        {
            const uint32_t page = store.children[b.children + c];
            const uint32_t pageStore = store.pageStores[page];
            if (pageStore == kNone || pageStore == kPendingStore)
            {
                Request request;
                request.priority = entry.priority + 1.f;
                request.asset = store.asset;
                request.chunk = store.chunk;
                request.store = storeIndex;
                request.generation = store.generation;
                request.page = page;
                request.blobs = store.pages[page].blobs;
                requests.push_back(request);
                continue;
            }
            for (uint32_t head : mStores[pageStore]->heads)
                children.push_back(makeHandle(pageStore, head));
        }
    }
    else
        for (uint32_t c = 0; c < b.childCount; ++c)
            children.push_back(makeHandle(storeIndex, store.children[b.children + c]));
}

bool CloudResidency::cutRefinable(uint64_t handle) const
{
    const Store& store = *mStores[handle >> 32];
    const Brick& b = store.bricks[uint32_t(handle)];
    return b.childCount > 0 || (store.kind == StoreKind::Coarse && b.record.level == kChunkLevel && b.record.childMask != 0);
}

bool CloudResidency::update(const CloudSea& sea, const CloudView& view, const std::vector<uint32_t>& changedSlots)
{
    const auto startTime = std::chrono::steady_clock::now();
    ++mFrame;
    bool changed = false;
    const auto& tiles = sea.getTiles();
    auto* pRenderContext = mpDevice->getRenderContext();
    // A cut on the worker (view.cutAsync) reads the stores, the loaded bricks, the visibility caches and the sea's tiles. Until it
    // returns, nothing here that changes those runs - no pages, loads or releases - and the frame keeps the last cut; HSTRCloud holds
    // the sea's tile changes back meanwhile (cutInFlight), so changed slots mean the walk is already done. Maps, unmaps and fades do
    // run: the walk reads what they change from the snapshot it started with.
    std::optional<CutWalk> finished;
    if (mCutJob.valid())
    {
        if (!changedSlots.empty() || mCutJob.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
            finished = mCutJob.get();
        else
        {
            // Maps, fades and unmaps run beside the walk: they change only what it reads from its snapshot (mMappedSnapshot), and
            // the merge revisits the bricks they touched. Loads, pages, releases and tile changes still wait for it.
            mStaged.clear();
            mCommitGroups.clear();
            bool running = false;
            {
                FALCOR_PROFILE(pRenderContext, "maps");
                mapReady(running);
            }
            {
                FALCOR_PROFILE(pRenderContext, "fades");
                processFadeEnds(running);
            }
            finishFrame(sea, view, startTime);
            return running;
        }
    }
    for (uint32_t slot : changedSlots)
    {
        if (slot < mVisibility.size())
            mVisibility[slot].clear(); // Another instance now: its bricks sit elsewhere.
        if (mDesc.gpuSun)
            releaseSunField(slot);
        mInstances[slot] = tiles[slot].instance;
        const HSTRCloudInstance& i = tiles[slot].instance;
        if (tiles[slot].occupied)
        {
            mTileForward[slot] = inverse(float3x3{i.row0.x, i.row0.y, i.row0.z, i.row1.x, i.row1.y, i.row1.z, i.row2.x, i.row2.y, i.row2.z});
            mInstances[slot].sunClass = sunClassOf(i);
            if (mDesc.gpuSun)
                acquireSunField(slot);
        }
        mInstancesDirty = true;
    }

    // Arrived pages become stores.
    {
        FALCOR_PROFILE(pRenderContext, "pages");
        std::vector<Completion> completions;
        {
            std::lock_guard lock(mIoMutex);
            for (Completion& completion : mCompletions)
                mWaiting.push_back(std::move(completion));
            mCompletions.clear();
        }
        // Pages whose payload is in GPU memory (DirectStorage finished, or uploaded now) are ready.
        for (size_t i = 0; i < mWaiting.size();)
        {
            Completion& completion = mWaiting[i];
            if (completion.ticket != CloudPayloadPool::kNone)
            {
                const int state = mpPayload->pollLoad(completion.ticket);
                if (state == 0)
                {
                    ++i;
                    continue;
                }
                completion.ticket = CloudPayloadPool::kNone;
                completion.ok &= state > 0;
            }
            else if (completion.ok)
                mpPayload->upload(completion.request.payloadWord, completion.payload);
            completions.push_back(std::move(completion));
            mWaiting[i] = std::move(mWaiting.back());
            mWaiting.pop_back();
        }
        for (Completion& completion : completions)
        {
            const Request& request = completion.request;
            const uint32_t payloadBytes = request.blobs.payload.raw;
            bool stored = false;
            if (request.page == kNone)
            {
                uint32_t& chunkStore = mAssets[request.asset].chunkStores[request.chunk];
                if (chunkStore == kPendingStore)
                {
                    chunkStore = kNone;
                    if (completion.ok)
                    {
                        const uint32_t store = createStore(
                            StoreKind::Chunk, request.asset, request.chunk, std::move(completion.page), mAssets[request.asset].chunkBrick[request.chunk],
                            request.payloadWord, payloadBytes
                        );
                        mAssets[request.asset].chunkStores[request.chunk] = store;
                        stored = true;
                    }
                }
            }
            else if (request.store < mStores.size() && mStores[request.store]->generation == request.generation &&
                     mStores[request.store]->pageStores[request.page] == kPendingStore)
            {
                mStores[request.store]->pageStores[request.page] = kNone;
                // The level-2 brick's parent: the level-3 brick above it in the chunk store.
                Store& chunkStore = *mStores[request.store];
                const uint3 parentCoord = BrickHeader{chunkStore.pages[request.page].coord}.brick() / 2u;
                uint64_t parent = kNoHandle;
                for (uint32_t b = 0; b < chunkStore.bricks.size(); ++b)
                    if (all(chunkStore.bricks[b].record.brick() == parentCoord))
                        parent = makeHandle(request.store, b);
                if (completion.ok && parent != kNoHandle)
                {
                    const uint32_t store = createStore(
                        StoreKind::Page, request.asset, request.chunk, std::move(completion.page), parent, request.payloadWord, payloadBytes
                    );
                    mStores[store]->parentStore = request.store;
                    mStores[store]->parentPage = request.page;
                    mStores[request.store]->pageStores[request.page] = store;
                    stored = true;
                }
            }
            if (!stored)
                mpPayload->free(request.payloadWord, payloadBytes);
        }
    }

    // The cut: greedy refinement by priority from each nearby instance's top brick, within the atlas budget. It is redone when the
    // camera or the sea changed, or newly arrived pages or bricks can refine further; a static, converged view costs nothing.
    // A move under a quarter voxel or a turn under a quarter degree changes no brick's level or frustum membership that the guard band
    // and the 30-frame refresh below do not cover; re-cutting for it cost the whole cut (~100 ms on the sea) on every frame of a slow
    // camera.
    // With a motion envelope (view.cutMargin) the last cut holds until the camera leaves it or turns by cutTurn, which the frustum's
    // guard band covers, and arrivals re-cut every fourth frame at most.
    const bool envelope = view.cutMargin > 0.f;
    const float turnLimit = envelope ? view.cutTurn : 0.25f;
    auto rowAngleChanged = [&](uint32_t row)
    {
        const float3 a = view.viewProjection.getRow(row).xyz();
        const float3 c = mCutViewProjection.getRow(row).xyz();
        const float la = length(a);
        const float lc = length(c);
        return la <= 0.f || lc <= 0.f || dot(a, c) < std::cos(math::radians(turnLimit)) * la * lc || std::abs(la - lc) > 1e-3f * lc;
    };
    const float3 moved = view.position - mCutPosition;
    const float moveLimit = std::max(0.25f * sea.getVoxelWorld(), envelope ? mCutMarginWorld : 0.f);
    const bool viewChanged = dot(moved, moved) > moveLimit * moveLimit || rowAngleChanged(0) || rowAngleChanged(1) || rowAngleChanged(3);
    const bool arrivals = mCutCommits > 0 && (!envelope || mFrame >= mCutFrame + 4);
    const bool runCut = mCutFrame == 0 || viewChanged || !changedSlots.empty() || arrivals || mFrame >= mCutFrame + 30;
    std::vector<Request> requests;
    // With view.cutAsync the walk runs on a worker (launched at the end of this function, taken on in a later frame); otherwise here.
    const bool async = view.cutAsync && mCutFrame != 0;
    std::optional<CutWalk> walk = std::move(finished);
    if (!walk && runCut && !async)
    {
        FALCOR_PROFILE(pRenderContext, "cut");
        beginCut(sea, view);
        walk = walkCut(sea, view);
    }
    // Taking a cut on: its bricks already carry it in their spare state slot, and the worker built what changes, so the frame flips
    // the slot and applies short lists.
    // MEASURED before (4K sea flight at 20 units a frame, 800k visits, 130k desired and 150k mapped bricks): marking the visits and
    // rebuilding the lists here cost 19 ms on every frame that took on a cut.
    const bool didCut = walk.has_value();
    if (didCut)
    {
        FALCOR_PROFILE(pRenderContext, "merge");
        mCutId = walk->id;
        mCutSlot ^= 1u;
        mMergeFrame = mCutFrame;
        requests = std::move(walk->requests);
        mDesired = std::move(walk->desired);
        mToLoad = std::move(walk->toLoad);
        mToLoadNext = 0;
        mToMap = std::move(walk->toMap);
        for (uint64_t handle : walk->fading)
            updateFade(handle);
        if (mDesc.gpuSun)
        {
            // The walk wrote the scheduler states into the spare table.
            mSunStateLive ^= 1u;
            mSunStateSync = std::move(walk->stateChanged);
            for (size_t block = 0; block < walk->stateBlocksDirty.size(); ++block)
                mSunStateBlocksDirty[block] |= walk->stateBlocksDirty[block];
        }
        // Bricks mapped, unmapped or faded while the walk ran: the walk saw them as they were when it started.
        const std::vector<uint64_t> touched = std::move(mWalkTouched);
        mWalkTouched.clear();
        for (uint64_t handle : touched)
        {
            const Brick& b = brick(handle);
            if (mDesc.gpuSun && b.gpu != kNone)
            {
                mSunState[mSunStateLive][b.gpu] = sunStateOf(cutOf(b), mCutId, b.mapped);
                mSunStateSync.push_back(b.gpu);
                mSunStateBlocksDirty[b.gpu / kBrickBlock] = 1;
            }
            if (b.mapped)
                updateFade(handle);
            else if (desired(b) && (b.flags & kLoaded))
                mToMap.push_back(handle);
        }
        mStats.cutPops = walk->pops;
        mStats.cutOrdered = walk->ordered;
        mStats.cutMargin = mCutMarginWorld / sea.getVoxelWorld();
        ++mStats.cuts;
        mStats.cutTotalMs = walk->milliseconds;
    }
    FALCOR_PROFILE(pRenderContext, "apply");
    mStats.desired = uint32_t(mDesired.size());

    // Reconstruct desired bricks whose parents are in the atlas (mToLoad has parents first, and a parent staged this frame commits in
    // an earlier dispatch than its children); then map those whose parents are mapped (mToMap has parents first too). Both lists
    // only shrink between cuts, so a frame touches what is still to do rather than the whole cut.
    // MEASURED before (4K sea flight, 130k desired and mapped bricks): walking all of them every frame cost 3.9 ms of CPU.
    mStaged.clear();
    std::optional<ScopedProfilerEvent> phase(std::in_place, pRenderContext, "loads");
    bool held = false; // A brick waits on its parent: the cursor stays before it.
    for (size_t i = mToLoadNext; i < mToLoad.size() && mStaged.size() < mDesc.loadsPerFrame; ++i)
    {
        const uint64_t handle = mToLoad[i];
        const Brick& b = brick(handle);
        if (!(b.flags & kLoaded))
        {
            if (b.parent != kNoHandle && !(brick(b.parent).flags & kLoaded))
            {
                held = true;
                continue;
            }
            if (!commit(handle))
                break;
        }
        if (!held)
            mToLoadNext = i + 1;
    }
    phase.reset();
    phase.emplace(pRenderContext, "maps");
    mapReady(changed);
    changed |= !mStaged.empty();
    mCutCommits += uint32_t(mStaged.size());
    phase.reset();
    // A cut's requests replace the queue. Only a cut's: an empty list between cuts emptied the queue of everything not yet started.
    if (didCut)
    {
        FALCOR_PROFILE(pRenderContext, "enqueue");
        enqueue(std::move(requests));
    }

    // Staging order: coarsest level first, one dispatch per level.
    {
        FALCOR_PROFILE(pRenderContext, "stage");
        std::vector<uint32_t> order(mStaged.size());
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(
            order.begin(), order.end(), [&](uint32_t a, uint32_t c) { return brick(mStaged[a]).record.level > brick(mStaged[c]).record.level; }
        );
        std::vector<HSTRCloudStaging> infos(mStaged.size());
        mCommitGroups.clear();
        for (uint32_t i = 0; i < order.size(); ++i)
        {
            infos[i] = mStagingInfo[order[i]];
            const uint32_t level = brick(mStaged[order[i]]).record.level;
            if (i == 0 || level != brick(mStaged[order[i - 1]]).record.level)
                mCommitGroups.push_back({i, 0});
            ++mCommitGroups.back().count;
        }
        std::copy(infos.begin(), infos.end(), mStagingInfo.begin());
    }

    // Fades that end by this frame: faded in, or faded out and unmapped (they stay in the atlas).
    phase.emplace(pRenderContext, "fades");
    processFadeEnds(changed);

    // Page and chunk stores nothing has used for a while are released (pages before their chunks); at once when the payload pool
    // filled, down to those used this frame.
    phase.reset();
    phase.emplace(pRenderContext, "release");
    const bool releaseNow = mReleaseStores;
    mReleaseStores = false;
    if (mFrame % 60 == 0 || releaseNow)
        for (StoreKind kind : {StoreKind::Page, StoreKind::Chunk})
            for (uint32_t s = 0; s < mStores.size(); ++s)
            {
                Store& store = *mStores[s];
                if (store.kind != kind || store.bricks.empty() || store.loaded > 0 || store.lastUsedFrame + (releaseNow ? 1 : 120) > mFrame)
                    continue;
                // The merge used the store of every brick it desired, so a store unused since cannot hold one.
                bool busy = false;
                if (store.lastUsedFrame >= mMergeFrame)
                    for (const Brick& b : store.bricks)
                        busy |= desired(b);
                for (uint32_t pageStore : store.pageStores)
                    busy |= pageStore != kNone;
                if (!busy)
                    releaseStore(s);
            }

    phase.reset();
    // The next cut starts once this one is applied, so the worker never overlaps the loads, maps and releases above.
    if (async && runCut)
    {
        FALCOR_PROFILE(pRenderContext, "launch");
        beginCut(sea, view);
        std::packaged_task<CutWalk()> task([this, &sea, view]() { return walkCut(sea, view); });
        mCutJob = task.get_future();
        if (!mCutThread.joinable())
            mCutThread = std::thread([this] { cutWorker(); });
        {
            std::lock_guard lock(mCutMutex);
            mCutTask = std::move(task);
            mCutQueued = true;
        }
        mCutWake.notify_one();
    }
    finishFrame(sea, view, startTime);
    return changed;
}

void CloudResidency::finishFrame(const CloudSea& sea, const CloudView& view, std::chrono::steady_clock::time_point startTime)
{
    auto* pRenderContext = mpDevice->getRenderContext();
    {
        FALCOR_PROFILE(pRenderContext, "sunBakes");
        scheduleSunBakes(sea, view);
        if (mDesc.gpuSun && (mGpuSunFrame.run || mGpuSunFrame.info.ageRows > 0))
            mpSunFrameInfo->setBlob(&mGpuSunFrame.info, 0, sizeof(HSTRCloudSunFrame));
    }
    {
        FALCOR_PROFILE(pRenderContext, "upload");
        upload();
        if (mActiveFades > 0)
        {
            const HSTRCloudFadeFrame frame{mFrame & kCloudFadeFrameMask, uint32_t(mBricks.size()), mFadeStep, 0};
            mpFadeFrame->setBlob(&frame, 0, sizeof(frame));
        }
    }
    mStats.mapped = uint32_t(mMappedList.size());
    mStats.mapBacklog = uint32_t(mToMap.size());
    mStats.unmapBacklog = uint32_t(mUnmapQueue.size());
    mStats.activeFades = mActiveFades;
    mStats.loaded = uint32_t(mLoadedList.size());
    mStats.committed = uint32_t(mStaged.size());
    // Desired bricks not loaded yet: those past the load cursor (only a brick held back by its parent can be loaded beyond it).
    const uint32_t waiting = uint32_t(mToLoad.size() - mToLoadNext);
    {
        std::lock_guard lock(mIoMutex);
        mStats.pending = uint32_t(mQueue.size() + mCompletions.size() + mWaiting.size()) + mInFlight + waiting;
    }
    mStats.payloadMB = double(mpPayload->getUsedBytes()) / (1024.0 * 1024.0);
    mStats.nodesUsed = uint32_t(mNodes.size() / 64 - mFreeNodes.size());
    mStats.pagesLoaded = mPageStores;
    mStats.residentMB = (double(mStats.slotsUsed) * kBrickValues + double(mNodes.size()) * 4 + double(mBricks.size()) * sizeof(HSTRCloudBrick)) /
                        (1024.0 * 1024.0);
    mStats.cutMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
}

void CloudResidency::cutWorker()
{
    for (;;)
    {
        std::packaged_task<CutWalk()> task;
        {
            std::unique_lock lock(mCutMutex);
            mCutWake.wait(lock, [&] { return mCutStop || mCutQueued; });
            if (mCutStop)
                return;
            task = std::move(mCutTask);
            mCutQueued = false;
        }
        task();
    }
}

void CloudResidency::beginCut(const CloudSea& sea, const CloudView& view)
{
    mCutCommits = 0;
    mCutPosition = view.position;
    mCutViewProjection = view.viewProjection;
    mCutFrame = mFrame;
    mCutMarginWorld = view.cutMargin * mCutMarginScale * sea.getVoxelWorld();
    mVisibility.resize(sea.getTiles().size());
    // What the walk reads of the mapped bricks, as of now (a copy of a byte per GPU brick).
    std::copy(mMappedState.begin(), mMappedState.end(), mMappedSnapshot.begin());
    mWalkTouched.clear();
}

CloudResidency::CutWalk CloudResidency::walkCut(const CloudSea& sea, const CloudView& view)
{
    const auto start = std::chrono::steady_clock::now();
    const auto& tiles = sea.getTiles();
    // The spare scheduler state table catches up with the live one: the entries the last cut changed.
    std::vector<uint32_t>& spareState = mSunState[mSunStateLive ^ 1u];
    for (uint32_t gpu : mSunStateSync)
        spareState[gpu] = mSunState[mSunStateLive][gpu];
    mSunStateSync.clear();
    // Two hardware threads stay with the frame (its thread and the driver's), so a walk on the worker does not compete with it.
    const uint32_t tasks = std::clamp(std::thread::hardware_concurrency(), 3u, 18u) - 2u;
    // Two phases, because the work is not spread over the instances: the one or two nearest hold most of the fine bricks. The first
    // walks each instance's coarse bricks (one thread per group of slots) and stops at the level-3 bricks; the second walks those
    // subtrees, in parallel over all instances together. Below level 3 no brick measures visibility, so the second phase never
    // touches a slot's visibility cache.
    struct Task
    {
        std::vector<CutVisit> visits;
        std::vector<Request> requests;
        std::vector<CutEntry> frontier;
        uint32_t pops = 0;
    };
    std::vector<Task> coarse(tasks);
    std::vector<Task> fine(tasks);
    auto expand = [&](Task& task, std::vector<CutEntry>& stack, bool stopAtChunks)
    {
        std::vector<uint64_t> children;
        while (!stack.empty())
        {
            const CutEntry entry = stack.back();
            stack.pop_back();
            if (stopAtChunks && brick(entry.handle).record.level <= 3)
            {
                task.frontier.push_back(entry);
                continue;
            }
            ++task.pops;
            cutChildren(entry, children, task.requests);
            for (uint64_t child : children)
            {
                float visibility = entry.visibility;
                const float priority = brickPriority(sea, entry.slot, child, view, visibility);
                task.visits.push_back({child, priority, entry.slot});
                if (priority > 0.f && cutRefinable(child))
                    stack.push_back({priority, child, entry.slot, visibility});
            }
        }
    };
    auto walkCoarse = [&](uint32_t index)
    {
        std::vector<CutEntry> stack;
        for (uint32_t slot = index; slot < tiles.size(); slot += tasks)
        {
            CutEntry top;
            if (!cutSeed(sea, view, slot, top))
                continue;
            coarse[index].visits.push_back({top.handle, top.priority, slot});
            if (cutRefinable(top.handle))
                stack.push_back(top);
            expand(coarse[index], stack, true);
        }
    };
    std::vector<CutEntry> frontier;
    auto walkFine = [&](uint32_t index)
    {
        // Round robin, so the subtrees of one near instance spread over every thread.
        std::vector<CutEntry> stack;
        for (size_t i = index; i < frontier.size(); i += tasks)
            stack.push_back(frontier[i]);
        expand(fine[index], stack, false);
    };
    auto parallel = [&](const std::function<void(uint32_t)>& body)
    {
        std::vector<std::future<void>> workers;
        for (uint32_t index = 1; index < tasks; ++index)
            workers.push_back(std::async(std::launch::async, body, index));
        body(0);
        for (auto& worker : workers)
            worker.get();
    };
    parallel(walkCoarse);
    for (const Task& task : coarse)
        frontier.insert(frontier.end(), task.frontier.begin(), task.frontier.end());
    parallel(walkFine);

    // The cut's state goes into the bricks' spare slot, under the next cut id. A visit marks its brick at the highest priority any
    // instance gives it; merged in slot order, coarse phase first, which keeps parents before children.
    const uint32_t next = mCutSlot ^ 1u;
    const uint32_t budget = uint32_t(0.9f * float(mBricks.size()));
    CutWalk walk;
    walk.id = mCutId + 1;
    auto desire = [&](uint64_t handle, float priority, uint32_t slot)
    {
        CutState& state = brick(handle).cut[next];
        storeOf(handle).lastUsedFrame = mCutFrame;
        const uint16_t sunClass = uint16_t(1u << mInstances[slot].sunClass);
        if (state.id != walk.id)
        {
            state = {walk.id, priority, sunClass};
            walk.desired.push_back(handle);
        }
        else
        {
            state.priority = std::max(state.priority, priority);
            state.sunClasses |= sunClass;
        }
    };
    for (const std::vector<Task>* phase : {&coarse, &fine})
        for (const Task& task : *phase)
        {
            for (const CutVisit& visit : task.visits)
                desire(visit.handle, visit.priority, visit.slot);
            walk.requests.insert(walk.requests.end(), task.requests.begin(), task.requests.end());
            walk.pops += task.pops;
        }
    // The walk refines every brick with a positive priority, whatever the order; only when that desires more than the atlas budget
    // does the order matter, and the cut runs again as the greedy one (under a fresh id, so the walk's marks do not count).
    // MEASURED (sea, settled, 90k refinements, 127k bricks): the single-threaded heap cut took ~100 ms, the heap alone a quarter of it.
    walk.ordered = walk.desired.size() > budget;
    // An envelope that does not fit is dropped rather than rationed - the budget goes to the current view - and the next ones are
    // halved until they fit; one that leaves a third of the budget free grows back.
    if (walk.ordered && mCutMarginWorld > 0.f)
    {
        mCutMarginWorld = 0.f;
        mCutMarginScale = mCutMarginScale > 0.125f ? 0.5f * mCutMarginScale : 0.f;
    }
    else if (view.cutMargin > 0.f && walk.desired.size() < 2 * budget / 3)
        mCutMarginScale = std::min(1.f, std::max(2.f * mCutMarginScale, 0.125f));
    if (walk.ordered)
    {
        ++walk.id;
        walk.desired.clear();
        walk.requests.clear();
        walk.pops = 0;
        std::vector<CutEntry> heap;
        auto queue = [&](const CutEntry& entry)
        {
            desire(entry.handle, entry.priority, entry.slot);
            if (entry.priority > 0.f && cutRefinable(entry.handle))
            {
                heap.push_back(entry);
                std::push_heap(heap.begin(), heap.end());
            }
        };
        for (uint32_t slot = 0; slot < tiles.size(); ++slot)
        {
            CutEntry top;
            if (cutSeed(sea, view, slot, top))
                queue(top);
        }
        std::vector<uint64_t> children;
        while (!heap.empty() && walk.pops < 200000)
        {
            std::pop_heap(heap.begin(), heap.end());
            const CutEntry entry = heap.back();
            heap.pop_back();
            ++walk.pops;
            cutChildren(entry, children, walk.requests);
            size_t newBricks = 0;
            for (uint64_t child : children)
                newBricks += brick(child).cut[next].id != walk.id ? 1 : 0;
            if (walk.desired.size() + newBricks > budget)
                continue;
            for (uint64_t child : children)
            {
                float visibility = entry.visibility;
                const float priority = brickPriority(sea, entry.slot, child, view, visibility);
                queue({priority, child, entry.slot, visibility});
            }
        }
    }

    // What the frame applies. Loads go most important first within each level, coarsest level first (a parent before its children);
    // the walk leaves parents first but not in priority order, and the loads per frame are capped.
    for (uint64_t handle : walk.desired)
    {
        const Brick& b = brick(handle);
        if (!(b.flags & kLoaded))
            walk.toLoad.push_back(handle);
        if (b.gpu == kNone || !(mMappedSnapshot[b.gpu] & kSnapMapped))
            walk.toMap.push_back(handle);
    }
    std::sort(
        walk.toLoad.begin(), walk.toLoad.end(),
        [&](uint64_t a, uint64_t c)
        {
            const Brick& ba = brick(a);
            const Brick& bc = brick(c);
            return ba.record.level != bc.record.level ? ba.record.level > bc.record.level : ba.cut[next].priority > bc.cut[next].priority;
        }
    );
    // Mapped bricks fade in while desired, and out once undesired with no mapped children; the scheduler's states follow the cut.
    if (mDesc.gpuSun)
        walk.stateBlocksDirty.assign(mSunStateBlocksDirty.size(), 0);
    // Maps and fades run meanwhile, so what they change comes from the snapshot (and the owners stay: loads wait for the walk).
    for (uint32_t gpu = 0; gpu < uint32_t(mMappedSnapshot.size()); ++gpu)
    {
        const uint8_t snapshot = mMappedSnapshot[gpu];
        if (!(snapshot & kSnapMapped))
            continue;
        const uint64_t handle = mBrickOwner[gpu];
        const Brick& b = brick(handle);
        const bool inCut = b.cut[next].id == walk.id;
        // Only bricks whose fade turns under this cut (updateFade decides again when it is taken on).
        const int current = (snapshot & kSnapFadeIn) ? 1 : (snapshot & kSnapFadeOut) ? -1 : 0;
        const bool full = current == 0 && (snapshot & kSnapFull);
        const int direction = inCut ? (full ? 0 : 1) : ((snapshot & kSnapChildren) ? 0 : -1);
        if (direction != current)
            walk.fading.push_back(handle);
        if (mDesc.gpuSun)
        {
            const uint32_t state = sunStateOf(inCut ? b.cut[next] : b.cut[mCutSlot], walk.id, true);
            if (state != spareState[gpu])
            {
                spareState[gpu] = state;
                walk.stateChanged.push_back(gpu);
                walk.stateBlocksDirty[gpu / kBrickBlock] = 1;
            }
        }
    }
    // Page requests, most important first (enqueue keeps the head of the list).
    std::sort(walk.requests.begin(), walk.requests.end(), [](const Request& a, const Request& b) { return a.priority > b.priority; });
    walk.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return walk;
}

bool CloudResidency::commit(uint64_t handle)
{
    Brick& b = brick(handle);
    if ((mFreeSlots.empty() || mFreeBricks.empty()) && (!evict(2 * mDesc.loadsPerFrame, 2 * mDesc.loadsPerFrame) || mFreeSlots.empty() || mFreeBricks.empty()))
        return false;
    const Store& store = storeOf(handle);
    const Brick* parent = b.parent != kNoHandle ? &brick(b.parent) : nullptr;
    b.gpu = mFreeBricks.back();
    mFreeBricks.pop_back();
    b.slot = mFreeSlots.back();
    mFreeSlots.pop_back();
    ++mStats.slotsUsed;
    mBrickOwner[b.gpu] = handle;
    mMappedState[b.gpu] = 0;
    HSTRCloudBrick& gpu = mBricks[b.gpu];
    gpu.slot = b.slot;
    gpu.info = b.record.level | (uint32_t(b.record.childMask) << 4);
    gpu.parent = parent ? parent->gpu : kCloudRefNone;
    gpu.fade = 0.f;
    gpu.fadeBase = 0.f;
    gpu.fadeStart = 0;
    gpu.minValue = b.record.valueMin;
    gpu.range = b.record.valueRange;
    touchBrick(b.gpu);

    const uint32_t n = uint32_t(mStaged.size());
    mStaged.push_back(handle);
    HSTRCloudStaging& info = mStagingInfo[n];
    const bool predicted = (b.record.flags & kBrickPredicted) != 0;
    const uint3 parity = b.record.brick() % 2u;
    info.slot = b.slot;
    info.parentSlot = parent ? parent->slot : kCloudRefNone;
    info.parity = parity.x | (parity.y << 1) | (parity.z << 2);
    // The coefficients are already in the payload pool; decodeCloudResiduals reads them from there.
    info.coded = !predicted && b.record.payloadBytes > 0 && store.payloadWord != kNone ? 1u : 0u;
    info.step = b.record.step;
    info.payloadWord = info.coded ? store.payloadWord + b.record.payloadOffset / 4 : 0u;
    info.transform = b.record.transform;
    info.brick = b.gpu;
    info.parentMin = parent ? parent->record.valueMin : 0.f;
    info.parentRange = parent ? parent->record.valueRange : 0.f;
    info.valueMin = b.record.valueMin;
    info.valueRange = b.record.valueRange;

    b.flags |= kLoaded;
    if (parent)
        ++brick(b.parent).loadedChildren;
    ++storeOf(handle).loaded;
    mLoadedList.push_back(handle);
    return true;
}

bool CloudResidency::evict(uint32_t slotsNeeded, uint32_t metasNeeded)
{
    // Loaded, unmapped, undesired leaves of the loaded set, least recently desired first.
    std::vector<uint64_t> candidates;
    size_t kept = 0;
    for (uint64_t handle : mLoadedList)
    {
        const Brick& b = brick(handle);
        if (!(b.flags & kLoaded))
            continue;
        mLoadedList[kept++] = handle;
        if (!b.mapped && !desired(b) && b.loadedChildren == 0)
            candidates.push_back(handle);
    }
    mLoadedList.resize(kept);
    std::sort(candidates.begin(), candidates.end(), [&](uint64_t a, uint64_t c) { return lastDesired(brick(a)) < lastDesired(brick(c)); });
    uint32_t freedSlots = 0;
    uint32_t freedMetas = 0;
    for (uint64_t handle : candidates)
    {
        if (freedSlots >= slotsNeeded && freedMetas >= metasNeeded)
            break;
        const Brick& b = brick(handle);
        if (b.loadedChildren > 0)
            continue;
        ++freedSlots;
        ++freedMetas;
        unload(handle);
    }
    return freedMetas > 0;
}

void CloudResidency::unload(uint64_t handle)
{
    Brick& b = brick(handle);
    FALCOR_ASSERT((b.flags & kLoaded) && !b.mapped);
    mMappedState[b.gpu] = 0;
    mFreeSlots.push_back(b.slot);
    --mStats.slotsUsed;
    mBrickOwner[b.gpu] = kNoHandle;
    // Its sun bakes go with it (on the GPU when it schedules them).
    if (mDesc.gpuSun)
    {
        mSunRelease.push_back(b.gpu);
        mSunSched[b.gpu] = HSTRCloudSunSched{};
        mSunSchedBlocksDirty[b.gpu / kBrickBlock] = 1;
        mSunState[0][b.gpu] = mSunState[1][b.gpu] = 0u;
        mSunStateBlocksDirty[b.gpu / kBrickBlock] = 1;
    }
    else
    {
        for (uint32_t k = 0; k < kCloudSunBakesPerBrick; ++k)
        {
            const size_t pair = (size_t(b.gpu) * kCloudSunBakesPerBrick + k) * 2;
            if (mSunSlotTable[pair + 1] != kCloudRefNone)
                mFreeSunSlots.push_back(mSunSlotTable[pair + 1]);
            mSunSlotTable[pair] = kCloudRefNone;
            mSunSlotTable[pair + 1] = kCloudRefNone;
        }
        mSunTableBlocksDirty[b.gpu / kBrickBlock] = 1;
    }
    mFreeBricks.push_back(b.gpu);
    b.slot = kNone;
    b.gpu = kNone;
    b.flags &= ~kLoaded;
    if (b.parent != kNoHandle)
        --brick(b.parent).loadedChildren;
    --storeOf(handle).loaded;
}

bool CloudResidency::map(uint64_t handle)
{
    Brick& b = brick(handle);
    if (mFreeNodes.size() < 2 && b.record.level < kChunkLevel)
    {
        if (!mWarnedNodes)
            logWarning("HSTRCloud: cloud page node pool exhausted; raise cloudBrickPoolMB.");
        mWarnedNodes = true;
        return false;
    }
    b.mapped = true;
    mBricks[b.gpu].fade = 0.f;
    touchBrick(b.gpu);
    // MEASURED (4K sea flight at 20 units a frame, 400 maps and 200 unmaps a frame): the page table edits of all of them cost
    // 0.09 ms of the frame, so they stay on it.
    paint(storeOf(handle).asset, b.record, b.gpu);
    ++mStats.maps;
    markChanged(storeOf(handle).asset, b.record);
    noteMapped(handle);
    if (b.parent != kNoHandle)
    {
        ++brick(b.parent).mappedChildren;
        noteMapped(b.parent);
        updateFade(b.parent); // A parent fading out holds while it has mapped children.
    }
    b.mappedIndex = uint32_t(mMappedList.size());
    mMappedList.push_back(handle);
    updateFade(handle);
    if (mDesc.gpuSun)
        writeSunSched(handle);
    return true;
}

void CloudResidency::mapReady(bool& changed)
{
    // Bricks still in the atlas all map on the frame a cut wants them back: a bounded number a frame (parents first, as listed).
    size_t kept = 0;
    uint32_t maps = 0;
    for (size_t i = 0; i < mToMap.size(); ++i)
    {
        if (maps >= kMapsPerFrame)
        {
            // The rest waits untouched.
            kept = std::copy(mToMap.begin() + i, mToMap.end(), mToMap.begin() + kept) - mToMap.begin();
            break;
        }
        const uint64_t handle = mToMap[i];
        const Brick& b = brick(handle);
        if (b.mapped)
            continue;
        const bool parentMapped = b.parent == kNoHandle || brick(b.parent).mapped;
        if (!(b.flags & kLoaded) || !parentMapped || !map(handle))
            mToMap[kept++] = handle;
        else
        {
            ++maps;
            changed = true;
        }
    }
    mToMap.resize(kept);
}

void CloudResidency::noteMapped(uint64_t handle)
{
    const Brick& b = brick(handle);
    if (b.gpu == kNone)
        return;
    const HSTRCloudBrick& gpu = mBricks[b.gpu];
    const int direction = fadeDirection(gpu);
    uint8_t state = 0;
    state |= b.mapped ? kSnapMapped : 0;
    state |= b.mappedChildren > 0 ? kSnapChildren : 0;
    state |= direction > 0 ? kSnapFadeIn : direction < 0 ? kSnapFadeOut : gpu.fade >= 1.f ? kSnapFull : 0;
    mMappedState[b.gpu] = state;
    if (mCutJob.valid())
        mWalkTouched.push_back(handle);
}

float CloudResidency::fadeOf(const HSTRCloudBrick& gpu, uint32_t frame) const
{
    const int direction = fadeDirection(gpu);
    if (direction == 0)
        return gpu.fade;
    const float delta = mFadeStep * float((frame - gpu.fadeStart) & kCloudFadeFrameMask);
    return direction > 0 ? std::min(1.f, gpu.fadeBase + delta) : std::max(0.f, gpu.fadeBase - delta);
}

void CloudResidency::updateFade(uint64_t handle)
{
    const Brick& b = brick(handle);
    if (!b.mapped)
        return;
    const HSTRCloudBrick& gpu = mBricks[b.gpu];
    // As of the last frame shown: a fade that starts now has advanced a step in this frame.
    const float fade = fadeOf(gpu, mFrame - 1);
    int direction = 0;
    if (desired(b))
        direction = fade < 1.f ? 1 : 0;
    else if (b.mappedChildren == 0)
        direction = -1;
    if (direction != fadeDirection(gpu))
        setFade(handle, direction);
}

void CloudResidency::setFade(uint64_t handle, int direction)
{
    Brick& b = brick(handle);
    HSTRCloudBrick& gpu = mBricks[b.gpu];
    const uint32_t shown = mFrame - 1;
    const float fade = fadeOf(gpu, shown);
    const int previous = fadeDirection(gpu);
    if ((previous != 0) != (direction != 0))
        mActiveFades = direction != 0 ? mActiveFades + 1 : mActiveFades - 1;
    ++b.fadeSerial;
    gpu.fade = fade;
    gpu.fadeBase = fade;
    gpu.fadeStart = (shown & kCloudFadeFrameMask) | (direction > 0 ? kCloudFadeIn : direction < 0 ? kCloudFadeOut : 0u);
    touchBrick(b.gpu);
    noteMapped(handle);
    if (direction == 0)
        return;
    // It ends when it reaches 1 or 0, at the latest fadeFrames from now; an end already due is handled this frame.
    const float left = direction > 0 ? 1.f - fade : fade;
    const uint32_t steps = std::min(uint32_t(std::ceil(std::max(0.f, left / mFadeStep - 1e-3f))), uint32_t(mFadeEnds.size() - 1));
    ++mStats.fadeStarts;
    const uint32_t end = std::max(shown + steps, mFrame);
    mFadeEnds[end % mFadeEnds.size()].push_back({handle, b.fadeSerial, mStores[handle >> 32]->generation});
}

CloudResidency::MappedAudit CloudResidency::auditMapped() const
{
    // A running walk writes only the spare cut state, which this does not read.
    MappedAudit audit;
    for (uint64_t handle : mMappedList)
    {
        const Brick& b = brick(handle);
        const HSTRCloudBrick& gpu = mBricks[b.gpu];
        if (fadeDirection(gpu) != 0 && ((mFrame - (gpu.fadeStart & kCloudFadeFrameMask)) & kCloudFadeFrameMask) > mDesc.fadeFrames + 1)
            ++audit.stale;
        if (desired(b))
            continue;
        if (fadeDirection(gpu) < 0)
            ++audit.fadingOut;
        else if (b.mappedChildren > 0)
            ++audit.held;
        else
            ++audit.idle;
    }
    return audit;
}

void CloudResidency::processFadeEnds(bool& changed)
{
    changed |= mActiveFades > 0;
    const uint32_t ring = uint32_t(mFadeEnds.size());
    const uint32_t count = std::min(mFrame - mFadeProcessed, ring);
    auto valid = [&](const FadeEnd& end)
    {
        const uint32_t s = uint32_t(end.handle >> 32);
        if (s >= mStores.size() || !mStores[s] || mStores[s]->generation != end.generation || uint32_t(end.handle) >= mStores[s]->bricks.size())
            return false;
        const Brick& b = brick(end.handle);
        return b.fadeSerial == end.serial && b.mapped;
    };
    // Ended fades in settle; ended fades out queue for their unmap.
    auto sweep = [&](std::vector<FadeEnd>& bucket)
    {
        for (const FadeEnd& end : bucket)
        {
            if (!valid(end))
            {
                ++mStats.fadeVoid;
                continue;
            }
            ++mStats.fadeEnds;
            HSTRCloudBrick& gpu = mBricks[brick(end.handle).gpu];
            changed = true;
            if (fadeDirection(gpu) > 0)
            {
                // Held at 1 exactly, whatever the rounding of the steps.
                setFade(end.handle, 0);
                gpu.fade = gpu.fadeBase = 1.f;
                noteMapped(end.handle);
            }
            else
                mUnmapQueue.push_back(end);
        }
        bucket.clear();
    };
    for (uint32_t k = 0; k < count; ++k)
        sweep(mFadeEnds[(mFrame - count + 1 + k) % ring]);
    mFadeProcessed = mFrame;
    // Bricks that faded out read as their parent already, so their unmaps spread over frames (a cut's fade-outs all end together),
    // oldest first. A parent whose last mapped child unmaps here may end at once, in this frame's bucket.
    auto& current = mFadeEnds[mFrame % ring];
    for (uint32_t unmaps = 0; unmaps < kUnmapsPerFrame;)
    {
        if (mUnmapQueue.empty())
        {
            if (current.empty())
                break;
            sweep(current);
            continue;
        }
        const FadeEnd end = mUnmapQueue.front();
        mUnmapQueue.pop_front();
        if (!valid(end) || fadeDirection(mBricks[brick(end.handle).gpu]) >= 0)
            continue;
        ++unmaps;
        setFade(end.handle, 0);
        unmap(end.handle);
        unlistMapped(end.handle);
    }
    sweep(current); // What the last unmaps started waits in the queue, not for this bucket's next turn.
}

void CloudResidency::unlistMapped(uint64_t handle)
{
    Brick& b = brick(handle);
    const uint64_t last = mMappedList.back();
    mMappedList[b.mappedIndex] = last;
    brick(last).mappedIndex = b.mappedIndex;
    mMappedList.pop_back();
    b.mappedIndex = kNone;
}

uint32_t CloudResidency::sunStateOf(const CutState& state, uint32_t cutId, bool mapped) const
{
    // The scheduler reads priorities only as histogram bins (sunCandidateBin, sunHolderBin), and classes and priority only of bricks
    // in the cut: anything finer would change the entry at every cut without changing a decision.
    const bool inCut = state.id == cutId;
    const uint32_t bin = inCut ? uint32_t(std::clamp(int(state.priority * 16.f), 0, int(kCloudSunBins) - 1)) : 0u;
    const uint32_t flags = (inCut ? state.sunClasses | kCloudSunInCut : 0u) | (mapped ? kCloudSunMapped : 0u);
    return flags | (bin << kCloudSunBinShift);
}

void CloudResidency::writeSunSched(uint64_t handle)
{
    // Both state tables take it, unless a walk is writing the spare one: then the live one, and the merge redoes the entry.
    const Brick& b = brick(handle);
    mSunSched[b.gpu] = HSTRCloudSunSched{b.record.coord, mStores[handle >> 32]->asset};
    mSunSchedBlocksDirty[b.gpu / kBrickBlock] = 1;
    const uint32_t state = sunStateOf(cutOf(b), mCutId, b.mapped);
    mSunState[mSunStateLive][b.gpu] = state;
    if (mCutJob.valid())
        mWalkTouched.push_back(handle);
    else
        mSunState[mSunStateLive ^ 1u][b.gpu] = state;
    mSunStateBlocksDirty[b.gpu / kBrickBlock] = 1;
}

void CloudResidency::unmap(uint64_t handle)
{
    Brick& b = brick(handle);
    const uint32_t replacement = b.parent != kNoHandle ? brick(b.parent).gpu : kCloudRefNone;
    replace(storeOf(handle).asset, b.record, b.gpu, replacement);
    ++mStats.unmaps;
    markChanged(storeOf(handle).asset, b.record);
    if (fadeDirection(mBricks[b.gpu]) != 0)
        setFade(handle, 0);
    b.mapped = false;
    mBricks[b.gpu].fade = 0.f;
    noteMapped(handle);
    if (b.parent != kNoHandle)
    {
        // An undesired parent fades out once its last mapped child has gone.
        Brick& parent = brick(b.parent);
        --parent.mappedChildren;
        noteMapped(b.parent);
        if (parent.mappedChildren == 0)
            updateFade(b.parent);
    }
    if (mDesc.gpuSun)
        writeSunSched(handle);
}

void CloudResidency::paintEntry(uint32_t& entry, uint32_t ref, uint32_t level)
{
    if (entry != kCloudRefNone && (entry & kCloudRefNode))
    {
        const uint32_t node = entry & ~kCloudRefNode;
        for (uint32_t k = 0; k < 64; ++k)
            paintEntry(mNodes[size_t(node) * 64 + k], ref, level);
        touchNode(node);
    }
    else if (entry == kCloudRefNone || levelOf(mBricks[entry]) > level)
        entry = ref;
}

void CloudResidency::replaceEntry(uint32_t& entry, uint32_t ref, uint32_t replacement)
{
    if (entry != kCloudRefNone && (entry & kCloudRefNode))
    {
        const uint32_t node = entry & ~kCloudRefNode;
        const uint32_t* cells = &mNodes[size_t(node) * 64];
        for (uint32_t k = 0; k < 64; ++k)
            replaceEntry(mNodes[size_t(node) * 64 + k], ref, replacement);
        touchNode(node);
        // A node whose cells all hold the same brick collapses back into its parent entry.
        const uint32_t first = cells[0];
        if (first != kCloudRefNone && (first & kCloudRefNode))
            return;
        for (uint32_t k = 1; k < 64; ++k)
            if (cells[k] != first)
                return;
        entry = first;
        mFreeNodes.push_back(node);
    }
    else if (entry == ref)
        entry = replacement;
}

uint32_t CloudResidency::ensureNode(uint32_t& entry)
{
    if (entry != kCloudRefNone && (entry & kCloudRefNode))
        return entry & ~kCloudRefNode;
    const uint32_t node = mFreeNodes.back();
    mFreeNodes.pop_back();
    std::fill_n(mNodes.begin() + size_t(node) * 64, 64, entry);
    entry = kCloudRefNode | node;
    touchNode(node);
    if (mDesc.gpuSun)
        mSunNodeResets.push_back(node); // Its fine invalidation entries belonged to another cell.
    return node;
}

void CloudResidency::paint(uint32_t assetID, const BrickHeader& record, uint32_t ref)
{
    const CloudAsset& asset = *mAssetRecords[assetID];
    const uint32_t level = record.level;
    const uint3 lo = record.brick() * (8u << level);
    const uint3 hi = min(lo + (8u << level), asset.dims);
    const uint3 chunkLo = lo / kChunkVoxels;
    const uint3 chunkHi = (hi + kChunkVoxels - 1u) / kChunkVoxels;
    for (uint32_t cz = chunkLo.z; cz < chunkHi.z; ++cz)
        for (uint32_t cy = chunkLo.y; cy < chunkHi.y; ++cy)
            for (uint32_t cx = chunkLo.x; cx < chunkHi.x; ++cx)
            {
                const uint3 chunk(cx, cy, cz);
                const size_t d = mAssets[assetID].directoryOffset + asset.chunkIndex(chunk);
                touchDirectory(d);
                if (level >= kChunkLevel)
                {
                    paintEntry(mDirectory[d], ref, level);
                    continue;
                }
                const uint32_t node = ensureNode(mDirectory[d]);
                const uint3 origin = chunk * kChunkVoxels;
                const uint3 cellLo = (max(lo, origin) - origin) / 32u;
                const uint3 cellHi = (min(hi, origin + kChunkVoxels) - origin + 31u) / 32u;
                for (uint32_t z = cellLo.z; z < cellHi.z; ++z)
                    for (uint32_t y = cellLo.y; y < cellHi.y; ++y)
                        for (uint32_t x = cellLo.x; x < cellHi.x; ++x)
                        {
                            const size_t cell = size_t(node) * 64 + x + 4 * (y + 4 * z);
                            touchNode(node);
                            if (level >= 2)
                            {
                                paintEntry(mNodes[cell], ref, level);
                                continue;
                            }
                            const uint32_t sub = ensureNode(mNodes[cell]);
                            const uint3 cellOrigin = origin + uint3(x, y, z) * 32u;
                            const uint3 subLo = (max(lo, cellOrigin) - cellOrigin) / 8u;
                            const uint3 subHi = (min(hi, cellOrigin + 32u) - cellOrigin + 7u) / 8u;
                            for (uint32_t sz = subLo.z; sz < subHi.z; ++sz)
                                for (uint32_t sy = subLo.y; sy < subHi.y; ++sy)
                                    for (uint32_t sx = subLo.x; sx < subHi.x; ++sx)
                                        paintEntry(mNodes[size_t(sub) * 64 + sx + 4 * (sy + 4 * sz)], ref, level);
                            touchNode(sub);
                        }
            }
}

void CloudResidency::replace(uint32_t assetID, const BrickHeader& record, uint32_t ref, uint32_t replacement)
{
    // Only the entries inside the brick's region can name it: walk those, collapsing nodes that became uniform.
    const CloudAsset& asset = *mAssetRecords[assetID];
    const uint32_t level = record.level;
    const uint3 lo = record.brick() * (8u << level);
    const uint3 hi = min(lo + (8u << level), asset.dims);
    const uint3 chunkLo = lo / kChunkVoxels;
    const uint3 chunkHi = (hi + kChunkVoxels - 1u) / kChunkVoxels;
    auto isNode = [](uint32_t entry) { return entry != kCloudRefNone && (entry & kCloudRefNode); };
    auto collapse = [&](uint32_t& entry)
    {
        const uint32_t node = entry & ~kCloudRefNode;
        const uint32_t* cells = &mNodes[size_t(node) * 64];
        if (isNode(cells[0]))
            return;
        for (uint32_t k = 1; k < 64; ++k)
            if (cells[k] != cells[0])
                return;
        entry = cells[0];
        mFreeNodes.push_back(node);
    };
    for (uint32_t cz = chunkLo.z; cz < chunkHi.z; ++cz)
        for (uint32_t cy = chunkLo.y; cy < chunkHi.y; ++cy)
            for (uint32_t cx = chunkLo.x; cx < chunkHi.x; ++cx)
            {
                const uint3 chunk(cx, cy, cz);
                const size_t d = mAssets[assetID].directoryOffset + asset.chunkIndex(chunk);
                touchDirectory(d);
                if (level >= kChunkLevel || !isNode(mDirectory[d]))
                {
                    replaceEntry(mDirectory[d], ref, replacement);
                    continue;
                }
                const uint32_t node = mDirectory[d] & ~kCloudRefNode;
                const uint3 origin = chunk * kChunkVoxels;
                const uint3 cellLo = (max(lo, origin) - origin) / 32u;
                const uint3 cellHi = (min(hi, origin + kChunkVoxels) - origin + 31u) / 32u;
                for (uint32_t z = cellLo.z; z < cellHi.z; ++z)
                    for (uint32_t y = cellLo.y; y < cellHi.y; ++y)
                        for (uint32_t x = cellLo.x; x < cellHi.x; ++x)
                        {
                            uint32_t& cell = mNodes[size_t(node) * 64 + x + 4 * (y + 4 * z)];
                            if (level >= 2 || !isNode(cell))
                            {
                                replaceEntry(cell, ref, replacement);
                                continue;
                            }
                            const uint32_t sub = cell & ~kCloudRefNode;
                            const uint3 cellOrigin = origin + uint3(x, y, z) * 32u;
                            const uint3 subLo = (max(lo, cellOrigin) - cellOrigin) / 8u;
                            const uint3 subHi = (min(hi, cellOrigin + 32u) - cellOrigin + 7u) / 8u;
                            for (uint32_t sz = subLo.z; sz < subHi.z; ++sz)
                                for (uint32_t sy = subLo.y; sy < subHi.y; ++sy)
                                    for (uint32_t sx = subLo.x; sx < subHi.x; ++sx)
                                        replaceEntry(mNodes[size_t(sub) * 64 + sx + 4 * (sy + 4 * sz)], ref, replacement);
                            touchNode(sub);
                            collapse(cell);
                        }
                touchNode(node);
                collapse(mDirectory[d]);
            }
}

void CloudResidency::touchDirectory(size_t index)
{
    mDirectoryDirty[0] = std::min(mDirectoryDirty[0], index);
    mDirectoryDirty[1] = std::max(mDirectoryDirty[1], index + 1);
}

void CloudResidency::touchNode(size_t node)
{
    mNodeBlocksDirty[node / kNodeBlock] = 1;
}

void CloudResidency::touchBrick(uint32_t gpu)
{
    mBrickBlocksDirty[gpu / kBrickBlock] = 1;
}

void CloudResidency::upload()
{
    if (mInstancesDirty)
    {
        mpInstances->setBlob(mInstances.data(), 0, mInstances.size() * sizeof(HSTRCloudInstance));
        mInstancesDirty = false;
    }
    if (mDirectoryDirty[0] < mDirectoryDirty[1])
    {
        mpDirectory->setBlob(
            mDirectory.data() + mDirectoryDirty[0], mDirectoryDirty[0] * sizeof(uint32_t), (mDirectoryDirty[1] - mDirectoryDirty[0]) * sizeof(uint32_t)
        );
        mDirectoryDirty[0] = SIZE_MAX;
        mDirectoryDirty[1] = 0;
    }
    // Runs of dirty blocks upload as one range: every upload costs its own staging, mapping and copy command, and a cut dirties
    // blocks all over the scheduler table.
    auto uploadRuns = [](std::vector<uint8_t>& dirty, size_t blockElements, const void* pData, size_t elements, size_t elementSize, Buffer* pBuffer)
    {
        for (size_t block = 0; block < dirty.size();)
        {
            if (!dirty[block])
            {
                ++block;
                continue;
            }
            size_t end = block;
            while (end < dirty.size() && dirty[end])
                dirty[end++] = 0;
            const size_t first = block * blockElements;
            const size_t count = std::min(elements, end * blockElements) - first;
            pBuffer->setBlob(static_cast<const uint8_t*>(pData) + first * elementSize, first * elementSize, count * elementSize);
            block = end;
        }
    };
    uploadRuns(mNodeBlocksDirty, size_t(kNodeBlock) * 64, mNodes.data(), mNodes.size(), sizeof(uint32_t), mpNodes.get());
    uploadRuns(mBrickBlocksDirty, kBrickBlock, mBricks.data(), mBricks.size(), sizeof(HSTRCloudBrick), mpBricks.get());
    if (!mStaged.empty())
        mpStagingInfo->setBlob(mStagingInfo.data(), 0, mStaged.size() * sizeof(HSTRCloudStaging));
    if (!mSunBakes.empty())
        mpSunBakes->setBlob(mSunBakes.data(), 0, mSunBakes.size() * sizeof(HSTRCloudSunBake));
    if (mDesc.gpuSun)
    {
        uploadRuns(mSunSchedBlocksDirty, kBrickBlock, mSunSched.data(), mSunSched.size(), sizeof(HSTRCloudSunSched), mpSunSched.get());
        const auto& state = mSunState[mSunStateLive];
        uploadRuns(mSunStateBlocksDirty, kBrickBlock, state.data(), state.size(), sizeof(uint32_t), mpSunState.get());
        return; // The GPU owns the sun slot table.
    }
    const size_t pairsPerBrick = 2 * kCloudSunBakesPerBrick;
    uploadRuns(
        mSunTableBlocksDirty, kBrickBlock * pairsPerBrick, mSunSlotTable.data(), mSunSlotTable.size(), sizeof(uint32_t), mpSunSlotTable.get()
    );
}

uint32_t CloudResidency::sunClassOf(const HSTRCloudInstance& instance)
{
    // The rows are a scaled signed permutation: normalised and rounded, one of eight matrices per class.
    float3x3 m;
    const float4 rows[3] = {instance.row0, instance.row1, instance.row2};
    for (uint32_t r = 0; r < 3; ++r)
    {
        const float3 row = normalize(rows[r].xyz());
        for (uint32_t c = 0; c < 3; ++c)
            m[r][c] = std::round(row[c]);
    }
    for (uint32_t k = 0; k < mSunClasses.size(); ++k)
        if (all(mSunClasses[k][0] == m[0]) && all(mSunClasses[k][1] == m[1]) && all(mSunClasses[k][2] == m[2]))
            return k;
    FALCOR_CHECK(mSunClasses.size() < kCloudSunClasses, "HSTRCloud: more than {} cloud orientation classes.", kCloudSunClasses);
    mSunClasses.push_back(m);
    return uint32_t(mSunClasses.size() - 1);
}

void CloudResidency::markChanged(uint32_t assetID, const BrickHeader& record)
{
    // Mapping or unmapping a brick changes what samples read inside its region only.
    AssetState& state = mAssets[assetID];
    const uint3 lo = record.brick() * (8u << record.level);
    const uint3 hi = min(lo + (8u << record.level), mAssetRecords[assetID]->dims);
    const uint3 cellLo = lo / 32u;
    const uint3 cellHi = min((hi + 31u) / 32u, state.changeDims);
    state.lastChange = mFrame;
    if (mDesc.gpuSun)
    {
        // stampSunChanges applies it on the GPU.
        auto pack = [](uint3 c) { return c.x | (c.y << 10) | (c.z << 20); };
        mSunChanges.push_back({assetID, pack(cellLo), pack(cellHi - 1u)});
        return;
    }
    for (uint32_t z = cellLo.z; z < cellHi.z; ++z)
        for (uint32_t y = cellLo.y; y < cellHi.y; ++y)
            for (uint32_t x = cellLo.x; x < cellHi.x; ++x)
                state.changeFrame[x + state.changeDims.x * (size_t(y) + state.changeDims.y * size_t(z))] = mFrame;
}

void CloudResidency::acquireSunField(uint32_t slot)
{
    const uint32_t pair = mInstances[slot].asset * kCloudSunClasses + mInstances[slot].sunClass;
    mSlotSunPair[slot] = pair;
    ++mSunFieldUsers[pair];
    if (mSunFieldBlocks[pair] != kNone)
        return;
    // A pair keeps its block while no other pair needs it: losing it drops every bake of the pair, sea-wide, and a sea window
    // shifting by a tile briefly leaves classes without an instance.
    uint32_t block = kNone;
    if (!mFreeSunFieldBlocks.empty())
    {
        block = mFreeSunFieldBlocks.back();
        mFreeSunFieldBlocks.pop_back();
    }
    else
    {
        for (uint32_t other = 0; other < mSunFieldBlocks.size() && block == kNone; ++other)
            if (mSunFieldBlocks[other] != kNone && mSunFieldUsers[other] == 0)
            {
                block = mSunFieldBlocks[other];
                mSunFieldBlocks[other] = kNone;
            }
        FALCOR_CHECK(block != kNone, "HSTRCloud: no free sun invalidation block.");
    }
    // The pair's bakes held from an earlier use missed the changes since: a block starts at the current frame.
    mSunFieldBlocks[pair] = block;
    mSunBlockResets.push_back(block);
    mSunFieldBlocksDirty = true;
}

void CloudResidency::releaseSunField(uint32_t slot)
{
    const uint32_t pair = mSlotSunPair[slot];
    if (pair == kNone)
        return;
    mSlotSunPair[slot] = kNone;
    --mSunFieldUsers[pair];
}

void CloudResidency::uploadSunChanges()
{
    // Changes over the same cells stamp the same entries: most are fine bricks sharing a cell.
    std::sort(
        mSunChanges.begin(),
        mSunChanges.end(),
        [](const HSTRCloudSunChange& a, const HSTRCloudSunChange& b)
        { return std::tie(a.asset, a.first, a.last) < std::tie(b.asset, b.first, b.last); }
    );
    mSunChanges.erase(
        std::unique(
            mSunChanges.begin(),
            mSunChanges.end(),
            [](const HSTRCloudSunChange& a, const HSTRCloudSunChange& b) { return a.asset == b.asset && a.first == b.first && a.last == b.last; }
        ),
        mSunChanges.end()
    );
    if (mSunChanges.size() > mSunChangeCapacity)
    {
        // More than the buffer holds: the whole field restarts instead.
        mSunResetAll = true;
        mSunChanges.clear();
    }
    if (mSunResetAll)
    {
        mSunBlockResets.clear();
        for (uint32_t block : mSunFieldBlocks)
            if (block != kNone)
                mSunBlockResets.push_back(block);
        mSunNodeResets.resize(mNodes.size() / 64);
        std::iota(mSunNodeResets.begin(), mSunNodeResets.end(), 0u);
        mSunResetAll = false;
    }
    if (mSunFieldBlocksDirty)
    {
        // The shader reads a block's first entry.
        std::vector<uint32_t> offsets(mSunFieldBlocks.size(), kCloudRefNone);
        for (size_t pair = 0; pair < offsets.size(); ++pair)
            if (mSunFieldBlocks[pair] != kNone)
                offsets[pair] = mSunFieldBlocks[pair] * mSunFieldBlockSize;
        mpSunFieldBlocks->setBlob(offsets.data(), 0, offsets.size() * sizeof(uint32_t));
        mSunFieldBlocksDirty = false;
    }
    // resetSunField restarts these ranges on the GPU: the blocks' first entries, then the nodes'.
    auto unique = [](std::vector<uint32_t>& list)
    {
        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
    };
    unique(mSunBlockResets);
    unique(mSunNodeResets);
    std::vector<uint32_t> resets;
    resets.reserve(mSunBlockResets.size() + mSunNodeResets.size());
    for (uint32_t block : mSunBlockResets)
        resets.push_back(block * mSunFieldBlockSize);
    for (uint32_t node : mSunNodeResets)
        resets.push_back(mSunFineBase + node * kCloudSunFineEntries);
    if (!resets.empty())
        mpSunResets->setBlob(resets.data(), 0, resets.size() * sizeof(uint32_t));
    mGpuSunFrame.info.blockResets = uint32_t(mSunBlockResets.size());
    mGpuSunFrame.info.nodeResets = uint32_t(mSunNodeResets.size());
    mGpuSunFrame.info.blockLength = mSunFieldBlockSize;
    mGpuSunFrame.info.fineBase = mSunFineBase;
    mSunBlockResets.clear();
    mSunNodeResets.clear();
    mGpuSunFrame.info.changeCount = uint32_t(mSunChanges.size());
    if (!mSunChanges.empty())
        mpSunChanges->setBlob(mSunChanges.data(), 0, mSunChanges.size() * sizeof(HSTRCloudSunChange));
    mSunChanges.clear();
}

bool CloudResidency::sunBakeStale(uint32_t assetID, const BrickHeader& record, float3 direction, float reach, uint32_t bakeFrame) const
{
    // A bake read the bricks mapped over its texels (with their trilinear support) swept towards the sun by the reach. If any of
    // that region was mapped or unmapped since, the bake no longer equals the live march.
    const AssetState& state = mAssets[assetID];
    if (state.lastChange <= bakeFrame)
        return false;
    const float scale = float(1u << record.level);
    const float3 origin = float3(record.brick() * (8u << record.level));
    const float3 sweep = reach * direction;
    const float3 lo = origin - 1.5f * scale - 0.5f + min(sweep, float3(0.f)) - scale;
    const float3 hi = origin + 9.5f * scale - 0.5f + max(sweep, float3(0.f)) + scale;
    const int3 maximum = int3(state.changeDims) - 1;
    const int3 first = clamp(int3(floor(lo / 32.f)), int3(0), maximum);
    const int3 last = clamp(int3(floor(hi / 32.f)), int3(0), maximum);
    for (int32_t z = first.z; z <= last.z; ++z)
        for (int32_t y = first.y; y <= last.y; ++y)
            for (int32_t x = first.x; x <= last.x; ++x)
                if (state.changeFrame[x + state.changeDims.x * (size_t(y) + state.changeDims.y * size_t(z))] > bakeFrame)
                    return true;
    return false;
}

void CloudResidency::scheduleSunBakes(const CloudSea& sea, const CloudView& view)
{
    mSunBakes.clear();
    mGpuSunFrame.run = false;
    mGpuSunRan = false;
    if (mDesc.gpuSun)
    {
        // ageSunField runs every frame, idle or not: a few field rows each, so every row is visited every 1024 frames, well
        // within the 32768 frames a 16-bit age has left once clamped.
        HSTRCloudSunFrame& info = mGpuSunFrame.info;
        info = {};
        info.frame = mFrame;
        info.generation = view.sunGeneration;
        info.reachScale = view.sunNearVoxels * sea.getVoxelWorld();
        info.ageRow = mSunAgeRow;
        info.ageRows = std::min((mSunFieldRows + 1023) / 1024, mSunFieldRows - mSunAgeRow);
        mSunAgeRow = mSunAgeRow + info.ageRows >= mSunFieldRows ? 0 : mSunAgeRow + info.ageRows;
    }
    if (view.sunNearVoxels <= 0.f || mDesc.sunBakesPerFrame == 0 || mSunClasses.empty())
    {
        if (mDesc.gpuSun && !mSunChanges.empty())
        {
            // Nothing stamps while baking is off: the whole field restarts when it resumes.
            mSunResetAll = true;
            mSunChanges.clear();
        }
        return;
    }
    if (mDesc.gpuSun)
    {
        // The GPU schedules (HSTRCloud runs the passes over what upload() sends); the CPU only describes the frame.
        const float3 sunDirection = normalize(view.sunBakeDirection);
        SunScheduleInputs inputs;
        inputs.cutFrame = mCutId;
        inputs.generation = view.sunGeneration;
        for (const AssetState& state : mAssets)
            inputs.lastChange = std::max(inputs.lastChange, state.lastChange);
        inputs.freeSlots = mSunRelease.size(); // Releases change the pool: never idle across one.
        inputs.mapped = mMappedList.size();
        inputs.direction = sunDirection;
        mGpuSunInputs = inputs;
        const bool fieldChanged =
            mSunFieldBlocksDirty || mSunResetAll || !mSunBlockResets.empty() || !mSunNodeResets.empty() || !mSunChanges.empty();
        if (mSunRelease.empty() && !fieldChanged && mGpuSunIdleValid && inputs == mGpuSunIdle)
        {
            mStats.sunBakesFrame = 0;
            return;
        }
        for (size_t c = 0; c < mSunClasses.size(); ++c)
            mSunDirections[c] = float4(normalize(mul(mSunClasses[c], sunDirection)), 0.f);
        mpSunDirections->setBlob(mSunDirections.data(), 0, mSunDirections.size() * sizeof(float4));
        const uint32_t releases = uint32_t(std::min(mSunRelease.size(), mSunSched.size()));
        if (releases > 0)
            mpSunRelease->setBlob(mSunRelease.data(), 0, releases * sizeof(uint32_t));
        mSunRelease.clear();
        mGpuSunFrame.run = true;
        mGpuSunRan = true;
        mGpuSunFrame.info.releaseCount = releases;
        mGpuSunFrame.info.classCount = uint32_t(mSunClasses.size());
        mGpuSunFrame.info.bakeMax = mDesc.sunBakesPerFrame;
        mGpuSunFrame.info.capacity = uint32_t(mSunSched.size());
        uploadSunChanges();
        return;
    }
    // For every mapped brick and every orientation class that desired it: a bake pair with the current key whose read region has not
    // been mapped or unmapped since, or a candidate bake. Candidates bake most important brick first; until then, and when a brick
    // already holds kCloudSunBakesPerBrick wanted bakes or no sun slot is free, those samples take the live march.
    //
    // MEASURED (4K, single-asset sea of 64 instances, 2026-09-15): the sea's 8 orientation classes all want the coarse bricks every
    // instance shares, so 4 bakes per brick left the most important candidates unplaceable and baking stalled at 0.2% of the queries
    // answered; 8 pairs per brick plus the eviction below reach 96% (near) / 95% (farside) / 77% (sea, where the slot pool runs out
    // at 271k bakes for 315k wanted). The bake pass itself costs 0.07 ms per frame at 256 bakes, so the budget has room: a camera
    // flight leaves up to 190k bakes waiting and pays for it in the live march (41 -> 124 ms at 4 units per frame).
    const float3 sun = normalize(view.sunBakeDirection);
    // MEASURED (4K sea, settled, slot pool full): 18 ms of CPU every frame to conclude, every frame, that nothing can be baked - the
    // walk below visits every mapped brick and, with no free slot, sorts every held pair as well. A repeat with the same inputs
    // reaches the same conclusion, and the stats it set still stand.
    SunScheduleInputs inputs;
    inputs.cutFrame = mCutId;
    inputs.generation = view.sunGeneration;
    for (const AssetState& state : mAssets)
        inputs.lastChange = std::max(inputs.lastChange, state.lastChange);
    inputs.freeSlots = mFreeSunSlots.size();
    inputs.mapped = mMappedList.size();
    inputs.direction = sun;
    if (mSunScheduleIdleValid && inputs == mSunScheduleIdle)
    {
        mStats.sunBakesFrame = 0;
        return;
    }
    auto reachOf = [&](uint32_t assetID)
    { return view.sunNearVoxels * sea.getVoxelWorld() / (mAssetRecords[assetID]->voxelWorld * sea.getFitScale()); };
    struct Candidate
    {
        float priority;
        uint64_t handle;
        uint32_t sunClass;
    };
    std::vector<Candidate> candidates;
    std::vector<float3> classDirections(mSunClasses.size());
    for (size_t c = 0; c < mSunClasses.size(); ++c)
        classDirections[c] = normalize(mul(mSunClasses[c], sun));
    // Each brick reads and writes only its own pairs of the slot table, so the walk runs in parallel chunks of the mapped list; the
    // counters, dirty blocks and candidates of each chunk merge afterwards, in chunk order (the candidate order the serial walk had).
    // MEASURED (sea flight, 127k mapped bricks, 8 classes): ~50 ms a frame serially whenever the cut moved.
    struct Chunk
    {
        std::vector<Candidate> candidates;
        std::vector<uint32_t> dirtyBlocks;
        uint32_t baked = 0;
        uint32_t waiting = 0;
    };
    const size_t chunkCount = std::clamp<size_t>(std::thread::hardware_concurrency(), 1, 16);
    const size_t chunkSize = (mMappedList.size() + chunkCount - 1) / std::max<size_t>(chunkCount, 1);
    std::vector<Chunk> chunks(chunkCount);
    auto parallel = [&](const std::function<void(size_t)>& body)
    {
        std::vector<std::future<void>> workers;
        for (size_t index = 1; index < chunkCount; ++index)
            workers.push_back(std::async(std::launch::async, body, index));
        body(0);
        for (auto& worker : workers)
            worker.get();
    };
    // Each chunk keeps only its most important consideredMax candidates (a bounded min-heap): the pass considers no more than that
    // many, and collecting every waiting pair (100k+ on a moving sea) and sorting them cost several ms a frame.
    const size_t consideredMax = size_t(4 * mDesc.sunBakesPerFrame);
    auto higher = [](const Candidate& a, const Candidate& c) { return a.priority > c.priority; };
    auto keep = [&](std::vector<Candidate>& heap, const Candidate& candidate)
    {
        if (heap.size() == consideredMax)
        {
            if (candidate.priority <= heap.front().priority)
                return;
            std::pop_heap(heap.begin(), heap.end(), higher);
            heap.pop_back();
        }
        heap.push_back(candidate);
        std::push_heap(heap.begin(), heap.end(), higher);
    };
    auto scan = [&](size_t index)
    {
        Chunk& chunk = chunks[index];
        const size_t first = index * chunkSize;
        const size_t last = std::min(mMappedList.size(), first + chunkSize);
        for (size_t i = first; i < last; ++i)
        {
            const uint64_t handle = mMappedList[i];
            const Brick& b = brick(handle);
            if (!desired(b))
                continue; // Cached but outside the cut: its bakes are the first evicted.
            const CutState& cut = cutOf(b);
            const uint32_t assetID = mStores[handle >> 32]->asset;
            const float reach = reachOf(assetID);
            uint32_t needs = 0;
            for (uint32_t sunClass = 0; sunClass < mSunClasses.size(); ++sunClass)
            {
                if (!(cut.sunClasses & (1u << sunClass)))
                    continue;
                const uint32_t key = (view.sunGeneration << 4) | sunClass;
                const size_t base = size_t(b.gpu) * kCloudSunBakesPerBrick;
                bool baked = false;
                for (uint32_t k = 0; k < kCloudSunBakesPerBrick && !baked; ++k)
                {
                    if (mSunSlotTable[(base + k) * 2] != key)
                        continue;
                    if (!sunBakeStale(assetID, b.record, classDirections[sunClass], reach, mSunBakeFrames[base + k]))
                        baked = true;
                    else
                    {
                        mSunSlotTable[(base + k) * 2] = kCloudRefNone; // Keeps its slot for the rebake.
                        chunk.dirtyBlocks.push_back(b.gpu / kBrickBlock);
                    }
                }
                if (baked)
                {
                    ++chunk.baked;
                    continue;
                }
                ++chunk.waiting;
                needs |= 1u << sunClass;
            }
            // Only if a pair can take a bake: every pair holding a wanted bake leaves the brick's other classes on the live march.
            uint32_t held = 0;
            for (uint32_t k = 0; k < kCloudSunBakesPerBrick; ++k)
            {
                const uint32_t key = mSunSlotTable[(size_t(b.gpu) * kCloudSunBakesPerBrick + k) * 2];
                held += key != kCloudRefNone && (key >> 4) == view.sunGeneration && (cut.sunClasses & (1u << (key & 15u))) ? 1 : 0;
            }
            for (uint32_t sunClass = 0; sunClass < mSunClasses.size() && held < kCloudSunBakesPerBrick; ++sunClass)
                if (needs & (1u << sunClass))
                {
                    keep(chunk.candidates, {cut.priority, handle, sunClass});
                    ++held;
                }
        }
    };
    auto* pRenderContext = mpDevice->getRenderContext();
    {
        FALCOR_PROFILE(pRenderContext, "scan");
        parallel(scan);
    }
    FALCOR_PROFILE(pRenderContext, "assign");
    mStats.sunBaked = 0;
    mStats.sunWaiting = 0;
    mStats.sunStale = 0;
    for (const Chunk& chunk : chunks)
    {
        mStats.sunBaked += chunk.baked;
        mStats.sunWaiting += chunk.waiting;
        mStats.sunStale += uint32_t(chunk.dirtyBlocks.size());
        for (uint32_t block : chunk.dirtyBlocks)
            mSunTableBlocksDirty[block] = 1;
        candidates.insert(candidates.end(), chunk.candidates.begin(), chunk.candidates.end());
    }
    const size_t considered = std::min(candidates.size(), consideredMax);
    std::partial_sort(
        candidates.begin(), candidates.begin() + considered, candidates.end(), [](const Candidate& a, const Candidate& c) { return a.priority > c.priority; }
    );
    // When the pool is short, slots held by the least important bakes (unwanted pairs and bricks outside the cut first) move to more
    // important candidates.
    struct Holder
    {
        float priority;
        uint32_t pair;
    };
    std::vector<Holder> holders;
    size_t nextHolder = 0;
    if (considered > 0 && mFreeSunSlots.size() < std::min(considered, size_t(mDesc.sunBakesPerFrame)))
    {
        // Only the least important sunBakesPerFrame holders can be handed over, and only to a candidate more important than they
        // are: a bounded max-heap keeps those, where this used to collect every held pair (~800k on a moving sea) and sort them. One
        // heap per chunk of the mapped list, in parallel (the pool is full on a moving sea, so this ran every frame), then merged.
        const size_t needed = mDesc.sunBakesPerFrame;
        const float best = candidates.front().priority;
        auto lower = [](const Holder& a, const Holder& c) { return a.priority < c.priority; };
        std::vector<std::vector<Holder>> heaps(chunkCount);
        auto collect = [&](size_t index)
        {
            std::vector<Holder>& heap = heaps[index];
            heap.reserve(needed);
            const size_t first = index * chunkSize;
            const size_t last = std::min(mMappedList.size(), first + chunkSize);
            for (size_t i = first; i < last; ++i)
            {
                const Brick& b = brick(mMappedList[i]);
                const bool inCut = desired(b);
                const CutState& cut = cutOf(b);
                for (uint32_t k = 0; k < kCloudSunBakesPerBrick; ++k)
                {
                    const uint32_t pair = b.gpu * kCloudSunBakesPerBrick + k;
                    if (mSunSlotTable[pair * 2 + 1] == kCloudRefNone)
                        continue;
                    const uint32_t key = mSunSlotTable[pair * 2];
                    const bool wanted =
                        inCut && key != kCloudRefNone && (key >> 4) == view.sunGeneration && (cut.sunClasses & (1u << (key & 15u)));
                    const float priority = wanted ? cut.priority : (inCut ? -1.f : -2.f);
                    if (priority >= best || (heap.size() == needed && priority >= heap.front().priority))
                        continue;
                    if (heap.size() == needed)
                    {
                        std::pop_heap(heap.begin(), heap.end(), lower);
                        heap.pop_back();
                    }
                    heap.push_back({priority, pair});
                    std::push_heap(heap.begin(), heap.end(), lower);
                }
            }
        };
        {
            FALCOR_PROFILE(pRenderContext, "holders");
            parallel(collect);
        }
        for (const std::vector<Holder>& heap : heaps)
            holders.insert(holders.end(), heap.begin(), heap.end());
        const size_t kept = std::min(holders.size(), needed);
        std::partial_sort(holders.begin(), holders.begin() + kept, holders.end(), lower);
        holders.resize(kept);
    }
    for (size_t c = 0; c < considered && mSunBakes.size() < mDesc.sunBakesPerFrame; ++c)
    {
        const Brick& b = brick(candidates[c].handle);
        const size_t base = size_t(b.gpu) * kCloudSunBakesPerBrick;
        // A pair of this brick to (re)bake into: an invalid one holding a slot, one baked for a class nobody wants any more (or an old
        // generation), else an empty one taking a free slot.
        uint32_t chosen = kNone;
        for (uint32_t k = 0; k < kCloudSunBakesPerBrick && chosen == kNone; ++k)
        {
            const uint32_t key = mSunSlotTable[(base + k) * 2];
            const bool unwanted = key == kCloudRefNone || (key >> 4) != view.sunGeneration || !(cutOf(b).sunClasses & (1u << (key & 15u)));
            if (mSunSlotTable[(base + k) * 2 + 1] != kCloudRefNone && unwanted)
                chosen = k;
        }
        for (uint32_t k = 0; k < kCloudSunBakesPerBrick && chosen == kNone && !mFreeSunSlots.empty(); ++k)
            if (mSunSlotTable[(base + k) * 2 + 1] == kCloudRefNone)
            {
                chosen = k;
                mSunSlotTable[(base + k) * 2 + 1] = mFreeSunSlots.back();
                mFreeSunSlots.pop_back();
            }
        for (uint32_t k = 0; k < kCloudSunBakesPerBrick && chosen == kNone; ++k)
        {
            if (mSunSlotTable[(base + k) * 2 + 1] != kCloudRefNone)
                continue;
            // Skip holders already rebaked this frame (a candidate's own pair, or one handed over earlier).
            while (nextHolder < holders.size() && mSunBakeFrames[holders[nextHolder].pair] == mFrame)
                ++nextHolder;
            if (nextHolder == holders.size() || holders[nextHolder].priority >= candidates[c].priority)
                break;
            const uint32_t pair = holders[nextHolder++].pair;
            chosen = k;
            mSunSlotTable[(base + k) * 2 + 1] = mSunSlotTable[pair * 2 + 1];
            mSunSlotTable[pair * 2] = kCloudRefNone;
            mSunSlotTable[pair * 2 + 1] = kCloudRefNone;
            mSunTableBlocksDirty[(pair / kCloudSunBakesPerBrick) / kBrickBlock] = 1;
        }
        if (chosen == kNone)
            continue;
        const uint32_t assetID = storeOf(candidates[c].handle).asset;
        const uint32_t sunClass = candidates[c].sunClass;
        const float sourceVoxelWorld = mAssetRecords[assetID]->voxelWorld * sea.getFitScale();
        HSTRCloudSunBake job;
        job.brick = b.gpu;
        job.slot = mSunSlotTable[(base + chosen) * 2 + 1];
        job.asset = assetID;
        job.level = b.record.level;
        job.origin = float3(b.record.brick() * 8u);
        job.reach = reachOf(assetID);
        job.direction = normalize(mul(mSunClasses[sunClass], sun));
        job.cap = kCloudSunDepthOpaque / (kCloudSeaMinScale * sourceVoxelWorld);
        mSunBakes.push_back(job);
        mSunSlotTable[(base + chosen) * 2] = (view.sunGeneration << 4) | sunClass;
        mSunBakeFrames[base + chosen] = mFrame;
        mSunTableBlocksDirty[b.gpu / kBrickBlock] = 1;
    }
    mStats.sunBakesFrame = uint32_t(mSunBakes.size());
    mStats.sunSlotsFree = uint32_t(mFreeSunSlots.size());
    // Idle only if this pass staged nothing and left the slots as it found them (an eviction without a bake still moved them).
    mSunScheduleIdleValid = mSunBakes.empty() && mFreeSunSlots.size() == inputs.freeSlots;
    mSunScheduleIdle = inputs;
}

void CloudResidency::bind(const ShaderVar& var) const
{
    var["hstrCloudInstances"] = mpInstances;
    var["hstrCloudAssets"] = mpAssets;
    var["hstrCloudDirectory"] = mpDirectory;
    var["hstrCloudNodes"] = mpNodes;
    var["hstrCloudBricks"] = mpBricks;
    var["hstrCloudAtlas"] = mpAtlas;
    var["hstrCloudOccupancy"] = mpOccupancy;
    var["hstrCloudSunAtlas"] = mpSunAtlas;
    var["hstrCloudSunBakes"] = mpSunBakes;
    var["hstrCloudSunSlots"] = mpSunSlotTable;
    var["hstrCloudPayload"] = mpPayload->getBuffer();
    var["hstrCloudResiduals"] = mpResiduals;
    var["hstrCloudStagingInfo"] = mpStagingInfo;
}

void CloudResidency::bindGpuSun(const ShaderVar& var) const
{
    var["hstrCloudSunSlots"] = ref<Buffer>();
    var["hstrCloudSunBakes"] = ref<Buffer>();
    var["hstrCloudSunSlotsOutput"] = mpSunSlotTable;
    var["hstrCloudSunBakesOutput"] = mpSunBakes;
    var["hstrCloudSunSched"] = mpSunSched;
    var["hstrCloudSunState"] = mpSunState;
    var["hstrCloudSunFrames"] = mpSunFrames;
    var["hstrCloudSunFree"] = mpSunFree;
    var["hstrCloudSunFreeTop"] = mpSunFreeTop;
    var["hstrCloudSunCounters"] = mpSunCounters;
    var["hstrCloudSunHistogram"] = mpSunHistogram;
    var["hstrCloudSunSelect"] = mpSunSelect;
    var["hstrCloudSunNeeds"] = mpSunNeeds;
    var["hstrCloudSunCandidates"] = mpSunCandidates;
    var["hstrCloudSunHolders"] = mpSunHolders;
    var["hstrCloudSunRelease"] = mpSunRelease;
    var["hstrCloudSunDirections"] = mpSunDirections;
    var["hstrCloudSunField"] = mpSunField;
    var["hstrCloudSunFieldBlocks"] = mpSunFieldBlocks;
    var["hstrCloudSunFieldLevels"] = mpSunFieldLevels;
    var["hstrCloudSunChanges"] = mpSunChanges;
    var["hstrCloudSunResets"] = mpSunResets;
    var["hstrCloudSunFrameInfo"] = mpSunFrameInfo;
}

void CloudResidency::readGpuSunStats()
{
    if (!mDesc.gpuSun)
        return;
    // A frame the scheduler did not run keeps the last run's counts; it staged nothing itself.
    mStats.sunBakesFrame = mGpuSunRan ? mpSunCounters->getElement<uint32_t>(2) : 0;
    mStats.sunBaked = mpSunCounters->getElement<uint32_t>(3);
    mStats.sunWaiting = mpSunCounters->getElement<uint32_t>(4);
    mStats.sunStale = mpSunCounters->getElement<uint32_t>(5);
    mStats.sunSlotsFree = mpSunFreeTop->getElement<uint32_t>(0);
}
} // namespace hstrcloud
