/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "CloudResidency.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <optional>
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
    mpBricks = mpDevice->createStructuredBuffer(
        sizeof(HSTRCloudBrick), uint32_t(mBricks.size()), shaderResource, MemoryType::DeviceLocal, mBricks.data(), false
    );
    mpAtlas = mpDevice->createTexture3D(
        kAtlasBricksXY * 10u,
        kAtlasBricksXY * 10u,
        atlasDepth * 10u,
        ResourceFormat::R8Unorm,
        1,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
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
    std::sort(requests.begin(), requests.end(), [](const Request& a, const Request& b) { return a.priority > b.priority; });
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
    const auto& mean = sea.getMean();
    float depth = 0.f;
    for (uint32_t i = 0; i < steps && depth < 6.f; ++i)
    {
        const float3 p = view.position + delta * ((float(i) + 0.5f) / float(steps));
        const int3 v = int3(floor((p - desc.origin) / voxel));
        if (v.y < 0 || v.y >= int32_t(dims.y))
            continue;
        const int32_t x = (v.x % int32_t(dims.x) + int32_t(dims.x)) % int32_t(dims.x);
        const int32_t z = (v.z % int32_t(dims.z) + int32_t(dims.z)) % int32_t(dims.z);
        depth += mean[size_t(x) + size_t(dims.x) * (size_t(v.y) + size_t(dims.y) * size_t(z))] * view.densityScale * dt;
    }
    return std::exp(-depth);
}

float CloudResidency::brickPriority(const CloudSea& sea, uint32_t slot, Brick& b, const CloudView& view, float& visibility) const
{
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

    const float3 nearest = clamp(view.position, lo, hi);
    const float distance = math::length(nearest - view.position);
    if (distance > view.maxDistance)
        return -1.f;
    const float voxelWorld = scale * instance.sourceVoxelWorld;
    const float pixels = voxelWorld / (std::max(distance, 0.5f * voxelWorld) * view.pixelAngle);
    // Hysteresis: a brick whose children are mapped keeps them until its error falls a quarter level below the threshold.
    const float hysteresis = b.mappedChildren > 0 ? 0.25f : 0.f;
    if (std::log2(std::max(pixels / mDesc.lodPixels, 1e-6f)) - view.lodBias + hysteresis <= 0.f)
        return -1.f;
    // The image error a brick's missing detail causes is its voxels' screen size scaled by how much of it reaches the camera: the
    // transmittance from the camera through the always-resident proxy (measured for coarse bricks, inherited by finer ones, and
    // re-measured once the camera moved by a tenth of the distance), and a small share outside the frustum swept towards the sun.
    const float3 sweep = view.sunDirection * view.sunReach;
    const bool inside = inFrustum(view, min(lo, lo + sweep), max(hi, hi + sweep));
    if (inside && level >= 3)
    {
        const float3 moved = view.position - b.visibilityCamera;
        if (dot(moved, moved) > 0.01f * distance * distance + 1.f)
        {
            b.visibility = transmittance(sea, view, nearest);
            b.visibilityCamera = view.position;
        }
        visibility = b.visibility;
    }
    const float importance = (inside ? 1.f : 0.125f) * std::max(visibility, 0.01f);
    return std::log2(std::max(pixels * importance / mDesc.lodPixels, 1e-6f)) - view.lodBias + hysteresis;
}

bool CloudResidency::update(const CloudSea& sea, const CloudView& view, const std::vector<uint32_t>& changedSlots)
{
    const auto startTime = std::chrono::steady_clock::now();
    ++mFrame;
    bool changed = false;
    const auto& tiles = sea.getTiles();
    for (uint32_t slot : changedSlots)
    {
        mInstances[slot] = tiles[slot].instance;
        const HSTRCloudInstance& i = tiles[slot].instance;
        if (tiles[slot].occupied)
            mTileForward[slot] = inverse(float3x3{i.row0.x, i.row0.y, i.row0.z, i.row1.x, i.row1.y, i.row1.z, i.row2.x, i.row2.y, i.row2.z});
        mInstancesDirty = true;
    }

    auto* pRenderContext = mpDevice->getRenderContext();
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
    const bool viewChanged = any(view.position != mCutPosition) || any(view.viewProjection.getRow(0) != mCutViewProjection.getRow(0)) ||
                             any(view.viewProjection.getRow(1) != mCutViewProjection.getRow(1)) ||
                             any(view.viewProjection.getRow(2) != mCutViewProjection.getRow(2));
    const bool runCut = mCutFrame == 0 || viewChanged || !changedSlots.empty() || mCutCommits > 0 || mFrame >= mCutFrame + 30;
    struct Entry
    {
        float priority;
        uint64_t handle;
        uint32_t slot;
        bool operator<(const Entry& other) const { return priority < other.priority; }
    };
    std::vector<Entry> heap;
    std::vector<uint64_t> desired;
    std::vector<Request> requests;
    if (runCut)
    {
        mCutFrame = mFrame;
        mCutCommits = 0;
        mCutPosition = view.position;
        mCutViewProjection = view.viewProjection;
    }
    const uint32_t budget = uint32_t(0.9f * float(mBricks.size()));
    uint32_t bricksDesired = 0;
    auto desire = [&](uint64_t handle, float priority, float visibility, uint32_t slot)
    {
        Brick& b = brick(handle);
        storeOf(handle).lastUsedFrame = mFrame;
        if (b.desiredFrame != mCutFrame)
        {
            b.desiredFrame = mCutFrame;
            b.priority = priority;
            desired.push_back(handle);
            ++bricksDesired;
        }
        else
            b.priority = std::max(b.priority, priority);
        if (b.record.level < 3)
            b.visibility = visibility;
        if (priority > 0.f)
        {
            heap.push_back({priority, handle, slot});
            std::push_heap(heap.begin(), heap.end());
        }
    };
    for (uint32_t slot = 0; runCut && slot < tiles.size(); ++slot)
    {
        const CloudSea::Tile& tile = tiles[slot];
        if (!tile.occupied || mAssets[tile.instance.asset].top == kNoHandle)
            continue;
        // Instances whose proxy already projects below the pixel threshold need no fine bricks.
        const float distance = math::length(clamp(view.position, tile.worldMin, tile.worldMax) - view.position);
        const float proxyPixels = sea.getVoxelWorld() / (std::max(distance, 1e-3f) * view.pixelAngle);
        if (std::log2(proxyPixels / mDesc.lodPixels) - view.lodBias <= 0.f || distance > view.maxDistance)
            continue;
        const uint64_t top = mAssets[tile.instance.asset].top;
        float visibility = 1.f;
        desire(top, std::max(brickPriority(sea, slot, brick(top), view, visibility), 1e-3f), visibility, slot);
    }
    std::optional<ScopedProfilerEvent> cutScope;
    if (runCut)
        cutScope.emplace(pRenderContext, "cut");
    std::vector<uint64_t> children;
    uint32_t pops = 0;
    while (!heap.empty() && pops < 200000)
    {
        std::pop_heap(heap.begin(), heap.end());
        const Entry entry = heap.back();
        heap.pop_back();
        ++pops;
        const uint32_t storeIndex = uint32_t(entry.handle >> 32);
        Store& store = *mStores[storeIndex];
        Brick& b = store.bricks[uint32_t(entry.handle)];
        const CloudAsset& asset = *mAssetRecords[store.asset];
        children.clear();
        if (store.kind == StoreKind::Coarse && b.record.level == kChunkLevel)
        {
            // Children: the level-3 bricks of the chunk page.
            if (b.record.childMask == 0 || !all(b.record.brick() < asset.chunkDims))
                continue;
            const uint32_t chunk = asset.chunkIndex(b.record.brick());
            const uint32_t chunkStore = mAssets[store.asset].chunkStores[chunk];
            if (asset.chunks[chunk].meta.raw == 0)
                continue;
            if (chunkStore == kNone || chunkStore == kPendingStore)
            {
                Request request;
                request.priority = entry.priority + 1.f; // Pages precede the bricks they unlock.
                request.asset = store.asset;
                request.chunk = chunk;
                request.blobs = asset.chunks[chunk];
                requests.push_back(request);
                continue;
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
                for (uint32_t i = 0; i < mStores[pageStore]->bricks.size(); ++i)
                    if (mStores[pageStore]->bricks[i].record.level == kPageLevel)
                        children.push_back(makeHandle(pageStore, i));
            }
        }
        else
            for (uint32_t c = 0; c < b.childCount; ++c)
                children.push_back(makeHandle(storeIndex, store.children[b.children + c]));
        uint32_t newBricks = 0;
        for (uint64_t child : children)
            newBricks += brick(child).desiredFrame != mCutFrame ? 1 : 0;
        if (bricksDesired + newBricks > budget)
            continue;
        for (uint64_t child : children)
        {
            float visibility = b.visibility;
            const float priority = brickPriority(sea, entry.slot, brick(child), view, visibility);
            desire(child, priority, visibility, entry.slot);
        }
    }
    cutScope.reset();
    if (runCut)
        mDesired.swap(desired);
    FALCOR_PROFILE(pRenderContext, "apply");
    mStats.desired = uint32_t(mDesired.size());

    // Reconstruct desired bricks whose parents are in the atlas (parents come first in the cut, and a parent staged this frame
    // commits in an earlier dispatch than its children); map those whose parents are mapped.
    mStaged.clear();
    for (uint64_t handle : mDesired)
    {
        Brick& b = brick(handle);
        const bool parentLoaded = b.parent == kNoHandle || (brick(b.parent).flags & kLoaded);
        if (!(b.flags & kLoaded) && parentLoaded && mStaged.size() < mDesc.loadsPerFrame)
            commit(handle);
        const bool parentMapped = b.parent == kNoHandle || (brick(b.parent).flags & kMapped);
        if ((b.flags & kLoaded) && !(b.flags & kMapped) && parentMapped)
            changed |= map(handle);
    }
    changed |= !mStaged.empty();
    mCutCommits += uint32_t(mStaged.size());
    enqueue(std::move(requests));

    // Staging order: coarsest level first, one dispatch per level.
    {
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

    // Fades: desired bricks blend in; the rest blend out once their children are gone, then unmap (they stay in the atlas).
    const float fadeStep = 1.f / float(std::max(1u, mDesc.fadeFrames));
    for (size_t i = 0; i < mMappedList.size();)
    {
        const uint64_t handle = mMappedList[i];
        Brick& b = brick(handle);
        if (b.desiredFrame == mCutFrame)
        {
            if (b.fade < 1.f)
            {
                b.fade = std::min(1.f, b.fade + fadeStep);
                mBricks[b.gpu].fade = b.fade;
                touchBrick(b.gpu);
                changed = true;
            }
            ++i;
            continue;
        }
        if (b.mappedChildren > 0)
        {
            ++i;
            continue;
        }
        b.fade -= fadeStep;
        changed = true;
        if (b.fade > 0.f)
        {
            mBricks[b.gpu].fade = b.fade;
            touchBrick(b.gpu);
            ++i;
            continue;
        }
        unmap(handle);
        mMappedList[i] = mMappedList.back();
        mMappedList.pop_back();
    }

    // Page and chunk stores nothing has used for a while are released (pages before their chunks); at once when the payload pool
    // filled, down to those used this frame.
    const bool releaseNow = mReleaseStores;
    mReleaseStores = false;
    if (mFrame % 60 == 0 || releaseNow)
        for (StoreKind kind : {StoreKind::Page, StoreKind::Chunk})
            for (uint32_t s = 0; s < mStores.size(); ++s)
            {
                Store& store = *mStores[s];
                if (store.kind != kind || store.bricks.empty() || store.loaded > 0 || store.lastUsedFrame + (releaseNow ? 1 : 120) > mFrame)
                    continue;
                bool busy = false;
                for (const Brick& b : store.bricks)
                    busy |= b.desiredFrame == mCutFrame;
                for (uint32_t pageStore : store.pageStores)
                    busy |= pageStore != kNone;
                if (!busy)
                    releaseStore(s);
            }

    {
        FALCOR_PROFILE(pRenderContext, "upload");
        upload();
    }
    mStats.mapped = uint32_t(mMappedList.size());
    mStats.loaded = uint32_t(mLoadedList.size());
    mStats.committed = uint32_t(mStaged.size());
    uint32_t waiting = 0;
    for (uint64_t handle : mDesired)
        waiting += (brick(handle).flags & kLoaded) ? 0 : 1;
    {
        std::lock_guard lock(mIoMutex);
        mStats.pending = uint32_t(mQueue.size() + mCompletions.size() + mWaiting.size()) + mInFlight + waiting;
    }
    mStats.payloadMB = double(mpPayload->getUsedBytes()) / (1024.0 * 1024.0);
    mStats.nodesUsed = uint32_t(mNodes.size() / 64 - mFreeNodes.size());
    mStats.pagesLoaded = 0;
    for (const auto& store : mStores)
        mStats.pagesLoaded += store->kind != StoreKind::Coarse && !store->bricks.empty() ? 1 : 0;
    mStats.residentMB = (double(mStats.slotsUsed) * kBrickValues + double(mNodes.size()) * 4 + double(mBricks.size()) * sizeof(HSTRCloudBrick)) /
                        (1024.0 * 1024.0);
    mStats.cutMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
    return changed;
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
    HSTRCloudBrick& gpu = mBricks[b.gpu];
    gpu.slot = b.slot;
    gpu.info = b.record.level | (uint32_t(b.record.childMask) << 4);
    gpu.parent = parent ? parent->gpu : kCloudRefNone;
    gpu.fade = 0.f;
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
        if (!(b.flags & kMapped) && b.desiredFrame != mCutFrame && b.loadedChildren == 0)
            candidates.push_back(handle);
    }
    mLoadedList.resize(kept);
    std::sort(candidates.begin(), candidates.end(), [&](uint64_t a, uint64_t c) { return brick(a).desiredFrame < brick(c).desiredFrame; });
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
    FALCOR_ASSERT((b.flags & (kLoaded | kMapped)) == kLoaded);
    mFreeSlots.push_back(b.slot);
    --mStats.slotsUsed;
    mBrickOwner[b.gpu] = kNoHandle;
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
    b.flags |= kMapped;
    b.fade = 0.f;
    mBricks[b.gpu].fade = 0.f;
    touchBrick(b.gpu);
    paint(storeOf(handle).asset, b.record, b.gpu);
    if (b.parent != kNoHandle)
        ++brick(b.parent).mappedChildren;
    mMappedList.push_back(handle);
    return true;
}

void CloudResidency::unmap(uint64_t handle)
{
    Brick& b = brick(handle);
    const uint32_t replacement = b.parent != kNoHandle ? brick(b.parent).gpu : kCloudRefNone;
    replace(storeOf(handle).asset, b.record, b.gpu, replacement);
    b.flags &= ~kMapped;
    b.fade = 0.f;
    if (b.parent != kNoHandle)
        --brick(b.parent).mappedChildren;
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
    for (size_t block = 0; block < mNodeBlocksDirty.size(); ++block)
    {
        if (!mNodeBlocksDirty[block])
            continue;
        const size_t first = block * kNodeBlock * 64;
        const size_t count = std::min(mNodes.size() - first, size_t(kNodeBlock) * 64);
        mpNodes->setBlob(mNodes.data() + first, first * sizeof(uint32_t), count * sizeof(uint32_t));
        mNodeBlocksDirty[block] = 0;
    }
    for (size_t block = 0; block < mBrickBlocksDirty.size(); ++block)
    {
        if (!mBrickBlocksDirty[block])
            continue;
        const size_t first = block * kBrickBlock;
        const size_t count = std::min(mBricks.size() - first, size_t(kBrickBlock));
        mpBricks->setBlob(mBricks.data() + first, first * sizeof(HSTRCloudBrick), count * sizeof(HSTRCloudBrick));
        mBrickBlocksDirty[block] = 0;
    }
    if (!mStaged.empty())
        mpStagingInfo->setBlob(mStagingInfo.data(), 0, mStaged.size() * sizeof(HSTRCloudStaging));
}

void CloudResidency::bind(const ShaderVar& var) const
{
    var["hstrCloudInstances"] = mpInstances;
    var["hstrCloudAssets"] = mpAssets;
    var["hstrCloudDirectory"] = mpDirectory;
    var["hstrCloudNodes"] = mpNodes;
    var["hstrCloudBricks"] = mpBricks;
    var["hstrCloudAtlas"] = mpAtlas;
    var["hstrCloudPayload"] = mpPayload->getBuffer();
    var["hstrCloudResiduals"] = mpResiduals;
    var["hstrCloudStagingInfo"] = mpStagingInfo;
}
} // namespace hstrcloud
