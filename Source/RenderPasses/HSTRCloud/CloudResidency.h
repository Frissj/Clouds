/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "CloudSea.h"
#include "CloudPayloadPool.h"

#include <atomic>
#include <memory>

namespace hstrcloud
{
struct CloudResidencyDesc
{
    std::vector<std::filesystem::path> files; ///< The packages the pages stream from (CloudLibrary::files).
    uint32_t poolMB = 256;                    ///< Atlas memory (1 KB per brick).
    uint32_t loadsPerFrame = 1024; ///< Bricks reconstructed on the GPU per frame.
    float lodPixels = 1.f;         ///< A brick is refined while its voxels span more pixels than this.
    uint32_t fadeFrames = 8;       ///< Frames a brick blends against its parent while streaming in or out.
    uint32_t ioThreads = 2;
    uint32_t payloadPoolMB = 128;  ///< GPU memory of the packed coefficients of resident pages.
    bool directStorage = true;     ///< Load page payloads with DirectStorage (GPU decompression where supported).
};

/// What the residency cut is computed for.
struct CloudView
{
    float3 position = float3(0.f);
    float4x4 viewProjection = float4x4::identity();
    float pixelAngle = 1e-3f; ///< World footprint of one pixel per unit distance.
    float lodBias = 0.f;      ///< Levels added to every footprint.
    float3 sunDirection = float3(0.f, 1.f, 0.f);
    float sunReach = 0.f;     ///< World distance camera samples march towards the sun on fine density.
    float densityScale = 1.f; ///< Extinction per unit of stored density.
    float maxDistance = 1e9f; ///< Nothing beyond is rendered.
};

/// Virtual memory for cloud density. The cut through every nearby instance's brick pyramid is chosen by importance each frame:
/// screen error of the brick's voxels, attenuated by the camera's transmittance to it through the always-resident proxy and by
/// frustum membership (swept towards the sun for shadowing detail). Pages of the library package stream on I/O threads (chunk
/// pages, then level-2 pages); a brick whose page is in memory is reconstructed on the GPU from its parent in the atlas plus its
/// residual (commitCloudBricks), parents in earlier dispatches than their children. Mapped bricks blend in from their parents,
/// unneeded bricks blend out and stay in the atlas until their slot is needed (least recently desired first). Transport never
/// reads fine bricks: it runs on the proxy, which is always resident.
class CloudResidency
{
public:
    struct Stats
    {
        uint32_t loaded = 0;
        uint32_t mapped = 0;
        uint32_t desired = 0;
        uint32_t pending = 0;
        uint32_t committed = 0;
        uint32_t slotsUsed = 0;
        uint32_t nodesUsed = 0;
        uint32_t pagesLoaded = 0;
        double residentMB = 0.0;
        double payloadMB = 0.0; ///< Payload pool in use.
        double cutMilliseconds = 0.0;
    };

    /// Staged bricks of one level: one commit dispatch each, coarsest first, so parents are in the atlas before their children.
    struct CommitGroup
    {
        uint32_t offset = 0;
        uint32_t count = 0;
    };

    CloudResidency(ref<Device> pDevice, const CloudSea& sea, const CloudResidencyDesc& desc);
    ~CloudResidency();

    /// Chooses the cut, streams pages and fills the staging buffers. Returns true when the density the camera sees changed.
    bool update(const CloudSea& sea, const CloudView& view, const std::vector<uint32_t>& changedSlots);

    void bind(const ShaderVar& var) const;
    ref<Texture> getAtlas() const { return mpAtlas; }
    /// Bricks staged this frame (decodeCloudResiduals runs over them before the commit groups).
    uint32_t getStagedCount() const { return uint32_t(mStaged.size()); }
    const std::vector<CommitGroup>& getCommitGroups() const { return mCommitGroups; }
    uint32_t getAtlasShift() const { return 6; }
    const Stats& getStats() const { return mStats; }

private:
    static constexpr uint32_t kNone = 0xFFFFFFFF;
    static constexpr uint32_t kPendingStore = 0xFFFFFFFE;
    static constexpr uint64_t kNoHandle = ~0ull;
    static constexpr uint16_t kLoaded = 1;
    static constexpr uint16_t kMapped = 4;

    enum class StoreKind : uint8_t
    {
        Coarse, ///< An asset's bricks of levels >= 4.
        Chunk,  ///< The level-3 bricks of one chunk and its page directory.
        Page,   ///< One level-2 brick and its level-1 and level-0 descendants.
    };

    struct Brick
    {
        BrickHeader record;
        uint64_t parent = kNoHandle;
        uint32_t children = kNone; ///< Offset into the store's child list (childCount entries; page indices for level-3 bricks).
        uint32_t gpu = kNone;      ///< Brick index on the GPU while loaded.
        uint32_t slot = kNone;     ///< Atlas slot while loaded.
        uint32_t desiredFrame = 0;
        float priority = 0.f;
        float visibility = 1.f;
        float3 visibilityCamera = float3(std::numeric_limits<float>::max()); ///< Camera position the visibility was measured from.
        float fade = 0.f;
        uint16_t flags = 0;
        uint8_t childCount = 0;
        uint8_t loadedChildren = 0;
        uint8_t mappedChildren = 0;
    };

    struct Store
    {
        StoreKind kind = StoreKind::Coarse;
        uint32_t asset = 0;
        uint32_t chunk = kNone;
        uint32_t generation = 0;
        uint32_t parentStore = kNone; ///< Page stores: their chunk store.
        uint32_t parentPage = kNone;  ///< Page stores: their entry in the chunk store's directory.
        std::vector<Brick> bricks;
        std::vector<uint32_t> children; ///< Store-local brick indices, or page indices for level-3 bricks.
        uint32_t payloadWord = kNone;   ///< The page payload's range in the payload pool.
        uint32_t payloadBytes = 0;
        std::vector<PageEntry> pages;     ///< Chunk stores: level-2 page directory.
        std::vector<uint32_t> pageStores; ///< Chunk stores: store of each page, kNone or kPendingStore.
        uint32_t loaded = 0;
        uint32_t lastUsedFrame = 0;
    };

    struct AssetState
    {
        uint32_t coarseStore = kNone;
        uint64_t top = kNoHandle;
        uint32_t directoryOffset = 0;
        std::vector<uint32_t> chunkStores; ///< Per chunk: store index, kNone or kPendingStore.
        std::vector<uint64_t> chunkBrick;  ///< Per chunk: handle of its level-4 brick.
    };

    struct Request
    {
        float priority = 0.f;
        uint32_t asset = 0;
        uint32_t chunk = 0;
        uint32_t store = kNone; ///< Level-2 pages: the chunk store and its generation.
        uint32_t generation = 0;
        uint32_t page = kNone;
        PageBlobs blobs;
        uint32_t payloadWord = kNone; ///< Allocated in the payload pool when queued.
        bool operator<(const Request& other) const { return priority < other.priority; }
    };

    struct Completion
    {
        Request request;
        bool ok = false;
        DecodedPage page;
        std::vector<uint8_t> payload;        ///< Without DirectStorage: the decompressed payload to upload.
        uint32_t ticket = CloudPayloadPool::kNone; ///< With DirectStorage: the payload load to wait for.
    };

    static uint64_t makeHandle(uint32_t store, uint32_t index) { return (uint64_t(store) << 32) | index; }
    Brick& brick(uint64_t handle) { return mStores[handle >> 32]->bricks[uint32_t(handle)]; }
    const Brick& brick(uint64_t handle) const { return mStores[handle >> 32]->bricks[uint32_t(handle)]; }
    Store& storeOf(uint64_t handle) { return *mStores[handle >> 32]; }

    uint32_t createStore(StoreKind kind, uint32_t asset, uint32_t chunk, DecodedPage page, uint64_t parentHandle, uint32_t payloadWord, uint32_t payloadBytes);
    void releaseStore(uint32_t store);
    void ioWorker();
    void enqueue(std::vector<Request> requests);
    void resetPending(const Request& request);

    float brickPriority(const CloudSea& sea, uint32_t slot, Brick& b, const CloudView& view, float& visibility) const;
    float transmittance(const CloudSea& sea, const CloudView& view, float3 target) const;
    bool inFrustum(const CloudView& view, float3 lo, float3 hi) const;

    bool commit(uint64_t handle);
    bool evict(uint32_t slotsNeeded, uint32_t metasNeeded);
    void unload(uint64_t handle);
    bool map(uint64_t handle);
    void unmap(uint64_t handle);
    void paint(uint32_t asset, const BrickHeader& record, uint32_t ref);
    void replace(uint32_t asset, const BrickHeader& record, uint32_t ref, uint32_t replacement);
    void paintEntry(uint32_t& entry, uint32_t ref, uint32_t level);
    void replaceEntry(uint32_t& entry, uint32_t ref, uint32_t replacement);
    uint32_t ensureNode(uint32_t& entry);
    void touchDirectory(size_t index);
    void touchNode(size_t node);
    void touchBrick(uint32_t gpu);
    void upload();

    ref<Device> mpDevice;
    CloudResidencyDesc mDesc;
    std::vector<AssetState> mAssets;
    std::vector<std::unique_ptr<Store>> mStores;
    std::vector<uint32_t> mFreeStores;
    std::vector<uint64_t> mMappedList;
    std::vector<uint64_t> mLoadedList;
    std::vector<float3x3> mTileForward; ///< Per slot: asset source voxel to tile-local domain voxel (linear part).

    // GPU mirrors.
    std::vector<uint32_t> mDirectory;
    std::vector<uint32_t> mNodes; ///< 64 entries per node.
    std::vector<uint32_t> mFreeNodes;
    std::vector<HSTRCloudBrick> mBricks;
    std::vector<uint32_t> mFreeBricks;
    std::vector<uint32_t> mFreeSlots;
    std::vector<uint64_t> mBrickOwner;
    std::vector<HSTRCloudStaging> mStagingInfo;
    std::vector<uint64_t> mStaged; ///< Handles staged this frame, in staging order before grouping.
    std::vector<CommitGroup> mCommitGroups;
    size_t mDirectoryDirty[2] = {SIZE_MAX, 0};
    std::vector<uint8_t> mNodeBlocksDirty;  ///< Per 256 nodes.
    std::vector<uint8_t> mBrickBlocksDirty; ///< Per 4096 bricks.
    bool mInstancesDirty = true;
    std::vector<HSTRCloudInstance> mInstances;

    ref<Buffer> mpInstances;
    ref<Buffer> mpAssets;
    ref<Buffer> mpDirectory;
    ref<Buffer> mpNodes;
    ref<Buffer> mpBricks;
    ref<Texture> mpAtlas;
    ref<Buffer> mpResiduals; ///< decodeCloudResiduals output, 512 floats per staged brick.
    ref<Buffer> mpStagingInfo;
    std::unique_ptr<CloudPayloadPool> mpPayload;

    // I/O.
    std::mutex mIoMutex;
    std::condition_variable mIoWake;
    std::vector<Request> mQueue; ///< Max-heap by priority.
    std::vector<Completion> mCompletions;
    std::vector<Completion> mWaiting; ///< Pages whose meta arrived and whose payload DirectStorage is still loading.
    bool mReleaseStores = false;      ///< The payload pool filled: release unused stores now.
    uint32_t mInFlight = 0;
    bool mStop = false;
    std::vector<std::thread> mIoThreads;
    std::vector<const CloudAsset*> mAssetRecords;

    uint32_t mFrame = 1;
    uint32_t mCutFrame = 0;         ///< Frame of the last cut; desiredFrame == mCutFrame marks its bricks.
    std::vector<uint64_t> mDesired; ///< Bricks of the last cut, parents before children.
    float3 mCutPosition = float3(std::numeric_limits<float>::max());
    float4x4 mCutViewProjection;
    uint32_t mCutCommits = 0; ///< Bricks committed or pages arrived since the last cut.
    Stats mStats;
    bool mWarnedNodes = false;
};
} // namespace hstrcloud
