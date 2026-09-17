/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "CloudSea.h"
#include "CloudPayloadPool.h"

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <unordered_map>

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
    uint32_t sunBakesPerFrame = 256; ///< Bricks whose sun depth bakeCloudSun bakes per frame.
    bool gpuSun = false;             ///< Sun bakes are scheduled on the GPU (the scheduler passes of HSTRCloud.cs.slang).
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
    float sunNearVoxels = 0.f;    ///< HSTRCloudParams::sunNearVoxels (0: no baked sun depth).
    uint32_t sunGeneration = 0;   ///< HSTRCloudParams::cloudSunGeneration.
    float3 sunBakeDirection = float3(0.f, 1.f, 0.f); ///< Sun direction of that generation's bakes (the sun within cloudSunBakeAngle).
    float densityScale = 1.f; ///< Extinction per unit of stored density.
    float maxDistance = 1e9f; ///< Nothing beyond is rendered.
    /// Motion envelope (0: off): the cut is chosen for any camera within this many voxels of the position it was cut at, and is not
    /// redone until the camera leaves that envelope or turns by more than cutTurn degrees. It trades atlas slots for cut time.
    /// MEASURED (sea, live residency, 2026-09-17): flying 2 units a frame, 48 cuts in 48 frames -> 7 at 4 or 8 voxels (residency
    /// 29.9 -> 18.6 / 19.7 ms, 2-4% more bricks); at 16 the extra bricks' sun bakes cost more than the cuts saved, at 64+ the atlas
    /// payload fills. 20 units a frame: 43 -> 37 ms. HSTRCloud defaults to 8.
    float cutMargin = 0.f;
    float cutTurn = 3.f;
    /// The cut's walk runs on a worker while frames keep the last cut, and is applied in the first frame after it returns.
    bool cutAsync = false;
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
        uint32_t sunBaked = 0;      ///< Mapped bricks whose sun depth is baked for their instance's key.
        uint32_t sunWaiting = 0;    ///< Mapped bricks still on the live march (not yet baked, or neighbourhood still streaming).
        uint32_t sunBakesFrame = 0; ///< Bakes staged this frame.
        uint32_t sunSlotsFree = 0;  ///< Free sun atlas slots.
        uint32_t sunStale = 0;      ///< Bakes dropped this frame because their neighbourhood was mapped or unmapped since.        // The last cut.
        uint32_t cutPops = 0;      ///< Bricks refined.
        bool cutOrdered = false;   ///< Whether the atlas budget bound, forcing the priority-ordered cut.
        double cutTotalMs = 0.0;
        float cutMargin = 0.f; ///< The motion envelope of the last cut, in voxels.
        uint32_t cuts = 0;     ///< Cuts run so far.
        // Page table edits so far (maps and unmaps), and what waits.
        uint32_t maps = 0;
        uint32_t unmaps = 0;
        uint32_t mapBacklog = 0;   ///< Loaded desired bricks not yet mapped.
        uint32_t unmapBacklog = 0; ///< Faded out, not yet unmapped.
        uint32_t activeFades = 0;  ///< Bricks whose fade runs.
        uint32_t fadeStarts = 0;   ///< Fades started or turned so far.
        uint32_t fadeEnds = 0;     ///< Fade ends handled so far (settled or queued for the unmap).
        uint32_t fadeVoid = 0;     ///< End entries whose fade had already been replaced.
    };
    /// Mapped bricks the cut does not want, by why they are still mapped (auditMapped walks every mapped brick: diagnostics only).
    struct MappedAudit
    {
        uint32_t fadingOut = 0;
        uint32_t held = 0; ///< Mapped children keep them.
        uint32_t idle = 0; ///< Neither: nothing will unmap them.
        uint32_t stale = 0; ///< Fades still running fadeFrames after they started: their end was missed.
    };
    MappedAudit auditMapped() const;

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
    ref<Buffer> getOccupancy() const { return mpOccupancy; }
    ref<Texture> getSunAtlas() const { return mpSunAtlas; }
    /// Sun bakes staged this frame (bakeCloudSun runs over them after the commits).
    uint32_t getSunBakeCount() const { return uint32_t(mSunBakes.size()); }
    /// Bricks staged this frame (decodeCloudResiduals runs over them before the commit groups).
    uint32_t getStagedCount() const { return uint32_t(mStaged.size()); }
    const std::vector<CommitGroup>& getCommitGroups() const { return mCommitGroups; }
    uint32_t getAtlasShift() const { return 6; }
    const Stats& getStats() const { return mStats; }
    /// Whether a cut is running on the worker (CloudView::cutAsync). The sea must not apply tile changes meanwhile: the walk reads them.
    bool cutInFlight() const { return mCutJob.valid(); }
    /// Whether advanceCloudFades has fades to run this frame, and what it binds.
    bool fadesRunning() const { return mActiveFades > 0; }
    uint32_t getBrickCapacity() const { return uint32_t(mBricks.size()); }
    const ref<Buffer>& getBricks() const { return mpBricks; }
    const ref<Buffer>& getFadeFrame() const { return mpFadeFrame; }
    void waitForCut() const
    {
        if (mCutJob.valid())
            mCutJob.wait();
    }

    /// What this frame's GPU sun scheduling (CloudResidencyDesc::gpuSun) runs with.
    struct GpuSunFrame
    {
        bool run = false;        ///< The scheduling passes run (ageSunField runs whenever info.ageRows > 0).
        HSTRCloudSunFrame info = {}; ///< Uploaded to hstrCloudSunFrameInfo.
    };
    const GpuSunFrame& getGpuSunFrame() const { return mGpuSunFrame; }
    /// HSTRCloud dispatched the frame: a frame without a residency update (frozen) must not replay its releases, resets and changes.
    void gpuSunDispatched() { mGpuSunFrame = {}; }
    /// Binds the scheduler's buffers for writing (and unbinds the camera's views of the same buffers).
    void bindGpuSun(const ShaderVar& var) const;
    /// The scheduler's buffers to clear before its scan.
    ref<Buffer> getGpuSunCounters() const { return mpSunCounters; }
    ref<Buffer> getGpuSunHistogram() const { return mpSunHistogram; }
    /// Reads the scheduler's counts of the last frame into the stats (a GPU readback: for scripts, not the frame).
    void readGpuSunStats();
    /// The scheduler goes idle while its inputs stay as they were in a run that staged nothing. HSTRCloud reads that count back
    /// asynchronously: it calls gpuSunQueued when it queues the read for this frame's run, and gpuSunRead when the count arrives.
    void gpuSunQueued() { mGpuSunQueuedInputs = mGpuSunInputs; }
    void gpuSunRead(uint32_t bakes)
    {
        mGpuSunIdleValid = bakes == 0;
        mGpuSunIdle = mGpuSunQueuedInputs;
    }

private:
    static constexpr uint32_t kNone = 0xFFFFFFFF;
    static constexpr uint32_t kPendingStore = 0xFFFFFFFE;
    static constexpr uint64_t kNoHandle = ~0ull;
    static constexpr uint16_t kLoaded = 1;
    static constexpr uint16_t kToMap = 16;  ///< In mToMap.

    enum class StoreKind : uint8_t
    {
        Coarse, ///< An asset's bricks of levels >= 4.
        Chunk,  ///< The level-3 bricks of one chunk and its page directory.
        Page,   ///< One level-2 brick and its level-1 and level-0 descendants.
    };

    /// What one cut decided for a brick.
    struct CutState
    {
        uint32_t id = 0;      ///< The cut (mCutId) that desired the brick last in this slot.
        float priority = 0.f;
        uint16_t sunClasses = 0; ///< Orientation classes of the instances that desired it (bit per class).
    };

    struct Brick
    {
        BrickHeader record;
        uint64_t parent = kNoHandle;
        uint32_t children = kNone; ///< Offset into the store's child list (childCount entries; page indices for level-3 bricks).
        uint32_t gpu = kNone;      ///< Brick index on the GPU while loaded.
        uint32_t slot = kNone;     ///< Atlas slot while loaded.
        uint32_t mappedIndex = kNone; ///< Position in mMappedList while mapped.
        /// The current cut's state is cut[mCutSlot]; the walk of the next one writes the other slot, so taking it on is a flip.
        CutState cut[2];
        uint32_t fadeSerial = 0; ///< Bumped by every fade change (a FadeEnd of an older one is void).
        uint16_t flags = 0;
        /// In the page table. Its own field: maps and unmaps run beside a cut's walk, which reads flags (and the mapped state only
        /// from mMappedSnapshot).
        bool mapped = false;
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
        std::vector<uint32_t> heads;    ///< Page stores: the level-2 bricks, which hang from the chunk store.
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
        uint3 changeDims = uint3(0);
        std::vector<uint32_t> changeFrame; ///< CPU scheduler: per 32^3-voxel cell, last frame a brick over it was mapped or unmapped.
        uint32_t lastChange = 0;
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

    float brickPriority(const CloudSea& sea, uint32_t slot, uint64_t handle, const CloudView& view, float& visibility);
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
    void scheduleSunBakes(const CloudSea& sea, const CloudView& view);
    void markChanged(uint32_t asset, const BrickHeader& record);
    bool sunBakeStale(uint32_t asset, const BrickHeader& record, float3 direction, float reach, uint32_t bakeFrame) const;
    uint32_t sunClassOf(const HSTRCloudInstance& instance);

    ref<Device> mpDevice;
    CloudResidencyDesc mDesc;
    std::vector<AssetState> mAssets;
    std::vector<std::unique_ptr<Store>> mStores;
    std::vector<uint32_t> mFreeStores;
    std::vector<uint64_t> mMappedList;
    std::vector<uint64_t> mLoadedList;
    // The apply step touches only what can change (the cut merge rebuilds these; the rest is events):
    size_t mToLoadNext = 0;          ///< mToLoad before it is loaded.
    std::vector<uint64_t> mToMap;    ///< Desired bricks not yet mapped, parents first (kToMap).
    uint32_t mMergeFrame = 0;        ///< Frame of the last cut merge (every desired brick's store was used then).
    uint32_t mPageStores = 0;        ///< Non-empty chunk and page stores.
    // Fades: mapped bricks fade in while desired, and out once undesired with no mapped children (then unmap). A fade runs on the
    // GPU from its start (HSTRCloudBrick::fadeStart); the frame handles only where one starts, turns, stops or ends.
    // MEASURED before (4K sea flight at 20 units a frame): stepping every fading brick every frame cost 0.13 ms a frame and up to
    // 0.95 ms after a cut, plus the upload of every brick block it touched.
    struct FadeEnd
    {
        uint64_t handle;
        uint32_t serial;     ///< Brick::fadeSerial of the fade it ends.
        uint32_t generation; ///< Of the brick's store (a released store's handles are void).
    };
    static constexpr uint32_t kUnmapsPerFrame = 512;
    static constexpr uint32_t kMapsPerFrame = 1024;
    std::vector<std::vector<FadeEnd>> mFadeEnds;
    std::deque<FadeEnd> mUnmapQueue; ///< Faded out, waiting for their unmap. ///< Ring of fadeFrames + 1 buckets, by the frame a fade ends.
    uint32_t mFadeProcessed = 0;                 ///< The last frame whose fade ends were handled.
    uint32_t mActiveFades = 0;                   ///< Bricks whose fade runs.
    float mFadeStep = 1.f;
    ref<Buffer> mpFadeFrame;                     ///< HSTRCloudFadeFrame.
    /// A GPU brick's fade at a frame (advanceCloudFades computes the same).
    float fadeOf(const HSTRCloudBrick& gpu, uint32_t frame) const;
    /// Starts, turns or stops a mapped brick's fade for what it is now (desired, mapped children).
    void updateFade(uint64_t handle);
    /// Its fade from now: 1 in, -1 out, 0 held at its current value.
    void setFade(uint64_t handle, int direction);
    int fadeDirection(const HSTRCloudBrick& gpu) const
    {
        return (gpu.fadeStart & kCloudFadeIn) ? 1 : (gpu.fadeStart & kCloudFadeOut) ? -1 : 0;
    }
    void processFadeEnds(bool& changed);
    // Maps, unmaps and fades run beside a cut's walk (it takes about three frames on the sea; they waited for it, so each frame
    // between walks did three frames of them). The walk reads what they change from a snapshot taken when it starts, per GPU
    // brick; the merge revisits the bricks they touched meanwhile.
    static constexpr uint8_t kSnapMapped = 1;
    static constexpr uint8_t kSnapChildren = 2; ///< Mapped children.
    static constexpr uint8_t kSnapFadeIn = 4;
    static constexpr uint8_t kSnapFadeOut = 8;
    static constexpr uint8_t kSnapFull = 16;    ///< Held at a fade of 1.
    std::vector<uint8_t> mMappedState;          ///< Current, per GPU brick.
    std::vector<uint8_t> mMappedSnapshot;       ///< The running walk's copy.
    std::vector<uint64_t> mWalkTouched;         ///< Bricks changed while the walk ran.
    /// Refreshes a brick's mMappedState entry (and records it while a walk runs).
    void noteMapped(uint64_t handle);
    /// Maps what the cut wants and its parents hold, up to kMapsPerFrame.
    void mapReady(bool& changed);
    void unlistMapped(uint64_t handle);
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
    ref<Buffer> mpOccupancy; ///< kCloudCellWords words per GPU brick: 4-bit cell density bounds (occupancyCloudBricks).
    ref<Texture> mpSunAtlas; ///< Baked sun optical depth: its own pool of slots laid out like density atlas slots.
    ref<Buffer> mpSunBakes;
    ref<Buffer> mpSunSlotTable;
    std::vector<HSTRCloudSunBake> mSunBakes;
    std::vector<uint32_t> mSunSlotTable;  ///< Per GPU brick, kCloudSunBakesPerBrick pairs (key, sun slot); key kCloudRefNone: not valid.
    std::vector<uint32_t> mSunBakeFrames; ///< Per pair: frame it was baked.
    std::vector<uint32_t> mFreeSunSlots;
    std::vector<uint8_t> mSunTableBlocksDirty; ///< Per 4096 bricks.
    std::vector<float3x3> mSunClasses; ///< Signed permutation of each orientation class (HSTRCloudInstance::sunClass).
    /// What the last scheduleSunBakes read, when it staged nothing: an unchanged repeat would stage nothing again, so it is skipped.
    struct SunScheduleInputs
    {
        uint32_t cutFrame = 0;
        uint32_t generation = 0;
        uint32_t lastChange = 0;
        size_t freeSlots = 0;
        size_t mapped = 0;
        float3 direction = float3(0.f);
        bool operator==(const SunScheduleInputs& o) const
        {
            return cutFrame == o.cutFrame && generation == o.generation && lastChange == o.lastChange && freeSlots == o.freeSlots &&
                   mapped == o.mapped && all(direction == o.direction);
        }
    };
    SunScheduleInputs mSunScheduleIdle;
    bool mSunScheduleIdleValid = false;
    // GPU sun bake scheduling (mDesc.gpuSun). The GPU owns mpSunSlotTable, the bake frames and the free slots; the CPU writes what
    // the scheduler needs of each GPU brick, the residency changes and the bricks whose bakes are freed.
    void writeSunSched(uint64_t handle);
    /// A sea slot's instance takes (or gives up) its (asset, class) pair's block of the invalidation field.
    void acquireSunField(uint32_t slot);
    void releaseSunField(uint32_t slot);
    /// Uploads the field's block table, the ranges to reset and the changes to stamp into mGpuSunFrame.
    void uploadSunChanges();
    GpuSunFrame mGpuSunFrame;
    bool mGpuSunRan = false; ///< Whether the last residency update ran the scheduler (the stats' bake count is its).
    SunScheduleInputs mGpuSunInputs;       ///< This frame's.
    SunScheduleInputs mGpuSunQueuedInputs; ///< Those of the run whose bake count is being read back.
    SunScheduleInputs mGpuSunIdle;
    bool mGpuSunIdleValid = false;
    std::vector<HSTRCloudSunSched> mSunSched; ///< Per GPU brick: where it is (written when it maps).
    std::vector<uint8_t> mSunSchedBlocksDirty; ///< Per 4096 bricks.
    /// Per GPU brick, the scheduler state (hstrCloudSunState), twice: mSunState[mSunStateLive] is uploaded, and a cut's walk writes
    /// its changes into the other, so taking the cut on swaps them. The walk first copies the entries the previous cut changed
    /// (mSunStateSync) from the live one; the frame writes both, and only while no walk runs.
    /// MEASURED before (4K sea flight at 20 units a frame): writing the ~150k changed 16-byte entries on the frame cost 0.57 ms per
    /// cut, and uploading them 0.56 ms.
    std::array<std::vector<uint32_t>, 2> mSunState;
    uint32_t mSunStateLive = 0;
    std::vector<uint32_t> mSunStateSync;
    std::vector<uint8_t> mSunStateBlocksDirty; ///< Per 4096 bricks.
    std::vector<uint32_t> mSunRelease;         ///< GPU bricks unloaded since the last scheduling.
    // The invalidation field (HSTRCloudTypes.slang, kCloudSunClasses): level >= 2 blocks per (asset, class) pair, then a row per
    // directory node for levels 0-1.
    std::vector<HSTRCloudSunChange> mSunChanges; ///< Since the last scheduling run.
    std::vector<uint32_t> mSunFieldBlocks;       ///< Per asset, kCloudSunClasses entries: block index or kNone.
    std::vector<uint32_t> mSunFieldUsers;        ///< Per (asset, class): sea slots using it.
    std::vector<uint32_t> mSunFieldLevels;       ///< Per asset, kCloudSunFieldLevels entries: a level's offset in a block.
    std::vector<uint32_t> mSlotSunPair;          ///< Per sea slot: its (asset, class) pair, or kNone.
    std::vector<uint32_t> mFreeSunFieldBlocks;
    std::vector<uint32_t> mSunBlockResets;       ///< Blocks newly taken: every bake of their pair is stale.
    std::vector<uint32_t> mSunNodeResets;        ///< Directory nodes created since the last run.
    bool mSunResetAll = false;                   ///< Changes went unstamped: every block and node restarts.
    bool mSunFieldBlocksDirty = true;
    uint32_t mSunFieldBlockSize = 0;
    uint32_t mSunFineBase = 0;
    uint32_t mSunFieldRows = 0;
    uint32_t mSunAgeRow = 0;
    uint32_t mSunChangeCapacity = 0;
    std::vector<float4> mSunDirections;
    ref<Buffer> mpSunSched;
    ref<Buffer> mpSunState;
    ref<Buffer> mpSunFrames;
    ref<Buffer> mpSunFree;
    ref<Buffer> mpSunFreeTop;
    ref<Buffer> mpSunCounters;
    ref<Buffer> mpSunHistogram;
    ref<Buffer> mpSunSelect;
    ref<Buffer> mpSunNeeds;
    ref<Buffer> mpSunCandidates;
    ref<Buffer> mpSunHolders;
    ref<Buffer> mpSunRelease;
    ref<Buffer> mpSunDirections;
    ref<Texture> mpSunField;
    ref<Buffer> mpSunResets;
    ref<Buffer> mpSunFrameInfo;
    ref<Buffer> mpSunFieldBlocks;
    ref<Buffer> mpSunFieldLevels;
    ref<Buffer> mpSunChanges;    ref<Buffer> mpResiduals; ///< decodeCloudResiduals output, 512 floats per staged brick.
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
    uint32_t mCutFrame = 0;         ///< Frame of the last cut.
    uint32_t mCutId = 0;            ///< Increases with every cut; cut[mCutSlot].id == mCutId marks its bricks.
    uint32_t mCutSlot = 0;
    const CutState& cutOf(const Brick& b) const { return b.cut[mCutSlot]; }
    bool desired(const Brick& b) const { return b.cut[mCutSlot].id == mCutId; }
    /// The last cut that desired a brick (the eviction order).
    static uint32_t lastDesired(const Brick& b) { return std::max(b.cut[0].id, b.cut[1].id); }
    std::vector<uint64_t> mToLoad;  ///< Unloaded bricks of the last cut, coarsest level first, then by priority.

    /// Camera transmittance to one instance of a brick of level >= 3, per sea slot and brick: every instance of an asset shares its
    /// bricks, and a per-brick value was one instance's visibility handed to all the others. Per slot, so the cut's per-instance
    /// walks never share one.
    struct VisibilityEntry
    {
        float visibility = 1.f;
        float3 camera = float3(0.f); ///< Camera position it was measured from.
        uint32_t generation = 0;     ///< The store's generation: a reused store index is another brick.
    };
    std::vector<std::unordered_map<uint64_t, VisibilityEntry>> mVisibility;

    struct CutEntry
    {
        float priority;
        uint64_t handle;
        uint32_t slot;
        float visibility; ///< The brick's visibility in this instance, which its children inherit.
        bool operator<(const CutEntry& other) const { return priority < other.priority; }
    };
    struct CutVisit
    {
        uint64_t handle;
        float priority;
        uint32_t slot;
    };
    /// The top brick of a slot's instance, if the instance needs fine bricks at all.
    bool cutSeed(const CloudSea& sea, const CloudView& view, uint32_t slot, CutEntry& entry);
    /// The bricks a queued brick refines into that are in memory; page requests for those that are not.
    void cutChildren(const CutEntry& entry, std::vector<uint64_t>& children, std::vector<Request>& requests) const;
    /// Whether a brick has children to refine into (expanding a leaf does nothing, and most bricks are leaves).
    bool cutRefinable(uint64_t handle) const;
    /// A cut ready to take on: its bricks carry it in their spare CutState slot, and the lists the frame applies are built.
    struct CutWalk
    {
        std::vector<Request> requests;
        uint32_t pops = 0;
        uint32_t id = 0;      ///< The cut's mCutId.
        bool ordered = false; ///< The atlas budget bound, forcing the priority-ordered cut.
        double milliseconds = 0.0;
        std::vector<uint64_t> desired; ///< Parents before children.
        std::vector<uint64_t> toLoad;  ///< Coarsest level first, then by priority.
        std::vector<uint64_t> toMap;
        std::vector<uint64_t> fading;  ///< Mapped bricks that start or keep fading.
        std::vector<uint32_t> stateChanged;      ///< GPU bricks whose scheduler state the walk changed (in the spare mSunState).
        std::vector<uint8_t> stateBlocksDirty;   ///< The same, per 4096 bricks.
    };
    /// Records the view a cut is chosen for (the next cuts are measured against it).
    void beginCut(const CloudSea& sea, const CloudView& view);
    /// The cut, up to what the frame applies (on the worker with CloudView::cutAsync). It writes only the spare CutState slot of
    /// bricks, store use frames and the per-slot visibility caches; the frame changes none of what it reads until it is taken on.
    CutWalk walkCut(const CloudSea& sea, const CloudView& view);
    /// The scheduler entry of a GPU brick under a cut's state (priorities at the scheduler's bin resolution).
    uint32_t sunStateOf(const CutState& state, uint32_t cutId, bool mapped) const;
    /// The end of every update: sun bake scheduling, uploads and stats.
    void finishFrame(const CloudSea& sea, const CloudView& view, std::chrono::steady_clock::time_point startTime);
    std::future<CutWalk> mCutJob;
    /// The async cut's own thread, started with the first cut: launching one per cut cost the frame the thread's creation.
    void cutWorker();
    std::thread mCutThread;
    std::mutex mCutMutex;
    std::condition_variable mCutWake;
    std::packaged_task<CutWalk()> mCutTask; ///< Guarded by mCutMutex.
    bool mCutQueued = false;
    bool mCutStop = false;
    std::vector<uint64_t> mDesired; ///< Bricks of the last cut, parents before children.
    float3 mCutPosition = float3(std::numeric_limits<float>::max());
    float4x4 mCutViewProjection;
    uint32_t mCutCommits = 0; ///< Bricks committed or pages arrived since the last cut.
    float mCutMarginWorld = 0.f; ///< The motion envelope the running (or last) cut is chosen for, in world units.
    float mCutMarginScale = 1.f; ///< Share of CloudView::cutMargin that fitted the atlas budget lately.
    Stats mStats;
    bool mWarnedNodes = false;
};
} // namespace hstrcloud
