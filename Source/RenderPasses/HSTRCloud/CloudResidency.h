/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "CloudSea.h"
#include "CloudPayloadPool.h"
#include "Core/Pass/ComputePass.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_map>

/// HSTR_CLOUD_STREAMING 0 (default): whole-library residency. Every chunk and level-2 page of every asset is read, decompressed and
/// uploaded once when the residency is created (preloadLibrary), and kept: the cut only chooses which bricks to reconstruct into the
/// atlas and map. The sea's instances all share a handful of assets (the Intel half sea: 5 clouds), so the whole library is small
/// next to what streaming it saved, and streaming it failed on the sunset sea (2026-09-29):
///   - Livelock: an async walk re-requested pages that arrived while it ran; enqueue re-claimed them (kPendingStore over the
///     arrived store), the next walk asked again, and the same 256 pages loaded on every cut forever (48k requests a cut, the
///     identical 256 accepted, ~6k pages stored and released unused per 300 frames). Nothing below them refined: level-3/4 bricks
///     drawn where level 0 was wanted, the box-shaped haze in the path tracer, the exact march and the beam alike.
///   - With that fixed (enqueue skips loaded targets): desired rose to the 243k atlas budget, but loads fed only when a ~350 ms
///     walk is taken on (about every 60th frame) mapped ~6k bricks per 300 frames - 69k of 243k after 3000 frames - and the
///     128 MB payload pool filled by frame ~900 (40k refused allocations by frame 3000).
/// HSTR_CLOUD_STREAMING 1: the page streaming as it was (requests from the cut, enqueue, IO workers, DirectStorage or CPU GDeflate,
/// arrivals into stores, release of unused stores), with the livelock fix and 1024 loads a cut. Options if streaming comes back:
///   - Feed the IO queue every frame from the last walk's sorted requests instead of only when a walk is taken on. enqueue writes
///     kPendingStore into the page links the running walk reads, so track pending in mLoading only (the walk never reads it).
///   - Size the payload pool from demand: grow (allocate larger, copy, rebind) on sustained allocation failures up to a VRAM cap,
///     or size it at creation from the payload bytes of the first unrationed walk's desired set.
///   - Grow the atlas when the cut is rationed (walk.ordered) for several cuts, up to the 3D texture limit (~835k bricks at sun
///     pool x1; 2048 texels deep).
///   - Watchdog: warn when the same accepted request set repeats K cuts, or arrivals are released without ever being desired
///     (both mean a livelock), and expose "settled" (nothing pending) in cloudStats for benchmarks and reference traces to wait on.
#ifndef HSTR_CLOUD_STREAMING
#define HSTR_CLOUD_STREAMING 0
#endif

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
    /// Sun atlas slots per density slot (1-3): a brick is baked once per orientation class that uses it, so a sea of several classes
    /// (the two-layer sea) or a long sun reach wants more bakes than bricks. 2 bytes a texel: ~528 MB per unit at a 256 MB pool.
    uint32_t sunPoolScale = 1;
    /// 1 byte a texel instead of 2: the depth log-encoded (cloudSunEncode), so pool 3 costs less than pool 2 at 16 bits.
    bool sunAtlas8 = false;
    /// A second sun atlas, every bake's texels with its brick's density beside them (r sun, g density; the density's apron repeats
    /// its edge), so a lit sample reads both from one filtered fetch (cloudSunPackedRead). 0 none, 1 RG8 (8-bit sun code), 2 RG16F.
    uint32_t sunPacked = 0;
    bool gpuSun = false;            ///< Sun bakes are scheduled on the GPU (the scheduler passes of HSTRCloud.cs.slang).
};

/// What the residency cut is computed for.
struct CloudView
{
    float3 position = float3(0.f);
    float4x4 viewProjection = float4x4::identity();
    float pixelAngle = 1e-3f; ///< World footprint of one pixel per unit distance.
    float lodBias = 0.f;      ///< Levels added to every footprint.
    float visibilityFloor = 0.01f; ///< Least importance the proxy transmittance to a brick can give it (brickPriority).
    float outsideImportance = 0.125f; ///< Importance of a brick outside the frustum and its sun sweep (brickPriority).
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
    /// Threshold cut (cutThreshold): a brick the last cut held is admitted as if its parent ranked this many levels higher.
    /// MEASURED (stick1, 4K sunset walk at 2 units a frame, 60-frame arms x2 in one launch): bricks entering 226k / 268k / 87k / 268k
    /// / 285k at stick 0 / 0.5 / 1 / 2 / 0 again - noise, the sanity arm 2 no lower; of the entering bricks only 9-11% had left within
    /// the last ~4 cuts (cutReentered) at stick 0, so hysteresis can remove ~10% of the churn at most: it is new bricks coming into view.
    /// Budget-bound (ordered) cut: a brick the last cut refined is refined ahead of others by this many levels of priority, so
    /// the cut does not trade the refinements it holds for ones barely ahead of them (stall1: ~140k of 243k desired bricks replaced
    /// within ~100 walk frames; stick1: 50-107k a cut, and still ~100k a cut once parked - the mapped-children hysteresis feeds the
    /// half-loaded mapped set back into the cut). Keyed on the cut's own last answer, not on what is mapped. 0: off.
    /// MEASURED, left off (stick2, 4K sunset walk, arms in one launch, eviction beside the walk on): bricks entering per cut, stick
    /// 1 / 0: 43k / 36k (2 cuts each; stick 3 took no cut in its chunks). The parked oscillation (stick1) was the load stall, not the
    /// cut: with loads and evictions beside the walk the mapped set reaches all 243k desired within ~120 parked frames.
    /// Keyed on mapped children instead (stick1 run): 53k entering at 1 against 107k at 0, one cut each - noise at that count.
    /// stick3 (residency fixed, 120-frame arms x2 in one launch, 2-4 cuts an arm): entering a cut 89k / 62k / 45k / 38k at stick 0 /
    /// 0.5 / 1 / 0 again, HSTRCloud GPU 3.36 / 4.07 / 4.11 / 3.56 ms a frame - less churn, no faster frame.
    float cutStick = 0.f;
    /// The cut's walk runs on a worker while frames keep the last cut, and is applied in the first frame after it returns.
    bool cutAsync = false;
    /// Budget-bound cut by threshold: the parallel walk expands only bricks at or above a slack below the last cut's admission
    /// threshold, and a walk over the budget keeps the bricks whose refining parent ranks highest (admission priority), parents
    /// first, instead of walking again single-threaded with a heap. The same set as the heap's greedy wherever priorities fall
    /// from parent to child (cutphase1: 268 of ~800k children rose).
    /// MEASURED (4K sunset walk, 2 units a frame, 120-frame chunks, one launch; cutphase1 / cutthresh1): the heap cut ran on every
    /// walking cut (the parallel walk desired 750-820k against the 243k budget): ~500-650 ms of worker time a cut (parallel 180-235,
    /// heap 250-370), one cut every ~60 frames. Threshold: the heap set and the threshold set differ in 96 of 243,302 bricks
    /// (check), 6 cuts in 120 frames against 1-2, ~290 ms a cut, the pruned walk desiring 344-444k. Walks straddle the arms, so the
    /// per-arm totals mix them.
    bool cutThreshold = true;
    /// DIAGNOSTIC: unpruned threshold cut beside the heap cut (which is taken on); counts the bricks the two sets differ in.
    bool cutThresholdCheck = false;
    /// The walk's coarse phase level by level over every instance at once, each level's priorities (and their visibility rays) in
    /// parallel over its children; otherwise one thread per group of slots, which leaves the nearest instance's rays on one thread.
    /// The same visits, priorities and cut either way.
    /// MEASURED (4K sunset walk at 2 units a frame, arms alternating every 20 frames, walks bucketed by the configuration they ran
    /// under): with slot groups the busiest thread is the coarse phase (critical 0.183 G cycles ~ 73 ms wall, waves2); waves with
    /// threads taking chunks of entries (waves3) coarse 95.7 -> 23.3 ms, walk 191 -> 123 ms (lowprio1: 203 -> 135). But the cut
    /// cadence does not follow (1 cut every chunk in every arm: the walk is not what limits it), the walk's thread cycles are the
    /// same (0.96 G), and frames are slower while its burst runs: p50 4.48 / 4.38 against 4.03 / 3.95 (lowprio1), every chunk
    /// worse in waves3 (4.48 / 3.96 / 4.41 / 4.43 against 3.97 / 3.67 / 3.59 / 3.56). Off until the frame stops paying for it.
    bool cutWaves = false;
    /// The walk's lists (entered, toLoad / toMap, left, fading, scheduler states) built in parallel ranges, in the same order.
    bool cutTailParallel = false;
    /// Visibility refreshes traced on the GPU (CloudResidency::takeVisibilityBatch) instead of by the walk's threads: the walk keeps
    /// a brick's last visibility (a new brick its parent's) until the GPU's arrives, applied before the next walk.
    /// MEASURED (gpuvis1 / gpuvis2, 4K sunset walk at 2 units a frame): walk 156 -> 91 ms, coarse 0.869 -> 0.136 G thread cycles;
    /// frame mean neutral (4.07 / 4.09 against 4.29 / 3.90), motion score over 0.02 5.40 / 5.49% against 5.53 / 5.45%.
    bool cutGpuVisibility = false;
    /// Sea tile changes taken on while an async walk runs (with cutGpuVisibility: the walk then reads no sea density): the walk
    /// reads the slots as beginCut copied them, a changed slot's visibility is cleared when the walk is taken (which re-cuts), and
    /// GPU visibility traced for a slot's earlier instance is dropped (per-slot epochs). Off: changed slots wait for the walk.
    bool walkBesideTileChanges = false;
    /// The async walk's threads below normal priority, so its parallel bursts yield to the frame and the residency worker.
    /// MEASURED (lowprio1, waves on): no help - arm mean 4.70 against 4.63 at normal priority (slot groups 4.45 / 4.21), p50 4.38
    /// against 4.48. The frame does not lose cores to the burst; what it loses is unknown (memory or cache are candidates).
    bool cutLowPriority = false;
    /// Visibility rays (transmittance): the reference loop (0), the same samples stepping tile coordinates instead of dividing (1),
    /// or 1 checked against the reference (2, DIAGNOSTIC: Stats::cutRayMismatchesTotal); 5 as 1 with each corner ray stopped once
    /// it cannot raise the brick's best visibility, and 6 that checked against the corners' full rays. (3 and 4 are gone.)
    /// MEASURED and REMOVED (blocked1): 3 / 4 read a copy of the sea's mean in 4 x 4 x 4 blocks, built with every tile: 0.55 ms per
    /// 1k rays against stepped 0.55 - the proxy's memory layout is not the limit.
    /// MEASURED (4K sunset walk at 2 units a frame, coarse phase per 1k rays): reference 0.78-1.09 ms, stepped 0.80 / 0.52-0.55
    /// (cutrays1 / later runs), cut off
    /// 0.54 against stepped 0.52 (cutoff1; ray cycles a refresh 31.5k against ~32k: corner rays rarely stop early). Mismatches 0
    /// of 4.7M checked rays (modes 2, 4, 6). Rays are 93% of the coarse phase's thread cycles (walkcyc1), ~5k cycles each.
    uint32_t cutRays = 0;
    /// DIAGNOSTIC: fade ends a frame handles in their bucket's order (0), sorted by handle so siblings, their parent and their
    /// pages come together (1), or sorted with the next ends' bricks prefetched (2). MEASURED (cpuprof9, 4K sunset walk at 2 units a
    /// frame, arms alternating every 20 frames, one launch): CPU per fade-out end 737 / 780 / 847 ns for 0 / 1 / 2 and 952 for the
    /// anchor 0 repeated last - no gain beyond the drift; fades in ~150-230 ns. The fade-out end (unmap) is not limited by the order
    /// its bricks are visited in.
    uint32_t fadeEndOrder = 0;
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
        uint4 pendingParts = uint4(0); ///< DIAGNOSTIC: pending as queued, in flight on the IO worker, completions, payload waits.
        uint32_t arrivalsStored = 0;   ///< DIAGNOSTIC: loaded pages that became stores (cumulative).
        uint32_t arrivalsDropped = 0;  ///< DIAGNOSTIC: loaded pages freed on arrival, their store no longer pending (cumulative).
        uint32_t arrivalsFailed = 0;   ///< DIAGNOSTIC: loads that failed (cumulative).
        uint32_t storesReleased = 0;   ///< DIAGNOSTIC: page and chunk stores released (cumulative).
        uint32_t payloadAllocFailed = 0; ///< DIAGNOSTIC: load requests refused by a full payload pool (cumulative).
        uint32_t releaseNowFrames = 0; ///< DIAGNOSTIC: frames that released every unused store at once (cumulative).
        uint32_t committed = 0;
        uint64_t committedTotal = 0;  ///< DIAGNOSTIC: bricks committed (cumulative) ...
        uint32_t committedCapped = 0; ///< ... and frames that committed the full load cap (cumulative).
        double cutWalkMsTotal = 0.0;  ///< DIAGNOSTIC: worker milliseconds of the cuts taken on (cumulative) ...
        uint32_t cutBusyFrames = 0;   ///< ... frames that found a walk still running ...
        uint32_t walkSlotChanges = 0; ///< DIAGNOSTIC (walkBesideTileChanges): changed slots taken on beside a running walk, cumulative.
        uint32_t cutIdleFrames = 0;   ///< ... and frames with no walk running that did not start one (cumulative).
        double cutParallelMsTotal = 0.0;   ///< DIAGNOSTIC: the walks' parallel phases (cumulative) ...
        double cutHeapMsTotal = 0.0;       ///< ... and ordered heap cuts (cumulative).
        double cutCoarseMsTotal = 0.0;     ///< DIAGNOSTIC: CutWalk::coarseMs .. rays, cumulative.
        double cutFineMsTotal = 0.0;
        double cutDesireMsTotal = 0.0;
        double cutTailMsTotal = 0.0;
        uint64_t cutVisitsTotal = 0;
        uint64_t cutRefreshesTotal = 0;
        uint64_t cutRaysTotal = 0;
        uint64_t cutRayMismatchesTotal = 0;
        uint64_t gpuVisRequests = 0; ///< DIAGNOSTIC (cutGpuVisibility): refreshes sent to the GPU, results applied, batches, and
        uint64_t gpuVisApplied = 0;  ///< results dropped (their brick's slot or store changed meanwhile), cumulative.
        uint64_t gpuVisBatches = 0;
        uint64_t gpuVisDropped = 0;
        /// DIAGNOSTIC: by the configuration a walk ran under (CutWalk::config; walks straddle benchmark arms): walks, their total
        /// and coarse, fine, desire, middle (threshold trim and heap) and tail milliseconds, rays.
        static constexpr size_t kCutConfigs = 32;
        std::array<uint32_t, kCutConfigs> cutConfigWalks{};
        std::array<double, kCutConfigs> cutConfigMs{};
        std::array<double, kCutConfigs> cutConfigCoarseMs{};
        std::array<double, kCutConfigs> cutConfigFineMs{};
        std::array<double, kCutConfigs> cutConfigDesireMs{};
        std::array<double, kCutConfigs> cutConfigMidMs{};
        std::array<double, kCutConfigs> cutConfigTailMs{};
        std::array<uint64_t, kCutConfigs> cutConfigRays{};
        std::array<uint64_t, kCutConfigs> cutConfigBusyCycles{}; ///< CutWalk::coarseBusyCycles .. coarseRayCycles.
        std::array<uint64_t, kCutConfigs> cutConfigPriorityCycles{};
        std::array<uint64_t, kCutConfigs> cutConfigRayCycles{};
        std::array<uint64_t, kCutConfigs> cutConfigSerialCycles{};
        std::array<uint64_t, kCutConfigs> cutConfigCriticalCycles{};
        std::array<uint64_t, kCutConfigs> cutConfigRefreshes{};
        uint32_t cutUnorderedDesired = 0;  ///< DIAGNOSTIC: the last walk's parallel desired count (budget: 0.9 x atlas).
        uint64_t cutPriorityRises = 0;     ///< DIAGNOSTIC: children prioritised above their parent (cumulative).
        uint32_t cutThresholdTrims = 0;    ///< Threshold cuts trimmed to the budget (cumulative) ...
        uint32_t cutUnderfilled = 0;       ///< ... and pruned walks that desired under 95% of it (cumulative).
        uint64_t cutCheckBricks = 0;       ///< DIAGNOSTIC (cutThresholdCheck): heap cut bricks compared (cumulative) ...
        uint64_t cutCheckMismatches = 0;   ///< ... and bricks in one set but not the other (cumulative).
        // DIAGNOSTIC: why loads stall once the camera moves (mapped 243k -> ~100k on the walk, framelog1). Cumulative except where noted.
        uint32_t commitFails = 0;     ///< Frames whose loads stopped on a full atlas that eviction could not free.
        uint32_t evictCalls = 0;      ///< Evictions run.
        uint32_t evictFreed = 0;      ///< Bricks they unloaded.
        uint32_t evictCandidates = 0; ///< Last eviction: loaded, unmapped, undesired leaves found.
        uint32_t evictScans = 0;        ///< DIAGNOSTIC: scans of the loaded list for eviction candidates, cumulative.
        uint64_t updateUs = 0;          ///< DIAGNOSTIC: CPU time of update() wherever it ran, cumulative (microseconds) ...
        uint64_t workerUpdateUs = 0;    ///< ... of which on the residency's own thread (launchUpdate) ...
        uint64_t takeWaitUs = 0;        ///< ... and the frame's wait for those at takeUpdate: the rest ran beside the frame.
        /// DIAGNOSTIC: the update thread's cycles while running (QueryThreadCycleTime) against the TSC over its updates: their
        /// ratio is the share of the update's wall time the thread was on a core.
        uint64_t workerRunCycles = 0;
        uint64_t workerWallCycles = 0;
        uint64_t workerKernelUs = 0; ///< DIAGNOSTIC: the update thread's kernel-mode time over its updates (GetThreadTimes, ticks).
        uint64_t workerUserUs = 0;   ///< ... and user-mode time.
        uint64_t replayUs = 0;         ///< ... and takeUpdate's recording of what the update queued (the frame thread's).
        uint32_t workerUpdates = 0;     ///< DIAGNOSTIC: updates launched on the residency's own thread, cumulative.
        /// DIAGNOSTIC: update()'s phases, CPU microseconds, cumulative (the profiler's scopes see only sampled frames on the frame's
        /// thread): merge (taking a cut on), loads (evict included), evict, maps, stage, fades, sun scheduling, upload, cut launch.
        std::array<uint64_t, 9> phaseUs{};
        /// DIAGNOSTIC (setStepTiming): TSC cycles of the steps of unmap, map and the fade ends (CloudResidency::Step).
        std::array<uint64_t, 40> stepCycles{};
        uint64_t walkTouchedCalls = 0;  ///< DIAGNOSTIC: bricks noted as changed beside a walk, cumulative (repeats included).
        uint64_t walkTouchedListed = 0; ///< DIAGNOSTIC: of those, listed for the merge (once a brick a walk).
        uint32_t loadHeld = 0;        ///< Last frame: desired bricks skipped because their parent was not loaded.
        uint32_t loadScanned = 0;     ///< Last frame: load-list entries visited.
        uint32_t freeSlots = 0;       ///< Last frame: free atlas slots.
        uint32_t freeBricks = 0;      ///< Last frame: free brick table entries.
        uint32_t uploadRuns = 0;      ///< DIAGNOSTIC: dirty-block runs uploaded by setBlob (nodes, bricks, sun tables; cumulative).
        uint64_t uploadBytes = 0;     ///< DIAGNOSTIC: their bytes, and the scatter packs' (cumulative).
        uint32_t scatterJobs = 0;     ///< DIAGNOSTIC: tables uploaded by the scatter instead (cumulative).
        uint32_t scatterChecks = 0;   ///< DIAGNOSTIC: frames whose GPU tables were compared with the CPU's (cumulative).
        uint32_t sunStampRuns = 0;    ///< DIAGNOSTIC: change uploads for stampSunChanges (cumulative).
        uint32_t sunStampRaw = 0;     ///< DIAGNOSTIC: changes before the dedupe (cumulative).
        uint32_t sunStampChanges = 0; ///< DIAGNOSTIC: changes stamped (cumulative).
        uint32_t sunStampStores = 0;  ///< DIAGNOSTIC: the last scheduling run's stamp stores (hstrCloudSunCounters 8).
        uint32_t sunStampFresh = 0;   ///< ... of those, under stampSkipSame, the ones that changed the entry (9).
        uint32_t sunStampCheckStores = 0; ///< DIAGNOSTIC (cloudSunStampFineCheck): the check pass's stores (10)
        uint32_t sunStampCheckFresh = 0;  ///< ... and the entries it changed: the fine stamp's misses (1) or extras (2); must be 0.
        uint32_t sunFineRuns = 0;      ///< DIAGNOSTIC (cloudSunStampFine): runs stamped by stampSunFine (cumulative).
        uint32_t sunFineFallbacks = 0; ///< DIAGNOSTIC: runs over the candidate list's capacity, stamped in full instead (cumulative).
        uint32_t sunFineCells = 0;     ///< DIAGNOSTIC: candidate cells over all runs, fallbacks included (cumulative).
        uint32_t sunFineCellsPeak = 0; ///< DIAGNOSTIC: the most candidate cells in one run.
        uint32_t sunFineUploadBytes = 0; ///< DIAGNOSTIC: bytes stampSunFine's inputs uploaded (cumulative).
        uint32_t sunFineWaves = 0;     ///< DIAGNOSTIC: the last run's stampSunFine (class, cell) waves (hstrCloudSunCounters 12),
        uint32_t sunFinePassed = 0;    ///< ... those whose union box held a change (13),
        uint32_t sunFineTested = 0;    ///< ... and the entries they tested (14).
        uint32_t scatterMismatches = 0; ///< DIAGNOSTIC: entries that differed (cumulative).
        uint32_t slotsUsed = 0;
        uint32_t nodesUsed = 0;
        uint32_t pagesLoaded = 0;
        uint32_t pageRegions = 0;  ///< Canonical hierarchy regions invalidated this frame.
        uint32_t pageExpanded = 0; ///< Page candidates before bitmap deduplication.
        uint32_t pageUnique = 0;   ///< Exact compacted pages resolved by the GPU.
        double residentMB = 0.0;
        double payloadMB = 0.0; ///< Payload pool in use.
        double cutMilliseconds = 0.0;
        uint32_t sunBaked = 0;      ///< Mapped bricks whose sun depth is baked for their instance's key.
        uint32_t sunWaiting = 0;    ///< Mapped bricks still on the live march (not yet baked, or neighbourhood still streaming).
        uint32_t sunBakesFrame = 0; ///< Bakes staged this frame.
        uint32_t sunSlotsFree = 0;  ///< Free sun atlas slots.
        uint32_t sunRuns = 0;       ///< DIAGNOSTIC: GPU sun scheduling runs (cumulative).
        uint32_t sunIdleFrames = 0; ///< DIAGNOSTIC: frames the GPU sun scheduler idled on a zero-bake run's inputs (cumulative).
        uint32_t sunStale = 0;      ///< Bakes dropped this frame because their neighbourhood was mapped or unmapped since.        // The last cut.
        uint32_t cutPops = 0;      ///< Bricks refined.
        bool cutOrdered = false;   ///< Whether the atlas budget bound, forcing the priority-ordered cut.
        uint32_t cutOrderedCount = 0; ///< DIAGNOSTIC: cuts taken on that were ordered (cumulative).
        uint32_t cutEntered = 0;      ///< DIAGNOSTIC: bricks entering the desired set over all cuts taken on (cumulative).
        uint32_t cutLeft = 0;         ///< DIAGNOSTIC: bricks leaving it (cumulative).
        uint32_t cutReentered = 0;    ///< DIAGNOSTIC: of cutEntered, bricks that had left within the last ~4 cuts (cumulative).
        std::array<uint32_t, 8> cutEnteredLevel{}; ///< DIAGNOSTIC: cutEntered by level (cumulative).
        double cutTotalMs = 0.0;
        float cutMargin = 0.f; ///< The motion envelope of the last cut, in voxels.
        uint32_t cuts = 0;     ///< Cuts run so far.
        // Page table edits so far (maps and unmaps), and what waits.
        uint32_t maps = 0;
        uint32_t unmaps = 0;
        uint32_t mapBacklog = 0;   ///< Desired bricks not mapped yet (waiting on a parent or a load).
        uint32_t activeFades = 0;  ///< Bricks whose fade runs.
        uint32_t fadeStarts = 0;   ///< Fades started or turned so far.
        uint32_t fadeEnds = 0;     ///< Fade ends handled so far.
        uint32_t fadeVoid = 0;     ///< End entries whose fade had already been replaced.
        /// DIAGNOSTIC: of fadeEnds, fades in (settled); the rest unmap. MEASURED with per-end timers (cpuprof8, 4K sunset walk at 2
        /// units a frame, 120 frames, 565k ends): ~200 ns a fade-in end, ~860 a fade-out end, of which replace ~285 and writeSunSched
        /// ~95; the timers themselves cost ~0.3 ms a walk frame and were taken out.
        uint32_t fadeEndIns = 0;
        uint32_t fadeParallelBuckets = 0; ///< DIAGNOSTIC: fade end buckets swept by several threads (setParallelFades).
        uint32_t mapParallelRuns = 0;     ///< DIAGNOSTIC: map queues mapped by several threads (setParallelMaps).
        uint32_t fadeParallelEnds = 0;    ///< DIAGNOSTIC: fade ends in buckets swept by several threads.
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

    /// Var: a ShaderVar, or anything indexed by field name like one (HSTRCloud's cached fields).
    template<typename Var>
    void bind(const Var& var) const
    {
        var["hstrCloudInstances"] = mpInstances;
        var["hstrCloudAssets"] = mpAssets;
        var["hstrCloudDirectory"] = mpDirectory;
        var["hstrCloudNodes"] = mpNodes;
        var["hstrCloudPages"] = mpPages;
        var["hstrCloudLevelPages"] = mpLevelPages;
        var["hstrCloudSkirtMasks"] = mpSkirtMasks;
        var["hstrCloudBricks"] = mpBricks;
        var["hstrCloudAtlas"] = mpAtlas;
        var["hstrCloudOccupancy"] = mpOccupancy;
        var["hstrCloudSunAtlas"] = mpSunAtlas;
        var["hstrCloudSunPacked"] = mpSunPacked;
        var["hstrCloudSunBakes"] = mpSunBakes;
        var["hstrCloudSunSlots"] = mpSunSlotTable;
        var["hstrCloudSunResolved"] = mpSunResolved;
        var["hstrCloudPayload"] = mpPayload->getBuffer();
        var["hstrCloudResiduals"] = mpResiduals;
        var["hstrCloudStagingInfo"] = mpStagingInfo;
    }
    void bindPageUpdates(const ShaderVar& var) const;
    ref<Texture> getAtlas() const { return mpAtlas; }
    ref<Buffer> getOccupancy() const { return mpOccupancy; }
    ref<Buffer> getResiduals() const { return mpResiduals; } ///< DIAGNOSTIC (cloudCommitCheck).
    bool isBrickLoaded(uint32_t gpu) const { return gpu < mBrickOwner.size() && mBrickOwner[gpu] != kNoHandle; } ///< DIAGNOSTIC
    ref<Texture> getSunAtlas() const { return mpSunAtlas; }
    ref<Texture> getSunPacked() const { return mpSunPacked; } ///< CloudResidencyDesc::sunPacked (null when off).
    /// Sun bakes staged this frame (bakeCloudSun runs over them after the commits).
    uint32_t getSunBakeCount() const { return uint32_t(mSunBakes.size()); }
    /// Bricks staged this frame (decodeCloudResiduals runs over them before the commit groups).
    uint32_t getStagedCount() const { return uint32_t(mStaged.size()); }
    const std::vector<CommitGroup>& getCommitGroups() const { return mCommitGroups; }
    uint32_t getAtlasShift() const { return 6; }
    const Stats& getStats() const { return mStats; }
    /// Whether a cut is running on the worker (CloudView::cutAsync). The sea must not apply tile changes meanwhile: the walk reads them.
    bool cutInFlight() const { return mCutJob.valid(); }
    /// Whether that cut is still walking: one that finished is taken by the next update, which accepts changed slots with it.
    bool cutRunning() const { return mCutJob.valid() && mCutJob.wait_for(std::chrono::seconds(0)) != std::future_status::ready; }

    /// CloudView::cutGpuVisibility: the visibility refreshes a walk asked for, traced on the GPU (CutVisibility.cs.slang) and handed
    /// back; the cache takes them before the next walk. Both calls are safe beside a running update or walk (their own lock).
    struct VisibilityRequest
    {
        uint64_t handle;
        uint32_t slot;
        uint32_t generation;
        float3 lo; ///< The brick's world bounds.
        float3 hi;
        uint32_t epoch = 0; ///< The slot's epoch the walk saw (mSlotEpoch).
    };
    struct VisibilityBatch
    {
        float3 camera = float3(0.f); ///< The walk's camera: the rays start there.
        std::vector<VisibilityRequest> requests;
    };
    /// The oldest batch of requests not yet traced, if any.
    bool takeVisibilityBatch(VisibilityBatch& batch);
    /// A traced batch: each request's visibility, in order.
    void putVisibility(VisibilityBatch&& batch, std::vector<float>&& visibility);
    /// Whether advanceCloudFades has fades to run this frame, and what it binds.
    bool fadesRunning() const { return mActiveFades > 0; }
    /// Whether an outdated sun bake keeps answering until it rebakes in place (HSTRCloudSunFrame::keepStale).
    void setSunKeepStale(bool keep) { mSunKeepStale = keep; }
    /// Bakes staged a frame at most, under CloudResidencyDesc::sunBakesPerFrame (the buffers' size); 0: that many.
    void setSunBakesCap(uint32_t cap) { mSunBakesCap = cap; }
    /// Whether a mapping change outdates only the bakes of its brick's level or finer (HSTRCloudSunFrame::levelStamps).
    void setSunLevelStamps(bool levelStamps) { mSunLevelStamps = levelStamps; }
    /// Whether scanSunBakes sums its atomics per wave and bin first (HSTRCloudSunFrame::scanWave).
    void setSunScanWave(bool scanWave) { mSunScanWave = scanWave; }
    /// Whether stampSunChanges runs a thread per change and class (HSTRCloudSunFrame::stampSplit).
    void setSunStampSplit(bool stampSplit) { mSunStampSplit = stampSplit; }
    void setSunStampRows(uint32_t rows) { mSunStampRows = std::max(rows, 1u); }
    void setSunStampSkipSame(bool skip) { mSunStampSkipSame = skip; }
    void setSunStampFine(bool fine) { mSunStampFine = fine; }
    void setSunFineUploadAll(bool all) { mSunFineUploadAll = all; }
    void setSunEvery(uint32_t every) { mSunEvery = std::max(every, 1u); }
    /// asyncResidency: the context this frame's uploads and scatters record into (the compute queue's), or nullptr for the device's
    /// render context. On, freed atlas slots, GPU brick indices and sun slots are quarantined: none is written for a new brick or
    /// bake before a later frame, so the main queue's frame running beside this one never reads another brick's data through a
    /// mapping it still holds.
    void setAsyncContext(ComputeContext* pContext) { mpAsyncContext = pContext; }
    /// The context residency work records into this frame.
    ComputeContext* getContext() const;
    /// update() on the residency's own thread while the frame records the rest: its uploads and dispatches are queued rather than
    /// recorded, and its profiler scopes are off (Falcor's profiler is the frame thread's). The update keeps the asyncResidency
    /// state of the frame that launched it. Nothing else may touch the residency until takeUpdate (or waitUpdate, to read it).
    void launchUpdate(const CloudSea& sea, const CloudView& view, std::vector<uint32_t> changedSlots);
    /// Whether a launched update has not been taken yet.
    bool updatePending() const { return mUpdateJob.valid(); }
    /// The next update's frame number, when updates do not come every frame (HSTRCloud cloudResidencySkipBusy): fades, sun cadence
    /// and the other frame counts then run in frames, not updates. Called before launchUpdate.
    void setFrameClock(uint32_t frame) { mNextFrame = frame; }
    /// DIAGNOSTIC: the frame clocks and what the updates did, for HSTRCloud's cloudResidencyTrace (not beside a running update).
    std::string traceState() const
    {
        return fmt::format(
            "frame {} next {} cutFrame {} fadeProcessed {} activeFades {} maps {} fadeEnds {} cuts {} toMap {} mapQueue {} toLoad {}/{} "
            "cutInFlight {}",
            mFrame, mNextFrame, mCutFrame, mFadeProcessed, mActiveFades, mStats.maps, mStats.fadeEnds, mStats.cuts, mToMap.size(),
            mMapQueue.size(), mToLoadNext, mToLoad.size(), mCutJob.valid()
        );
    }
    /// DIAGNOSTIC: false copies each deferred upload into a vector of its own (the arena's A/B anchor).
    void setDeferArena(bool on) { mDeferArena = on; }
    uint32_t getFrame() const { return mFrame; }
    /// The fade frame the GPU's advanceCloudFades reads, at a given frame (HSTRCloud writes it on frames without an update).
    HSTRCloudFadeFrame fadeFrameAt(uint32_t frame) const { return {frame & kCloudFadeFrameMask, uint32_t(mBricks.size()), mFadeStep, 0}; }
    /// Whether a launched update has finished (takeUpdate would not wait).
    bool updateReady() const { return mUpdateJob.valid() && mUpdateJob.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }
    /// Waits for a launched update without taking it; the residency may then be read.
    void waitUpdate() const
    {
        if (mUpdateJob.valid())
            mUpdateJob.wait();
    }
    /// Waits for a launched update, records what it queued into getContext() in its order, and returns what update() returned.
    bool takeUpdate();
    void setLoadsBesideCut(bool on) { mLoadsBesideCut = on; }
    /// Bricks committed a frame, at most desc.loadsPerFrame (the staging capacity); a runtime cap, so it changes without a rebuild.
    void setLoadCap(uint32_t cap) { mLoadCap = std::clamp(cap, 1u, mDesc.loadsPerFrame); }
    void setScatterUploads(bool on) { mScatterUploads = on; }
    /// A table scatters whenever its changed elements cost fewer bytes than its dirty blocks (up to half the table), not only up
    /// to an eighth of it.
    void setScatterDense(bool on) { mScatterDense = on; }
    /// DIAGNOSTIC: times the steps of unmap, map and the fade ends into Stats::stepCycles (an rdtsc pair a step).
    void setStepTiming(bool on) { mStepTiming = on; }
    /// Fade ends swept by several threads (mParallelFades); false: one (the A/B anchor).
    void setParallelFades(bool on) { mParallelFades = on; }
    /// Maps by several threads (mParallelMaps); false: one (the A/B anchor).
    void setParallelMaps(bool on) { mParallelMaps = on; }
    /// The cut merge's lists by several threads (mParallelMerge); false: one (the A/B anchor).
    void setParallelMerge(bool on) { mParallelMerge = on; }
    /// Upload's scatter gathers by several threads (mParallelUpload); false: one (the A/B anchor).
    void setParallelUpload(bool on) { mParallelUpload = on; }
    void setParallelMin(uint32_t count) { mParallelMin = std::max(count, 1u); }
    /// The most threads a parallel residency step uses (up to 16 and the hardware's).
    void setParallelThreads(uint32_t count) { mParallelThreads = std::max(count, 1u); }
    /// The sun changes' sort and the fine stamp's passes by several threads (mParallelSun); false: as before (the A/B anchor).
    void setParallelSun(bool on) { mParallelSun = on; }
    /// evict's late candidates: only those that joined since the last call sorted, merged into its sorted remainder (false: all
    /// sorted every call, the A/B anchor).
    void setEvictLateMerge(bool on) { mEvictLateMerge = on; }
    /// mapReady by events (mMapEvents); false scans the whole map list every update (the A/B anchor).
    void setMapEvents(bool on)
    {
        mMapReseed |= on && !mMapEvents;
        mMapEvents = on;
    }
    /// Stats::stepCycles entries.
    enum Step : uint32_t
    {
        kStepReplace,     ///< unmap: the page table (its dirtyPages too).
        kStepDirtyPages,  ///< paint / replace: dirtyPages alone.
        kStepUnmapMark,   ///< unmap: markChanged and the beam change.
        kStepUnmapNotes,  ///< unmap: its own fade, flags, noteMapped, noteEvictCandidate.
        kStepUnmapParent, ///< unmap: the parent's count, noteMapped, updateFade.
        kStepUnmapSun,    ///< unmap: writeSunSched.
        kStepEndValid,    ///< fade ends: the entry's check.
        kStepEndSetFade,  ///< fade ends out: setFade(0) before the unmap.
        kStepEndUnlist,   ///< fade ends out: unlistMapped.
        kStepEndIn,       ///< fade ends in: all of it.
        kStepPaint,       ///< map: the page table (its dirtyPages too).
        kStepMapMark,     ///< map: markChanged.
        kStepMapNotes,    ///< map: noteMapped, the parent's count, noteMapped and updateFade, the list.
        kStepMapFade,     ///< map: its own updateFade.
        kStepMapSun,      ///< map: writeSunSched.
        kStepMergeLeft,   ///< cut merge: the bricks the cut left (noteEvictCandidate).
        kStepMergeFading, ///< cut merge: the walk's fading list (updateFade).
        kStepMergeSun,    ///< cut merge: the sun scheduler states the walk wrote.
        kStepMergeTouched, ///< cut merge: the bricks changed beside the walk.
        kStepFadePartition, ///< parallel fade ends: the bucket split by chunk.
        kStepFadeThreads,   ///< ... the threads' sweeps (wall time).
        kStepFadeMerge,     ///< ... their sinks merged.
        kStepFadeSerial,    ///< ... the serial sweep after (coarse stores; whole buckets below setParallelMin).
        kStepMapCoarse,     ///< parallel maps: the coarse stores' and the split by chunk.
        kStepMapThreads,    ///< ... the threads' maps (wall time).
        kStepMapMerge,      ///< ... their sinks merged.
        kStepMapSerial,     ///< ... the serial pass after (retries; whole queues below setParallelMin).
        kStepSunSort,       ///< uploadSunChanges: the changes sorted and deduplicated.
        kStepSunResets,     ///< ... the field resets.
        kStepSunFine,       ///< ... buildSunFineStamp.
        kStepSunSend,       ///< ... the changes uploaded.
        kStepEvictSort,     ///< evict: the late candidates sorted.
        kStepEvictCheck,    ///< ... each candidate checked as it is taken.
        kStepEvictUnload,   ///< ... unload.
        kStepCommit,        ///< commit, less its eviction.
        kStepCount
    };
    void addScatterChecks(uint32_t frames) { mScatterCheck += frames; }
    bool scatterCheckPending() const { return mScatterCheck > 0; } ///< Its blocking readbacks need the residency's work serial.
    uint32_t getBrickCapacity() const { return uint32_t(mBricks.size()); }
    const ref<Buffer>& getBricks() const { return mpBricks; }
    const ref<Buffer>& getPages() const { return mpPages; }
    const ref<Buffer>& getLevelPages() const { return mpLevelPages; }
    /// The instance records as last uploaded (one per sea tile slot) and each asset's page table: for rtSpanProbe.
    const std::vector<HSTRCloudInstance>& getInstanceRecords() const { return mInstances; }
    uint32_t getAssetCount() const { return uint32_t(mAssets.size()); }
    uint3 getAssetDims(uint32_t asset) const { return mAssetRecords[asset]->dims; }
    uint32_t getAssetPageOffset(uint32_t asset) const { return mAssets[asset].pageOffset; }
    const ref<Buffer>& getSkirtMasks() const { return mpSkirtMasks; }
    uint32_t getPageCount() const { return uint32_t(mPages.size()); }
    const ref<Buffer>& getSunResolved() const { return mpSunResolved; }
    const ref<Buffer>& getSunTouched() const { return mpSunTouched; }
    /// GPU bricks whose level or parent the CPU wrote since the last resolveCloudSunSlots (loads), for hstrCloudSunTouched.
    std::vector<uint32_t>& getSunTouchList() { return mSunTouchList; }
    /// Bricks whose density settled since the caller last cleared this (fade-in ends, unmaps), as asset, level, packed coordinates,
    /// 0: what the persistent beam image invalidates (markBeamChanges). The caller clears it every frame.
    std::vector<uint4>& getBeamChanges() { return mBeamChanges; }
    const ref<Buffer>& getDirtyPageRegions() const { return mpDirtyPageRegions; }
    const ref<Buffer>& getDirtyPageBits() const { return mpDirtyPageBits; }
    const ref<Buffer>& getDirtyPages() const { return mpDirtyPages; }
    const ref<Buffer>& getDirtyPageCount() const { return mpDirtyPageCount; }
    const ref<Buffer>& getDirtyPageArgs() const { return mpDirtyPageArgs; }
    uint32_t getDirtyPageRegionCount() const { return uint32_t(mDirtyPageRegions.size()); }
    uint32_t getDirtyPageWorkCount() const { return mDirtyPageWorkCount; }
    /// The skirt masks this frame's page changes can reach (resolveCloudSkirtRegions); UINT64_MAX past the region buffer.
    uint32_t getSkirtRegionCount() const { return uint32_t(mSkirtRegions.size()); }
    uint64_t getSkirtWorkCount() const { return mSkirtWorkCount; }
    void readPageStats() { mStats.pageUnique = mpDirtyPageCount->getElement<uint32_t>(0); }
    void pageUpdatesDispatched()
    {
        mStats.pageRegions = uint32_t(mDirtyPageRegions.size());
        mStats.pageExpanded = mDirtyPageWorkCount;
        mDirtyPageRegions.clear();
        mDirtyPageWorkCount = 0;
        mSkirtRegions.clear();
        mSkirtWorkCount = 0;
    }
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
        bool releaseOnly = false; ///< cloudSunEvery's frames between runs: only the release pass (info.releaseCount).
        HSTRCloudSunFrame info = {}; ///< Uploaded to hstrCloudSunFrameInfo.
    };
    const GpuSunFrame& getGpuSunFrame() const { return mGpuSunFrame; }
    /// DIAGNOSTIC (cloudSunStampFineCheck): the scheduler's frame record, rewritten between check passes.
    void setGpuSunFrameInfo(const HSTRCloudSunFrame& info) const { mpSunFrameInfo->setBlob(&info, 0, sizeof(HSTRCloudSunFrame)); }
    /// HSTRCloud dispatched the frame: a frame without a residency update (frozen) must not replay its releases, resets and changes.
    void gpuSunDispatched() { mGpuSunFrame = {}; }
    /// Binds the scheduler's buffers for writing (and unbinds the camera's views of the same buffers).
    void bindGpuSun(const ShaderVar& var) const;
    /// The scheduler's buffers to clear before its scan.
    ref<Buffer> getGpuSunCounters() const { return mpSunCounters; }
    /// The bake jobs (hstrCloudSunBakes): CPU-staged, or written by assignSunBakes.
    const ref<Buffer>& getSunBakes() const { return mpSunBakes; }
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
        uint16_t refined = 0;    ///< The ordered cut refined it (its children were desired too).
        float admit = 0.f;       ///< CloudView::cutThreshold: the highest priority of a parent that refined into it (seeds: max).
    };

    struct Brick
    {
        BrickHeader record;
        uint64_t parent = kNoHandle;
        uint32_t children = kNone; ///< Offset into the store's child list (childCount entries; page indices for level-3 bricks).
        uint32_t gpu = kNone;      ///< Brick index on the GPU while loaded.
        uint32_t slot = kNone;     ///< Atlas slot while loaded.
        uint32_t mappedIndex = kNone; ///< Position in mMappedList while mapped.
        uint32_t loadedIndex = kNone; ///< Position in mLoadedList while loaded.
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
        uint32_t walkTouched = 0; ///< mWalkTouchedEpoch when last listed in mWalkTouched (listed once a walk).
        uint32_t leftCut = 0;     ///< DIAGNOSTIC: the cut taken on when it last left the desired set (0: never).
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
        uint32_t pageOffset = 0;
        uint32_t levelPageBase = 0; ///< First word of the asset's resolved level pages (HSTRCloudAsset::levelPageBase).
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
    /// Whole-library residency (HSTR_CLOUD_STREAMING 0): reads and decodes every chunk and level-2 page (in parallel over assets).
    struct PreloadedPage
    {
        DecodedPage meta;
        std::vector<uint8_t> payload;
    };
    struct PreloadedChunk
    {
        uint32_t chunk = 0;
        DecodedPage meta;
        std::vector<uint8_t> payload;
        std::vector<PreloadedPage> pages; ///< In the order of meta.pages.
    };
    std::vector<std::vector<PreloadedChunk>> readLibrary(const std::vector<CloudAsset>& assets, uint64_t& payloadBytes) const;
    /// Creates the chunk and page stores of readLibrary's pages, linked as arriving pages were, their payloads uploaded.
    void createLibraryStores(std::vector<std::vector<PreloadedChunk>>& library);
    void enqueue(std::vector<Request> requests);
    void resetPending(const Request& request);

    float brickPriority(const CloudSea& sea, uint32_t slot, uint64_t handle, const CloudView& view, float& visibility);
    /// `maxDepth` (CloudView::cutRays 5 / 6): stop once the optical depth reaches it (the reference stops at 6).
    float transmittance(const CloudSea& sea, const CloudView& view, float3 target, float maxDepth = 6.f) const;
    bool inFrustum(const CloudView& view, float3 lo, float3 hi) const;

    bool commit(uint64_t handle, bool evictOk);
    void loadReady(bool evictOk); ///< Commits desired bricks from the load cursor, loadsPerFrame at most.
    void stageCommits();          ///< Orders this frame's staged bricks coarsest level first, one commit group per level.
    bool evict(uint32_t slotsNeeded, uint32_t metasNeeded);
    void unload(uint64_t handle);
    bool map(uint64_t handle);
    void unmap(uint64_t handle);
    void paint(uint32_t asset, const BrickHeader& record, uint32_t ref);
    void replace(uint32_t asset, const BrickHeader& record, uint32_t ref, uint32_t replacement);
    void paintEntry(uint32_t& entry, uint32_t ref, uint32_t level);
    void replaceEntry(uint32_t& entry, uint32_t ref, uint32_t replacement);
    void dirtyPages(uint32_t asset, const BrickHeader& record);
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
    std::vector<uint4> mBeamChanges; ///< See getBeamChanges.
    // The apply step touches only what can change (the cut merge rebuilds these; the rest is events):
    size_t mToLoadNext = 0;          ///< mToLoad before it is loaded.
    std::vector<uint64_t> mToMap;    ///< Desired bricks not yet mapped, parents first (kToMap).
    /// setMapEvents: mapReady takes only the bricks whose readiness may have changed - the cut's list once (mMapReseed), then each
    /// brick committed while desired, and the bricks waiting on a parent once it maps - instead of the whole of mToMap every update.
    /// stepsplit1 (4K sunset walk, worker on, step timers): the maps phase 101 ms over 80 frames, map() itself 50.5 of it; the rest
    /// was the scan of mToMap, 10-70k bricks waiting on their loads, each a random brick read. MEASURED (mapevents1, same walk,
    /// arms alternating): the phase less map() 28 / 39 ms (events) against 68 (scan) over 80 frames, 0.66 / 0.60 against 0.91 us a
    /// mapped brick; chunk-end mapped counts in the scan arm's range.
    bool mMapEvents = true;
    bool mMapReseed = true;
    std::vector<uint64_t> mMapQueue;
    std::unordered_multimap<uint64_t, uint64_t> mMapWaitParent; ///< Parent -> loaded desired child waiting for it to map.
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
    std::vector<std::vector<FadeEnd>> mFadeEnds; ///< Ring of fadeFrames + 1 buckets, by the frame a fade ends.
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
    uint32_t mFadeEndOrder = 0; ///< CloudView::fadeEndOrder of the frame.
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
    std::vector<uint64_t> mWalkTouched;         ///< Bricks changed while the walk ran, each once (noteWalkTouched).
    uint32_t mWalkTouchedEpoch = 1;             ///< Bumped whenever mWalkTouched is cleared.
    /// evict's candidates, least recently desired first: built by one scan of the loaded list and taken in order (each checked again
    /// as it is taken) until a store is released or it runs out. Bricks that become candidates meanwhile - an unmap, a parent's last
    /// loaded child unloaded, a cut that no longer desires them, a walk's snapshot gone - go on mEvictLate, merged in by key.
    std::vector<std::pair<uint32_t, uint64_t>> mEvictOrder;
    size_t mEvictNext = 0;
    bool mEvictValid = false;
    std::vector<std::pair<uint32_t, uint64_t>> mEvictLate;
    /// setEvictLateMerge: mEvictLate's first entries, sorted by key (the last call's remainder); what joined since follows.
    /// MEASURED (evictmerge1, 4K sunset walk, step timers): neutral - the sort was 1.0 / 1.6 ms over 80 frames against 1.1 sorting
    /// all every call; evict is its unloads (21-34 ms) and candidate checks (5-10).
    size_t mEvictLateSorted = 0;
    bool mEvictLateMerge = true;
    bool evictCandidate(const Brick& b) const;
    void noteEvictCandidate(uint64_t handle);
    void scanEvictCandidates();
    /// Lists a brick changed beside the walk, once a walk. Every map, unmap, fade and sun state write listed it again: the merge
    /// redid each brick per listing and re-queued it to load and map as often. MEASURED (4K sunset walk, 2 units a frame): 2.20M
    /// listings -> 0.69M bricks; merge/touched 0.84 -> 0.30 ms a walk frame (cpuprof2 -> cpuprof3).
    void noteWalkTouched(uint64_t handle);
    /// Refreshes a brick's mMappedState entry (and records it while a walk runs).
    void noteMapped(uint64_t handle);
    /// Maps what the cut wants and its parents hold.
    void mapReady(bool& changed);
    void unlistMapped(uint64_t handle);
    /// setParallelFades: a bucket's fade ends swept by several threads, one chunk's bricks per thread. A chunk's page and chunk
    /// stores (levels 0-3) and its level-4 brick are touched by its own fades only, page table included; the coarse stores' ends
    /// (levels 4+, whose parents span chunks) run serially after. While a thread sweeps, everything shared it would write goes to its
    /// ResidencySink (tSink), merged in afterwards. stepsplit1 (4K sunset walk): the fades phase 160 ms over 80 frames, the largest of
    /// the update, ~660 ns a fade-out end and ~214 ns a fade-in end, one thread. MEASURED (parfades1, same walk, arms alternating, up
    /// to 16 threads): fades 45 / 62 ms for 236k / 334k ends (~190 ns an end) against 177 ms for 306k (579 ns), walk mean 6.20 / 7.45
    /// against 8.03 ms, p95 12.6 / 19.5 against 23.5. Exact where it can be checked (parfades2): 0 scatter mismatches and 0 stale
    /// skirt masks over 60 checked frames an arm. Motion score (parfades3, HSTR_MOTION_SCORE 2, arms reordered so neither config
    /// holds the first position): > 0.02 at 5.48 / 5.63% parallel against 5.36 / 5.55% serial.
    struct ResidencySink
    {
        std::vector<FadeEnd>* pCurrent = nullptr; ///< The partition being swept (ends due in the bucket append to it)...
        uint32_t currentBucket = 0;               ///< ... the bucket's ring index.
        std::vector<FadeEnd> serial;              ///< Ends due in the bucket of coarse-store bricks.
        std::vector<std::pair<uint32_t, FadeEnd>> laterEnds; ///< Ends due in other buckets.
        std::vector<uint4> beamChanges;
        std::vector<HSTRCloudDirtyPageRegion> dirtyRegions; ///< Without work counts (the merge counts them on).
        std::vector<HSTRCloudDirtyPageRegion> skirtRegions;
        std::vector<HSTRCloudSunChange> sunChanges;
        std::vector<uint32_t> changedAssets;
        std::vector<uint32_t> bricks, nodes, scheds, states; ///< Scatter elements newly marked (their flags are set in place).
        size_t directoryLo = SIZE_MAX, directoryHi = 0;
        std::vector<uint32_t> freeNodes;
        std::vector<uint64_t> walkTouched;
        std::vector<std::pair<uint32_t, uint64_t>> evictLate;
        std::vector<uint64_t> unlist;
        int64_t activeFades = 0;
        uint32_t unmaps = 0, fadeEnds = 0, fadeVoid = 0, fadeEndIns = 0, fadeStarts = 0; ///< As in Stats.
        uint64_t walkTouchedCalls = 0, walkTouchedListed = 0;
        bool changed = false;
        // Maps (setParallelMaps):
        std::vector<uint32_t> nodePool; ///< Free page nodes set aside for this thread's maps (ensureNode).
        std::vector<uint32_t> sunNodeResets;
        std::vector<uint64_t> mapped;                     ///< Into mMappedList in the merge.
        std::unordered_multimap<uint64_t, uint64_t> wait; ///< Parent -> child waiting for it, both of this thread's chunks.
        std::vector<std::pair<uint64_t, uint64_t>> waitShared; ///< Parent (a coarse store's) -> child: mMapWaitParent in the merge.
        std::vector<uint64_t> released; ///< Bricks it mapped whose mMapWaitParent children it took (the merge erases them).
        std::vector<uint64_t> retry;    ///< Bricks it could not map (its nodes ran out): the serial pass.
        uint32_t maps = 0;
        // Cut merge (setParallelMerge):
        std::vector<uint64_t> toMap, toLoad; ///< Onto mToMap / mToLoad in the merge.
        std::vector<uint32_t> stateSync;     ///< Onto mSunStateSync.
    };
    /// setParallelMerge: the cut merge's per-brick lists (left, fading, touched) visited by several threads, by chunk as the fade
    /// ends. stepsplit2 (4K sunset walk, the fades and maps parallel): the merge 57-86 ms over 80 frames, all on the 3-4 frames that
    /// took a cut (~20 ms each): touched 38-45, fading 13-31, left 3-6, sun states 3. MEASURED (parmerge1, same walk, with
    /// setParallelUpload): the merge 26 / 34 ms over 3 / 4 cuts against 84 over 4 (touched 15 / 17 against 48, fading 5 / 8 against
    /// 26); walk mean 5.05 / 5.63 against 5.93 ms, p95 8.5 / 11.8 against 19.0. parmerge2 (arms ser, par, par2, ser2;
    /// HSTR_MOTION_SCORE 2): 0 scatter mismatches and 0 stale skirt masks over 60 checked frames an arm; > 0.02 at 5.54 / 5.69%
    /// parallel against 6.07 / 5.88% serial.
    bool mParallelMerge = true;
    /// setParallelMin: the fewest fade ends, queued maps or merge bricks the threads take (fewer: this thread). MEASURED (parmin1,
    /// 4K sunset walk): at 2048, 99% of the fade ends were already in buckets the threads took; 256 / 512 changed nothing (176 /
    /// 184 ns an end against 176 / 180).
    size_t mParallelMin = 2048;
    uint32_t mParallelThreads = 16; ///< setParallelThreads.
    size_t parallelThreads() const { return std::clamp<size_t>(std::min<size_t>(std::thread::hardware_concurrency(), mParallelThreads), 1, 16); }
    /// setParallelSun. stepsplit3 (4K sunset walk, 20 sun runs over 80 frames): the changes' sort 27-42 ms (15-19k changes a run),
    /// buildSunFineStamp 30-33 ms, its per-asset passes on at most 5 threads (5 assets). MEASURED (psun1, same walk, arms
    /// alternating): the sun phase 34 / 35 ms against 98 with both on one thread; sort 9.1 / 8.6 against 22.8, fine stamp 24 / 26
    /// against 75 (and 30-33 per asset).
    bool mParallelSun = true;
    std::vector<std::vector<uint64_t>> mBrickParts; ///< parallelBricks' handles per thread.
    /// Visits handles, a chunk's on one thread (with its sink, merged afterwards, then the coarse stores' serially, sink null); all on
    /// this thread below a threshold or without parallelism.
    void parallelBricks(const std::vector<uint64_t>& handles, const std::function<void(uint64_t, ResidencySink*)>& visit, bool parallel);
    bool mParallelFades = true;
    /// setParallelMaps: mapReady's queue mapped by several threads, one chunk's bricks per thread as for the fade ends (the coarse
    /// stores' first, serially: parents map first). stepsplit1 / mapevents1 (4K sunset walk): map() ~430-660 ns a brick, the maps
    /// phase 75-122 ms over 80 frames, one thread. MEASURED (parmaps1, same walk, arms alternating, up to 16 threads): the maps
    /// phase 41 / 49 ms for 136k / 158k maps (0.30 us a map) against 85 ms for 125k (0.68 us); walk mean 5.45 / 6.22 against 6.21
    /// ms, p95 10.3 / 15.1 against 14.2. parmaps2 (arms ser, par, par2, ser2; HSTR_MOTION_SCORE 2): 0 scatter mismatches and 0 stale
    /// skirt masks over 60 checked frames an arm; > 0.02 at 5.35 / 5.40% parallel against 5.95 / 6.30% serial.
    bool mParallelMaps = true;
    std::vector<std::vector<uint64_t>> mMapParts; ///< The queue's fine bricks per thread.
    /// Maps one thread's partition into its sink (children released by its maps join entries).
    void mapPartition(std::vector<uint64_t>& entries, ResidencySink& sink);
    void mergeMapSink(ResidencySink& sink);
    /// The chunk-store index a brick's fade end or map is partitioned by (a page store's chunk store); kNone: a coarse store's.
    uint32_t partitionOf(uint64_t handle) const;
    std::vector<ResidencySink> mSinks;
    std::vector<std::vector<FadeEnd>> mFadeParts; ///< A bucket's ends per thread.
    bool fadeEndValid(const FadeEnd& end) const;
    /// One fade end: a fade in settles at 1, a fade out unmaps.
    void endFade(const FadeEnd& end, bool& changed);
    static thread_local ResidencySink* tSink; ///< This thread's sink while it sweeps a partition (else null: the members).
    /// The serial sweep of a bucket (ends appended meanwhile are swept too).
    void sweepFadeEnds(std::vector<FadeEnd>& bucket, bool& changed);
    /// One thread's partition, into its sink.
    void sweepFadePartition(std::vector<FadeEnd>& entries, ResidencySink& sink);
    void mergeResidencySink(ResidencySink& sink, uint32_t bucket);
    std::vector<float3x3> mTileForward; ///< Per slot: asset source voxel to tile-local domain voxel (linear part).
    /// What the walk reads of the slots, as beginCut copied them (CloudView::walkBesideTileChanges): the sea's instance tiles, their
    /// forward transforms, sun classes and epochs.
    std::vector<CloudSea::Tile> mWalkTiles;
    std::vector<float3x3> mWalkForward;
    std::vector<uint32_t> mWalkSunClass;
    std::vector<uint32_t> mWalkSlotEpoch;
    /// Per slot: bumped whenever the slot takes another instance; a GPU visibility result of an earlier epoch is dropped.
    std::vector<uint32_t> mSlotEpoch;
    /// Slots changed while the walk ran: their visibility caches are cleared when it is taken, and that update re-cuts.
    std::vector<uint32_t> mWalkChangedSlots;

    // GPU mirrors.
    std::vector<uint32_t> mDirectory;
    std::vector<uint32_t> mNodes; ///< 64 entries per node.
    std::vector<HSTRCloudVirtualPage> mPages;
    std::vector<HSTRCloudDirtyPageRegion> mDirtyPageRegions;
    uint32_t mDirtyPageWorkCount = 0;
    std::vector<HSTRCloudDirtyPageRegion> mSkirtRegions; ///< dirtyPages' regions widened by the skirt reach.
    uint64_t mSkirtWorkCount = 0;
    ref<Buffer> mpSkirtRegions;
    std::vector<uint32_t> mFreeNodes;
    std::vector<HSTRCloudBrick> mBricks;
    std::vector<uint32_t> mFreeBricks;
    std::vector<uint32_t> mFreeSlots;
    /// Freed this frame under asyncResidency (setAsyncContext): free again at the next update.
    std::vector<uint32_t> mRetiredBricks;
    std::vector<uint32_t> mRetiredSlots;
    ComputeContext* mpAsyncContext = nullptr;
    /// The running update's asyncResidency state (mpAsyncContext changes at the frame's ends, which a launched update spans).
    bool mUpdateAsync = false;
    /// launchUpdate: device work is queued (mDeferred) rather than recorded, and there are no profiler scopes (mpProfileContext null).
    bool mDeferDevice = false;
    RenderContext* mpProfileContext = nullptr;
    std::vector<std::function<void(ComputeContext*)>> mDeferred;
    std::vector<uint8_t> mDeferredBytes; ///< The deferred uploads' data (uploadBlob), until takeUpdate replays them.
    bool mDeferArena = true;             ///< setDeferArena.
    std::future<bool> mUpdateJob;
    std::thread mUpdateThread;
    std::mutex mUpdateMutex;
    std::condition_variable mUpdateWake;
    std::packaged_task<bool()> mUpdateTask; ///< Guarded by mUpdateMutex.
    bool mUpdateQueued = false;
    bool mUpdateStop = false;
    void updateWorker();
    /// A DeviceLocal buffer's update through getContext() (Buffer::setBlob records into the device's render context); queued with a
    /// copy of the bytes under launchUpdate.
    void uploadBlob(Buffer* pBuffer, const void* pData, size_t offset, size_t size);
    std::vector<uint64_t> mBrickOwner;
    std::vector<HSTRCloudStaging> mStagingInfo;
    std::vector<uint64_t> mStaged; ///< Handles staged this frame, in staging order before grouping.
    std::vector<CommitGroup> mCommitGroups;
    size_t mDirectoryDirty[2] = {SIZE_MAX, 0};
    std::vector<uint8_t> mNodeBlocksDirty;  ///< Per 256 nodes.
    std::vector<uint8_t> mBrickBlocksDirty; ///< Per 4096 bricks.
    /// The elements of a GPU table changed this frame (beside its block flags): uploadTables scatters them (CloudScatter.cs.slang)
    /// while they are few, and falls back to the dirty blocks' runs otherwise.
    struct ScatterSet
    {
        std::vector<uint32_t> list;
        std::vector<uint8_t> flag;
        void reset(size_t elements)
        {
            flag.assign(elements, 0);
            list.clear();
        }
        void mark(uint32_t element)
        {
            if (element < flag.size() && !flag[element])
            {
                flag[element] = 1;
                list.push_back(element);
            }
        }
        /// Marks it into another list (a ResidencySink's; distinct elements only from distinct threads).
        void mark(uint32_t element, std::vector<uint32_t>& into)
        {
            if (element < flag.size() && !flag[element])
            {
                flag[element] = 1;
                into.push_back(element);
            }
        }
        void clear()
        {
            for (uint32_t element : list)
                flag[element] = 0;
            list.clear();
        }
    };
    ScatterSet mNodeScatter;  ///< Page nodes (64 words each).
    ScatterSet mBrickScatter; ///< HSTRCloudBrick.
    ScatterSet mSchedScatter; ///< HSTRCloudSunSched (gpuSun).
    ScatterSet mStateScatter; ///< Live sun scheduler states (gpuSun).
    bool mScatterUploads = true;
    bool mScatterDense = false; ///< setScatterDense.
    /// setParallelUpload: upload's scatter gathers (a random table read and a flag clear an element) in up to 16 ranges.
    /// stepsplit2 (4K sunset walk, fades / maps / merge parallel): the upload phase 56-98 ms over 80 frames, ~0.5 MB a frame.
    /// MEASURED (parmerge1, with setParallelMerge): 28 / 36 ms for 32.9 / 46.0 MB (0.85 / 0.78 ms a MB) against 65 for 42.4 MB
    /// (1.53); exact (parmerge2: 0 scatter mismatches over 60 checked frames an arm).
    bool mParallelUpload = true;
    bool mStepTiming = false;   ///< setStepTiming.
    uint32_t mScatterCheck = 0; ///< DIAGNOSTIC: frames left to compare the GPU tables with the CPU's.
    std::vector<uint32_t> mScatterIndex;
    std::vector<uint32_t> mScatterData;
    ref<Buffer> mpScatterIndex;
    ref<Buffer> mpScatterData;
    ref<ComputePass> mpScatterUints;
    ref<ComputePass> mpScatterBricks;
    ref<ComputePass> mpScatterSched;
    bool mInstancesDirty = true;
    std::vector<HSTRCloudInstance> mInstances;

    ref<Buffer> mpInstances;
    ref<Buffer> mpAssets;
    ref<Buffer> mpDirectory;
    ref<Buffer> mpNodes;
    ref<Buffer> mpPages;
    ref<Buffer> mpLevelPages; ///< Resolved level pages of every asset (resolveDirtyCloudPages writes them with the pages).
    ref<Buffer> mpSkirtMasks; ///< Per page, the neighbours its skirt reads (resolveCloudSkirtMasks rewrites them after page changes).
    ref<Buffer> mpSunResolved; ///< Per brick and orientation class, the sun bake a sample uses (resolveCloudSunSlots, per frame).
    ref<Buffer> mpDirtyPageRegions;
    ref<Buffer> mpDirtyPageBits;
    ref<Buffer> mpDirtyPages;
    ref<Buffer> mpDirtyPageCount;
    ref<Buffer> mpDirtyPageArgs;
    ref<Buffer> mpBricks;
    ref<Texture> mpAtlas;
    ref<Buffer> mpOccupancy; ///< kCloudCellWords words per GPU brick: 4-bit cell density bounds (occupancyCloudBricks).
    ref<Texture> mpSunAtlas; ///< Baked sun optical depth: its own pool of slots laid out like density atlas slots.
    ref<Texture> mpSunPacked; ///< The sun atlas' slots with each bake's brick density beside it (CloudResidencyDesc::sunPacked).
    ref<Buffer> mpSunBakes;
    ref<Buffer> mpSunSlotTable;
    std::vector<HSTRCloudSunBake> mSunBakes;
    std::vector<uint32_t> mSunSlotTable;  ///< Per GPU brick, kCloudSunBakesPerBrick pairs (key, sun slot); key kCloudRefNone: not valid.
    std::vector<uint32_t> mSunBakeFrames; ///< Per pair: frame it was baked.
    std::vector<uint32_t> mFreeSunSlots;
    uint32_t mSunSlotCount = 0; ///< Sun atlas slots (sunPoolScale times the density slots).
    bool mSunKeepStale = true;
    uint32_t mSunBakesCap = 0;
    bool mSunLevelStamps = true;
    bool mSunScanWave = false;
    bool mSunStampSplit = false;
    uint32_t mSunStampRows = 1;
    bool mSunStampSkipSame = false;
    /// cloudSunStampFine: levels 0-1 stamped by stampSunFine, one thread per (candidate cell's node, class, entry) testing its support
    /// box against per-asset prefix sums of the run's changed cells, instead of per change over overlapping sweeps.
    bool mSunStampFine = false;
    std::vector<HSTRCloudAsset> mGpuAssetRecords; ///< The assets as uploaded (dims, top level, voxel size for the stamp grids).
    std::vector<uint32_t> mSunChangeSat;          ///< Per asset (offset, grid) headers, then the two prefix-sum grids per asset.
    std::vector<uint2> mSunFineCells;             ///< This run's candidate cells: (asset, x | y << 10 | z << 20).
    ref<Buffer> mpSunChangeSat;
    bool mSunChangeSatUploaded = false; ///< Its header (per-asset offsets and dimensions) is on the GPU.
    bool mSunFineUploadAll = false;     ///< DIAGNOSTIC cloudSunFineUploadAll: every asset's grids each run (the A/B anchor).
    ref<Buffer> mpSunFineCells;
    uint32_t mSunFineCellCapacity = 0;
    uint32_t mSunEvery = 1;
    uint32_t mSunRunFrame = 0; ///< mFrame of the last scheduling run (cloudSunEvery counts frames from it).
    uint32_t mSunFreeStack = 0;      ///< The free sun slot stack the next scheduling run pops (HSTRCloudSunFrame::freePop).
    bool mLoadsBesideCut = true; ///< Loads run on the frames a walk is on the worker (free slots only).
    uint32_t mLoadCap = 1024;    ///< setLoadCap.
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
        uint32_t freeStack = 0; ///< The free slot stack the run pops (sun slot quarantine alternates two).
        bool operator==(const SunScheduleInputs& o) const
        {
            return cutFrame == o.cutFrame && generation == o.generation && lastChange == o.lastChange && freeSlots == o.freeSlots &&
                   mapped == o.mapped && all(direction == o.direction) && freeStack == o.freeStack;
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
    void buildSunFineStamp(); ///< cloudSunStampFine: this run's changed-cell prefix sums and candidate cells.
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
    ref<Buffer> mpSunTouched; ///< Per GPU brick: the last scheduler frame that wrote its slot pairs (resolveCloudSunSlots).
    std::vector<uint32_t> mSunTouchList;
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
    /// A request's target: a chunk page (asset, chunk) or a level-2 page (store, page, generation).
    using LoadKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
    static LoadKey loadKey(const Request& r) { return r.page == kNone ? LoadKey{0u, r.asset, r.chunk, 0u} : LoadKey{1u, r.store, r.page, r.generation}; }
    std::set<LoadKey> mLoading; ///< Main thread: targets queued, in flight, or arrived and not yet stored.
    bool mReleaseStores = false;     ///< The payload pool filled: release unused stores now.
    size_t mReleaseCursor = 0;        ///< The store the incremental release sweep examined last.
    uint32_t mInFlight = 0;
    bool mStop = false;
    std::vector<std::thread> mIoThreads;
    std::vector<const CloudAsset*> mAssetRecords;

    uint32_t mFrame = 1;
    uint32_t mNextFrame = 0; ///< setFrameClock: the next update's mFrame (0: the last one's + 1).
    uint32_t mCutFrame = 0;        ///< Frame of the last cut.
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
        bool valid = false;          ///< Measured (a new entry is not).
        uint32_t requested = 0;      ///< CloudView::cutGpuVisibility: the frame of the walk that sent it to the GPU (0: none pending).
    };
    /// CloudView::cutGpuVisibility: the running walk's requests (under mVisBuildMutex), and the batches waiting for the GPU and the
    /// traced ones waiting for the next walk (under mVisMutex).
    std::mutex mVisBuildMutex;
    std::vector<VisibilityRequest> mVisBuilding;
    std::mutex mVisMutex;
    std::deque<VisibilityBatch> mVisOutbox;
    std::vector<std::pair<VisibilityBatch, std::vector<float>>> mVisInbox;
    void applyVisibility();
    /// Each slot's cache in shards by handle, each emplace under its shard's striped lock, so walk threads can share a slot (the
    /// waves); an entry is written only by the thread that visits its brick in that slot, and a rehash moves no entry.
    static constexpr uint32_t kVisibilityShards = 64;
    static constexpr uint32_t kVisibilityLocks = 256;
    std::vector<std::array<std::unordered_map<uint64_t, VisibilityEntry>, kVisibilityShards>> mVisibility;
    std::array<std::mutex, kVisibilityLocks> mVisibilityLocks;
    VisibilityEntry& visibilityEntry(uint32_t slot, uint64_t handle);

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
        float admit; ///< The refining parent's priority (CutState::admit).
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
        uint32_t entered = 0;                ///< DIAGNOSTIC: desired bricks the last cut did not desire.
        uint32_t reentered = 0;              ///< DIAGNOSTIC: ... of them, left within the last ~4 cuts.
        std::array<uint32_t, 8> enteredLevel{}; ///< ... by level.
        double milliseconds = 0.0;
        double parallelMs = 0.0;        ///< DIAGNOSTIC: the two parallel phases ...
        double heapMs = 0.0;            ///< ... and the ordered (heap) cut, when the budget binds.
        double coarseMs = 0.0;          ///< DIAGNOSTIC: the coarse phase, the fine phase and the serial merge of the visits ...
        double fineMs = 0.0;
        double desireMs = 0.0;
        double tailMs = 0.0;            ///< (the lists the frame applies, after the heap)
        double midMs = 0.0;             ///< DIAGNOSTIC: between the desire merge and the tail (threshold trim, heap).
        uint32_t visRequests = 0;       ///< CloudView::cutGpuVisibility: refreshes sent to the GPU.
        uint64_t visits = 0;           ///< ... the visits, the visibility refreshes and their transmittance rays.
        uint64_t refreshes = 0;
        uint64_t rays = 0;
        uint64_t rayMismatches = 0;     ///< (CloudView::cutRays 2)
        uint64_t coarseBusyCycles = 0;  ///< DIAGNOSTIC: the coarse phase's rdtsc cycles over its threads (waves: their parallel
        uint64_t coarsePriorityCycles = 0; ///< ranges), in brickPriority, in visibility refreshes.
        uint64_t coarseRayCycles = 0;
        uint64_t coarseSerialCycles = 0;   ///< DIAGNOSTIC: the waves' serial parts (cutWaves) ...
        uint64_t coarseCriticalCycles = 0; ///< ... and the critical path: the waves' parallel wall, or the busiest slot group.
        uint32_t config = 0;            ///< DIAGNOSTIC: the view's cutWaves | cutTailParallel << 1 | ray mode << 2 (0 reference,
                                        ///< 1 stepped, 2 blocked (removed), 3 cut off).
        uint32_t unorderedDesired = 0;  ///< DIAGNOSTIC: what the parallel walk desired (over the budget when ordered).
        uint32_t priorityRises = 0;     ///< DIAGNOSTIC: children the parallel walk gave a higher priority than their parent.
        bool thresholdTrimmed = false;  ///< CloudView::cutThreshold: trimmed to the budget ...
        bool underfilled = false;       ///< ... or pruned to under 95% of it.
        uint32_t checkBricks = 0;       ///< DIAGNOSTIC (cutThresholdCheck): the heap cut's bricks ...
        uint32_t checkMismatches = 0;   ///< ... and the bricks the threshold and heap sets differ in.
        std::vector<uint64_t> desired; ///< Parents before children.
        std::vector<uint64_t> toLoad;  ///< Coarsest level first, then by priority.
        std::vector<uint64_t> toMap;
        std::vector<uint64_t> fading;  ///< Mapped bricks that start or keep fading.
        std::vector<uint64_t> left;    ///< Bricks the last cut desired and this one does not (evict's new candidates).
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
public:
    /// DIAGNOSTIC: the cut walk records every expansion in this instance slot (~0u: off); read with walkTrace().
    uint32_t mTraceSlot = ~0u;
    std::string walkTrace()
    {
        std::lock_guard lock(mTraceMutex);
        std::string enqueues;
        for (const std::string& line : mEnqueueTrace)
            enqueues += line + "\n";
        return mTrace + enqueues;
    }
    /// DIAGNOSTIC: the desired cut per level: bricks, and the 10th / 50th percentile of their cut priority.
    std::string cutLevels() const;
private:
    std::mutex mTraceMutex;
    std::string mTrace;
    std::vector<std::string> mEnqueueTrace;
    std::string mTraceBuilding;
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
    float mCutPrune = 0.f;       ///< CloudView::cutThreshold: the next walk expands nothing below this priority (0: everything).
    Stats mStats;
    bool mWarnedNodes = false;
};
} // namespace hstrcloud
