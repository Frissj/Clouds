#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

// Radiance components shared by the HST frame and the path-traced reference, as bits of hstComponents, referenceShow,
// compareSubstitute and compareTarget. The HST approximates them by: background transmittance * sky; octave 0 of the sun
// residual; octaves 1-3 of the sun residual; the HST camera source (sky, plus hstSunFraction of the sun).
static const uint kComponentBackground = 1;  ///< Sky seen through the medium without scattering.
static const uint kComponentSunSingle = 2;   ///< Sun light scattered once.
static const uint kComponentSunMultiple = 4; ///< Sun light scattered two or more times.
static const uint kComponentSkyScattered = 8; ///< Sky light scattered one or more times.
static const uint kComponentCount = 4;

// hstrBeamFusedState (runBeamDirtyFused): claim counters of the fused dirty chain, then the build's decisions it reads. One per
// 128-byte line (kFusedStride uints apart): all in one line, every idle wave's poll queued behind every working wave's atomic at
// one L2 slice - fused1, 4K: the kernel took 10-49 ms against ~1.4 for the passes it replaces.
static const uint kFusedStride = 32;
static const uint kFusedRayNext = 0 * kFusedStride;        ///< Query rays claimed.
static const uint kFusedBlocksDone = 1 * kFusedStride;     ///< Blocks whose rays all finished, published in order into the block queue.
static const uint kFusedBlocksReserved = 2 * kFusedStride; ///< Block queue slots reserved.
static const uint kFusedItemNext = 3 * kFusedStride;       ///< Tile items (a finished block's padded tiles) claimed.
static const uint kFusedItemsDone = 4 * kFusedStride;      ///< Tile items finished.
static const uint kFusedUnitsPublished = 5 * kFusedStride; ///< Listed units written and published, in order.
static const uint kFusedUnitNext = 6 * kFusedStride;       ///< Units claimed by the fused kernel; the units pass marches the rest.
static const uint kFusedUnitHandBack = 7 * kFusedStride;   ///< The lowest unit claimed past what was ever listed (~0: none).
static const uint kFusedWarp = 8 * kFusedStride;           ///< The warp field is built this build.
static const uint kFusedTolerance = 8 * kFusedStride + 1;  ///< beamPolicy's tile tolerance (asuint), copied out of hstrBeamWarpArgs[11].
static const uint kFusedTilesTested = 9 * kFusedStride;    ///< Stats: tiles the fused chain tested.
static const uint kFusedUnitsMarched = 9 * kFusedStride + 1; ///< Stats: units it marched itself.
// Diagnostic, after gFusedTiles' tile counters (never cleared, so it survives to the stats readback): bits of the waits that hit
// kFusedWaitLimit (1 block publish, 2 unit publish, 4 item wait, 8 idle loop), then the last give-up's counter value and what it
// waited for (idle loop: blocks published, blocks reserved, blocks, rays claimed, rays, items done, items, units published).
// 9, 10: block and unit publishes in progress (their counts at the first give-up in 15, 16); 11: publishes an abort broke out of,
// 12-14 the first one's counter value, awaited value, and bit << 24 | polls.
// 17: item waits an abort broke; 18-22 the first one's claim start, count << 24 | polls, lanes still waiting, the first waiting
// lane's queue position, and what its slot held then.
static const uint kFusedDiagCount = 23;
static const uint kFusedAbort = 9 * kFusedStride + 2;      ///< Diagnostic: a wait gave up; every wave leaves.
static const uint kFusedWaitLimit = 20000;                 ///< Polls before a wait gives up instead of hanging the GPU.
static const uint kFusedStateCount = 10 * kFusedStride;

static const uint kWorldCacheView = 8;   ///< debugView of the world-space radiance cache experiment.
static const uint kWorldCacheSlots = 25;   ///< Scalar slots per cache cell: unit sun SH 0-8, unit sky SH 9-17, sun extension 18-24.
static const uint kWorldCacheTextures = 7; ///< RGBA textures holding the cache slots (slot s in texture s / 4, channel s % 4).
static const uint kBeamView = 9;           ///< debugView of the world-cache frame reconstructed from hierarchical beam tiles.
static const uint kBeamMaxLevels = 4;      ///< Beam tile levels, each halving the tile edge.
/// Beam count entries: one per level and the march list, then (beamRefreshDebug 7) the steps marched and carried by root queries.
// The first kBeamMaxLevels + 3 entries retain the legacy level/refresh counters. The
// remaining entries are sparse-query instrumentation (see kBeamSparse* below).
static const uint kBeamCountSlots = 248;
/// Cell views (cellViews), per build: rays composed, content cells answered from a view, empty cells answered from a view, cells
/// marched exactly (no usable view), cells visited, views requested, views built, march steps of the exact chords.
static const uint kCellViewRays = 240;
static const uint kCellViewHits = 241;
static const uint kCellViewEmptyHits = 242;
static const uint kCellViewExact = 243;
static const uint kCellViewCells = 244;
static const uint kCellViewRequests = 245;
static const uint kCellViewBuilt = 246;
static const uint kCellViewSteps = 247;
static const uint kBeamDirtyBlocks = 28;     ///< Blocks the translation guard could not certify, and so the work this build generated.
static const uint kBeamDirtyUnverified = 29; ///< Of those, the ones it knows nothing about rather than the ones whose bound failed.
static const uint kBeamDirtyOwnMarched = 30;   ///< Units of listed blocks the dirty march marched.
static const uint kBeamDirtyApronMarched = 31; ///< Apron units (around listed blocks) it marched.
/// beamRepairProbe, four each: re-marched samples scored, kept in place within 0.02, reprojected within 0.02, either.
static const uint kBeamProbeUnits = 32;
static const uint kBeamProbeRays = 36;
/// beamRepairProbe: the kBeamProbeUnits and kBeamProbeRays counts again, weighted by march steps (units at +0, rays at +4).
static const uint kBeamProbeSteps = 40;
/// beamRepairProbe, per build: units' march steps taken and paid (wave longest x lanes), then the same for the dirty query's rays.
static const uint kBeamProbeDivergence = 48;
/// beamRepairProbe, per dirty query build, over whole dirty blocks rather than rays (a block's rays are consecutive lanes):
/// blocks scored, blocks whose every re-marched ray was within 0.02 kept in place, the march steps of those blocks, all steps of
/// scored blocks; then for a tight and a loose current-view geometry signature (every ray's opacity and opacity-weighted
/// distance unchanged within a bound), the steps of the blocks it would accept and of those it would accept wrongly.
static const uint kBeamProbeBlocks = 52;
/// beamRepairProbe witness matrix (beamProbeWitnesses): per signature (kBeamProbeSignatures: opacity; + centroid; + d25/d50/d75;
/// + sun at d50; + sun at all three and cache at d50; the third and fifth again at 0.05 / 5%; the witness's own full march within
/// 0.02 of its old radiance, within 0.01, and within 0.02 with the fifth) and policy (kBeamProbePolicies: one centre, two, four, all
/// eight, nearest old value, nearest two), query steps accepted and accepted wrongly; then each policy's witness steps; then, for
/// the signatures in kBeamProbeAcceptSignatures, the units' steps in accepted blocks and of those the ones within 0.02 kept in
/// place, and kept or reprojected.
static const uint kBeamProbeSignatures = 10;
static const uint kBeamProbePolicies = 6;
static const uint kBeamProbeMatrix = 60;
static const uint kBeamProbeWitnessSteps = 180; // kBeamProbeMatrix + 2 * signatures * policies
static const uint kBeamProbeUnitAccept = 186;   // kBeamProbeWitnessSteps + policies; 3 signatures x 6 policies x 3
static const uint kBeamProbeAcceptSignatures[3] = { 4, 7, 9 };
static const uint kBeamSparseRoots = 7;
static const uint kBeamSparseEmpty = 8;
static const uint kBeamSparseCandidates = 9;
static const uint kBeamSparseUnique = 10;
static const uint kBeamSparseFinal = 11;
static const uint kBeamSparseStarts = 12; ///< Four entries, one per starting level.
static const uint kBeamSparseAccepted = 16; ///< Four entries, accepted tiles per level.
static const uint kBeamSparseLevelUnique = 20; ///< Four entries, newly marched queries per level.
static const uint kBeamSparseCutResidualQueries = 24; ///< Cut-evaluated queries that still fine-marched at least one step.
static const uint kBeamSparseCutResidualSteps = 25;   ///< Fine-march steps remaining across cut-evaluated queries.
static const uint kBeamSparseCutPaged = 26;    ///< Cut cells a basis ray hit and took from their page.
static const uint kBeamSparseCutMarched = 27;  ///< Cut cells a basis ray hit whose page could not carry its optical depth.
static const uint kBeamQueueBuckets = 4;   ///< Lists of queued root queries by predicted cost (beamQueue).
static const uint kCompareBins = 64;       ///< Exact comparison histogram entries per row: log error bins, then the maximum.
static const float kCompareBinFloor = 1e-4f; ///< Upper edge of the first log error bin; 4 bins per octave above.

struct HSTRCutNode
{
    uint4 minLeft;
    uint4 maxRight;
    float4 statistics;
    float4 transmittancePage;
    float4 transmittanceCorners0;
    float4 transmittanceCorners1;
};

// Virtual cloud density. Every VDB of the cloud library is a sparse mip pyramid of 8^3 bricks (level L voxels cover 2^L source
// voxels) stored with a one-voxel apron, so a resident brick filters trilinearly in the atlas without its neighbours.
static const uint kCloudBrickCore = 8;
static const uint kCloudBrickEdge = 10;        ///< Stored edge: core plus apron.
static const uint kCloudChunkLevel = 4;        ///< Directory cells are level-4 bricks (128 source voxels).
static const uint kCloudRefNone = 0xFFFFFFFF;  ///< Directory/node entry: nothing mapped (the domain proxy applies).
static const uint kCloudRefNode = 0x80000000;  ///< Directory/node entry flag: index of a 4^3 node in the node pool.
/// Resolved level page word (hstrCloudLevelPages): kCloudRefNone (the proxy), a brick (index in bits 0-23, its level in bits
/// 24-27), or kCloudLevelEmpty with the mapped brick's level in bits 0-3 (the asked-for level lies in a missing child octant).
static const uint kCloudLevelEmpty = 0x40000000;
static const uint kCloudBrickConstant = 0x1000; ///< HSTRCloudBrick.info flag: the brick holds one value (minValue).
static const uint kCloudTransformHaar = 1;     ///< HSTRCloudStaging.transform: CloudFormat.h kTransformHaar.
static const float kCloudSeaMinScale = 0.65f;  ///< Smallest instance scale of the sea (CloudSea::makeTile draws 0.65-1).
static const float kCloudSunDepthOpaque = 12.f; ///< Sun optical depth past which nothing is lit (sunDepthAt stops).

/// One tile of the cloud sea domain: a library asset placed with a quarter-turn rotation, mirror, scale and offset.
struct HSTRCloudInstance
{
    float4 row0; ///< Domain voxel position to asset source voxel position (affine rows).
    float4 row1;
    float4 row2;
    uint asset;
    float sourceVoxelWorld; ///< World size of one source voxel of this instance.
    float proxyLevel;       ///< Pyramid level whose voxels are as coarse as the domain proxy: from here on the proxy is used.
    /// The instance's scale (sourceVoxelWorld over its asset's), which the lit sample's sun reach needs without the asset record;
    /// 0 for a tile with no cloud (what an occupied flag held here before, so the record stays 80 bytes, 16-byte aligned).
    float scale;
    uint sunClass; ///< Orientation class (the signed permutation of the rows): instances of one class share baked sun depth.
    /// The asset's resolved level pages (hstrCloudLevelPages), copied here so a camera sample never loads the asset record.
    uint levelPageBase;
    uint pageDims; ///< Level-0 pages per axis, 10 bits each (x | y << 10 | z << 20).
    uint topLevel;
};

struct HSTRCloudAsset
{
    uint3 dims; ///< Source voxels, multiples of 128.
    uint directoryOffset;
    uint3 chunkDims;
    uint topLevel;
    float sourceVoxelWorld; ///< World size of one source voxel at instance scale 1.
    uint pageOffset; ///< First level-0 virtual page in the persistent dense page table.
    /// First entry of the asset's resolved level pages: for every level L of the pyramid, one word per level-L brick naming what a
    /// sample asking for level L reads there (cloudLevelPage). Levels follow each other, each (pages + 2^L - 1) >> L per axis.
    uint levelPageBase;
    uint padAsset2;
};

/// One camera-hot virtual page. The parent and fade remain in HSTRCloudBrick, so transition state is not duplicated per page.
struct HSTRCloudVirtualPage
{
    uint physicalRef;
    uint mappedLevel;
};

/// A virtual region invalidated by one canonical CPU hierarchy edit. workEnd is the inclusive prefix end in expanded page work.
struct HSTRCloudDirtyPageRegion
{
    uint3 origin;
    uint asset;
    uint3 extent;
    uint workEnd;
};

/// A resident brick: an atlas slot (or a constant), its quantisation and its place in the pyramid.
struct HSTRCloudBrick
{
    uint slot;
    uint info;   ///< Level (bits 0-3), child mask (bits 4-11), kCloudBrickConstant.
    uint parent; ///< Brick index of the parent, kCloudRefNone at the top.
    float fade;  ///< Blend weight against the parent while the brick streams in or out.
    float minValue;
    float range; ///< density = minValue + range * t^2 for the stored unorm t.
    /// A running fade (kCloudFadeIn or kCloudFadeOut set, the frame it started below them) is advanced on the GPU
    /// (advanceCloudFades): fade = saturate(fadeBase +- (frame - start) / fadeFrames). The CPU writes a brick only when its fade
    /// starts, turns or ends.
    uint fadeStart;
    float fadeBase;
};
static const uint kCloudFadeIn = 0x80000000;
static const uint kCloudFadeOut = 0x40000000;
static const uint kCloudFadeFrameMask = 0x3fffffff;
/// What advanceCloudFades evaluates fades with.
struct HSTRCloudFadeFrame
{
    uint frame;      ///< CloudResidency's frame counter (kCloudFadeFrameMask bits).
    uint brickCount;
    float step;      ///< 1 / fadeFrames.
    uint padFade;
};

/// hstrCloudOccupancy words per brick: a 4-bit density bound per 2-voxel cell (cloudCellCode) in words 0-7, then the bound of each of
/// the brick's eight 4-voxel octants in word 8 (the brick's own bound is the maximum of those eight).
static const uint kCloudCellWords = 9;
/// Levels a sample of an unbaked brick climbs for a baked ancestor before the live march: to the root. The sun slot pool holds one
/// bake per density slot, and the sea's bricks need about three (orientation classes), so it settles full with ~126k bakes that never
/// run; climbing two levels, 1-7% of the samples a 4K walk re-marched still took the live march at every step, 4.5-4.9 ms of the
/// frame. To the root the live passes cost 0.005 ms at identical error (0.136% over 0.02, p99.9 0.0256).
static const uint kCloudSunAncestors = 15;
static const uint kCloudSunBakesPerBrick = 8; ///< Orientation classes one brick keeps baked sun depth for (hstrCloudSunSlots).

/// One brick's sun bake (bakeCloudSun): per atlas texel, the optical depth over reach source voxels towards the sun.
struct HSTRCloudSunBake
{
    uint brick;
    uint slot;        ///< Sun atlas slot.
    uint asset;
    uint level;
    float3 origin;    ///< Level-voxel position of the brick's first core voxel.
    float reach;      ///< Source voxels marched towards the sun.
    float3 direction; ///< Unit sun direction in the asset's source voxels.
    float cap;        ///< Source-length optical depth past which the march stops (opaque for every instance scale).
};

/// GPU sun bake scheduling (CloudResidencyDesc::gpuSun): what the scheduler needs of a GPU brick, written by the CPU. Where the brick
/// is (HSTRCloudSunSched) changes only when it maps; its state (hstrCloudSunState: the flags below, the priority bin above
/// kCloudSunBinShift) changes with every cut, so it is its own 4-byte table.
static const uint kCloudSunInCut = 0x10000;  ///< State flag: the last cut desired the brick.
static const uint kCloudSunMapped = 0x20000; ///< State flag: the brick is mapped.
static const uint kCloudSunBins = 256;       ///< Priority histogram bins of the scheduler (candidates, then holders).
static const uint kCloudSunBinShift = 24;    ///< State: the last cut's priority bin (floor(priority * 16), clamped to the bins).
/// hstrCloudSunCounters entries (cleared every frame): candidates emitted, holders emitted, bakes staged, bricks baked, bricks
/// waiting, stale bakes invalidated, then the partial-bin reservations of candidates and holders.
static const uint kCloudSunCounters = 8;
struct HSTRCloudSunSched
{
    uint coord;     ///< BrickHeader::coord: level-local brick coordinates, 10 bits per axis.
    uint asset;
};

/// Durable sparse beam interfaces. Tiles consume stable virtual query IDs backed
/// directly by the result buffer; the legacy dense lattice is not part of this path.
struct BeamResult
{
    uint depthCache01; ///< Half-packed cache.rg.
    uint depthCache23; ///< Half-packed cache.b and optical depth.
    uint single01;     ///< Half-packed single.rg.
    uint single23;     ///< Half-packed single.b and opacity-weighted distance.
};

struct BeamTile
{
    uint2 origin;
    uint level;
    uint q00;
    uint q10;
    uint q01;
    uint q11;
    uint qc;
};

/// Sun bake invalidation (CloudResidencyDesc::gpuSun). A bake of a level-L brick read the change cells (32^3 source voxels) over its
/// texels, their trilinear support and the sun sweep. The invalidation field holds, per brick position and orientation class, the
/// last frame a change landed in the region a bake there read: a map or unmap is a change over the cells of the brick's region, and
/// stampSunChanges writes the frame into every entry whose bakes that region reaches, so a bake is stale when its one entry is newer.
/// Every entry is one brick position, so the test is exact (it equals the per-cell test over the bake's region).
/// - Levels >= 2: a dense block per (asset, class) pair in use, one entry per brick position of each level.
/// - Levels 0 and 1: per directory node, kCloudSunFineEntries entries (8 level-1 and 64 level-0 positions per class). A node holds
///   the 8-voxel cells of one 32-voxel cell and exists exactly while fine bricks are mapped there, so fine entries cost memory only
///   where fine bricks are resident; a node is reset when it is created.
/// Entries are 16-bit frame epochs in an R16 texture (kCloudSunFieldWidth texels a row): a bake is stale when its entry's age is
/// below its own. Bakes kCloudSunAgeCap frames old are rebaked, and ageSunField clamps older entries to that age before they wrap.
static const uint kCloudSunClasses = 8;         ///< Orientation classes at most (the sea's quarter turns and mirror).
static const uint kCloudSunFieldLevels = 16;    ///< Levels per asset in hstrCloudSunFieldLevels.
static const uint kCloudSunFineEntries = 72 * kCloudSunClasses;
static const uint kCloudSunFieldWidth = 16384;
static const uint kCloudSunAgeCap = 32768;

/// What one frame's sun scheduling runs with (hstrCloudSunFrameInfo, one element). Kept out of HSTRCloudParams: a params change
/// rebuilds every binding of a pass, and the scheduler's many small dispatches then cost more CPU than GPU.
struct HSTRCloudSunFrame
{
    uint frame;        ///< CloudResidency's frame counter (bake and change frames are in it).
    uint releaseCount; ///< Bricks in hstrCloudSunRelease whose bakes are freed.
    uint classCount;   ///< Orientation classes in hstrCloudSunDirections.
    uint bakeMax;      ///< Bakes staged at most (the candidate list's capacity).
    uint capacity;     ///< GPU bricks (the scan range).
    uint changeCount;  ///< Residency changes in hstrCloudSunChanges to stamp.
    uint fineBase;     ///< First field entry of the per-node part.
    uint generation;   ///< HSTRCloudParams::cloudSunGeneration.
    uint ageRow;       ///< ageSunField's first field row.
    uint ageRows;
    uint blockResets;  ///< hstrCloudSunResets: pair blocks (blockLength entries each), then nodes (kCloudSunFineEntries each).
    uint nodeResets;
    uint blockLength;
    float reachScale;  ///< World length of the sun reach (sunNearVoxels sea voxels); over a source voxel's world size, its voxels.
    uint padSunFrame0;
    uint padSunFrame1;
};

/// The cloud sea's domain proxy is built on the GPU from the tiles' float16 volumes (HSTRCloud::uploadDomainExtinction): per changed
/// slot (hstrDomainRegions), its packed volume scattered from hstrDomainStaged with its extinction, the 4^3-voxel block maxima
/// around it, the dilated majorant and the 16-voxel occupancy. The world cache clear of changed tiles reads it too.
struct HSTRCloudDomainFrame
{
    uint regionCount;  ///< Changed slots in hstrDomainRegions.
    float scale;       ///< Density scale (the volumes are unscaled).
    uint tileVoxels;   ///< Domain voxels per tile edge.
    uint tiles;        ///< Tiles per window axis.
    uint3 dims;        ///< Domain voxels.
    uint cacheCellVoxels; ///< Domain voxels per world cache cell edge.
    uint3 blockDims;   ///< Majorant blocks (dims / 4).
    uint padDomain1;
    uint3 cacheDims;   ///< World cache cells.
    uint padDomain2;
};
static const uint kDomainBlock = 4;       ///< Voxels per majorant block edge.
static const uint kDomainOccupancy = 4;   ///< Majorant blocks per occupancy cell edge.
static const uint kDomainBatch = 8;       ///< Slots per scatter (hstrDomainStaged holds this many volumes).
static const uint kDomainRegionZero = 0x80000000; ///< Region flag: the slot is empty (its volume clears).
static const uint kDomainRegionKeep = 0x40000000; ///< Region flag: the GPU holds the slot's volume (only what derives from it rebuilds).
static const uint kDomainRegionSlot = 0x3fffffff;

/// One residency change: an asset's change cells [first, last] (10 bits per axis).
struct HSTRCloudSunChange
{
    uint asset;
    uint first;
    uint last;
};

/// A brick staged for GPU reconstruction (see reconstructBrick in CloudFormat.h): parent prediction plus dequantised residual,
/// requantised into its atlas slot.
struct HSTRCloudStaging
{
    uint slot;
    uint parentSlot;  ///< kCloudRefNone: the prediction is zero.
    uint parity;      ///< Brick coordinate & 1 per axis (bits 0-2).
    uint coded;       ///< 0: predicted only (no residual).
    float step;       ///< Coefficient = quantised value * step.
    uint payloadWord; ///< First word of the packed coefficients in hstrCloudPayload.
    float parentMin;
    float parentRange;
    float valueMin;
    float valueRange;
    uint transform; ///< CloudFormat.h kTransform*.
    uint brick;     ///< GPU brick index (its occupancy in hstrCloudOccupancy).
};

/// One cached cell view (cellViews): the transfer of a domain cell [cell - 0.5, cell + 0.5)^3 (voxel coordinates, voxel i centred
/// at i) as seen from camera, on a cellViewEdge^2 texel grid over the cell's projection. See buildCellViewTexel.
struct CellView
{
    float3 camera; ///< Build camera, voxel coordinates.
    float bandLow; ///< Camera distances (voxels, to the cell centre) over which the cell's density level and fade still hold.
    int3 cell;     ///< The cell, unwrapped (the map is indexed by its wrap into the domain).
    float bandHigh;
    uint flags;    ///< kCellViewEmpty.
    uint frame;    ///< cellViewFrame it was built in, plus one (0: never built).
    float fade;    ///< The sea's distance fade at the build distance.
    uint pad;
};
static const uint kCellViewEmpty = 1; ///< Nothing in the cell reaches the camera: no transmittance lost, no light.

struct HSTRCloudParams
{
    uint2 frameDim = { 0, 0 };
    uint baseSteps = 32;
    uint refinementLevel = 1;

    uint3 hstrLeafDims = { 0, 0, 0 };
    uint hstrCellWidth = 1;

    uint3 hstrExtinctionDims = { 0, 0, 0 };
    uint hstrRootNode = 0;

    uint2 hstrTileDims = { 0, 0 };
    uint hstrNodeCount = 0;
    uint hstrMaxTileNodes = 128;

    uint2 hstrLatticeDims = { 0, 0 }; ///< Tile-corner camera queries, (tileDims + 1).
    uint hstrTileSize = 8;
    uint maxMarchSteps = 4096;

    int3 hstrGridMin = { 0, 0, 0 };
    uint frameIndex = 0;

    uint activeRank = 24;
    float activeThreshold = 0.f;
    uint padHSTRGoal = 0;
    uint traceSpatialOrder = 2;

    uint hstrFaceDofs = 4;
    uint hstrTraceDofs = 24;
    uint hstrStorageRank = 24;
    uint correctionBudget = 262144;

    uint correctionAtomCount = 0;
    uint correctionFadeFrames = 8;
    uint adjointGoalCount = 0;
    uint compareTarget = 0xF; ///< Reference components the comparison measures against (kComponent* bits).

    uint ballisticBudget = 24;
    uint nearScatterBudget = 120;
    uint diffuseRankBudget = 24;
    uint maximumDiffuseRank = 0;

    uint3 schurWindowCenter = { 0, 0, 0 };
    uint schurWindowRadius = 8;

    uint operatorDictionarySize = 0;
    uint hstComponents = 0xF;     ///< Components the HST frame includes (kComponent* bits).
    uint referenceShow = 0xF;     ///< Reference components the reference view displays.
    uint compareSubstitute = 0;   ///< Reference components added to the HST frame before comparing (oracle substitution).

    float residualBlend = 1.f;
    float densityScale = 1.f;
    float skyScale = 1.f;   ///< Scale of the HST camera source (the scattered sky when hstSunFraction = 0), for calibration tests.
    float skyAmbient = 0.f; ///< When positive, replaces the HST camera source by this multiple of an opacity-weighted ambient sky.

    float3 sunDirection = float3(0.4319f, 0.8639f, 0.2699f);
    float anisotropy = 0.85f;

    float3 sunRadiance = float3(12.f, 10.5f, 8.f);
    float pad0 = 0.f;

    float3 skyRadiance = float3(0.25f, 0.4f, 0.8f);
    float pad1 = 0.f;

    float3 localSourceRadiance = float3(0.f);
    float pad2 = 0.f;

    float mixedFidelityThreshold = 0.15f;
    float operatorDictionaryTolerance = 1e-4f;
    float cutTransmittanceTolerance = 0.01f;
    float cutHysteresis = 0.25f; ///< Relative split/merge band for view-dependent cut acceptance.

    // World-space residual pages (2-voxel cells) and extinction majorants (4-voxel blocks).
    uint3 hstrFineDims = { 0, 0, 0 };
    uint referenceSamples = 0; ///< Path-traced frames already accumulated into the reference average.

    uint3 hstrMajorantDims = { 0, 0, 0 };
    uint debugView = 0;

    uint3 hstrCameraLightingDims = { 0, 0, 0 };
    uint cutParity = 0;

    float basisTolerance = 0.02f;  ///< Relative hierarchical surplus above which a tile descends to per-pixel integration.
    float basisHysteresis = 0.5f;  ///< A refined tile merges back once its surplus falls below this fraction of the tolerance.
    float residualTolerance = 0.02f; ///< Pages whose largest opacity-weighted sun deposit stays below this are not resident.
    float residualStrength = 1.4f;

    float stepOpticalDepth = 0.5f; ///< Target optical depth per majorant-bounded march step.
    /// Midpoint quadrature of trilinear extinction is accurate at one-voxel steps at any density. MEASURED (4K, per-pixel c15, 2026-09-16):
    /// 97% of the camera's density samples step at this minimum, so it, not stepOpticalDepth, sets the sample count - and the sample
    /// count is what the march costs. 1 -> 2 voxels: 29.2 -> 16.8 samples per pixel, near 39.0 -> 24.2 ms and sea 29.9 -> 18.8 ms, at
    /// 0.26% of pixels over 0.02 log error (p99.9 0.036, max 0.14). 1 -> 4 voxels: 10.8 samples, near 15.2 ms and sea 11.4 ms, but
    /// 2.2% over 0.02 (p99.9 0.12, max 0.34): too coarse for midpoint quadrature.
    /// MEASURED again on the beam view, which is the path that ships (4K, sea library, one thread per ray, temporal off, baked sun,
    /// against the per-pixel 1-voxel march): 2 voxels gives sea 8.63 ms at p99.9 0.022 and near 10.76 ms at p99.9 0.036; 4 voxels
    /// gives sea 5.39 ms at p99.9 0.061 and near 6.55 ms at p99.9 0.122. The near view degrades about twice as hard, which is what
    /// the aliasing reading predicts: close up the pixel footprint is fine, so a 4-voxel step is furthest above the rate the density
    /// needs, while in the distance the footprint already exceeds the step and nothing changes.
    /// MEASURED against the 4K path tracer (sea farside, 65 spp, 8x8 block log error, 2026-09-17): min 2 / max 2 0.1265, min 2 / max 1
    /// 0.0998, 1 / 1 0.0996 (the exact march 0.0984). A step is sized before its sample reads the brick, so with max 2 the first step
    /// from coarse density into a fine brick crossed ~5 of its voxels; the per-pixel 1-voxel comparison showed it as 51% of farside's
    /// pixels over 0.02. Capping at one voxel costs 3-8% (the 1-voxel march 60-78%). CloudSea.py ships min 2, max 1.
    float minStepVoxels = 1.f;
    float maxStepVoxels = 4.f;
    float compareMapScale = 0.f;   ///< >0: block comparisons overwrite the frame with their signed log error map, saturating here.

    // Sun transport carried by the voxel-resolution sun optical-depth pages: octave n scatters with energy a^n, extinction
    // b^n and phase anisotropy c^n g. Octave 0 is exact single scattering; the parameters are fitted to the path tracer.
    float octaveEnergy = 0.8f;     // Fitted against the path-traced reference (static, backlit, oblique views).
    float octaveExtinction = 0.4f;
    float octavePhase = 0.3f;
    float hstSunFraction = 0.f; ///< Share of the sun also injected into the HST boundary solve (0: HST carries the sky).

    float3 sunVoxelDirection = float3(0.f); ///< One voxel step towards the sun, in voxel coordinates.
    float sunVoxelWorldLength = 0.f;        ///< World length of that step.

    uint3 hstrOctaveDims = { 0, 0, 0 }; ///< 2-voxel grid of scatterer-weighted sun arrival for octaves 1-3.
    uint octaveBlurAxis = 0;
    float octaveBlurSigma = 1.f; ///< Lateral spread of multiply scattered sun light, in 2-voxel cells.
    float padHSTR10 = 0.f;
    float padHSTR11 = 0.f;
    float padHSTR12 = 0.f;

    // World-space radiance cache experiment (debugView kWorldCacheView).
    uint3 worldCacheDims = { 0, 0, 0 };
    uint worldCacheCellVoxels = 2; ///< Voxels per cache cell edge.

    uint worldCacheSamples = 0;   ///< Estimator passes (gather passes or photon batches) accumulated into the cache.
    uint worldCacheOrder = 2;     ///< Highest SH band the camera evaluates (0-2; light tracing fills bands 0-1 only).
    uint worldCacheEstimator = 1; ///< 0: one gather path per cell per pass; 1: light tracing with track-length deposition.
    uint worldCachePhotons = 262144; ///< Photons per light-tracing batch.

    float3 worldVoxelSize = float3(0.f);   ///< World size of one voxel (the grid axes are world aligned).
    float worldCacheDepositScale = 4096.f; ///< Fixed-point scale of the light-tracing deposition atomics.

    uint worldCacheSegments = 0; ///< 0: every batch traces whole photon paths; >0: a persistent photon pool advances this many
                                 ///< flight segments per photon per update, respawning photons as they die.
    /// Pixels per edge of the coarsest beam tiles. This sets a floor on the frame that no content can lower: the level-0 lattice is
    /// one corner and one centre per tile whatever the image holds. CloudSea.py runs 4 pixels at one level, which at 4K is about
    /// 1.04M rays - 12.5% of the frame - before a single pixel is refined. MEASURED (4K, sea library, minStepVoxels 2, one thread
    /// per ray, temporal off): that lattice costs 4.4 ms of the sea's 8.63 ms frame and 5.3 ms of the near view's 10.76 ms, against
    /// a per-pixel march of 27.17 and 38.32 ms. Per ray the queries are ordinary - 4.2 ns against 3.3 ns for an average camera pixel
    /// - while the 4-5% of pixels the tiles fail on cost 11.4 ns each, being the longest rays in the frame. So the frame is about
    /// 1.37M rays, and it is the lattice, not the refinement, that a coarser tiling with more levels has to trade against.
    ///
    /// MEASURED trading it, and the answer depends on the content, so there is no tiling to prefer here. Coarser tiles with levels
    /// (minStepVoxels 2, temporal off, against the per-pixel 1-voxel march) on the near view, one smooth cloud filling the screen:
    /// 4x1 10.92 ms at p99.9 0.036, 8x2 8.17 ms at p99.9 0.036, 16x3 9.32 ms at p99.9 0.051 - a quarter off for nothing at 8x2.
    /// On the sea, small distant clouds everywhere: 4x1 8.83 ms at p99.9 0.022, 8x2 10.23 ms at p99.9 0.036, 16x3 11.08 ms at
    /// p99.9 0.051 - the flat lattice wins on both axes. The reason is in the query lanes: on the sea an 8-pixel tile almost never
    /// accepts, so level 0 costs 2.9 ms and level 1 another 2.5 against 4.5 ms for the one flat lattice, and the hierarchy pays for
    /// both. A level only earns its lattice where tiles that size accept, and in a sea of high-frequency cloud they do not.
    /// CloudSea.py therefore keeps 4 pixels at one level.
    ///
    /// MEASURED again on the sea against the continuous metrics, one settle, drift 9.81 -> 9.89 ms: 4x1 9.81 ms at >0.02 0.111%,
    /// 8x1 7.96 at 0.287%, 8x2 8.51 at 0.306%, 16x2 8.24 at 0.801%, 16x3 8.41 at 0.815%, and 8x1 with the tolerance tightened to
    /// 0.02 8.78 at 0.183%. Quartering the lattice takes queries from 4.9 to 2.0 ms and hands 1.1 ms back to resolve as the marched
    /// share goes 3.9% -> 5.7%, so the net is 1.85 ms for 2.6x the error, and buying the quality back with the tolerance leaves 1 ms
    /// against a still worse image. Queries and resolve are 4.9 / 4.9: work moved out of one lands in the other.
    ///
    /// That is a result about UNIFORM tilings only. Every configuration above applies one tile size to the whole frame, and the near
    /// view's preference for 8x2 against the sea's for 4x1 is itself evidence that the best size is a property of the content under
    /// the tile, not of the frame. Nothing here tests admitting tiles at different sizes in different regions, and this must not be
    /// read as closing that.
    uint beamTileSize = 16;
    uint beamLevels = 3;          ///< Beam tile levels; tiles that fail at the finest level are marched per pixel.
    /// log(1 + radiance) surplus at a tile centre above which the tile is refined. NOT the knob that controls the refinement rate -
    /// beamTileRefines ORs this with the silhouette test and the guide, and on the sea the silhouette test fires first almost
    /// everywhere. Sweeping this 0.05 -> 0.5 leaves the marched share at 3.9% and the frame within 1%; setting it to 1e9 does not
    /// stop refinement at all. Go to beamEdgeContrast instead.
    float beamTolerance = 0.02f;

    uint worldCacheBands = 2; ///< Highest SH band the light tracer deposits (1: 4 coefficients per field, 2: 9).
    uint worldCacheWindow = 1; ///< 1: Hanning-window the evaluated SH bands against truncation ringing.
    uint worldCacheTextured = 1; ///< 1: the camera reads the cache from baked, hardware-filtered 3D textures.
    /// The camera march re-evaluates the cache every lightingStride steps and holds it between.
    /// MEASURED on the dirty passes and LEFT AT 1 (4K, OCT_T001, parallax 1, one settle, dirty_lighting_rate.py): walk 7.48 ms
    /// (anchor 7.44) -> stride 2 7.41 at 0.136% -> 0.154% over 0.02, stride 4 7.46 at 1.097%; sprint 9.20 -> 9.17 / 9.13, error
    /// unchanged. The cache read is not where the lighting half of a step goes. Holding the sun depth instead (cloudSunReuse 2 / 4
    /// voxels, shipping defines kept) is worth 0.75-0.9 ms walk and 0.5-0.7 sprint but puts 4.0-4.8% / 6.7-7.1% of pixels over
    /// 0.02 (p99.9 0.026 -> 0.072, sprint 0.015 -> 0.15): the sun depth is not smooth at one-voxel steps through cloud edges.
    uint lightingStride = 1;

    uint2 beamTileDims = { 0, 0 };    ///< Coarsest beam tiles.
    uint2 beamLatticeDims = { 0, 0 }; ///< Query lattice, one texel per beamLatticeStep pixels; every tile corner and centre is a texel.

    /// Transmittance range of the queries around a tile above which it is refined (silhouettes); 0: off. This, and NOT beamTolerance,
    /// is what decides how much of the frame is marched per pixel. beamTileRefines takes three independent tests - the surplus
    /// against beamTolerance, this silhouette test, and the guide - and on the sea the silhouette one fires first almost everywhere,
    /// so beamTolerance is inert: sweeping it 0.05 -> 0.5 leaves the marched share at 3.9% and moves the frame 1%. Anything trying to
    /// buy back quality or time through the refinement rate has to come here.
    ///
    /// MEASURED (sea, 4K, beam view, drift-anchored, first test repeated at the end), from 0.05:
    ///     0.05   9.43 ms   3.9% marched   0.110% over 0.02   p99.9 2.15e-02
    ///     0.1    8.99      3.4%           0.111%             2.15e-02
    ///     0.2    8.42      2.7%           0.112%             2.15e-02
    ///     0.5    7.46      1.7%           0.118%             2.56e-02
    ///     1.0    5.97      0.4%           0.185%             3.04e-02   (saturates; 2.0 is the same image)
    /// The sea is small distant clouds everywhere, so transmittance range is everywhere too, and 0.05 was marching ten times as many
    /// pixels per pixel as the image needed. DEFAULTED TO 0.2: a tenth off the frame with p99.9 unchanged and the share over 0.02
    /// moving 0.110 to 0.112, which is noise. 0.5 is there for 21% at 0.118%.
    ///
    /// It also reopens the tiling. beamTileSize 8 was measured twice as losing at equal quality, but both times against a silhouette
    /// test at 0.05 that was forcing refinement everywhere, so coarsening the tiles only moved work into a pass that was already
    /// over-firing. With the gate set sanely, 8 pixels at edge 1.0 gives 3.40 ms against 6.45 - the earlier conclusion was an
    /// artefact of a badly tuned neighbour, not a property of the hierarchy.
    float beamEdgeContrast = 0.2f;
    uint compareExact = 0;     ///< 1: the comparison measures against the stored exact frame instead of the reference.
    /// Threads that march consecutive segments of one query ray in parallel (divides 64). Every segment starts from unit
    /// transmittance, so splitting a ray defeats the march's early termination: segment 0 stops inside the first dense cloud it
    /// meets and segments 1..n march the rest of the ray in full, at full density, only for the composition to multiply what they
    /// return by a transmittance near zero. MEASURED (4K, sea library, beam view, minStepVoxels 2, baked sun, temporal off) going
    /// 4 -> 1: near 18.04 -> 10.99 ms and sea 13.68 -> 8.69 ms, with the query pass itself 11.0 -> 5.4 ms. The threads it was
    /// buying are not worth the work they duplicate; one thread per ray keeps the early termination that makes a camera pixel
    /// cheap. CloudSea.py had already overridden this to 1, so that is a correction to the default here rather than a saving on
    /// the sea path. What is left is not a small dispatch and not latency-bound: at CloudSea.py's 4-pixel tiles the level-0 lattice
    /// is about 1.04M rays (see beamTileSize), and at 4.2 ns each they run at much the same rate as ordinary camera pixels. The
    /// query pass costs what it costs because of how many rays the lattice asks for, not because of how they are scheduled.
    uint beamSegments = 1;
    uint beamPassLevel = 0;    ///< Tile level the current beam query or tile pass processes.

    uint2 beamLevelDims = { 0, 0 }; ///< Level map, one texel per finest tile.
    uint beamLatticeStep = 2;       ///< Pixels between lattice texels: half the finest tile.
    uint beamTemporal = 0;          ///< 1: query every level in one pass over the tiles refined last frame (one frame of lag).
    /// 1: a coarsest tile whose region was marched per pixel last frame refines straight to the next level, paying neither its centre
    /// query nor its own test. Its corners are still marched - they are the outer corners of its children, which query only edge
    /// midpoints and child centres - so this saves the centre half of the root lattice, not all of it. The tiling that wins is
    /// content-dependent (see beamTileSize):
    /// coarse tiles with levels pay off over one smooth cloud and lose over a sea of small ones, because there almost every coarse
    /// tile fails and the frame pays the coarse lattice before paying the fine one anyway. Choosing the entry level per region rather
    /// than per frame is what makes a coarse root safe to offer at all. The prediction is a pure function of the previous frame that
    /// the query pass and the tile pass both evaluate, so it can only cost rays, never correctness (see beamRootDifficult).
    ///
    /// MEASURED, and it does not pay (4K, 8-pixel roots over two levels, temporal off): near 9.39 -> 9.36 ms and sea 9.66 -> 10.83,
    /// the image bit-identical in both - same marched share, same error to every digit - which is the design working and the cost
    /// going the wrong way. The arithmetic is against it. Only about 4% of pixels are marched per pixel, so only about a tenth of
    /// root tiles are predicted difficult, and their centres are a twentieth of the root lattice; that is less than the four history
    /// reads the prediction costs at every centre query and every root test. The tiles it forces to refine would have failed their
    /// own test anyway, which is why the image never moves. The near view's win at 8x2 (11.12 -> 9.39) is the tiling itself, not the
    /// adaptivity. A cheaper predictor could turn the sea's loss into a wash, but the ceiling on the whole idea is that twentieth.
    uint beamAdaptiveRoot = 0;

    uint compareBlock = 1; ///< Comparisons average compareBlock x compareBlock pixel blocks first (1: per pixel).
    float worldCacheModulation = 0.f;       ///< b of the cache's exp(-b tau_sun) voxel-resolution modulation (0: off); set per frame.
    float worldCacheModulationDepth = 8.f;  ///< Sun optical depth at which the modulation stops falling.
    uint beamGuide = 0; ///< Beam tiles also tested against the full-resolution guide (beamTileGuideFails): 1 at every level, 2 above the finest.

    uint worldCacheZonalBands = 0;  ///< >0: the sun field's zonal part about the sun direction is windowed on its own and extended
                                    ///< by zonal bands 3 up to this (at most 8); 0: off.
    uint worldCacheZonalWindow = 0; ///< Window order of that zonal part (0: max(worldCacheZonalBands, 2)).
    uint worldCacheSunOrder = 2;    ///< 3: the sun field carries all of band 3 (7 extension slots) instead of zonal bands.
    float beamGuideDepth = 0.05f; ///< Guide test: log deviation of the distance at which a pixel's optical depth reaches one.

    float beamGuideSun = 0.05f; ///< Guide test: deviation of the sun attenuations exp(-tau) and exp(-b tau) at that distance.
    float sunNearVoxels = 0.f; ///< Camera samples march the sun optical depth exactly over this many voxels towards the sun, then
                               ///< read the voxel-centre field; 0: one-voxel midpoint step onto the field (enough for the
                               ///< adaptive march, which integrates the sun attenuation analytically).
    uint cloudEmptySkip = 1; ///< 1: camera marches jump over brick cells and child octants whose density is exactly zero.
    uint worldCacheSimilarity = 4; ///< >0: photons scatter exactly this many times, then continue in the similar isotropic
                                   ///< medium (extinction sigma (1 - a g), albedo a (1 - g) / (1 - a g)); 0: exact throughout.

    uint adaptiveMarch = 0; ///< 1: the cache views integrate camera segments analytically and bisect on estimated error
                            ///< (about twice the beam view's cost of the fixed march for ~0.001-0.002 less error).
    float marchTolerance = 0.01f;    ///< Accepted segment error relative to 1 + the pixel's expected radiance.
    float marchMinVoxels = 0.03f;    ///< Shortest bisected segment.
    float marchCoarseVoxels = 2.f;   ///< Shortest coarse segment in the medium (longer where the majorant allows).

    uint seaMode = 0;               ///< The cloud library domain repeats in world X/Z: a periodic sea with periodic transport.
    float seaViewDistance = 4000.f; ///< Maximum world distance marched through the cloud sea.
    uint cloudDomain = 0;           ///< 1: the medium is the cloud library domain (seaOrigin, seaVoxelSize), not the scene grid.
    uint cloudVirtual = 0;          ///< 1: camera samples read resident fine bricks where mapped.

    float3 seaOrigin = float3(0.f); ///< World position of the domain's minimum corner (voxel i is centred at seaOrigin + (i + 0.5) size).
    float cloudPixelAngle = 0.f;    ///< World footprint of one pixel per unit of distance.
    float3 seaVoxelSize = float3(1.f); ///< World size of one domain voxel.
    float cloudLodBias = 0.f;          ///< Added to the footprint level of camera samples.

    uint2 cloudTiles = { 1, 1 }; ///< Instances per domain axis (X, Z).
    uint cloudTileVoxels = 64;   ///< Domain voxels per tile edge.
    uint cloudAtlasShift = 6;    ///< log2 of the atlas bricks per X and Y axis.

    uint cloudCommitCount = 0; ///< Staged bricks the current commit dispatch reconstructs, from cloudCommitOffset.
    float cloudFineMinVoxels = 0.05f; ///< Shortest camera step inside fine bricks, in domain voxels.
    uint cloudCommitOffset = 0;
    uint cloudStagedCount = 0; ///< Staged bricks whose residuals decodeCloudResiduals decodes this frame.
    uint cloudPageRegionCount = 0; ///< Canonical hierarchy regions expanded into the unique dirty-page queue this frame.
    uint padCloudPage0 = 0;
    uint padCloudPage1 = 0;
    uint padCloudPage2 = 0;

    uint3 fineSunOffset = { 0, 0, 0 }; ///< First voxel of the sun page region computeFineSun updates.
    uint cloudSunGeneration = 0; ///< Changes with everything the baked sun depth depends on (sun, density scale, reach).
    uint3 fineSunSize = { 0, 0, 0 };   ///< Voxels of that region (0: the whole grid).
    uint cloudSunCache = 1; ///< 1: camera sun depth reads baked bricks where their key matches (else the live march).

    float3 cloudAtlasInvSize = float3(1.f); ///< Reciprocal texel dimensions of the brick atlas.
    float worldCacheKeep = 1.f;              ///< Share of the world cache deposits decayWorldCache keeps.

    float3 cloudSunAtlasInvSize = float3(1.f); ///< Reciprocal texel dimensions of the sun atlas (its own slot pool, same brick layout).
    uint marchProbe = 0; ///< kWorldCacheView: > 0 replaces the pixel with a per-pixel count of the camera march (probeMarch modes).

    uint cloudSunOldestGeneration = 0; ///< Oldest sun generation whose bakes still answer (only the sun direction changed since).
    float cloudThinDepth = 0.f; ///< Camera marches treat brick cells whose density bound gives at most this optical depth across as empty.
    /// 1: camera steps with a zero majorant cross the whole zero 16-voxel block (hstrMajorantZero). MEASURED (4K c15): sea 32.0 ->
    /// 29.1 ms, near 38.2 -> 40.7 ms - it pays where a view crosses a lot of empty sea and costs a little where it does not.
    /// Above 1, how many consecutive empty blocks one step may cross. The sea marches 30.00 steps a pixel of which 17.81 are these
    /// zero crossings, and no step length touches them - which is why every per-step result that paid on the near view barely moved
    /// the sea, and why this is the sea's own lever.
    /// MEASURED first as a second, 64-voxel level tested before the 16-voxel one, and that lost: sea 9.22 -> 11.07 ms against a
    /// baseline that drifted to 10.86 in the same run, for a bit-identical image, and no effect at all on the near view. The reason
    /// is a number that should have been checked first - cloudTileVoxels is 64, so a 64-voxel block is exactly one sea tile and
    /// nearly always holds some cloud. The coarse test almost never succeeded and every zero crossing paid an extra load for it.
    /// MEASURED again as this run walk, and it gives nothing either: sea 9.23 ms at one block against 10.39 at eight, where the
    /// same baseline re-measured at the end of the run had drifted to 10.92, so somewhere between nothing and 5%. The image stays
    /// bit-identical at every run length, which is the traversal being correct rather than the change doing anything. What it
    /// falsifies is the premise: a zero crossing costs its texture load, not the loop around it, and walking a run removes the loop.
    ///
    /// The step-count share is not the cost share, and the two must not be conflated: a density sample costs several times a zero
    /// crossing, so 17.81 of 30.00 steps is nowhere near 59% of the frame. Fitting the sea's frame against its density samples per
    /// pixel across step lengths 1, 2, 4, 8 and 16 in one settle gives
    ///     ms  ~  1.84 + 0.575 * (density samples per pixel)
    /// which predicts 8.76 ms at a 2-voxel step against 8.85 measured. So about 7 ms of that 8.85 ms frame is density sampling and
    /// only about 1.8 ms is not - the zero crossings among it, with the sky, the resolve, the lattice setup and the compositing.
    /// Empty traversal is a small part of a small remainder, which is why neither attempt on it moved anything.
    ///
    /// An earlier two-point fit here read 5.40 + 0.346 and was wrong by three times. It was taken across runs whose sea had settled
    /// differently - 656 frames with 88498 bakes still waiting, inheriting the near view's cut, against 1160 frames with 44964
    /// waiting when the sea settles alone - and the same 4-voxel configuration measured 8.52 ms in one and 5.82 ms in the other.
    /// Only compare within one process, one settle, with a repeat of the first test at the end to measure the drift.
    ///
    /// Nor is the error metric as sharp as it looks. referenceLogP999 is read out of a histogram binned four to the octave, so it
    /// moves only in 19% steps and two quite different images routinely report the same number. Price changes on the mean log error
    /// and the shares over 0.02 and 0.1, which are continuous, and treat a single-bin move in p99.9 as a direction, not a size.
    ///
    /// Where the density samples are decides what any codec can be worth. The sea takes 12.19 samples a pixel, 8.88 of them in
    /// bricks and the remaining 3.31 resolved by the proxy. What those 3.31 cost is NOT known from this fit and must not be taken
    /// from it: an attempt here to floor a brick-only codec at 1.84 + 0.575 * 3.31 = 3.74 ms was wrong, because it charged a
    /// proxy-resolved sample the all-sample average when cameraExtinctionUnfaded reads the proxy unconditionally and returns on it,
    /// so a proxy sample pays a fetch and a failed cloudInstanceAt while a brick sample pays that and the whole hierarchy and atlas
    /// on top. cloudCostProbe measures the three parts separately. (Those counts come from a run that settled the short way, 656
    /// frames with 88498 bakes waiting, so fewer bricks were mapped than in a clean settle and the brick share is understated.) Of the 12.19, a third read exactly zero and another third are non-empty but too faint to
    /// change the pixel (alpha <= 1e-6), so 64% of the sea's density sampling buys no image at all.
    ///
    /// That 64% is not, however, 64% of the time, and the fit above must not be read as saying it is - it averages over samples whose
    /// costs differ by an order of magnitude. Both halves were then measured directly and both are nearly worthless to attack: the
    /// faint third is worth 0.7 ms at usable quality and 1.7 ms at any quality (cloudThinDepth), and the zero third is worth 0.01 ms,
    /// because an empty sample short-circuits in the directory and never reaches the atlas (cloudTightReject). The time is in the 4.4
    /// samples a pixel that do contribute, at roughly 1.6 ms per sample per pixel.
    /// DEFAULTED TO 8 on 2026-09-16 after re-measuring it against a drift anchor. The earlier reading here - 9.23 ms at one block
    /// against 10.39 at eight - put the eight-block arm late in a run whose baseline drifted from 9.7 to 10.9, so the whole 1.16 ms
    /// it appeared to lose was thermal. Measured with the first test repeated after every later one, the ratio is 0.866 at one block
    /// against 0.854 at eight for the same image: about 1%, small but free.
    uint cloudZeroSkip = 8;
    /// 1: a camera step whose sample is provably zero skips the brick walk that would find that out (0: off).
    ///
    /// The march steps on the dilated majorant, which bounds the trilinear extinction anywhere in a 4-voxel block INCLUDING the half
    /// voxel of a neighbouring block its support reaches. hstrTightMajorant is the same grid undilated, so where it is zero the
    /// block's own texels are all zero, and a sample at least half a voxel inside every face of that block reads only those texels:
    /// its density is exactly zero, and the directory walk and atlas read that returned zero were the price of the dilation.
    ///
    /// The test is conservative and exact - no error of any kind - which is what distinguishes it from cloudThinDepth. It costs one
    /// Texture3D load against a brick walk, and if it pays, hstrMajorant and hstrTightMajorant share dimensions and can be packed
    /// into one float2 so the march fetches both in the load it already does per step.
    ///
    /// MEASURED as a count first (marchProbe 24/25, min == max step 2): the sea takes 4.96 samples a pixel in a tight-zero block, of
    /// which 2.13 are far enough inside to reject (16.6% of all its density samples, 52% of the 4.08 that read exactly zero); the
    /// near view 5.65 and 2.33. The rest sit in the boundary shell where the dilation is doing real work - which is why the tight-zero
    /// count exceeds the exactly-empty count: a boundary sample in a tight-zero block can still read density from its neighbour.
    ///
    /// AND IT IS WORTH NOTHING. Sea, 4K, beam view, min == max step 2, one settle (10.02 ms against 10.04 re-measured at the end):
    /// off 10.02, the load and the test with the answer thrown away 10.43, the test acted on 10.44. The rejection saves 0.01 ms -
    /// zero - and the entire 0.42 is the extra fetch. Packing the two majorants into one float2 would make the fetch free and buy
    /// nothing, so it is not worth doing. The error is bit-identical across all three, to every digit including the maximum, so the
    /// interior test is right and really is rejecting only zeros; there is simply nothing behind them.
    ///
    /// What that establishes is worth more than the parameter. An exactly empty density sample already costs almost nothing: the
    /// directory walk short-circuits on an empty cell code and never reaches the atlas. So "ms ~ 1.84 + 0.575 * density samples" is
    /// an average over samples of very different cost and must not be used to price a change that removes a particular KIND of
    /// sample. The sea's 12.19 samples a pixel are a third exactly zero and a third too faint to change the pixel, and neither third
    /// is where the time goes - cloudThinDepth buys 0.7 ms of usable time from the faint ones and this buys none from the zeros. The
    /// roughly 7 ms of density work sits in the 4.4 samples a pixel that actually read the atlas and contribute, about 1.6 ms each,
    /// which is also where the longitudinal oracle and any transfer or moment codec would act. Attack the contributing samples.
    uint cloudTightReject = 0;
    /// Timing oracle decomposing what one camera density sample costs (0: off, the real image). 1 returns the proxy extinction the
    /// moment it is read, so every sample costs what a proxy-resolved sample costs; 2 walks cloudInstanceAt and the sparse hierarchy
    /// and writes the sample, but returns before cloudBrickDensity, so the atlas fetch and its filtering are the only thing missing.
    /// Both render the wrong image deliberately - pin minStepVoxels == maxStepVoxels so the step schedule and the ray count cannot
    /// move, and read only the times.
    ///
    /// This exists because "ms ~ 1.84 + 0.575 * density samples" is an average over samples whose costs differ by an order of
    /// magnitude (see cloudTightReject), and so cannot price a change that removes one CLASS of sample. It was used that way once
    /// here - to put a 3.74 ms floor under a brick-only codec by charging the sea's 3.31 proxy-resolved samples 0.575 ms each - and
    /// that was wrong in its own commit's terms. cameraExtinctionUnfaded reads the proxy unconditionally before it looks for a brick,
    /// so a brick sample pays the proxy's cost and everything after it, while a proxy sample pays only the fetch and a failed
    /// cloudInstanceAt. The floor needs the proxy's own cost, which is what mode 1 measures.
    ///
    /// MEASURED (4K, beam view, min == max step 2, two independent processes). The arms reproduce closely - near 3.17 and 3.17
    /// proxy-only, 9.11 and 9.16 without the atlas; sea 3.88 and 4.03, 10.31 and 9.96 - which is what the reading rests on, because
    /// the baselines drifted badly in the second run (sea 9.89 to 10.89, 10%) and the attempt to hold the ray count fixed by raising
    /// beamTolerance did NOT work: the marched fraction came back identical, so tolerance is not what gates refinement here. The
    /// probe modes do move it (sea 3.9% full, 5.5% proxy-only, 4.5% no-atlas), and since proxy-only refines MOST its time is
    /// overstated, so everything attributed to the hierarchy below is understated.
    ///
    ///     sea   full ~10.0   no atlas ~10.0   proxy only ~4.0     so atlas ~0 ms, hierarchy ~6.0 ms
    ///     near  full ~12.2   no atlas  ~9.1   proxy only ~3.2     so atlas ~3.0 ms, hierarchy ~6.0 ms
    ///
    /// Mode 3 then splits that 6 ms, and cloudInstanceAt turns out to be almost none of it:
    ///
    ///                    proxy + fixed   cloudInstanceAt   walk + atlas    full
    ///     sea                 2.78            0.13             6.80        9.71
    ///     near                3.18            0.23             9.0        12.4
    ///
    /// AN EARLIER READING HERE SAID 2.77 ms FOR THE SEA, a quarter of its frame, and it was an artefact. The sea's instance cost
    /// measured 2.43, then 0.41, then 0.13 across runs; the large readings came only from runs where the sea followed the near view,
    /// and its settle is not a settle - "sun slots free 0, 88498 waiting" means the bake pool is exhausted and those bakes can never
    /// land, so which bricks hold one keeps churning as frames render. settle() waits on sunBakesFrame and not on sunWaiting, so it
    /// returns while the view is still moving. Interleaving an anchor between tests added about sixty frames before each later arm
    /// and shifted them along that churn, which is what exposed it. The near view settles properly (waiting 0) and had been saying
    /// 0.23 ms the whole time; a tenfold disagreement between the views was read as content asymmetry when it was a broken settle.
    ///
    /// Prefer the near view for per-sample cost work. Sea arms reproduce to a few percent within one run and not at all across
    /// settles, so read only differences between adjacent tests there, and never a difference spanning a run.
    ///
    /// So hoisting the instance is not worth doing - a 2D DDA over sea tiles would be chasing 0.13 ms - and the tile mask and the
    /// removal of the duplicate instance fetch beside it are kept because they are provably less work for a bit-identical image,
    /// not because either was measurable. What is left is the walk: about 6.8 ms of a 9.7 ms sea frame, with the atlas at roughly
    /// nothing. That is the one thing worth building for, and the chain is deeper than the
    /// node descent suggests: cloudAssetBrickAt goes directory, node, node, brick info, then a parent walk of up to fifteen dependent
    /// .parent and .info loads when the mapped brick is finer than the footprint asked for. A table keyed on (position, desired level)
    /// that stores the brick the residency cut says should answer removes the descent AND that loop. Weigh it against the note above
    /// residualAtVoxel first, which found these reads are cache hits: a table shortens a chain of hits rather than removing misses,
    /// but unlike the context cache it costs no persistent registers, so it does not lose the same way.
    ///
    /// Reading the density is free on the sea; the walk to find it is six of ten milliseconds. That also settles the floor properly:
    /// proxy-only runs all 12.19 samples a pixel in 4.0 ms, about 0.167 ms per proxy sample per pixel against the 0.575 average, so
    /// the sea's 3.31 proxy samples cost about 0.55 ms and a brick-only codec floors near 2.4 ms - under the target, not above it.
    ///
    /// And it says which codec. A representation compressing what is INSIDE a brick - moments, Gabor, Haar, analytic intervals -
    /// attacks the atlas fetch, which the sea measures at nothing. Caching the resolved brick context across a run attacks address
    /// arithmetic, and that was tried on 2026-09-15 (the note above residualAtVoxel): exact to 6e-9, 80% of lookups removed, and
    /// SLOWER at 46.5 -> 52.9 ms, because the context's registers cost more occupancy than the lookups cost latency. What is left is
    /// the shape of the walk. cloudAssetBrickAt chases directory -> node -> node -> brick with each address coming out of the
    /// previous load, so it is a dependent chain whose latency no arithmetic saving shortens. Fewer samples, or fewer DEPENDENT
    /// fetches per sample - a flatter index that answers "which brick, where in it" in one independent load.
    ///
    /// EXCEPT THAT NONE OF THAT IS WORTH BUILDING, and the stage probes say so before anything was written. Mode 5 minus mode 3 puts
    /// the directory/node/node descent at 1.4% of the frame - 0.14 ms - so a spatial page table replaces almost nothing. The parent
    /// climb looked like the rest of it at about 1.03 ms, but that delta was not schedule-clean, and marchProbe 28/29 settle it: the
    /// climb runs on 1.3% of the sea's brick samples and 0.1% of the near view's, one step each, 0.12 iterations a pixel. So an
    /// ancestor table keyed on (entry, desired level) is worth microseconds, and the 1.03 ms belongs to the octant test and to the
    /// schedule change mode 6 causes. Three structural proposals - the spatial page table, hoisting cloudInstanceAt with a sea-tile
    /// DDA (0.13 ms), and the ancestor table - each measured dead before being implemented. What is left in that 6.8 ms is the
    /// occupancy read, and its NET value is already positive: turning cloudEmptySkip off costs 0.68 ms.
    ///
    /// AND THE INTERVAL CODEC HAS A CEILING OF ABOUT 1.2x, measured before choosing a representation (marchProbe 26/27). What one
    /// compact interval operation could stand for is the CONTRIBUTING run, not the samples per brick visit: the empty and faint ones
    /// in a visit are nearly free, so replacing them buys nothing. The sea takes 4.38 contributing brick samples a pixel over 1.98
    /// distinct contributing entries - a run of 2.21 - and the near view 7.13 over 2.80, a run of 2.54, against 3.24 and 3.78
    /// counting every sample in the visit. An ideal codec costs one operation per entry, so it saves 2.40 events a pixel on the sea,
    /// and at the ~0.67 ms per brick sample per pixel implied above (6 ms of hierarchy over 8.94 brick samples) that is about 1.6 ms:
    /// 10.0 -> 8.4. The cause is geometric and not fixable by encoding - a brick is 8^3 at the sampled level, a 2-voxel step crosses
    /// it in about four steps, and only 2.2 of those contribute, so there is not enough depth along the ray to amortise against. It
    /// is the same answer d* = sqrt(r/z) = 0.87 gave from the other side (cloudTransferClasses): the brick is already the best unit,
    /// and the best unit is too small. So the choice between preintegrated (tau, centroid), Gabor, Haar moments and the rest barely
    /// matters - none can be worth more than about 1.6 ms - and it attacks the same events as the longitudinal oracle, so the two do
    /// not stack. What is NOT bounded by any of this is the number of rays: the per-sample cost is close to irreducible, the sample
    /// count is not.
    ///
    /// Except that the per-sample cost is NOT close to irreducible, which the split above shows and an earlier reading here denied.
    /// The frame factorises as events times cost per event, and every lever measured before this - the lattice, the faint and zero
    /// samples, an interval codec, the oracle - attacks only the events. The instance hoist and the page table attack the cost, so
    /// they are orthogonal to all of them and to each other. On the sea that is 2.77 + most of 3.70 against a 10.5 ms frame before
    /// the oracle's 1.5x applies to what is left, which is the first route to 3 ms that does not depend on something unbuilt.
    uint cloudCostProbe = 0;
    uint cloudTrapezoid = 0; ///< 1: camera steps integrate the trapezoid of their end extinctions instead of their midpoint's.
    uint cloudLocalStep = 0; ///< 1: camera steps are as long as the sample's own cell, octant and brick density bounds allow.
    /// Fraction of a step's length folded into its density sample's footprint (0: off). minStepVoxels above one samples the density
    /// below its Nyquist rate along the ray, so coarse steps lose detail rather than merely integrating it coarsely, and the brick
    /// pyramid already holds filtered levels that would not. MEASURED (4K, sea library, beam view, minStepVoxels 4, full step): the
    /// rays do get much cheaper - sea 8.63 -> 1.17 ms and near 10.76 -> 1.25 ms - but the frame is wrong, 16-25% of pixels past 0.02
    /// log error with p99.9 near 0.7, and no tile refines at all because every query has been smoothed into agreement. The pyramid
    /// filters isotropically while the sampling problem is only along the ray, so a full step's width throws away that much detail
    /// across the ray as well.
    ///
    /// MEASURED over fractions, to see what the transverse blur would bear, and there is no setting that pays: every nonzero
    /// fraction lands on the same p99.9 - 0.689 near and 0.819 sea - at 0.25 and 0.5 with 4-voxel steps and at 0.5 with 2-voxel
    /// steps alike, which is saturation rather than a trade. Worse, the partial settings are slower than no footprint at all: near
    /// s4 goes 6.66 -> 12.61 ms at fraction 0.25 and sea s4 5.56 -> 10.81 ms, because the marched share explodes (near 5.4% ->
    /// 18.3%, sea 4.0% -> 10.8%). That is the beam hierarchy's own constraint showing: a tile is tested by comparing its centre
    /// query against a prediction from the corner queries, so an estimator that shifts by an amount varying per ray makes
    /// neighbouring queries disagree and every tile refine. Only at full blur do they agree again, and then nothing refines and the
    /// frame is simply wrong. Leave it at 0; the longitudinal sampling problem needs a filter along the ray, which the isotropic
    /// brick pyramid cannot express.
    float cloudStepFootprint = 0.f;
    /// Density taps per camera march step (1: the midpoint rule), placed at Gauss-Legendre nodes. Raising minStepVoxels with it holds
    /// the taps per voxel fixed, which is the controlled test of whether a coarse step's error is quadrature error that a better
    /// estimator can remove at the same cost.
    ///
    /// MEASURED, and it is not (4K, beam view, one thread per ray, temporal off, against the per-pixel 1-voxel march; maxStepVoxels
    /// raised with minStepVoxels, which otherwise clamps the step to 4). The error tracks the step length and ignores everything
    /// else. Near: 1 tap over 2 voxels gives p99.9 0.036, and then 4-voxel steps give 0.122 whether they carry one tap or two, and
    /// 6- and 8-voxel steps give 0.205 at two or three taps alike. Sea: 0.022 at 1 tap over 2 voxels, 0.061 at one tap over 4 and
    /// 0.043 at two over 4 - extra nodes help a little there but nothing like enough. At a fixed tap budget the plain midpoint at
    /// the shorter step wins on both axes every time: sea q1 s2 8.98 ms at p99.9 0.022 against q2 s4 10.85 ms at 0.043, the latter
    /// slower despite equal taps because the node loop costs per-step overhead of its own.
    ///
    /// Two error terms live in a step, and separating them needs care with where the downstream sample sits. The centre node, which
    /// carries the source evaluation, the sun lookup and the empty-cell skip, is node cloudQuadrature / 2 - the exact midpoint at
    /// three nodes, but 0.789 of the way through the step at two. So one tap against two does not hold the source position fixed,
    /// and the two-node measurements above conflate a better tau with a displaced source. One tap against three does hold it.
    ///
    /// MEASURED that way, and tau is a real and roughly equal-proportion part of both views: at 4-voxel steps p99.9 goes 0.122 ->
    /// 0.086 near and 0.072 -> 0.051 sea, at 6-voxel steps 0.205 -> 0.145 and 0.205 -> 0.172. That is about 30% of the error, and it
    /// is why two nodes looked useless near - their tau gain was cancelled by the source moving off the midpoint, landing back on
    /// 0.122 exactly. The rest is the source varying under the extinction weighting, C = integral of T(s) sigma(s) S(s) ds, which
    /// sampling sigma cannot touch: even with tau nearly exact at three nodes, a 4-voxel step still carries 2.4 times the error of a
    /// 2-voxel one.
    ///
    /// Neither term is worth paying taps for, which is the part that decides the design. At equal cost the shorter step with fewer
    /// taps wins every time: q3 over 4 voxels is 10.96 ms at p99.9 0.086 against one tap over 2 voxels at 11.00 ms and 0.036, and
    /// q3 over 6 voxels is 7.94 ms at 0.145 against one tap over 4 voxels at 6.71 ms and 0.122 - beaten on both axes. So error goes
    /// as the step length, cost goes as the taps, and every within-step estimator of sigma is dominated; an exact analytic integral
    /// of the trilinear density would land on the same curve, because it would only address the 30%. Cheaper rays have to come from
    /// compositing events that each still stand for a short interval: an extinction-weighted source basis over the step
    /// (cloudSourceLinear), pre-integrated transfer along the ray, which this asset does not carry, or fewer rays.
    uint cloudQuadrature = 1;
    /// 1: a step composites its source as an extinction-weighted blend of the values at its two ends instead of one value standing
    /// for the whole interval. That is the error term cloudQuadrature isolated, and it costs nothing: the sample already sits at the
    /// step's end, so this step's end source is the next step's start source and there is still one evaluation per step. It is the
    /// runtime rehearsal of what a transfer codec would bake - the same basis, with the weights precomputed over a macro-segment
    /// rather than derived from one step's own tau.
    ///
    /// MEASURED, and it loses on both axes (4K, beam view, against the per-pixel 1-voxel march). Near: p99.9 0.036 -> 0.051 at
    /// 2-voxel steps and 0.122 -> 0.122 at 4, the time unchanged. Sea: 8.84 -> 11.10 ms and p99.9 0.026 -> 0.030 at 2 voxels,
    /// 6.90 -> 10.14 ms at 4. Two separate mechanisms, both attributable. The accuracy went backwards because needing the sample at
    /// the step's end switches the optical depth from the midpoint rule to the trapezoid, whose error is twice as large and of the
    /// opposite sign, and the source basis gained about what that lost. The time went backwards because a sample at the step's end
    /// weakens the empty-cell skip, which is the same effect that made the trapezoid 1.8x slower when it was first tried.
    ///
    /// The result does not settle the codec question, and it is worth being precise about why: these weights assume sigma constant
    /// across the step, whereas the whole point of a baked W_k = integral of T sigma b_k is that it carries the correlation between
    /// the extinction and the source at fine scale, which a constant-sigma weight throws away. So this says a cheap runtime version
    /// of the basis does not pay; it does not say a baked one cannot. What it does confirm is that the source's variation across a
    /// step is not low-order - a linear basis with reasonable weights recovers none of it - and any transfer representation has to
    /// carry the fine structure, not a smooth fit to it.
    uint cloudSourceLinear = 0;
    /// Octahedral grid edge of the transfer-reuse probe's direction classes, 0 to leave it off. A stored transfer operator has to be
    /// keyed by direction as well as entry point, so this counts how many rays would share one: the probe marks, per (asset brick,
    /// source-space direction class), which of the brick's 64 cells a ray entered through, and counts brick crossings. Crossings
    /// divided by distinct marks is the reuse, and it has to beat Z/(Z-1) - about 1.45 at the measured 3.2 events per crossing -
    /// before precomposing anything can pay.
    ///
    /// MEASURED (4K, beam view, minStepVoxels 2, one thread per ray, temporal off), and it clears break-even without clearing it by
    /// enough. Near, 8.86M crossings over 1.95M entries, 4.55 each; sea, 6.27M over 2.55M, 2.46 each. Almost none of that is lost to
    /// direction binning - 16, 64 and 256 classes give 5.37, 4.63, 4.55 near and 2.60, 2.47, 2.46 sea - because the camera is a point
    /// and the rays crossing one brick already arrive within a narrow cone. Nor is the sharing local: an entry cell is two source
    /// voxels, about two pixels, while the beam traces roughly one ray per eight, so what is being reused is the same asset brick
    /// entered at the same cell and angle from different sea tiles. That is CloudSea's instancing paying off, and it is the only
    /// reuse there is.
    ///
    /// It is not enough. Serving a crossing costs 1 + Z/R events against Z, so the brick work falls by Z/(1 + Z/R): 2.05x near but
    /// 1.40x on the sea, and brick samples are only 73% of the sea's density samples, so the frame gains 1.26x - 8.83 ms to about
    /// 7.0 - before a single cache lookup, eviction, bake dispatch or fade invalidation is paid for. Nor does a deeper unit rescue
    /// it, which was the obvious escape: four times deeper multiplies Z by four but a ray then crosses four times fewer units while
    /// the entries stay put, since distinct units fall with surface area about as fast as entry cells per unit rise. With Z = z d
    /// and R = r/d the gain z d / (1 + z d^2 / r) peaks at d = sqrt(r/z) = 0.87, so the brick is already the best unit and a node
    /// would give 1.03x. Precomposition needs reuse, reuse needs many rays through one entry, and a beam path with a two-pixel entry
    /// resolution does not have them.
    /// Fine density observations inside one camera step, taken at the pixel's own footprint (0: off). The step keeps its length; only
    /// its interior is resolved, which is the one-dimensional filter the ray actually needs, as against cloudStepFootprint's request
    /// for a coarser brick - a four-voxel footprint asks for a 4x4x4 low pass where the integration wants 1x1x4, and erasing the
    /// transverse detail is what left every beam query agreeing and no tile refining. Deliberately slow: it measures what the step
    /// would be worth if the quantity were free, not a way to make it free.
    uint cloudLongitudinalOracle = 0;
    /// 1: the step's source is evaluated at the opacity centroid M1/M0 of those observations rather than at the geometric midpoint.
    /// The centroid is where the step's extinction actually deposits its light, and moving one evaluation there costs nothing at
    /// runtime - only its position changes - so it is the cheapest possible attack on the source-variation error that survives an
    /// exact optical depth (see cloudQuadrature, where tau is only about 30% of a long step's error).
    ///
    /// MEASURED with 16 observations per step (4K, beam view, against the per-pixel 1-voxel march, min == max step, one settle per
    /// view with the first test repeated at the end: sea 9.76 against 9.78 ms, near 12.14 against 12.17). The marched share never
    /// moves (sea 3.9-4.0%, near 5.2-5.3%), so none of the gain is bought by refining more of the image.
    ///
    /// DO NOT PRICE THIS ON p99.9. referenceLogP999 is read out of a histogram binned four to the octave (kCompareBinFloor * 2^(b/4)
    /// in HSTRCloud.cpp), so it moves in steps of 19% and cannot resolve anything finer. It is why the sea and the near view report
    /// an identical 3.44e-01 at an 8-voxel step and an identical 2.56e-02 with the pair at four - two different images landing in one
    /// bin, not a coincidence. Every earlier reading of this parameter - the doubling, then 1.5-1.7x, then 1.25x at equal quality -
    /// was computed from that quantised number. Use the mean log error and the >0.02 and >0.1 shares, which are continuous.
    ///
    ///     sea, 4-voxel step   plain 1.00e-03 / 0.850% / 0.017%   tau only 8.11e-04 / 0.512% / 0.017%   pair 6.20e-04 / 0.168% / 0%
    ///     sea, 2-voxel plain  3.29e-04 / 0.111% / 0.001%    (mean log error / share over 0.02 / share over 0.1)
    ///     near, 4-voxel step  plain 2.33e-03 / 2.111% / 0.137%   tau only 1.74e-03 / 1.254% / 0.008%   pair 1.30e-03 / 0.221% / 0%
    ///     near, 2-voxel plain 5.72e-04 / 0.247% / 0.000%
    ///
    /// The metrics disagree about what it is worth, and the disagreement is the result. Against each view's own error curve the pair
    /// at four voxels matches a plain step of 2.97 voxels on the mean but 2.30 on the >0.02 share - 1.35x against 1.74x - because it
    /// does not improve the bulk of the image and the tail equally. On the near view, where a step carries more source variation, it
    /// is worth 2.07x: its 0.221% at four voxels is better than a plain two-voxel step's 0.247%.
    ///
    /// Where it really lands is the worst pixels. At an 8-voxel step the share over 0.1 goes 1.343% -> 0.011% on the sea and
    /// 3.764% -> 0.015% on the near view, two orders of magnitude, while the mean moves by a third. This is a fix for the blocky
    /// artefacts a long step leaves, not a uniform gain - which follows from the mechanism, since a misplaced source evaluation costs
    /// nothing where the source is flat and everything where it is not.
    ///
    /// Priced at equal quality on the >0.02 share, through each view's own cost curve and with the two scalars free: the sea would
    /// take a 3.48-voxel step at about 6.5 ms against 9.76, and the near view a 4.14-voxel step at about 7.1 ms against 12.14. So
    /// about 1.5x on the sea and 1.7x on the near view.
    ///
    /// The run that read a full doubling held maxStepVoxels at 8 while varying minStepVoxels, so its "2-voxel step" was really 2 to
    /// 8 voxels chosen by the majorant rule and only its 8 was a true fixed step. Pin minStepVoxels == maxStepVoxels when the step
    /// length is the variable.
    ///
    /// The first implementation moved tSample and midpoint to the centroid but left `sample` at the centre observation, and three
    /// consumers read a sample against the position it was taken at: sample.t (cloudLocalStep's next step), cloudEmptyExit (a box
    /// measured from sample.x, returned as an offset from the t it is given - so a centroid ahead of the observation skips past where
    /// density returns) and sunDepthAt (near field from the sample's brick entry, far field from the position). The source read a sun
    /// depth belonging to a point up to half a step away, in the direction the centroid had just decided the light was not. Fixing it
    /// moved the sea's p99.9 one bin at both four and eight voxels, in the right direction at both - which is all a metric binned four
    /// to the octave can say, and the reason the readings above are taken on the continuous ones instead.
    ///
    /// The split matters as much as the total, and it is not the same split in the bulk as in the tail. Over the whole image the two
    /// contribute about equally - of the 4-voxel gain, the optical depth carries 50% of the mean and 50% of the >0.02 share on the
    /// sea, 57% and 45% on the near view. An earlier reading of "a third" came from p99.9 and did not survive the continuous metrics.
    /// In the tail they separate completely: at four voxels the share over 0.1 is 0.017% plain, 0.017% with an exact optical depth
    /// alone - unchanged, to three figures, on both views - and 0.000% once the centroid moves the source. So the error that makes a
    /// long step look broken is entirely a question of where the step's one source evaluation is placed, and not of fine structure
    /// the step cannot represent; the optical depth buys an even share of the ordinary error and none of the worst of it. That is
    /// what makes the result worth anything: two scalars per position and
    /// direction - the longitudinal optical depth and the opacity centroid within it - are enough, where a transfer operator
    /// carrying the source itself was measured dead (cloudTransferClasses). Both are properties of the density alone, so unlike a
    /// transfer operator they survive every change of sun, cache and phase, and bakeCloudSun is already a per-voxel directional line
    /// integral at full transverse resolution - the template for computing them.
    ///
    /// What this does not establish is the price. Halving the compositing events does not halve the frame: the query lattice, the
    /// resolve and the per-pixel refinement do not scale with step length, and the observed 2-voxel to 4-voxel saving has ranged
    /// from 1.14x to 1.64x across runs on a laptop that throttles. Measure that scaling on a cool machine before costing any bake.
    uint cloudOracleCentroid = 0;
    uint cloudTransferClasses = 0;
    uint cloudTransferWords = 0; ///< Size of that buffer; its last word is the crossing counter.
    /// Domain voxels a camera march may move before it queries the sun depth again (0: every sample). MEASURED (4K c15, 0.1 voxels,
    /// which reuses about every second query): near 42.8 -> 42.3 ms, sea 30.4 -> 33.0 ms - nothing. Halving the sun fetches does not
    /// pay, the same way the brick context did not: what a sample costs is the sample, not the fetches inside it.
    float cloudSunReuse = 0.f;
    /// Camera transmittance at which a march stops (the rest of the ray is behind opaque cloud). An earlier reading here - near
    /// 42.6 -> 38.8 ms, sea 28.6 -> 30.4 - concluded that "the long-tailed rays are not what the frame waits for", and it was wrong
    /// twice over: both figures are the per-pixel view, which is the yardstick and not the shipping path, and the sea rose because
    /// the run drifted. RE-MEASURED on the beam path against a drift anchor: 0.02 is x0.938 of the frame for 0.111% -> 0.117% over
    /// 0.02, and 0.05 is x0.915 for 0.415%. The near view's 9% was in the comment the whole time, next to a sea number that buried it.
    ///
    /// Left at 1e-3 because it is not free - 0.02 is the cheapest quality the sea sells anywhere, but it is quality. Set it with
    /// cloudThinDepth 0.05 for about 13% together at 0.140%.
    float cloudMinTransmittance = 1e-3f;
    /// cloudTiles - 1 where the tile count is a power of two, else ~0u (the counts are square, so one scalar covers both axes).
    /// The sea wraps a possibly negative tile coordinate into the instance grid, which costs two integer modulos per density sample,
    /// and integer division is emulated on this hardware - tens of instructions each, paid 12.81 times a pixel on the sea, inside the
    /// cloudInstanceAt that cloudCostProbe mode 3 measures at a quarter of the sea's frame. For a power-of-two count the Euclidean
    /// wrap is exactly an AND, for negative coordinates too in two's complement, so the pair collapses to one instruction.
    ///
    /// This took the LAST slot rather than a place beside cloudTiles on purpose. HSTRCloudParams is hand-packed in 16-byte rows and
    /// cloudTiles, cloudTileVoxels and cloudAtlasShift fill one exactly; inserting there shifted every field after it and the shader
    /// silently read the wrong ones - every probe mode stopped working and every beam image came out identical and wrong. Add
    /// parameters by taking a pad slot or appending, never by inserting into a full row.
    uint cloudTileMask = ~0u;
    /// World Y of the lowest and highest majorant block holding any density, dilated by one block (the whole grid's height where
    /// nothing is resident). mediumInterval used the extinction grid's own floor and ceiling, so a sea ray began and ended in empty
    /// sky and spent most of its steps there - 17.81 of 30 a pixel are majorant-zero crossings. Clamping the slab to the band deletes
    /// those from the ray rather than skipping them a block at a time, and changes no pixel: outside the band there is no density to
    /// integrate. Unlike every step-length lever, this one is free of quality by construction.
    float2 seaContentY = float2(-1e30f, 1e30f);
    /// 0: march the whole grid height again, for measuring what the band is worth. MEASURED (sea, 4K, beam view, drift-anchored,
    /// first test repeated at the end): 9.92 -> 9.50 ms, x0.962, with the error identical to every digit including the maximum -
    /// 0.111% over 0.02, p99.9 2.15e-02 - which is what "lossless by construction" is supposed to look like and the reason to prefer
    /// this over every other lever measured today.
    ///
    /// It first measured 19% SLOWER, and the cause was a latent bug it exposed rather than the clamp: mediumInterval capped the ray
    /// at max(0, interval.x) + seaViewDistance, so the view distance ran from where the ray met the slab and not from the camera.
    /// While the slab was the whole grid and the camera sat inside it, interval.x was negative and the two agreed. Clamping to the
    /// content band made interval.x the positive distance to the band, and every ray then marched a view distance BEYOND it. The
    /// same bug would have bitten a camera flying above the grid with no clamp at all, which is the intended use of this renderer.
    uint cloudSlabClamp = 1;
    uint padHSTRCloud10 = 0;

    /// Measurement only, needs a stored exact frame: >0 replaces every beam tile test with the truth - a tile refines when at least
    /// this many of its pixels, reconstructed from its basis, are off the stored frame by more than beamOracleBar in the compare
    /// metric. The floor any real test could reach for a tiling.
    uint beamOracle = 0;
    float beamOracleBar = 0.02f;
    /// 1: coarsest tiles have no centre query - the root lattice is corners only, and accepted root tiles reconstruct bilinearly.
    uint beamCentreless = 0;
    uint beamGridCentres = 0; ///< Set per dispatch: the grid query pass marches tile centres (1) or corners (0).

    /// Temporal refresh of the beam view (0: off; one level, per-level build). Each build marches the root lattice points and the
    /// refined tiles of one cell of a beamRefresh x beamRefresh pattern, and reprojects the rest from the previous build at the
    /// opacity-weighted distance of their rays; points and pixels that do not reproject consistently are marched as well.
    uint beamRefresh = 0;
    uint beamFrame = 0;               ///< Set per build: the build counter that picks the refreshed cell.
    uint beamHistoryValid = 0;        ///< Set per build: whether the previous build's lattice, levels and pixels exist.
    float beamDepthTolerance = 0.05f; ///< Relative spread of distances across a reprojection footprint above which it is marched.
    // The fields above end half-way through a 16-byte row, and the shader starts a matrix on a fresh row while the host does not:
    // without these the matrix and everything after it were read 8 bytes off (HSTRCloud checks the two sizes at startup).
    uint padHSTRCloud13 = 0;
    uint padHSTRCloud14 = 0;
    float4x4 beamPrevViewProj;        ///< Set per build: view-projection of the previous build.
    float3 beamPrevCamera = { 0, 0, 0 }; ///< Set per build: camera position of the previous build.
    /// beamRefresh diagnostics: 1 counts cloud points whose first reprojection lands within half a pixel of themselves (in the list
    /// counts' slot 2), 2 also accepts every reprojection that lands in the frame, unchecked; 4, 5 and 6 skip the distance agreement,
    /// the accepted-tile requirement and the carry tolerance respectively.
    uint beamRefreshDebug = 0;
    /// beamRefresh: the largest angle, in pixels, between a carried value's previous ray and its current ray through the point it
    /// was found at (0: no limit). A volume ray is not a surface point: under parallax the new ray crosses the cloud along another path.
    float beamParallax = 0.f;
    /// beamRefresh: a carried lattice value that blends samples needs them to agree to this, in log(1 + cache + single) and in
    /// transmittance (0: no limit). The tile tests accept bases at beamTolerance; a carried value is resampled again every build.
    float beamCarryTolerance = 0.f;
    /// Queued root queries (one level, per-level build, single segments): a first pass carries what beamRefresh can and queues the
    /// rest into kBeamQueueBuckets lists by the step count the previous build marched there, and each list is marched in a dispatch
    /// of its own. A dispatch's threads run in batches that wait for their slowest ray, so a quarter of the rays marched in every 2 x 2
    /// block costs nearly what all of them do, and rays of very different lengths side by side cost what the longest does.
    ///
    /// MEASURED and OFF (sea, 4K, frozen residency, 2026-09-17): the hypothesis behind it is half right - with beamRefresh carrying 87%
    /// of the points the grid pass still costs 2.49 of 3.14 ms - but queuing loses anyway. Parked 5.72 -> 8.75 ms (buckets of 12
    /// steps), 10.17 (6), 8.95 (24); flying 2 units a frame 9.19 -> 9.47; the queued pixel march 1.76 -> 2.73. The lists fill in the
    /// order threads finish, so neighbouring rays stop being neighbours, and the march is memory-bound: losing that coherence costs
    /// more than similar batches save. A queue would have to keep spatial order within each list to be worth another try.
    uint beamQueue = 0;
    uint beamQueueBucket = 0;    ///< Set per dispatch: the list the queued march pass reads.
    uint beamQueueCapacity = 0;  ///< Set per build: entries per list.
    float beamQueueSteps = 12.f; ///< Steps per bucket of the predicted cost.
    uint beamQueueThreads = 1;   ///< Set per use: threads per queued entry (1 for root queries, the finest tile's pixels for tiles).
    /// beamRefresh: cells of the refresh pattern are blocks of this many lattice points (or finest tiles) a side, so a carried block
    /// runs no march at all instead of sharing its batches with the marched quarter.
    /// MEASURED (sea, 4K, r4, 2026-09-17): batching was not the loss - the carried points hold 77% of the steps (debug 7) but the
    /// carry itself cost ~2.6 ns a point against ~4 ns a march. Blocks of 4: parked 3.24 -> 2.95 ms at the same error, but flying
    /// 2 units a frame the error rises 0.152 -> 0.185%, and 8 is no better.
    uint beamRefreshBlock = 1;
    /// beamRefresh: the pattern size for tile centres (0: beamRefresh). The centres only feed the bubble term and the tile tests.
    /// MEASURED (before the carry fast path): r4 with centres at 8 carries 2% more points, 4.03 -> 3.93 ms parked and nothing flying.
    uint beamRefreshCentres = 0;
    /// Set per dispatch: the first row of the band of the path-traced reference being traced (HSTRCloud::mReferenceBandRows).
    uint referenceRowOffset = 0;
    /// The params upload is a raw copy, so the host and shader layouts must agree to the byte: the int3 below would start a new
    /// 16-byte row in the shader and pack tight on the host without this (the layout check in bindRenderer catches it).
    uint2 padCutOrigin = { 0, 0 };

    /// Cloud-sea transport cut. The sea has no CPU Schur tree, so its cut is the regular grid of 16^3-voxel domain cells placed as
    /// the camera-centred window of hstrLeafDims cells whose corner sits at this leaf cell (X/Z wrap into the torus, so the window
    /// tiles the reachable space exactly once; Y spans the whole layer and its origin is 0). Node coordinates are window-local,
    /// which keeps them small and exact however far the camera has flown from the domain origin: the traversal subtracts the
    /// origin from the ray once and every density lookup keeps the unshifted position.
    int3 hstrCutOriginLeaf = { 0, 0, 0 };
    uint hstrCutFlat = 0; ///< 1 while the flat domain cut replaces a CPU-built hierarchy (no parents, no ancestor antichain).

    /// Rotation-invariant beam basis frame (beamRefFrame).
    ///
    /// The beam basis is a function of WORLD DIRECTION, not of screen position: turning the camera does not invalidate a single
    /// basis ray, it only moves where that ray lands. Keying the basis to the screen lattice hides that, because the direction a
    /// pixel needs after a turn falls between the previous lattice points and the carried value has to be resampled through the
    /// previous basis - which is what beamReprojectQuery's "a turning camera, which reprojects exactly, still tripled the error"
    /// comment records, and why yaw is the most expensive motion there is.
    ///
    /// So the whole beam build - lattice, tiles, levels, verifier, basis - runs in a reference image that does NOT rotate with the
    /// camera. beamRefU/V/W are that image's camera basis (the anchor pose's, widened by beamRefMargin so the screen stays inside
    /// it) and a reference position maps to a direction exactly as cameraRayDirection does for a pixel. beamRefInverse is the
    /// inverse of [U V W], which sends a world direction back to a reference position for the resolve. The frame is re-anchored
    /// only when the camera turns far enough that the screen would leave it, and beamRefValid is 0 for that one build.
    float3 beamRefU = { 0, 0, 0 };
    float padBeamRefU = 0.f;
    float3 beamRefV = { 0, 0, 0 };
    float padBeamRefV = 0.f;
    float3 beamRefW = { 0, 0, 0 };
    float padBeamRefW = 0.f;
    float3 beamRefInverse0 = { 0, 0, 0 };
    float padBeamRefInverse0 = 0.f;
    float3 beamRefInverse1 = { 0, 0, 0 };
    float padBeamRefInverse1 = 0.f;
    float3 beamRefInverse2 = { 0, 0, 0 };
    float padBeamRefInverse2 = 0.f;
    uint2 beamFrameDim = { 0, 0 }; ///< The reference image's size in beam units (frameDim at 1 + beamRefMargin), or frameDim when off.
    uint beamRefFrame = 0;         ///< 1 while the beam build runs in the reference frame rather than in screen space.
    uint beamRefValid = 0;         ///< 0 on the build that re-anchors the frame: nothing of the previous basis is addressable.
    /// The screen's box in the beam image, with a tile of guard. Only what it covers is built: the margin the frame carries so a
    /// turn does not leave it holds no basis until it turns on screen, and a direction that turns ON screen has no history to carry
    /// whatever the margin is. So a turn marches the entering strip and nothing else.
    float4 beamScreenBounds = { 0, 0, 0, 0 };
    /// The same box for the build that wrote the history. A beam unit only holds a marched value if the previous build covered it,
    /// so this is what tells the per-pixel fallback that a unit which has just turned on screen has nothing to carry.
    float4 beamPrevScreenBounds = { 0, 0, 0, 0 };
    /// 1 when the camera has not translated since the build that wrote the persistent lattice. A rotation alone cannot move a
    /// basis point that is addressed by direction, so such a build reads and writes nothing for a carried point.
    uint beamStationary = 0;
    /// The camera still has the anchor orientation, so reference and screen coordinates differ by translation alone.
    uint beamRefAligned = 0;
    /// Column window of the reference comparison, [low, high). high 0 means the whole row. A whole-frame aggregate cannot show
    /// that a localised band is clean, so this restricts the same statistics to one, e.g. the last beamTileSize columns.
    uint compareColumnLow = 0;
    uint compareColumnHigh = 0;
    /// Tile-space origin of the grid query dispatch, so it can cover a REGION of the beam image rather than all of it.
    ///
    /// MEASURED, and bounding it to the screen box is worth NOTHING on its own: parked 0.91 -> 0.90 ms, yaw 1.87 -> 1.87, every
    /// prebuilt arm unchanged. The threads beamPointOnScreen rejects were already almost free - they exit after two reads - so
    /// the query pass's cost is not them. It is the threads that PASS the screen test and then find their point already valid,
    /// and those are inside the box by definition, which no dispatch bound can help. Removing that cost needs a compact list of
    /// the points that actually need work (the refresh phase, the entering strip, parallax failures); this origin is kept as the
    /// mechanism such a list would dispatch through, not as an optimisation in its own right.
    uint2 beamGridOrigin = { 0, 0 };
    /// When non-zero the grid query dispatch is GENERATED rather than swept: beamGridOrigin is the first refresh block instead of
    /// a point origin, and threads map through beamGridBlockStride to the blocks this build's refresh phase selects. Everything
    /// else in the image is carrying a value it already holds, which is exactly what the sweep was launching threads to discover.
    uint beamGridBlocks = 0;
    uint beamGridBlockStride = 1;
    /// 1 when the level-0 tile pass is generated from the same regions as the query dispatch rather than swept over every tile.
    uint beamTileGenerated = 0;
    /// 1 when the residual march covers the same regions as the query and tile passes instead of the whole beam image. A unit
    /// can only need marching where its tile's classification or its own carry could have changed, which is those regions.
    uint beamUnitGenerated = 0;
    /// PROBE, not a feature. 1 routes every lattice read through a page table instead of addressing the texture directly. The
    /// table is filled with the identity, so the data, the layout and every result are unchanged and only the cost of the
    /// indirection is measured: one dependent buffer read plus the page arithmetic, in a path the resolve walks thirteen times
    /// per pixel. A sparse paged direction atlas - the design that removes re-anchoring without allocating the sphere - cannot
    /// be built if that indirection is expensive here, and this measures it before anything is built on top of it.
    uint beamPageIndirect = 0;
    uint beamPageShift = 4;            ///< Lattice points per page edge, as a shift.
    uint2 beamPageDims = { 0, 0 };     ///< Pages covering the lattice.
    /// 1 when the beam image IS the octahedral sphere instead of a perspective rectangle anchored to a camera pose.
    ///
    /// Every world direction then has one fixed texel for as long as the renderer runs, so there is nothing to re-anchor: a turn
    /// changes which already-built directions are resolved and nothing else. That is the whole point. A re-anchor today discards
    /// a cache that is entirely correct - at 0.05 rad/frame a 65 degree view exposes 4.4% new directions per frame and the frame
    /// re-anchors every third one, re-marching 100% of the image - and the level-0 query pass it rebuilds is 4.16 ms of a 6.84 ms
    /// fast turn.
    ///
    /// The image stays a rectangle, so the lattice, the tiles, the level map, the residual, the carry and the generated dispatch
    /// are untouched: only beamRayDirection, beamImagePosition and beamPixelPosition change, because those are the only places
    /// the beam pipeline talks about the world. The octahedral map is continuous across z = 0 and costs no transcendentals (abs,
    /// divide, select); its only discontinuity is the square's border, which is the two half great circles x = 0 and y = 0 of the
    /// -z hemisphere. The frame is oriented with +z at the zenith, so that seam lies below the horizon.
    uint beamOct = 0;
    uint beamOctFull = 0;              ///< Build the whole sphere rather than the screen's footprint (isolates bounding bugs).
    /// Which world axis the octahedral map's +z points along: 0 the zenith, 1 world +x, 2 world +z.
    ///
    /// The map is continuous but its DERIVATIVE kinks on the great circle perpendicular to this axis, where a tile straddling it
    /// has a ray field the bilinear-plus-bubble basis cannot fit. With +z at the zenith that circle is the horizon, which is
    /// where a cloud camera spends all of its time; pointing the axis along the horizon instead trades it for a vertical circle
    /// that the view crosses at two azimuths. This exists to measure whether that kink is the octahedral image's remaining
    /// quality deficit before resolution is spent on it.
    uint beamOctAxis = 0;
    /// 1 when a failed tile's pixels are marched in SCREEN space rather than read out of the beam-image residual.
    ///
    /// A screen pixel never lands on an octahedral texel, so the resolve's bilinear over four marched units is a resampling the
    /// anchored rectangle never paid: measured, it alone scores 0.511% of pixels over 0.02 where the aligned rectangle scores
    /// 0.000%. Marching the pixel's own ray removes it outright instead of buying it back with resolution, at the cost of losing
    /// the residual's carry - every failed pixel marches every build. That is the trade this exposes: marching time against the
    /// memory a higher-resolution sphere would need.
    uint beamScreenResidual = 0;
    /// PROBE, not a feature: 1 lets a TRANSLATED point take the stationary carry path without testing its parallax.
    ///
    /// The image it produces is wrong - nothing is ever invalidated by moving - and the point is the timing. Measured, 98.93% of
    /// points survive translation at 20 units per frame, yet every one of the 1.067M swept points runs beamReprojectQuery, which
    /// reads a thirteen-sample neighbourhood to discover it. A per-block parallax certificate would let a surviving block skip
    /// that read entirely, and this prices what such a certificate could win before any of it is built.
    uint beamAssumeCarry = 0;
    /// 1 when a translated point may be certified as surviving by its BLOCK instead of by testing itself.
    ///
    /// The renderer asks a million points whether translation invalidated them to find the eleven thousand that it did - 98.93%
    /// survive at 20 units per frame - and each question reads a thirteen-sample neighbourhood through beamReprojectQuery. A
    /// block keeps the smallest distance any of its points has stored and the camera position it was last verified from. The
    /// stored depth of a sample and the depth predicted for it can then differ by at most |dP| * theta over the block's angular
    /// reach, so while
    ///
    ///     |P - verifiedFrom| * beamGuardAngle <= beamDepthTolerance * blockMinDistance
    ///
    /// every point in it still agrees and none of them need to be asked. Measured ceiling for this, with the test skipped
    /// outright: the level-0 query pass goes 0.86 ms to 0.07 at fly 20, and the whole frame 2.09 to 1.15.
    uint beamGuard = 0;
    /// Angular reach of a block in radians: its points, plus the two tiles a reconstruction can read past them. In the octahedral
    /// image a texel is anisotropic, so this uses the LARGEST singular value of the map - 2.36 at the diamond mid-edge - because
    /// the bound has to hold along the worst axis rather than on average.
    float beamGuardAngle = 0.f;
    /// Normalised-device units a single beam texel can span on screen, used to widen the octahedral on-screen test by the angular
    /// radius of whatever box is being tested. Zero disables the exact test and falls back to the bounding box.
    ///
    /// The octahedral footprint of the screen is CURVED, so bounding it with an axis-aligned box and padding that box by the
    /// largest gap between sampled directions over-covers badly - measured, a 0.05 rad/frame turn exposes 4.4% of the screen per
    /// frame and the residual was marching about ten times that. Projecting the direction itself costs three dot products and has
    /// no such slack.
    float beamOctNdcPerTexel = 0.f;
    /// The guard's block grid, and the block's edge in beam units, for the dispatch that covers only the blocks that failed.
    /// beamGuardDims starts a row deliberately: a uint2 is eight-byte aligned in the shader and placing it mid-row inserts
    /// padding the host struct does not have, which the layout check catches as a four-byte difference.
    uint2 beamGuardDims = { 0, 0 };
    uint beamDirtyBlockEdge = 0;
    uint beamDirtyCapacity = 0;
    /// Lattice points per axis in a guard block, which is NOT beamRefreshBlock. A block is beamRefreshBlock points of the LATTICE,
    /// the lattice is one point per beamLatticeStep beam pixels, and a query point is one per beamTileSize pixels - so the points
    /// per block are beamRefreshBlock * beamLatticeStep / beamTileSize. With a lattice step of 2 and a tile of 4 that is 2, not 4,
    /// and assuming 4 put every dirty point outside the tile grid: 77,895 threads dispatched, none of them in range, the blocks
    /// never verified, and so re-listed as invalidated on every frame for ever.
    uint beamDirtyPointStride = 1;
    /// The coarse level over the guard blocks: one cell per 1 << beamCoarseShift blocks on a side.
    ///
    /// A leaf certificate says the block is safe while |P - C| <= R. A coarse cell holds the same pair, chosen so that its
    /// sphere lies inside every one of its children's: anchored at the camera that last rebuilt it, with
    /// R = min over children of (R_child - |C_child - A|), which is the triangle inequality and so is exactly conservative.
    /// One test then answers for all its children, and - just as importantly - the octahedral on-screen test, which costs a
    /// decode and a projection, is paid once per cell instead of once per block.
    /// beamCoarseShift sits here, immediately after beamDirtyPointStride, so that the uint2 below starts eight-byte aligned. A
    /// uint2 placed mid-row is padded in the shader and not on the host, which the layout check reports as a four-byte difference.
    /// Two, so a cell is 4 x 4 blocks and its uncovered children fit a sixteen-bit mask (kBeamCoarseChildren). At three (8 x 8)
    /// one unverified child zeroed a cell of sixty-four: 720p oct walk, beamDirty 0.61 ms of a 0.62 ms query.
    uint beamCoarseShift = 2;
    /// First block the classification covers, when a caller bounds it. Kept at zero: see the note beside the classify dispatch.
    uint2 beamGuardOrigin = { 0, 0 };
    uint2 beamCoarseDims = { 0, 0 };
    uint beamCoarseCapacity = 0;
    /// Threads per ray in the dirty query: each marches an equal slice of the ray's volume interval from unit transmittance and
    /// the first composes them front to back. Rejected once as beamSegments on the full lattice (4 -> 1 took that pass 11.0 ->
    /// 5.4 ms) because slices lose early termination and so duplicate work - but that pass was a million rays and throughput-
    /// bound. The dirty query is a few hundred rays, a few waves on a GPU of thousands of lanes, so it costs the latency of its
    /// longest ray, and duplicated work is nearly free where latency is the whole cost. Must divide 64.
    ///
    /// MEASURED, 720p oct, bracketed (seg 1 run twice, warm-up arm first), query ms: walk 1 -> 0.34/0.28, 16 -> 0.15, 32 -> 0.090,
    /// 64 -> 0.055, frame 0.45 -> 0.19; sprint 1 -> 0.045, 64 -> 0.022 with the frame flat (0.55 / 0.52). Error statistics equal to
    /// every printed digit in every arm (not a pixel diff: slices restart marchBeam's state and its step and opacity caps, see
    /// buildBeamDirtyQueries). 64 is the ceiling the clamp allows and scaling had still not flattened there - at ~560 rays, i.e.
    /// ~36k threads. With thousands of dirty rays the query turns throughput-bound again, where slices are duplicated work.
    ///
    /// DEFAULT 1, and above 1 only a compiled variant (HSTR_BEAM_DIRTY_SLICES) so the shipped query carries no slice composition.
    /// MEASURED 4K, OCT_T001, beamGuardParallax 1 (dirty_segment_budget.py), HSTRCloud ms / query ms / >0.02 against the exact march:
    /// walk 1 slice 28.2 / 6.4 / 0.136%, 64 slices 62.0 / 38.2 / 0.554% (p99.9 0.026 -> 0.036); sprint 31.4 / 6.1 / 0.025% against
    /// 81.7 / 53.1 / 0.156%. Park and look never run the query (the sphere is prebuilt). So at 4K the query is always throughput-
    /// bound, and slicing also costs quality: the per-slice restarts are not negligible there. A per-build adaptive count (the most
    /// slices keeping rays x slices within a thread budget of 32k-256k) chose 1 in every one of those cases and was removed.
    uint beamDirtySegments = 1;
    /// Set by the host for the build that runs the fused dirty chain (runBeamDirtyFused): the units pass then marches only the
    /// units the fused kernel left (from hstrBeamFusedState[kFusedUnitNext]).
    uint beamDirtyFused = 0;
    /// The fused kernel builds the warp field (beamWarp without beamWarpAuto; under beamWarpAuto the GPU's decision says).
    uint beamFusedWarp = 0;
    /// DIAGNOSTIC (device-removed bisection): where runBeamDirtyFused stops - 0 all of it, 1 at once, 2 after the rays, 3 tile
    /// items without their tile tests, 4 everything but the units it marches itself.
    uint beamFusedStage = 0;
    /// PROBE (0 = off, compiled into the dirty query only as HSTR_ORDER_PROBE): counts over the rays the dirty query marches - steps,
    /// the longest ray, a log2 histogram, and how well the stored step count (lattice slice 2 .x) predicted them, with this many
    /// steps as the "long" threshold. See hstrBeamOrderProbe.
    uint beamOrderProbe = 0;
    /// Nonzero for the build that anchors the octahedral image when beamPrebuild is on: every block is built, not just the ones
    /// the screen touches (beamBoxOnScreen answers yes everywhere). See beamPrebuild on the host.
    uint beamBuildAll = 0;
    /// The parallax a guard block may accumulate since the camera it was marched from, in radians: beamGuardParallax texels of
    /// the beam image at its SMALLEST texel angle, so the bound holds along the axis where a radian is the most texels. A block
    /// stays valid while |P - capturedFrom| <= beamGuardParallaxAngle * blockMinDistance, because no point of it can have shifted
    /// by more than |dP| / distance. See beamCellCertified.
    float beamGuardParallaxAngle = 0.f;
    /// Levels of the guard's min-depth pyramid (hstrBeamGuardPyramid), and the one a build pass is writing.
    uint beamGuardPyramidLevels = 0;
    uint beamGuardPyramidLevel = 0;
    /// Changed-content regions in hstrBeamInvalidations for this build's invalidateBeamGuardBlocks.
    uint beamInvalidateCount = 0;
    /// beamWarpAuto: the share of classified on-screen guard blocks that have to be held for the warp to run (0: always); see
    /// decideBeamPolicy.
    float beamWarpAuto = 0.f;
    /// beamPolicy: the guard-driven beam view spends the error its held blocks do not, from the held share h this build classified
    /// (decideBeamPolicy): s = saturate((beamPolicyHeldHigh - h) / (beamPolicyHeldHigh - beamPolicyHeldLow)), and the tile
    /// tolerance lerp(beamTolerance, beamPolicyTolerance, s). 0: off.
    uint beamPolicy = 0;
    float beamPolicyTolerance = 0.f;
    float beamPolicyHeldLow = 0.f;
    float beamPolicyHeldHigh = 1.f;
    uint beamPolicyPad0 = 0; ///< With beamPolicyPad1 and 2, keeps the fields after these on their 16-byte alignment.
    uint beamPolicyPad1 = 0;
    uint beamPolicyPad2 = 0;
    /// Levels a sample of an unbaked brick climbs for a baked ancestor before sunDepthAt's live march (cloudSunSlotUp).
    uint cloudSunAncestors = kCloudSunAncestors;
    /// PROBE (beamRepairProbe only): 1 + signature * kBeamProbePolicies + policy makes the dirty query keep the OLD lattice values of
    /// every block that witness policy would have carried, while still marching them all, so the image shows what the carry would
    /// cost before anything is skipped. A carried block keeps its old values and is certified from the current camera
    /// (beamShadowCertify). 0: off.
    uint beamShadowCarry = 0;
    /// Cell views: a world-space transport cache keyed by domain cell, each view the cell's transfer seen from the camera that built
    /// it. Its only reader, the per-ray composeBeam, was removed as the wrong architecture (see the note where it was), so nothing
    /// requests builds now; the structures stay for a projected cell x BeamTile consumer. 0: off.
    uint cellViews = 0;
    uint cellViewEdge = 16;          ///< Texels along each side of a view.
    uint cellViewCapacity = 16384;   ///< Views the atlas holds (a ring: a build takes the oldest slot).
    uint cellViewBuildCapacity = 4096; ///< Builds one frame may queue.
    /// Lateral misregistration a view may accumulate before it is marched instead, in voxels: the camera's movement across the
    /// ray, over the distance to the build camera, times half the chord. Past half of it a view is still used and rebuilt.
    float cellViewDrift = 0.05f;
    uint cellViewMaxAge = 256;       ///< Frames after which a view is rebuilt (cache and lighting changes it does not see).
    uint cellViewFrame = 0;          ///< Advances every frame that composes.
    uint cellViewSlotsPerRow = 1;    ///< Views per atlas row.
    float cellViewAtlasInvWidth = 0.f; ///< Scalars, not a float2: see beamCoarseShift on mid-row vectors.
    float cellViewAtlasInvHeight = 0.f;
    uint cellViewRefine = 1;         ///< Second lookup at the chord's opacity centroid (1) or only at its midpoint (0).
    /// PROBE (beamPushProbe): the push architecture's work count, with no radiance. Every occupied domain cell (hstrPushCandidates,
    /// listed when the occupancy changes) is culled, projected once into the octahedral beam image and appended to the list of
    /// every pushTileSize^2 beam-texel tile its projection overlaps that the screen can see. See projectPushCell.
    uint pushTileSize = 16;
    /// Distance shells: 0 bins every cell at pushTileSize. Otherwise a cell nearer than pushShellDistance / 2^(k-1) voxels is binned
    /// at tiles of pushTileSize << k (k < pushShells), so that it spans about two tiles of its shell whatever its distance.
    uint pushShells = 0;
    float pushShellDistance = 0.f;
    uint pushTileTotal = 0;          ///< Tiles over every shell's grid.
    /// pushEval: one sample direction per pushSampleStep^2 beam texels over the screen's box, from sample (origin + i + 0.5) * step.
    uint pushSampleStep = 8;
    uint pushSampleOriginX = 0;
    uint pushSampleOriginY = 0;
    uint pushSampleDimsX = 0;
    uint pushSampleDimsY = 0;
    /// pushEval's cell payload: 0 cell views; else the cell operator (buildPushOperator) with 1 its extinction and the exact source
    /// of the chord (an oracle), 2 a constant source, 3 a linear source, 4 trilinear emission. See evaluatePushSample.
    uint pushEvalMode = 0;
    uint pushSlotCapacity = 0;       ///< Visible cells a frame can hold operators for (hstrPushSlotKeys).
    uint pushCandidateCapacity = 0;
    uint pushPieceCapacity = 0;
    uint pushListCapacity = 0;
    uint pushDilate = 0;             ///< Candidates: cells within this many cells of an occupied one (see listPushCandidate).
    uint pushShareMode = 2;          ///< countPushShare's oracles: 1 the payloads (experiments 2, 3), 2 the temporal one (4), 4 certification (5).
    float pushShareMotionX = 0.f;    ///< The camera's last displacement per frame (world), for the temporal oracle's ages.
    float pushShareMotionY = 0.f;
    float pushShareMotionZ = 0.f;
    uint pushEvalExactDepth = 0;     ///< pushEval modes 2-4: the chord's exact transmittance under the operator's source (see pushOperatorChord).
};
/// beamPushProbe counters (hstrPushCounts).
static const uint kPushCandidates = 0;     ///< Occupied cells listed (27-voxel occupancy), by the last listing.
static const uint kPushCandidatesOwn = 1;  ///< Of those, cells whose own voxel holds density.
static const uint kPushCulledDistance = 2; ///< Past the sea's view distance (its fade is zero there).
static const uint kPushCulledFrustum = 3;  ///< Bounding sphere outside the view frustum.
static const uint kPushVisible = 4;        ///< Cells projected.
static const uint kPushNear = 5;           ///< Cells the camera is inside or touching (their projection is the whole screen).
static const uint kPushPieces = 6;         ///< Octant pieces projected (a cell spanning the camera's axis planes splits).
static const uint kPushOverlaps = 7;       ///< Cell x tile pairs binned (on-screen tiles, pieces deduplicated).
static const uint kPushRectTiles = 8;      ///< Four: overlaps a cell's projected box would have at tiles of 4, 8, 16, 32 texels.
static const uint kPushMaxPerCell = 12;
static const uint kPushTilesNonEmpty = 13;
static const uint kPushMaxPerTile = 14;
static const uint kPushTilesOnScreen = 15;
static const uint kPushDropped = 16;       ///< Overlaps or pieces beyond the capacities (the counts above still include them).
static const uint kPushNearOverlaps = 17;  ///< Overlaps of cells nearer than 8 voxels.
static const uint kPushMidOverlaps = 18;   ///< Overlaps of cells between 8 and 32 voxels.
static const uint kPushHistCell = 20;      ///< 32 bins: cells by floor(log2(overlaps)) + 1 (bin 0: none).
static const uint kPushHistTile = 52;      ///< 32 bins: on-screen tiles by floor(log2(cells)) + 1.
static const uint kPushHistDistance = 84;  ///< 16 bins: visible cells by floor(log2(camera distance in voxels)) + 1.
static const uint kPushMinDistance = 100;  ///< ~asuint of the nearest visible cell's distance in voxels (0: none), by InterlockedMax.
static const uint kPushSortOverflow = 101; ///< Tiles with more cells than a sort group holds (kPushSortMax), left unsorted.
static const uint kPushSortMax = 1024;
static const uint kPushMaxShells = 16;
static const uint kPushShellCells = 102;    ///< Per shell: cells binned there.
static const uint kPushShellOverlaps = 118; ///< Per shell: their overlaps.
static const uint kPushSortTiles = 134;     ///< Tiles with two or more cells, listed for the sort.
/// pushEval (evaluatePushSample, comparePushSample), per build: samples evaluated; list entries they read; cells their ray crossed
/// (a chord), composited from a view, of those empty, and marched as residual chords, with the residual's march steps; samples that
/// went opaque before their lists ended; views requested; samples with any residual chord. Against the exact march of the same
/// direction: samples over 0.02 in log(1 + radiance) (any channel) and in transmittance, the sum of the log error x 1e4, its
/// maximum (asuint), and the exact march's steps.
static const uint kPushEvalSamples = 135;
static const uint kPushEvalEntries = 136;
static const uint kPushEvalCrossed = 137;
static const uint kPushEvalViewHits = 138;
static const uint kPushEvalViewEmpty = 139;
static const uint kPushEvalResidual = 140;
static const uint kPushEvalResidualSteps = 141;
static const uint kPushEvalTerminated = 142;
static const uint kPushEvalRequests = 143;
static const uint kPushEvalResidualSamples = 144;
static const uint kPushEvalOver02 = 145;
static const uint kPushEvalOverT = 146;
static const uint kPushEvalErrorSum = 147;
static const uint kPushEvalErrorMax = 148;
static const uint kPushEvalExactSteps = 149;
static const uint kPushSlotsDropped = 150; ///< Visible cells past pushSlotCapacity (not binned at all).
static const uint kPushCountSlots = 151;   ///< Then [+0] pieces, [+1] list entries, [+2] visible slots.
/// pushShare (countPushShare, histogramPushShare), per frame: the dirty query points listed (over capacity too), the dirty units;
/// rays walked, those that went opaque, their list crossings and the march steps inside them, and the same rays' exact march steps
/// (what the crossings must account for). Per shell: the cell x tile interactions some dirty ray crossed, their crossings, their
/// steps. And interactions by floor(log2(rays crossing it)), with the steps inside them (the cost-weighted distribution).
static const uint kPushShareQueries = 155;
static const uint kPushShareUnits = 156;
static const uint kPushShareRays = 157;
static const uint kPushShareOpaque = 158;
static const uint kPushShareCrossings = 159;
static const uint kPushShareSteps = 160;
static const uint kPushShareExactSteps = 161;
static const uint kPushShareShell = 162;     ///< 16 shells x (interactions, crossings, steps).
static const uint kPushShareHistCount = 210; ///< 16 bins.
static const uint kPushShareHistSteps = 226; ///< 16 bins.
/// The same rays marched whole with the chords' marcher: its steps; rays whose chord composite differs from it by over 0.02 in
/// transmittance, and in log(1 + radiance); the optical depth it holds beyond the chords' (clamped to [0, 10]) and in total, x 1000.
static const uint kPushShareFullSteps = 242;
static const uint kPushShareOverT = 243;
static const uint kPushShareOver02 = 244;
static const uint kPushShareDepthMissing = 245;
static const uint kPushShareDepthTotal = 246;
/// Rays off screen (the guard's reach), those of them over 0.02 in transmittance, and the rays over it by floor(log2(voxels to the
/// whole march's opacity centroid)) + 1.
static const uint kPushShareOffscreen = 247;
static const uint kPushShareOffscreenOverT = 248;
static const uint kPushShareMissDistance = 249; ///< 16 bins.
/// Experiment 2, per payload variant (0: each ray's own chords marched alone; r > 0: a lattice of (2^(r-1) + 1)^2 chords per tile,
/// read bilinearly), against the chords' composite: rays over 0.02 in log radiance and in transmittance, the error sum x 1e4, and
/// the lattice chords' steps (four a crossing, so per interaction it is (2^(r-1) + 1)^2 / 4 of the crossings' mean).
static const uint kPushSharePayloads = 5;
static const uint kPushSharePayloadOver02 = 265;
static const uint kPushSharePayloadOverT = 270;
static const uint kPushSharePayloadErrorSum = 275;
static const uint kPushSharePayloadSteps = 280;
static const uint kPushSharePayloadResidual = 285; ///< Crossings no lattice corner's ray meets the cell for (the ray's own chord).
/// Experiment 3, per payload (3^2, 5^2) and tolerance (0.005, 0.01, 0.02): interactions whose every crossing is within it, and
/// their chord steps.
static const uint kPushShareSafe = 290;
static const uint kPushShareDepthExtra = 302; ///< Optical depth the chords hold beyond the whole march's, x 1000.
static const uint kPushShareOverTMore = 303;  ///< Of kPushShareOverT, the rays whose chords are the more transparent (depth missed).
/// Experiment 4, the temporal oracle (pushShareMode 2), per cache (the old ray through the chord's midpoint 1 frame old; that ray
/// snapped to its beam texel's centre 1 and 4 frames old - see countPushShare) and tolerance (0.005, 0.01, 0.02): interactions
/// whose every crossing the cached exact result answers, and their chord steps; the chord steps of the crossings it answers; and
/// rays whose composite of reused crossings is over 0.02, per age.
static const uint kPushShareAgeSafe = 304;           ///< 18: per age, per tolerance, interactions then steps.
static const uint kPushShareAgeCrossingSteps = 322;  ///< 9: per age, per tolerance.
static const uint kPushShareAgeOver02 = 331;         ///< 3.
static const uint kPushShareEntriesDropped = 334;    ///< Crossings of list entries past the share buffers.
/// The snapped caches (1, 4 frames): crossings by how far the snapped ray passes from the chord's midpoint, in fine voxels, bin
/// floor(2 log2(offset)) + 32 (clamped to 0-47, so bin 32 starts at one fine voxel); and the largest offset (asuint).
static const uint kPushShareSnapOffsets = 335; ///< 2 x 48.
static const uint kPushShareSnapMax = 431;     ///< 2.
/// Experiment 5, certification (pushShareMode 4): the snapped 1-frame cache with a central-difference Jacobian in the 4 ray
/// coordinates about it. Per tolerance (0.005, 0.01, 0.02), 8 chord-step counts: [0] the snapped value is within it, [1] the
/// Jacobian's prediction is; [2] the Jacobian's predicted change accepts the snapped value, [3] of those, wrongly; [4] the
/// one-scalar L1 sensitivity bound accepts it, [5] wrongly; [6] the jet's curvature estimate accepts its own prediction, [7] wrongly.
static const uint kPushShareCertify = 433;         ///< 24.
static const uint kPushShareCertifyOver02 = 457;   ///< 6: per guard (J change, L1, jet self), at tolerance 0.005 then 0.02.
static const uint kPushShareCertifyUnkeyed = 463;  ///< Chord steps of crossings whose old ray has no beam texel.
/// Experiment 5 runs on every kPushShareCertifyStride-th block of 64 dirty rays (9 chord marches a crossing on all of them trips the TDR):
/// that sample's chord steps and rays, the denominators of the counts above.
static const uint kPushShareCertifyStride = 8;
static const uint kPushShareCertifySteps = 464;
static const uint kPushShareCertifyRays = 465;
/// Experiment 6, fine-brick reuse (pushShareMode 8): every dirty chord re-walked at the march's step rule, one record per visit of
/// an atlas brick (hstrPushBrickVisits: listing tile, entry | samples << 20); records written, dropped past capacity, and the walk's
/// brick samples (against pushShareSteps: the walk's step placement is the march's, not its exact steps).
static const uint kPushShareVisits = 466;
static const uint kPushShareVisitsDropped = 467;
static const uint kPushShareVisitSamples = 468;
/// The same walk's steps by what answered them: all steps; no majorant (air); an empty fine brick; the proxy with density; the proxy
/// at zero (the fine-brick samples are kPushShareVisitSamples).
static const uint kPushShareWalkSteps = 469;
static const uint kPushShareWalkAir = 470;
static const uint kPushShareWalkEmpty = 471;
static const uint kPushShareWalkProxy = 472;
static const uint kPushShareWalkProxyZero = 473;
/// Lane slots the lean march's waves issue: per wave, its longest ray's steps (contributing samples) times the wave's lanes.
static const uint kPushShareWalkWaveSteps = 474;
static const uint kPushShareWalkWaveLit = 475;
/// Contributing samples of the lean march by kind: alpha < 1e-3, alpha in [1e-3, 1e-2), step at maxStepVoxels, step at its
/// minimum, and transmittance in front below 0.25 - where an error-bounded or transmittance-scaled step could take fewer.
static const uint kPushShareWalkLitFaint = 476;
static const uint kPushShareWalkLitThin = 477;
static const uint kPushShareWalkLitAtMax = 478;
static const uint kPushShareWalkLitAtMin = 479;
static const uint kPushShareWalkLitDeep = 480;
static const uint kPushCountTotal = 481;

/// Span probe (hstrSpanCounts): rays recorded, spans, samples recorded, rays past kSpanPerRay spans (not evaluated), rays whose
/// evaluated T / single light differ from the march's, brick / proxy spans, samples evaluated, baked-sun samples, the largest
/// |dT| and relative |dL| (as float bits), sun fallbacks (no bake: the march's own fallback).
static const uint kSpanPerRay = 12;
static const uint kSpanRayWords = 5; ///< uint4s a ray: position | worldLength, direction | entryDistance, spans | T | opacity distance,
                                     ///< the march's single light, the evaluator's light | T.
static const uint kSpanRays = 0;
static const uint kSpanSpans = 1;
static const uint kSpanSamples = 2;
static const uint kSpanOverflow = 3;
static const uint kSpanMismatchT = 4;
static const uint kSpanMismatchL = 5;
static const uint kSpanBrickSpans = 6;
static const uint kSpanProxySpans = 7;
static const uint kSpanEvaluated = 8;
static const uint kSpanBaked = 9;
static const uint kSpanMaxDT = 10;
static const uint kSpanMaxDL = 11;
static const uint kSpanFallback = 12;
static const uint kSpanCountTotal = 16;

/// Per-tile camera basis: L(x, y) = C0 + Cx x + Cy y + Cxy x y + Cb 16 x (1 - x) y (1 - y) over the tile's unit square.
/// rgb is premultiplied cloud radiance and a is opacity, so a pixel reconstructs its source radiance as L.rgb / L.a and
/// multiplies it by its own full-resolution opacity.
struct HSTRTileBasis
{
    float4 c0;
    float4 cx;
    float4 cy;
    float4 cxy;
    float4 bubble;
    float4 info; ///< x: relative surplus, y: refined flag, z: minimum corner opacity, w: unused.
};

END_NAMESPACE_FALCOR
