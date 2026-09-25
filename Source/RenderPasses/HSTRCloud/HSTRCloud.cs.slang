/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
import Utils.Geometry.IntersectionHelpers;
import Utils.Math.Ray;
import Scene.Scene;
import HSTRCloudTypes;

/// 0 compiles sunDepthAt's live near march out (cloudSunLiveMarch): the camera program is the one the frame waits for, and every
/// register its fallback path needs costs the march occupancy. MEASURED (4K c15, settled): near 46.2 -> 38.8 ms, sea 31.3 -> 33.4 ms,
/// with identical frames - in a settled view every sun query is answered by a bake, so the march below never runs and only its
/// registers are felt.
///
/// MEASURED and REMOVED (4K beam dirty passes, 2026-09-22): deferring the fallback out of the dirty marches to a second pass that
/// marched only the samples reaching it. It looked worth 4.6 ms at walk, but that was the live marches themselves, not their
/// registers: the sun slot pool left 1-7% of re-marched samples without a bake two levels up (kCloudSunAncestors), and once they
/// climb to the root the deferred and inline programs time the same (walk 12.57 / 12.75 ms, sprint 16.14 / 16.29).
#ifndef HSTR_SUN_LIVE
#define HSTR_SUN_LIVE 1
#endif

/// A mask of groups of the march's experimental switches compiled to their shipping values (HSTRCloud::beamShipping says when that
/// is exact): a cbuffer value cannot be folded, so every one of them otherwise stays a live branch in the beam programs. The groups
/// exist because folding them all slowed one view while speeding up another.
#ifndef HSTR_SHIP
#define HSTR_SHIP 0
#endif
/// beamWarp: the resolve reads a held guard block where its capture camera saw the content (beamWarpPosition).
/// PROBE (mBeamStripProbe): the dirty query and unit passes' timing copies. 1 returns after the ray setup, 2 marches one step,
/// 3 the full march; each computes and never writes (beamStripSink). 0: the real passes.
#ifndef HSTR_STRIP
#define HSTR_STRIP 0
#endif
#ifndef HSTR_BEAM_WARP
#define HSTR_BEAM_WARP 0
#endif
#ifndef HSTR_BEAM_SPARSE_MIN_LEVEL
#define HSTR_BEAM_SPARSE_MIN_LEVEL 2
#endif
#ifndef HSTR_BEAM_SPARSE_CUT
#define HSTR_BEAM_SPARSE_CUT 1
#endif
/// beamRepairProbe: scores what a repair could predict for each re-marched sample. Set by the host on the dirty query and the
/// dirty unit march only, so no shipping program carries it.
#ifndef HSTR_BEAM_REPAIR_PROBE
#define HSTR_BEAM_REPAIR_PROBE 0
#endif
#ifndef HSTR_PUSH_SHARE
#define HSTR_PUSH_SHARE 0 // The dirty query lists the points it marches for countPushShare.
#endif
#define SHIP_BIT_cloudTransferClasses 1
#define SHIP_BIT_cloudCostProbe 2
#define SHIP_BIT_cloudEmptySkip 4
#define SHIP_BIT_cloudSunCache 8
#define SHIP_BIT_adaptiveMarch 16
#define SHIP_BIT_cloudTrapezoid 32
#define SHIP_BIT_cloudLocalStep 32
#define SHIP_BIT_cloudSourceLinear 32
#define SHIP_BIT_cloudStepFootprint 64
#define SHIP_BIT_cloudTightReject 128
#define SHIP_BIT_cloudQuadrature 128
#define SHIP_BIT_cloudLongitudinalOracle 128
#define SHIP_BIT_cloudOracleCentroid 128
#define SHIP_BIT_cloudSunReuse 256
#if HSTR_SHIP == 0
#define SHIP(name, value) (params.name)
#else
#define SHIP(name, value) ((((HSTR_SHIP) & SHIP_BIT_##name) != 0) ? (value) : (params.name))
#endif

static const uint kMaxTraceDofs = 24;
static const uint kMaxTileNodes = 128;
static const uint kSortGroupSize = 64;
static const float kMinTransmittance = 1e-3f;
static const float kPi = 3.14159265f;

/// A GPU brick's sun scheduler entry, from its two tables (HSTRCloudRenderer::sunSched).
struct SunSchedEntry
{
    float priority; ///< The last cut's priority, at bin resolution (bin / 16).
    uint flags;     ///< Orientation classes that desired it (bits 0-15), kCloudSunInCut, kCloudSunMapped.
    uint coord;     ///< BrickHeader::coord: level-local brick coordinates, 10 bits per axis.
    uint asset;
};

struct HSTRCorrectionAtom
{
    uint row;
    uint col;
    float value;
    float goalError;
};

/// One reference path's radiance, split into the kComponent* components.
struct ReferenceSample
{
    float3 background;
    float3 sunSingle;
    float3 sunMultiple;
    float3 sky;

    /// Component by bit position.
    float3 component(uint c) { return c == 0 ? background : (c == 1 ? sunSingle : (c == 2 ? sunMultiple : sky)); }
};

/// One photon of the persistent light-tracing pool.
struct Photon
{
    float4 positionWeight;      ///< Voxel position and emission weight.
    float4 directionThroughput; ///< World direction and Russian-roulette throughput.
    uint rng;
    uint state; ///< Bounce count, kPhotonSky, kPhotonAlive.
    uint2 pad;
};
static const uint kPhotonBounceMask = 0xFFFF;
static const uint kPhotonSky = 1u << 16;
static const uint kPhotonAlive = 1u << 17;
/// hstrBeamCoarseState: one bit per child of a 4 x 4 coarse cell (beamCoarseShift 2 is the most sixteen bits hold), and a flag.
static const uint kBeamCoarseChildren = 0xFFFF;
static const uint kBeamCoarseQueued = 1u << 16;
/// Root tiles around a dirty block whose test reads its basis (beamTransmittanceRange reaches two tiles), see testBeamDirtyTile.
static const uint kBeamDirtyApron = 2;

/// Reference averages of a set of components over the even-frame half, the odd-frame half, and both.
struct ReferenceHalves
{
    float3 first;
    float3 second;
    float3 both;
};

/// Split-frame terms along one camera ray, before albedo: world-cache radiance, single-scattered sun without its phase factor,
/// and optical depth (capped where the march terminates).
struct BeamSample
{
    float3 cache;
    float3 single;
    float depth;
    float distance; ///< Opacity-weighted distance from the camera (0: no opacity).
    float steps;    ///< March steps taken: the cost the next build predicts for a ray here (beamQueue).
};

/// Camera-ray sample of the adaptive cache march (integrateCachedSegment).
struct MarchSample
{
    float t;         ///< Ray parameter.
    float sigma;     ///< Extinction.
    float sunCache;  ///< Sun optical depth of the cache modulation (the voxel-centre field, capped as the deposits cap it).
    float sunSingle; ///< Sun optical depth of single scattering (sunDepthAt).
    float3 cache;    ///< Stored (demodulated) cache radiance towards the camera.
};

/// Light gathered over one camera segment from unit transmittance, and its optical depth.
struct SegmentLight
{
    float3 cache;
    float3 single;
    float depth;
};

static const uint kMarchStack = 8; ///< Pending right halves of bisected segments.

/// One photon's cache deposits in its current cell, summed in registers and flushed with atomics only when it leaves the cell:
/// in a dense cloud a photon scatters many times per cell, and per-segment atomics dominated the tracing cost.
struct DepositAccumulator
{
    int3 cell;       ///< Current cell (x < 0: none).
    uint field;      ///< 0: sun, 1: sky.
    float sums[16];  ///< Fixed-point sums of SH coefficients 0-8 and sun extension slots 9-15.
};

/// A primary ray restricted to the volume bounds, expressed in leaf space.
/// position is the entry point (t = 0), direction the entry-to-exit vector (t = 1).
struct CutRay
{
    float3 position;
    float3 direction;
    float worldLength;
    float entryDistance; ///< World distance from the camera to t = 0, for pixel footprints.
    float4 sunOctaves;    ///< Per-octave sun weight a^n p(theta; c^n g) for this ray's direction.
    float3 viewDirection; ///< Unit world-space direction of the camera ray.
};

/// Where a cloud density sample at asset source position x read: brick is set when it read the fine bricks of instance (entry at
/// level, for the footprint level desired). empty is set when the density there is exactly zero within the source-voxel box [lo, hi).
struct CloudSample
{
    bool brick;
    bool empty;
    bool cell;     ///< empty is a skippable cell of brick entry (cellCode: codes below it are skippable), not a missing octant.
    uint cellCode;
    uint bounds;   ///< Density bound codes around x: the 2-voxel cell (bits 0-3), its 4-voxel octant (4-7), the brick (8-11), and the
                   ///< cell's index in the brick (bits 12-13 x, 14-15 y, 16-17 z), which places the octant and brick boxes around it.
    float3 cellLo; ///< Corner of the cell's box in source voxels (its edge is 2 << level).
    float t;       ///< Ray parameter the sample was taken at (cloudLocalStep measures its step from there).
    float3 x;
    float3 lo;
    float3 hi;
    uint instance;
    uint entry;
    uint level;
    uint desired;
    uint mappedLevel; ///< Level of the brick the descent found, BEFORE the parent climb: level - mappedLevel is what the climb walked.
};

/// What the lean march (HSTR_SHIP bit 1024) keeps of a density sample for its sun lookup: the brick it read, or kCloudRefNone where
/// the proxy answered. Eight words instead of CloudSample's twenty-five, and only while the sample is being lit.
struct LeanHit
{
    uint entry;
    uint instance;
    uint levels; ///< level | desired << 8 | the instance's sunClass << 16.
    float3 x;
    /// The instance's sourceVoxelWorld and scale, taken while leanExtinction holds its record, so the lit sample's sun chain starts
    /// at the resolved bake instead of reloading the instance and then its asset.
    /// MEASURED (sunhit1-4, 4K, errors identical in every arm): a first version slower (walk 2.10 / 2.04 -> 2.13 / 2.07 ms) because
    /// it grew the instance record to 84 bytes, off its 16-byte alignment, for every density sample's instance load; at 80 bytes
    /// (scale took the occupied flag's word) within noise alone (sunhit2/3). With the march's per-ray registers cut - the
    /// policy's step scale out (sunhit3: sprint -0.07 ms) and voxelsPerUnit's reciprocal in place of two values - carry + reciprocal
    /// against reload in the same run (sunhit4): walk 2.066 / 2.033 -> 2.047 / 2.000 ms, sprint 2.414 / 2.416 -> 2.403 / 2.393.
    float scale;
    float near;
};

/// 4-bit bound on the stored density a brick cell's trilinear samples can read: 0 exactly zero, q in 1-14 at most 2^(2q - 30),
/// 15 unbounded.
uint cloudCellCode(float maximum)
{
    if (maximum <= 0.f)
        return 0u;
    return uint(clamp(ceil((log2(maximum) + 30.f) * 0.5f), 1.f, 15.f));
}

float cloudCellBound(uint code)
{
    return code >= 15u ? 1e30f : pow(2.f, 2.f * float(code) - 30.f);
}

struct HSTRCloudRenderer
{
    HSTRCloudParams params;
    RWStructuredBuffer<float4> hstrRadiance;
    StructuredBuffer<float> hstrLeafBasisLeft;
    StructuredBuffer<float> hstrLeafBasisRight;
    StructuredBuffer<float> hstrLeafBallistic;
    StructuredBuffer<float> hstrLeafNearScatter;
    StructuredBuffer<float> hstrLeafDiffuseLeft;
    StructuredBuffer<float> hstrLeafDiffuseRight;
    StructuredBuffer<uint> hstrLeafDiffuseRanks;
    StructuredBuffer<float> hstrLeafResidualBounds;
    StructuredBuffer<float4> hstrRootIncident;
    StructuredBuffer<float> hstrLeafAdjointResponses;
    StructuredBuffer<uint2> hstrCorrectionRanges;
    StructuredBuffer<HSTRCorrectionAtom> hstrCorrectionAtoms;
    StructuredBuffer<HSTRCutNode> hstrCutNodes;
    RWStructuredBuffer<HSTRCutNode> hstrDomainCutNodes;
    StructuredBuffer<uint> hstrCutNodeParents;
    RWStructuredBuffer<float4> hstrNodeProjection;
    RWStructuredBuffer<float2> hstrNodeDepth;
    RWStructuredBuffer<uint> hstrNodeOrder;
    StructuredBuffer<uint> hstrCutStatePrevious;
    RWStructuredBuffer<uint> hstrCutState;
    RWStructuredBuffer<uint> hstrTileNodeCounts;
    RWStructuredBuffer<uint> hstrTileNodes;
    Texture3D<float> hstrExtinction;
    Texture3D<float> hstrMajorant;
    Texture3D<float> hstrTightMajorant; ///< Undilated block maxima: bound the trilinear extinction inside the block only.
    Texture3D<uint> hstrOccupancy;      ///< 1 where a 16-voxel block holds any extinction.
    Texture3D<uint> hstrMajorantZero;   ///< 0 where every majorant block of a 16-voxel block is zero (camera marches cross it at once).
    // The sea's domain proxy, built on the GPU (HSTRCloudDomainFrame).
    Texture3D<float2> hstrDomainVolume;                 ///< Per domain voxel: unscaled mean density and conservative maximum.
    Texture3D<float2> hstrDomainBlocks;                 ///< Per majorant block: unscaled maximum and mean over its trilinear support.
    RWTexture3D<float2> hstrDomainVolumeOutput;
    StructuredBuffer<uint> hstrDomainStaged;            ///< Packed volumes of a batch's slots (CloudSea::TileVolume::packed).
    RWTexture3D<float> hstrExtinctionOutput;
    RWTexture3D<float2> hstrDomainBlocksOutput;
    RWTexture3D<float> hstrMajorantOutput;
    RWTexture3D<float> hstrTightMajorantOutput;
    RWTexture3D<uint> hstrOccupancyOutput;
    RWTexture3D<uint> hstrMajorantZeroOutput;
    StructuredBuffer<uint> hstrDomainRegions;           ///< Changed slots.
    StructuredBuffer<HSTRCloudDomainFrame> hstrDomainFrame;
    // Virtual cloud density (see CloudResidency): instances per sea tile, assets, the chunk directory and 4^3 page nodes whose
    // entries name the finest mapped brick, the resident bricks and their R8 atlas, and the staging of newly loaded bricks.
    StructuredBuffer<HSTRCloudInstance> hstrCloudInstances;
    StructuredBuffer<HSTRCloudAsset> hstrCloudAssets;
    StructuredBuffer<uint> hstrCloudDirectory;
    StructuredBuffer<uint> hstrCloudNodes;
    StructuredBuffer<HSTRCloudVirtualPage> hstrCloudPages;
    RWStructuredBuffer<HSTRCloudVirtualPage> hstrCloudPagesOutput;
    StructuredBuffer<uint> hstrCloudLevelPages;         ///< Resolved level pages (HSTRCloudAsset::levelPageBase).
    RWStructuredBuffer<uint> hstrCloudLevelPagesOutput;
    StructuredBuffer<HSTRCloudDirtyPageRegion> hstrCloudDirtyRegions;
    RWStructuredBuffer<uint> hstrCloudDirtyBits;
    RWStructuredBuffer<uint> hstrCloudDirtyPages;
    RWStructuredBuffer<uint> hstrCloudDirtyPageCount;
    RWStructuredBuffer<uint> hstrCloudDirtyPageArgs;
    StructuredBuffer<HSTRCloudBrick> hstrCloudBricks;
    Texture3D<float> hstrCloudAtlas;
    RWTexture3D<unorm float> hstrCloudAtlasOutput;
    StructuredBuffer<uint> hstrCloudPayload;                 ///< Packed coefficients of resident pages (DirectStorage writes them).
    RWStructuredBuffer<float> hstrCloudResiduals;            ///< Decoded residuals, 512 per staged brick (decodeCloudResiduals).
    StructuredBuffer<HSTRCloudStaging> hstrCloudStagingInfo;
    // Per GPU brick, two words: bit x + 4 y + 16 z set where the 2-voxel cell (x, y, z) of the brick's 4^3 can sample non-zero
    // density (occupancyCloudBricks).
    StructuredBuffer<uint> hstrCloudOccupancy;
    RWStructuredBuffer<uint> hstrCloudOccupancyOutput;
    /// Transfer-reuse probe (cloudTransferClasses): two mask words per (asset brick, source-space direction class) marking which of
    /// the brick's 64 cells a ray first entered it through, plus one global crossing counter at the end. A stored transfer operator
    /// would need one entry per bit set here and would replace one crossing each time, so crossings divided by bits set is the reuse
    /// the whole idea lives on.
    RWStructuredBuffer<uint> hstrTransferProbeOutput;
    // Baked sun optical depth per atlas texel (same slots as the density atlas) and this frame's bakes (bakeCloudSun).
    Texture3D<float> hstrCloudSunAtlas;
    RWTexture3D<float> hstrCloudSunAtlasOutput;
    StructuredBuffer<HSTRCloudSunBake> hstrCloudSunBakes;
    StructuredBuffer<uint> hstrCloudSunSlots; ///< Per GPU brick, kCloudSunBakesPerBrick pairs (key, sun atlas slot).
    /// Per GPU brick and orientation class (resolveCloudSunSlots, every frame): cloudSunSlotUp's answer - the sun atlas slot, and the
    /// brick whose bake it is with its level in bits 24-27.
    StructuredBuffer<uint2> hstrCloudSunResolved;
    RWStructuredBuffer<uint2> hstrCloudSunResolvedOutput;
    // GPU sun bake scheduling (CloudResidencyDesc::gpuSun): the scheduler owns the slot table, the bake frames and the free slots.
    RWStructuredBuffer<uint> hstrCloudSunSlotsOutput;
    RWStructuredBuffer<HSTRCloudSunBake> hstrCloudSunBakesOutput;
    RWStructuredBuffer<HSTRCloudBrick> hstrCloudBricksOutput; ///< advanceCloudFades only.
    StructuredBuffer<HSTRCloudFadeFrame> hstrCloudFadeFrame;
    StructuredBuffer<HSTRCloudSunSched> hstrCloudSunSched;  ///< Per GPU brick.
    StructuredBuffer<uint> hstrCloudSunState;               ///< Per GPU brick: flags and priority bin (kCloudSunBinShift).
    RWStructuredBuffer<uint> hstrCloudSunFrames;            ///< Per pair: the residency frame it was baked in.
    RWStructuredBuffer<uint> hstrCloudSunFree;              ///< Free sun atlas slots, a stack.
    RWStructuredBuffer<uint> hstrCloudSunFreeTop;           ///< Its size.
    RWStructuredBuffer<uint> hstrCloudSunCounters;          ///< kCloudSunCounters entries.
    RWStructuredBuffer<uint> hstrCloudSunHistogram;         ///< kCloudSunBins candidate bins, then kCloudSunBins holder bins.
    RWStructuredBuffer<uint> hstrCloudSunSelect;            ///< Candidate bin and count taken from it, holder bin and count, holders.
    RWStructuredBuffer<uint> hstrCloudSunNeeds;             ///< Per GPU brick: classes needing a bake (bits 0-15).
    RWStructuredBuffer<float4> hstrCloudSunCandidates;      ///< (brick, class, priority, -), cloudSunBakeMax of them.
    RWStructuredBuffer<uint> hstrCloudSunHolders;           ///< Pairs whose slots move to candidates, cloudSunBakeMax of them.
    StructuredBuffer<uint> hstrCloudSunRelease;             ///< Bricks whose bakes are freed this frame (unloaded).
    StructuredBuffer<float4> hstrCloudSunDirections;        ///< Per orientation class: sun direction in its source voxels.
    RWTexture2D<uint> hstrCloudSunField;                    ///< Sun bake invalidation epochs (R16), kCloudSunFieldWidth a row.
    StructuredBuffer<uint> hstrCloudSunFieldBlocks;         ///< Per asset, kCloudSunClasses entries: its level >= 2 block's first entry, or kCloudRefNone.
    StructuredBuffer<uint> hstrCloudSunFieldLevels;         ///< Per asset, kCloudSunFieldLevels entries: a level's offset in a block.
    StructuredBuffer<HSTRCloudSunChange> hstrCloudSunChanges; ///< This frame's residency changes (stampSunChanges).
    StructuredBuffer<uint> hstrCloudSunResets;              ///< First entries of the field ranges resetSunField restarts.
    StructuredBuffer<HSTRCloudSunFrame> hstrCloudSunFrameInfo; ///< This frame's scheduling (one element).
    StructuredBuffer<float> hstrCloudTileBatches; ///< Light-tracing batches deposited per sea tile since its content changed.
    StructuredBuffer<uint> hstrCloudTileReset;    ///< 1 for sea tiles whose world cache cells clearWorldCacheTiles clears.
    Texture3D<float4> hstrCameraLighting;
    RWTexture3D<float4> hstrCameraLightingOutput;

    // World-space residual pages: sun optical depth at every voxel, resident per leaf.
    Texture3D<float> hstrFineSun;
    RWTexture3D<float> hstrFineSunOutput;
    RWStructuredBuffer<uint> hstrLeafResidual;
    // Scatterer-weighted sun arrival for octaves 1-3 (rgb numerators, a weight), spread laterally.
    Texture3D<float4> hstrSunOctaves;
    RWTexture3D<float4> hstrSunOctavesOutput;

    // Camera basis.
    Texture2D<float4> hstrCameraQueries;
    RWTexture2D<float4> hstrCameraQueryOutput;
    Texture2D<float4> hstrTileCenters;
    RWTexture2D<float4> hstrTileCenterOutput;
    RWStructuredBuffer<HSTRTileBasis> hstrTileBasis;
    RWStructuredBuffer<uint> hstrTileState;

    // Path-traced reference: running sums per radiance component, in two halves that alternate by frame (slice
    // 2 * component + half, rgb sum and sample count), and per-row comparison against the HST frame.
    RWTexture2DArray<float4> hstrReferenceSum;
    RWStructuredBuffer<float4> hstrReferenceRowError;
    RWStructuredBuffer<float> hstrReferenceRowHistogram; ///< Exact comparisons: per row, kCompareBins - 1 log error bins, then the row's maximum.
    // World-space radiance cache experiment: 9 SH coefficients x RGB running sums per cell, and the light-tracing
    // batch's fixed-point deposits (9 SH coefficients for a unit sun, then 9 for a unit-colour sky, per cell).
    RWStructuredBuffer<float> hstrWorldCache;
    RWStructuredBuffer<int> hstrWorldCacheDeposit;
    // Persistent light-tracing pool and the number of photons it has emitted since the last restart.
    RWStructuredBuffer<Photon> hstrPhotonPool;
    RWStructuredBuffer<uint> hstrPhotonEmitted;
    // The cache means baked for the camera: slot s = 3k + c of the 27 in texture s / 4, channel s % 4, trilinear by hardware.
    Texture3D<float4> hstrWorldCacheTexture[kWorldCacheTextures];
    RWTexture3D<float4> hstrWorldCacheTextureOutput[kWorldCacheTextures];
    // Beam view: exact queries on the lattice of all tile corners and centres, slice 0 (cache rgb, optical depth), slice 1
    // (single scatter without phase, unused); the level that finalised every finest tile (beamLevels: marched per pixel);
    // the per-level lists of refined tiles (list beamLevels: finest tiles marched per pixel), their counts and indirect
    // arguments; and the stored exact frame for comparisons.
    Texture2DArray<float4> hstrBeamLattice;
    /// Identity page table for the indirection probe (params.beamPageIndirect).
    StructuredBuffer<uint> hstrBeamPageTable;
    /// Per block of lattice points: the smallest distance any of its points holds, and the camera it was last verified from.
    /// Together they certify that translation cannot have invalidated the block (params.beamGuard).
    RWTexture2D<uint> hstrBeamGuardDepth;
    /// Min-depth pyramid over the guard blocks, levels packed one after another (level 0 is hstrBeamGuardDepth as a float, never
    /// verified as unbounded). See beamCellRadius.
    RWStructuredBuffer<float> hstrBeamGuardPyramid;
    RWTexture2D<float4> hstrBeamGuardCamera;
    /// The blocks whose certificate failed this build, compacted, with the count and the indirect arguments that dispatch them.
    RWStructuredBuffer<uint> hstrBeamDirty;
    RWStructuredBuffer<uint> hstrBeamDirtyCount;
    RWStructuredBuffer<uint> hstrBeamDirtyArgs;
    /// Per guard block: the first dirty-list slot it was listed in this build, ~0 when not listed (cleared every build). Lets
    /// each tile the dirty tile pass covers be tested by exactly one block (beamDirtyTileOwner).
    RWTexture2D<uint> hstrBeamDirtyMark;
    /// The units the dirty march has to march, compacted (count in hstrBeamDirtyCount[1]), packed x | y << 16.
    RWStructuredBuffer<uint> hstrBeamDirtyUnits;
    /// World regions whose content (density or lighting) changed since the last build, as bounding spheres (xyz centre, w radius),
    /// params.beamInvalidateCount of them. See invalidateBeamGuardBlock.
    StructuredBuffer<float4> hstrBeamInvalidations;
    /// beamRepairProbe: the residual units and the lattice as they stood before this build re-marched them.
    Texture2D<float4> hstrBeamPixelsSnapshot;
    Texture2DArray<float4> hstrBeamLatticeSnapshot;
    /// beamRepairProbe: each block's guard camera before this build, i.e. where its old values were marched from, and the witness
    /// policies that would have carried it (kBeamProbeAcceptSignatures), for the unit march to score its residual against.
    Texture2D<float4> hstrBeamGuardCameraSnapshot;
    RWTexture2D<uint> hstrBeamProbeAccept;
    /// Cell views (cellViews, see buildCellViewTexel): per domain cell (wrapped into the domain), its view's slot plus one (0: none); per
    /// cell, the cellViewFrame + 1 it was last queued for a build in; the views; their texels in an atlas of cellViewEdge^2 tiles,
    /// (cache rgb, transmittance) and (single rgb, opacity centroid as a fraction of the chord); the cells queued this frame, how
    /// many, the ring cursor that hands out slots, and the build dispatch's arguments.
    RWTexture3D<uint> hstrCellMap;
    RWTexture3D<uint> hstrCellRequest;
    /// Per domain cell: 1 where a sample inside it can read density (any of its 27 voxels has a nonzero conservative maximum).
    Texture3D<uint> hstrCellOccupancy;
    RWTexture3D<uint> hstrCellOccupancyOutput;
    /// beamPushProbe (see projectPushCell): the occupied cells and how many ([0] listed, [1] of them with density of their own);
    /// this frame's projected pieces (cell key, first and past-last tile, the cell's first piece and shell << 27); per tile of
    /// every shell its count and then scatter cursor, and its list offset; the tiles to sort; the lists; counters; the indirect
    /// arguments (projection, scatter, sort).
    RWStructuredBuffer<uint> hstrPushCandidates;
    RWStructuredBuffer<uint> hstrPushCandidateCount;
    RWStructuredBuffer<uint4> hstrPushPieces;
    RWStructuredBuffer<uint> hstrPushTileCounts;
    RWStructuredBuffer<uint> hstrPushTileOffsets;
    RWStructuredBuffer<uint> hstrPushSortTiles;
    RWStructuredBuffer<float4> hstrPushSamples; ///< pushEval: per sample, the composited radiance and transmittance (a < 0: none).
    /// Per visible cell (the slot the lists hold): its packed key; and its operator, 27 points of packed halves (sigma, sun
    /// emission rgb, cache emission rgb), see buildPushOperator.
    RWStructuredBuffer<uint> hstrPushSlotKeys;
    RWStructuredBuffer<uint4> hstrPushOps;
    RWStructuredBuffer<uint> hstrPushList;
    RWStructuredBuffer<uint> hstrPushCounts;
    RWStructuredBuffer<uint> hstrPushArgs;
    /// Span probe (spanProbe, HSTR_SHIP bit 262144): every dirty ray the lean march takes, as its ordered runs of samples in one
    /// brick (or the proxy) - hstrSpanRays holds 4 words a ray (the ray, then its span count, T, opacity distance and single light
    /// as the march found them), hstrSpanRecords kSpanPerRay spans of 2 words a ray, hstrSpanCounts the counters (kSpan*) and
    /// hstrSpanArgs the evaluator's dispatch. evaluateSpans then integrates the same samples from the spans alone.
    RWStructuredBuffer<uint4> hstrSpanRays;
    RWStructuredBuffer<uint4> hstrSpanRecords;
    RWStructuredBuffer<uint> hstrSpanCounts;
    RWStructuredBuffer<uint> hstrSpanArgs;
    /// pushShare: the lattice points the dirty query marched this frame (packed beam position), and per list entry (cell x tile)
    /// the dirty rays that crossed it and the march steps they spent inside it. See countPushShare.
    RWStructuredBuffer<uint> hstrPushShareRays;
    RWStructuredBuffer<uint4> hstrPushShareEntries; ///< And z, w: the largest crossing error under the 3^2, 5^2 payloads (asuint).
    RWStructuredBuffer<uint4> hstrPushShareAges;    ///< Per list entry: the largest crossing error reusing 1, 4, 16 frames old.
    RWStructuredBuffer<uint2> hstrPushBrickVisits;  ///< pushShareMode 8: per brick visit of a dirty chord, tile id and entry | samples << 20.
    RWStructuredBuffer<CellView> hstrCellViews;
    Texture2D<float4> hstrCellTexels;
    Texture2D<float4> hstrCellTexelsSingle;
    RWTexture2D<float4> hstrCellTexelsOutput;
    RWTexture2D<float4> hstrCellTexelsSingleOutput;
    RWStructuredBuffer<int4> hstrCellBuild;
    RWStructuredBuffer<uint> hstrCellCounters;
    RWStructuredBuffer<uint> hstrCellCursor;
    RWStructuredBuffer<uint> hstrCellArgs;
    /// The coarse certificate level: xyz is the camera the cell was last rebuilt from, w the radius it is safe within.
    RWTexture2D<float4> hstrBeamCoarse;
    /// Per coarse cell: the children its certificate does NOT cover (low 16 bits) and whether it is queued for a rebuild (bit 16).
    RWTexture2D<uint> hstrBeamCoarseState;
    RWStructuredBuffer<uint> hstrBeamDescend;      ///< Guard blocks this build has to classify one by one.
    RWStructuredBuffer<uint> hstrBeamDescendCount; ///< [0] blocks in hstrBeamDescend, [1] cells in hstrBeamRebuild.
    RWStructuredBuffer<uint> hstrBeamRebuild;      ///< Coarse cells whose certificate this build has to rebuild.
    /// Reference frame: per 8 x 8 resolve group holding failed-tile pixels, its packed group position and a 64-bit pixel mask.
    RWStructuredBuffer<uint> hstrBeamResidualList;
    RWStructuredBuffer<uint> hstrBeamResidualArgs; ///< [0] groups queued, [1..3] the residual resolve's dispatch, [4..6] its warped one.
    /// beamWarpAuto (writeBeamWarpArgs): [0..2] the warp field's dispatch, [3..5] the warped resolve's, [6..8] the plain resolve's -
    /// one of the two resolves gets no groups - and [9] the decision (1: warp), kept across builds that do not classify.
    /// beamPolicy (decideBeamPolicy): [11] the tile tolerance, as float bits ([10] unused).
    RWStructuredBuffer<uint> hstrBeamWarpArgs;
    RWTexture2DArray<float4> hstrBeamLatticeOutput;
    /// beamWarp: per lattice point, the offset from its image position to where its content was captured (buildBeamWarpField).
    Texture2D<float2> hstrBeamWarpField;
    RWTexture2D<float2> hstrBeamWarpFieldOutput;
    Texture2D<uint> hstrBeamLevel;
    RWTexture2D<uint> hstrBeamLevelOutput;
    Texture2DArray<float4> hstrBeamLatticePrev; ///< beamRefresh: the previous build's lattice.
    Texture2D<uint> hstrBeamLevelPrev;          ///< beamRefresh: the previous build's level map.
    Texture2D<float4> hstrBeamPixelPrev;        ///< beamRefresh: the previous build's marched pixels (rgb radiance, a distance).
    Texture2D<float4> hstrBeamPixels;           ///< beamRefFrame: this build's marched beam units, which the resolve samples.
    RWTexture2D<float4> hstrBeamPixelOutput;    ///< beamRefresh: this build's marched pixels.
    RWStructuredBuffer<uint> hstrBeamLists;
    RWStructuredBuffer<uint> hstrBeamCounts;
    RWStructuredBuffer<uint> hstrBeamArgs;
    RWStructuredBuffer<uint> hstrBeamSparseCandidates; ///< Candidate tiles, partitioned by level.
    RWStructuredBuffer<uint> hstrBeamSparseArgs;       ///< emit and verify indirect arguments.
    RWStructuredBuffer<BeamResult> hstrBeamResults;
    RWStructuredBuffer<BeamTile> hstrBeamFinalTiles;
    RWTexture2D<uint> hstrBeamQueryMap; ///< Query ID per lattice coordinate; 0xffffffff is unclaimed.
    RWTexture2D<uint> hstrBeamTileMap;  ///< Final BeamTile ID per finest cell; sentinels are empty/fallback.
    RWStructuredBuffer<uint> hstrBeamQueue;       ///< beamQueue: kBeamQueueBuckets lists of beamQueueCapacity packed positions.
    RWStructuredBuffer<uint> hstrBeamQueueCounts; ///< beamQueue: entries per list.
    RWStructuredBuffer<uint> hstrBeamQueueArgs;   ///< beamQueue: dispatch arguments per list (bound by its own pass only).
    // Temporal beam tiles: last frame's lists and counts, and per finest cell the levels at which the tile starting there was
    // refined (bit per level), last frame's and this frame's.
    StructuredBuffer<uint> hstrBeamPreviousLists;
    StructuredBuffer<uint> hstrBeamPreviousCounts;
    Texture2D<uint> hstrBeamHistory;
    RWTexture2D<uint> hstrBeamHistoryOutput;
    // Full-resolution beam guide (beamGuidePixel).
    Texture2D<float4> hstrBeamGuide;
    RWTexture2D<float4> hstrBeamGuideOutput;
    Texture2D<float4> hstrExactFrame;

    SamplerState hstrExtinctionSampler;
    SamplerState hstrLinearSampler;
    SamplerState hstrLinearClampSampler;
    RWTexture2D<float4> color;
    RWTexture2D<float> transportError;
    RWTexture2D<float4> cutStats;

    float phaseHG(float cosine) { return phaseHG(cosine, params.anisotropy); }

    float phaseHG(float cosine, float g)
    {
        const float denominator = max(1e-4f, 1.f + g * g - 2.f * g * cosine);
        return (1.f - g * g) / (4.f * kPi * denominator * sqrt(denominator));
    }

    float densityAt(GridVolume volume, Grid grid, float3 worldPosition, inout Grid::Accessor accessor)
    {
        const float3 localPosition = mul(volume.data.invTransform, float4(worldPosition, 1.f)).xyz;
        return max(0.f, grid.lookupLinearIndex(localPosition, accessor)) * volume.data.densityScale * params.densityScale;
    }

    uint leafIndex(uint3 p) { return p.x + params.hstrLeafDims.x * (p.y + params.hstrLeafDims.y * p.z); }

    float3 sampleLeafFace(uint leaf, uint face, float3 localPosition)
    {
        const uint order = params.traceSpatialOrder;
        const uint faceDofs = params.hstrFaceDofs;
        const uint base = params.hstrTraceDofs * leaf + faceDofs * face;
        if (order == 1)
            return hstrRadiance[base].rgb;

        const uint faceAxis = face / 2;
        const uint tangent0 = faceAxis == 0 ? 1 : 0;
        const uint tangent1 = faceAxis == 2 ? 1 : 2;
        const float2 samplePosition = saturate(float2(localPosition[tangent0], localPosition[tangent1])) * order - 0.5f;
        const int2 sampleBase = int2(floor(samplePosition));
        const float2 f = frac(samplePosition);
        float3 result = 0.f;
        for (uint v = 0; v < 2; ++v)
            for (uint u = 0; u < 2; ++u)
            {
                const uint2 q = uint2(clamp(sampleBase + int2(u, v), int2(0), int2(order) - 1));
                const float2 w = select(uint2(u, v) == 0, 1.f - f, f);
                result += hstrRadiance[base + q.x + order * q.y].rgb * w.x * w.y;
            }
        return result;
    }

    float3 hstrField(float3 p, uint face)
    {
        const int3 base = int3(floor(p));
        const float3 f = frac(p);
        float3 result = 0.f;
        for (uint z = 0; z < 2; ++z)
            for (uint y = 0; y < 2; ++y)
                for (uint x = 0; x < 2; ++x)
                {
                    const int3 q = clamp(base + int3(x, y, z), int3(0), int3(params.hstrLeafDims) - 1);
                    const float3 w = select(uint3(x, y, z) == 0, 1.f - f, f);
                    const float3 localPosition = p - float3(q) + 0.5f;
                    result += sampleLeafFace(leafIndex(uint3(q)), face, localPosition) * w.x * w.y * w.z;
                }
        return result;
    }

    float hstrResidualBound(float3 indexPosition)
    {
        const float3 p = (indexPosition - float3(params.hstrGridMin)) / float(params.hstrCellWidth) - 0.5f;
        const int3 base = int3(floor(p));
        const float3 f = frac(p);
        float result = 0.f;
        for (uint z = 0; z < 2; ++z)
            for (uint y = 0; y < 2; ++y)
                for (uint x = 0; x < 2; ++x)
                {
                    const int3 q = clamp(base + int3(x, y, z), int3(0), int3(params.hstrLeafDims) - 1);
                    const float3 w = select(uint3(x, y, z) == 0, 1.f - f, f);
                    result += hstrRadiance[params.hstrTraceDofs * leafIndex(uint3(q))].a * w.x * w.y * w.z;
                }
        return result;
    }

    float3 hstrLighting(float3 indexPosition, float3 viewDirection)
    {
        const float3 p = (indexPosition - float3(params.hstrGridMin)) / float(params.hstrCellWidth) - 0.5f;
        const float3 outgoing = -normalize(viewDirection);
        float3 result = 0.f;
        result += abs(outgoing.x) * hstrField(p, outgoing.x < 0.f ? 0u : 1u);
        result += abs(outgoing.y) * hstrField(p, outgoing.y < 0.f ? 2u : 3u);
        result += abs(outgoing.z) * hstrField(p, outgoing.z < 0.f ? 4u : 5u);
        return result / max(1e-4f, abs(outgoing.x) + abs(outgoing.y) + abs(outgoing.z));
    }

    // ---------------------------------------------------------------------------------------------
    // Field lookups. v denotes voxel coordinates relative to the grid minimum (voxel i is centred at
    // v = i) and leaf-space positions are v / hstrCellWidth.

    /// Virtual instancing for a cloud sea. One VDB, hierarchy and cache stay resident while X/Z repeat around the camera.
    float3 seaVoxel(float3 v)
    {
        if (params.seaMode == 0)
            return v;
        const float3 dims = float3(params.hstrExtinctionDims);
        v.x -= floor(v.x / dims.x) * dims.x;
        v.z -= floor(v.z / dims.z) * dims.z;
        return v;
    }

    /// Transport extinction: the always-resident proxy (the scene grid's extinction, or the cloud sea domain's mean density).
    float extinctionAtVoxel(float3 v)
    {
        v = seaVoxel(v);
        return hstrExtinction.SampleLevel(hstrExtinctionSampler, (v + 0.5f) / float3(params.hstrExtinctionDims), 0.f);
    }

    /// Density of one resident brick at source position x (voxel centres at integers, i its voxel), before the density scale.
    float cloudBrickDensity(uint brickIndex, float3 x, int3 i)
    {
        const HSTRCloudBrick brick = hstrCloudBricks[brickIndex];
        if ((brick.info & kCloudBrickConstant) != 0)
            return brick.minValue;
        const uint level = brick.info & 15u;
        const float3 levelPosition = (x + 0.5f) / float(1u << level) - 0.5f;
        const int3 brickCoord = (i >> int(level)) >> 3;
        const float3 local = clamp(levelPosition - float3(brickCoord * 8), -0.5f, 7.5f);
        const uint shift = params.cloudAtlasShift;
        const uint mask = (1u << shift) - 1u;
        const uint3 slot = uint3(brick.slot & mask, (brick.slot >> shift) & mask, brick.slot >> (2u * shift));
        const float3 texel = float3(slot * kCloudBrickEdge) + 1.5f + local;
        const float t = hstrCloudAtlas.SampleLevel(hstrLinearClampSampler, texel * params.cloudAtlasInvSize, 0.f);
        return brick.minValue + brick.range * t * t;
    }

    /// Ray parameter up to which the camera extinction stays zero after a sample at ray parameter t returned box: the box's exit
    /// along the ray, but no further than where the footprint selects another level (or the proxy), where the zero no longer holds.
    /// Ray parameter (per unit of dx, from box.x) where a skippable cell box of a brick's 4^3 cells stops being followed by skippable
    /// cells of the same brick along source direction dx: the entry of the first cell that is not, or the brick's exit.
    ///
    /// MEASURED (4K near): worth almost nothing on its own - empty samples stayed at 26.8% of the density samples, so an empty cell
    /// is nearly always followed by an occupied one (a 2-voxel cell at the footprint's level is about a pixel wide, and the empty
    /// ones are the cloud's own surface). It costs one buffer read per empty sample and keeps the exit exact, so it stays.
    float cloudCellRunExit(CloudSample box, float3 dx)
    {
        const float size = box.hi.x - box.lo.x; // Cell edge in source voxels.
        const float3 origin = floor((box.lo + 0.5f) / (4.f * size)) * (4.f * size) - 0.5f;   // The brick's first cell corner.
        int3 cell = int3(floor((box.lo - origin) / size + 0.5f));
        const int3 stepDirection = int3(sign(dx));
        float3 next = 1e30f;
        float3 delta = 1e30f;
        for (uint a = 0; a < 3; ++a)
            if (abs(dx[a]) > 1e-12f)
            {
                next[a] = ((dx[a] > 0.f ? box.hi[a] : box.lo[a]) - box.x[a]) / dx[a];
                delta[a] = size / abs(dx[a]);
            }
        const uint base = box.entry * kCloudCellWords;
        float exit = 0.f;
        [loop]
        for (uint k = 0; k < 10; ++k)
        {
            const uint a = next.x <= next.y && next.x <= next.z ? 0u : (next.y <= next.z ? 1u : 2u);
            exit = next[a];
            cell[a] += stepDirection[a];
            if (cell[a] < 0 || cell[a] > 3 || exit >= 1e29f)
                return exit;
            const uint index = uint(cell.x + 4 * cell.y + 16 * cell.z);
            if (((hstrCloudOccupancy[base + (index >> 3u)] >> ((index & 7u) * 4u)) & 15u) >= box.cellCode)
                return exit;
            next[a] += delta[a];
        }
        return exit;
    }

    /// Ray parameter where a camera step from t whose majorant is zero may meet density again: the exit of t's 16-voxel block when every
    /// majorant block of it is zero (hstrMajorantZero), else t. Only in the sea's vertical range (its X/Z wrap).
    ///
    /// MEASURED (4K): zero-majorant steps 21.0 -> 14.1 per pixel on the near view and 27.0 -> 17.9 on the sea view, and pixels still
    /// marching after 32 steps 62% -> 49% (near) and 74% -> 41% (sea). Exact: the dilated majorant already covers the trilinear
    /// support, so a block of zero majorant blocks holds no density a sample could read.
    /// majorantZeroExit, and where the 16-voxel block is not all zero, the exit of the zero 4-voxel majorant block t lies in and of
    /// up to three more zero blocks behind it: a zero dilated majorant already bounds the trilinear support, so they hold nothing.
    float majorantZeroExitFine(CutRay ray, float t)
    {
        const float coarse = majorantZeroExit(ray, t);
        if (coarse > t)
            return coarse;
        const float3 dv = ray.direction * float(params.hstrCellWidth);
        const float nudge = 0.125f / max(length(dv), 1e-9f);
        float cursor = t;
        [loop]
        for (uint k = 0; k < 4; ++k)
        {
            const float3 p = seaVoxel((ray.position + (cursor + (k == 0 ? 0.f : nudge)) * ray.direction) * float(params.hstrCellWidth));
            const int3 b = int3(floor(p * 0.25f));
            if (any(b < 0) || any(b >= int3(params.hstrMajorantDims)) || hstrMajorant.Load(int4(b, 0)) > 0.f)
                break;
            const float3 lo = float3(b) * 4.f;
            float exit = 1e30f;
            for (uint a = 0; a < 3; ++a)
                if (abs(dv[a]) > 1e-12f)
                    exit = min(exit, ((dv[a] > 0.f ? lo[a] + 4.f : lo[a]) - p[a]) / dv[a]);
            cursor += (k == 0 ? 0.f : nudge) + max(exit, 0.f);
        }
        return max(cursor, t);
    }

    float majorantZeroExit(CutRay ray, float t)
    {
        const float3 v = seaVoxel((ray.position + t * ray.direction) * float(params.hstrCellWidth));
        const uint3 dims = (params.hstrMajorantDims + 3u) / 4u;
        const int3 block = int3(floor(v / 16.f));
        if (params.cloudZeroSkip == 0 || any(block < 0) || any(block >= int3(dims)) || hstrMajorantZero.Load(int4(block, 0)) != 0)
            return t;
        // A sea ray spends most of its steps crossing nothing - 17.81 of 30.00 per pixel - and no step length touches those, so the
        // only way to shorten them is to cross more per step. cloudZeroSkip is how many consecutive empty blocks one step may cross.
        // Each still costs the one load it costs now, but the march's own per-step work - the majorant fetch, the step rule, the loop
        // - is paid once rather than once per block. The cursor lands exactly on the last empty block's face, so nothing non-empty is
        // ever skipped; the quarter-voxel nudge only moves the probe off that face into the block being tested.
        const float3 dv = ray.direction * float(params.hstrCellWidth);
        const float nudge = 0.25f / max(length(dv), 1e-9f);
        float cursor = t;
        [loop]
        for (uint k = 0; k < params.cloudZeroSkip; ++k)
        {
            const float3 p = k == 0 ? v : seaVoxel((ray.position + (cursor + nudge) * ray.direction) * float(params.hstrCellWidth));
            const int3 b = k == 0 ? block : int3(floor(p / 16.f));
            if (k > 0 && (any(b < 0) || any(b >= int3(dims)) || hstrMajorantZero.Load(int4(b, 0)) != 0))
                break;
            const float3 lo = float3(b) * 16.f;
            float exit = 1e30f;
            for (uint a = 0; a < 3; ++a)
                if (abs(dv[a]) > 1e-12f)
                    exit = min(exit, ((dv[a] > 0.f ? lo[a] + 16.f : lo[a]) - p[a]) / dv[a]);
            cursor += (k == 0 ? 0.f : nudge) + max(exit, 0.f);
        }
        return max(cursor, t);
    }

    /// How far (in ray parameter) a march standing at a brick sample may go before its optical depth can exceed budget: the coarsest of
    /// the sample's nested boxes - its 2-voxel cell, the 4-voxel octant around it, the whole brick - whose density bound allows the
    /// whole crossing, else the length that bound allows inside the innermost one. The bounds hold only inside their own box, so every
    /// candidate is clamped to that box's exit: the step never outruns the bound that justified it.
    ///
    /// MEASURED (4K c15, 2026-09-16): near 39.7 -> 35.8 ms and sea 31.6 -> 28.5 ms, but against a 0.5-voxel reference it lands at
    /// 39.1 ms with 0.54% of pixels over 0.02 log error - where a flat 2-voxel step reaches 26.5 ms at 0.29%. Dominated on both axes,
    /// so it stays off (cloudLocalStep). The reason is in the profile: the samples the march spends its time on sit at cloud
    /// boundaries, and a conservative maximum over a cell that holds any dense texel is high, so the bounds only open up in the thin
    /// regions that were already cheap. A bound hierarchy cannot make boundary work sparser; only sampling them less often can.
    float cloudLocalStep(CutRay ray, CloudSample sample, float budget, float densityScale)
    {
        const HSTRCloudInstance instance = hstrCloudInstances[sample.instance];
        const float3 dv = ray.direction * float(params.hstrCellWidth);
        const float3 dx = float3(dot(instance.row0.xyz, dv), dot(instance.row1.xyz, dv), dot(instance.row2.xyz, dv));
        const float cellSize = 2.f * float(1u << sample.level);
        const int3 cell = int3((sample.bounds >> 12u) & 3u, (sample.bounds >> 14u) & 3u, (sample.bounds >> 16u) & 3u);
        // The optical depth of a box crossing is at most its bound times the world length of the crossing.
        const float perLength = densityScale * ray.worldLength;
        float allowed = 0.f;
        [unroll]
        for (uint k = 0; k < 3; ++k)
        {
            const float size = cellSize * float(1u << k);
            const float3 lo = sample.cellLo - cellSize * float3(cell & ((1 << int(k)) - 1));
            float exit = 1e30f;
            for (uint a = 0; a < 3; ++a)
                if (abs(dx[a]) > 1e-12f)
                    exit = min(exit, ((dx[a] > 0.f ? lo[a] + size : lo[a]) - sample.x[a]) / dx[a]);
            exit = max(exit, 0.f);
            const float bound = cloudCellBound((sample.bounds >> (4u * k)) & 15u) * perLength;
            const float step = min(exit, bound > 0.f ? budget / bound : 1e30f);
            allowed = max(allowed, step);
            if (step < exit)
                break; // The bound, not the box, is what stops this step: a coarser box can only bound worse.
        }
        // Past the footprint's next level the samples read another brick, whose density these bounds do not cover.
        if (params.cloudPixelAngle > 0.f && ray.worldLength > 0.f)
        {
            const float nextLevel = min(float(sample.level) + 1.f, instance.proxyLevel);
            const float nextFootprint = instance.sourceVoxelWorld * pow(2.f, max(nextLevel - params.cloudLodBias, 0.f));
            allowed = min(allowed, max(0.f, (nextFootprint / params.cloudPixelAngle - ray.entryDistance) / ray.worldLength - sample.t));
        }
        return allowed;
    }

    /// Octahedral direction class of a unit vector on an edge x edge grid: equal-area enough for counting, and continuous across the
    /// hemisphere boundary so neighbouring directions land in neighbouring classes.
    uint octahedralClass(float3 d, uint edge)
    {
        const float3 a = d / max(dot(abs(d), float3(1.f)), 1e-12f);
        float2 p = a.xy;
        if (a.z < 0.f)
            p = (1.f - abs(float2(a.y, a.x))) * float2(a.x >= 0.f ? 1.f : -1.f, a.y >= 0.f ? 1.f : -1.f);
        const float2 uv = saturate(p * 0.5f + 0.5f);
        const uint2 c = min(uint2(uv * float(edge)), edge - 1u);
        return c.x + edge * c.y;
    }

    /// Records one ray's entry into a brick for the transfer-reuse probe. The direction is taken in the asset's own frame, so the
    /// instance's quarter-turn and mirror fold into the key instead of multiplying it: two tiles of the sea that present the same
    /// brick at the same angle share one entry however they are oriented in the world.
    void recordTransferCrossing(CutRay ray, CloudSample sample)
    {
        const uint edge = SHIP(cloudTransferClasses, 0u);
        const HSTRCloudInstance instance = hstrCloudInstances[sample.instance];
        const float3 dv = ray.direction * float(params.hstrCellWidth);
        const float3 dx = float3(dot(instance.row0.xyz, dv), dot(instance.row1.xyz, dv), dot(instance.row2.xyz, dv));
        const uint direction = octahedralClass(normalize(dx), edge);
        // The 4 x 4 x 4 cell the sample sits in is already decoded in sample.bounds, and stands for where the ray came in.
        const uint cell = ((sample.bounds >> 12u) & 3u) + 4u * ((sample.bounds >> 14u) & 3u) + 16u * ((sample.bounds >> 16u) & 3u);
        const uint key = 2u * (sample.entry * edge * edge + direction);
        const uint bit = 1u << (cell & 31u);
        uint previous;
        InterlockedOr(hstrTransferProbeOutput[key + (cell >> 5u)], bit, previous);
        // A bit that was not set is an entry a transfer cache would have had to produce; every crossing is one it could serve.
        if ((previous & bit) == 0u)
            InterlockedAdd(hstrTransferProbeOutput[params.cloudTransferWords - 2u], 1u);
        InterlockedAdd(hstrTransferProbeOutput[params.cloudTransferWords - 1u], 1u);
    }

    float cloudEmptyExit(CutRay ray, float t, CloudSample box) { return cloudEmptyExitOf(ray, t, box, hstrCloudInstances[box.instance]); }

    /// cloudEmptyExit for a sample whose instance is already loaded.
    float cloudEmptyExitOf(CutRay ray, float t, CloudSample box, HSTRCloudInstance instance)
    {
        const float3 dv = ray.direction * float(params.hstrCellWidth);
        const float3 dx = float3(dot(instance.row0.xyz, dv), dot(instance.row1.xyz, dv), dot(instance.row2.xyz, dv));
        float exit = box.cell ? cloudCellRunExit(box, dx) : 1e30f;
        if (!box.cell)
            for (uint a = 0; a < 3; ++a)
                if (abs(dx[a]) > 1e-12f)
                    exit = min(exit, ((dx[a] > 0.f ? box.hi[a] : box.lo[a]) - box.x[a]) / dx[a]);
        exit = max(exit, 0.f);
        if (params.cloudPixelAngle > 0.f && ray.worldLength > 0.f)
        {
            const float footprintLevel = log2(max(rayFootprint(ray, t) / instance.sourceVoxelWorld, 1.f)) + params.cloudLodBias;
            const float nextLevel = min(floor(max(footprintLevel, 0.f)) + 1.f, instance.proxyLevel);
            if (nextLevel > footprintLevel)
            {
                const float footprint = instance.sourceVoxelWorld * pow(2.f, max(nextLevel - params.cloudLodBias, 0.f));
                exit = min(exit, (footprint / params.cloudPixelAngle - ray.entryDistance) / ray.worldLength - t);
            }
        }
        return t + max(exit, 0.f);
    }

    /// Camera extinction at domain voxel position v for a sample whose pixel footprint spans footprint world units: the finest
    /// mapped brick no finer than the footprint needs, blended against its parent while it streams, else the proxy. fineVoxels
    /// returns the sampled voxel edge in domain voxels (1 for the proxy), which bounds the next march step.
    float cameraExtinctionAtVoxel(float3 v, float footprint, inout float fineVoxels)
    {
        CloudSample sample;
        return cameraExtinctionAtVoxel(v, footprint, fineVoxels, sample);
    }

    /// cameraExtinctionAtVoxel that also returns where the sample read (the brick lookup sunDepthAt reuses) and, where the density is
    /// exactly zero, the source-space box around v in which it stays zero at v's footprint level (cloudEmptyExit).
    float cameraExtinctionAtVoxel(float3 v, float footprint, inout float fineVoxels, out CloudSample sample)
    {
        // The sea fades out over the last 30% of the view distance, so tiles entering the window never pop in.
        const float distanceFade = params.seaMode != 0 && params.cloudPixelAngle > 0.f
                                       ? saturate((1.f - footprint / (params.cloudPixelAngle * params.seaViewDistance)) / 0.3f)
                                       : 1.f;
        return distanceFade * cameraExtinctionUnfaded(v, footprint, fineVoxels, sample);
    }

    /// The sea tile instance around domain position v and v's asset source position x, for a sample whose footprint selects
    /// footprintLevel. false in the proxy regime: no virtual density, an empty tile, or a footprint as coarse as the proxy.
    bool cloudInstanceAt(float3 v, float footprint, out uint instanceIndex, out float3 x, out float footprintLevel)
    {
        HSTRCloudInstance unused;
        return cloudInstanceAt(v, footprint, instanceIndex, x, footprintLevel, unused);
    }

    /// cloudInstanceAt that also hands back the instance it loaded. Callers that go on to read the instance - which is every camera
    /// density sample that finds a brick - would otherwise fetch the same 64 bytes a second time on the next line.
    bool cloudInstanceAt(float3 v, float footprint, out uint instanceIndex, out float3 x, out float footprintLevel,
                         out HSTRCloudInstance instanceOut)
    {
        instanceIndex = 0;
        x = 0.f;
        footprintLevel = 0.f;
        instanceOut = {};
        if (params.cloudVirtual == 0)
            return false;
        const float tileVoxels = float(params.cloudTileVoxels);
        const float2 worldTile = floor((v.xz + 0.5f) / tileVoxels);
        const int2 tiles = int2(params.cloudTiles);
        // Euclidean wrap into the instance grid. For a power-of-two count the mask is exact for negative coordinates too, which is
        // what the general form below costs two emulated integer divisions to reproduce (cloudTileMask).
        const int2 slot = params.cloudTileMask != ~0u ? (int2(worldTile) & int2(int(params.cloudTileMask)))
                                                     : (int2(worldTile) % tiles + tiles) % tiles;
        instanceIndex = slot.x + tiles.x * slot.y;
        const HSTRCloudInstance instance = hstrCloudInstances[instanceIndex];
        instanceOut = instance;
        if (!(instance.scale > 0.f)) // An empty tile.
            return false;
        footprintLevel = log2(max(footprint / instance.sourceVoxelWorld, 1.f)) + params.cloudLodBias;
        if (footprintLevel >= instance.proxyLevel)
            return false;
        const float3 p = float3(v.x - worldTile.x * tileVoxels, v.y, v.z - worldTile.y * tileVoxels);
        x = float3(dot(instance.row0.xyz, p) + instance.row0.w, dot(instance.row1.xyz, p) + instance.row1.w, dot(instance.row2.xyz, p) + instance.row2.w);
        return true;
    }

    /// The brick a sample of asset at source position x with footprint level desired reads: the finest mapped brick no finer than
    /// desired (sample.entry, sample.level; sample.instance is left to the caller). false outside the asset (the proxy applies).
    /// Where the density is exactly zero sample.empty is set and sample.level is the level of the zero region's voxels: a missing
    /// child octant of a coarser brick, or an empty cell of the brick (checked only for settled bricks when settledOnly, since a
    /// fading brick also reads its parent).
    bool cloudAssetBrickAt(HSTRCloudAsset asset, float3 x, uint desired, bool settledOnly, out CloudSample sample)
    {
        return cloudAssetBrickAt(asset, x, desired, settledOnly, 0.f, sample);
    }

    /// cloudAssetBrickAt that also counts as empty the thin cells: those whose stored density bound is at most thinDensity / 2^level
    /// (thinDensity: the bound whose optical depth across a level-0 cell is the tolerance; cells of coarser levels are longer).
    /// thinDensity 0: only exactly empty cells.
    ///
    /// MEASURED (4K near, cloudThinDepth 0.001): density samples 28.85 -> 28.13 per pixel, of which the ones too faint to change the
    /// pixel (alpha <= 1e-6) only fell 25.0% -> 23.8%. Those samples are not in thin cells: they are the trilinear tails of cells
    /// that also hold dense texels, so a cell bound cannot reject them.
    ///
    /// That is true and it is not the whole story, because 0.001 is a tolerance far below anything that matters. RE-MEASURED on the
    /// sea's beam path against a drift anchor: 0.05 is x0.942 of the frame for 0.111% -> 0.129% over 0.02. An earlier sweep priced
    /// this at 0.70 ms by reading raw milliseconds from positions four and five of an eight-test run, which is halfway up a 12%
    /// thermal ramp. Still 0 by default - it costs quality - but it is one of the two cheapest things the sea sells.
    bool cloudAssetBrickAt(HSTRCloudAsset asset, float3 x, uint desired, bool settledOnly, float thinDensity, out CloudSample sample)
    {
        sample = {};
        sample.x = x;
        sample.desired = desired;
        sample.entry = kCloudRefNone;
        const int3 i = int3(floor(x + 0.5f));
        if (any(i < 0) || any(i >= int3(asset.dims)))
            return false;
        const uint3 brick = uint3(i) >> 3u;
        const uint3 brickDims = asset.dims >> 3u;
        const HSTRCloudVirtualPage page = hstrCloudPages[asset.pageOffset + brick.x + brickDims.x * (brick.y + brickDims.y * brick.z)];
        uint entry = page.physicalRef;
        if (entry == kCloudRefNone)
            return false;
        // 5 stops with the directory/node descent done and nothing after it, so mode 5 minus mode 3 is the descent alone - the part a
        // flat page table would replace - and mode 2 minus mode 5 is the parent climb, the octant test and the occupancy read, which
        // it would not. Returning false sends the caller to the proxy, which is what the other cost modes do.
        if (SHIP(cloudCostProbe, 0u) == 5)
            return false;
        sample.brick = true;
        uint level = page.mappedLevel;
        sample.mappedLevel = level;
        if (desired < level)
        {
            // The mapped brick is coarser than the footprint asks for: where its child does not exist the density is zero, for every
            // voxel of the missing child's octant (no descendant of an unmapped child is mapped).
            const uint3 child = (uint3(i) >> (level + 2u)) & 1u;
            const uint childBit = 1u << (child.x + 2u * child.y + 4u * child.z);
            if (((hstrCloudBricks[entry].info >> 4) & childBit) == 0)
            {
                sample.empty = true;
                sample.lo = float3((uint3(i) >> (level + 2u)) << (level + 2u)) - 0.5f;
                sample.hi = sample.lo + float(1u << (level + 2u));
                sample.level = level - 1u;
                return true;
            }
        }
        [loop]
        for (uint k = 0; k < 15 && level < desired && hstrCloudBricks[entry].parent != kCloudRefNone; ++k)
        {
            entry = hstrCloudBricks[entry].parent;
            level = hstrCloudBricks[entry].info & 15u;
        }
        sample.entry = entry;
        sample.level = level;
        // 6 adds the octant test and the parent climb to mode 5 but stops before the occupancy read, splitting the rest in two.
        if (SHIP(cloudCostProbe, 0u) == 6)
            return false;
        cloudBrickCellTest(entry, level, i, x, settledOnly, thinDensity, sample);
        return true;
    }

    /// cloudAssetBrickAt from the resolved level pages: one word instead of the page, the octant test and the parent climb. The
    /// same sample (sample.mappedLevel is the level read, not the page's finest), from the instance alone.
    bool cloudAssetBrickAtFlat(HSTRCloudInstance instance, float3 x, uint desired, bool settledOnly, float thinDensity, out CloudSample sample)
    {
        sample = {};
        sample.x = x;
        sample.desired = desired;
        sample.entry = kCloudRefNone;
        const int3 i = int3(floor(x + 0.5f));
        const uint3 pages = uint3(instance.pageDims & 1023u, (instance.pageDims >> 10) & 1023u, instance.pageDims >> 20);
        if (any(i < 0) || any(i >= int3(pages * 8u)))
            return false;
        const uint level = min(desired, instance.topLevel);
        uint base = instance.levelPageBase;
        for (uint l = 0; l < level; ++l)
        {
            const uint3 d = (pages + (1u << l) - 1u) >> l;
            base += d.x * d.y * d.z;
        }
        const uint3 levelDims = (pages + (1u << level) - 1u) >> level;
        const uint3 c = (uint3(i) >> 3u) >> level;
        const uint word = hstrCloudLevelPages[base + c.x + levelDims.x * (c.y + levelDims.y * c.z)];
        if (word == kCloudRefNone)
            return false;
        sample.brick = true;
        if ((word & kCloudLevelEmpty) != 0)
        {
            const uint mapped = word & 15u;
            sample.empty = true;
            sample.lo = float3((uint3(i) >> (mapped + 2u)) << (mapped + 2u)) - 0.5f;
            sample.hi = sample.lo + float(1u << (mapped + 2u));
            sample.level = mapped - 1u;
            sample.mappedLevel = mapped;
            return true;
        }
        sample.entry = word & 0xFFFFFFu;
        sample.level = (word >> 24) & 15u;
        sample.mappedLevel = sample.level;
        cloudBrickCellTest(sample.entry, sample.level, i, x, settledOnly, thinDensity, sample);
        return true;
    }

    /// The empty-cell part of cloudAssetBrickAt: marks sample empty where x lies in a skippable 2-voxel cell of brick entry, and
    /// records the cell, octant and brick bounds around it.
    void cloudBrickCellTest(uint entry, uint level, int3 i, float3 x, bool settledOnly, float thinDensity, inout CloudSample sample)
    {
        if (SHIP(cloudEmptySkip, 1u) != 0 && (!settledOnly || hstrCloudBricks[entry].fade >= 1.f))
        {
            // An empty 2-voxel cell of the brick: every trilinear texel it can read is zero. Voxels of the cell whose finest mapped
            // brick is a descendant walk back up to this brick, so the whole cell reads it.
            const float levelScale = float(1u << level);
            const float3 brickOrigin = float3(((uint3(i) >> level) >> 3u) * 8u);
            const float3 local = clamp((x + 0.5f) / levelScale - 0.5f - brickOrigin, -0.5f, 7.4999f);
            const uint3 cell = uint3(clamp(int3(floor((local + 0.5f) * 0.5f)), 0, 3));
            const uint index = cell.x + 4u * cell.y + 16u * cell.z;
            const uint code = (hstrCloudOccupancy[entry * kCloudCellWords + (index >> 3u)] >> ((index & 7u) * 4u)) & 15u;
            // The bounds of the cell, of its octant and of the whole brick, for the march's step length (cloudLocalStep).
            const uint octants = hstrCloudOccupancy[entry * kCloudCellWords + 8u];
            const uint octant = (cell.x >> 1u) + 2u * (cell.y >> 1u) + 4u * (cell.z >> 1u);
            uint brickCode = 0;
            for (uint o = 0; o < 8; ++o)
                brickCode = max(brickCode, (octants >> (o * 4u)) & 15u);
            sample.bounds = code | (((octants >> (octant * 4u)) & 15u) << 4u) | (brickCode << 8u) |
                            (cell.x << 12u) | (cell.y << 14u) | (cell.z << 16u);
            sample.cellLo = (brickOrigin + 2.f * float3(cell)) * levelScale - 0.5f;
            // Codes below cellCode are skippable: bound(q) = 2^(2q - 30) <= thin  <=>  q <= (log2(thin) + 30) / 2.
            const float thin = thinDensity / levelScale;
            sample.cellCode = thin > 0.f ? uint(clamp(floor((log2(thin) + 30.f) * 0.5f) + 1.f, 1.f, 15.f)) : 1u;
            if (code < sample.cellCode)
            {
                sample.empty = true;
                sample.cell = true;
                sample.lo = (brickOrigin + 2.f * float3(cell)) * levelScale - 0.5f;
                sample.hi = sample.lo + 2.f * levelScale;
            }
        }
    }

    float cameraExtinctionUnfaded(float3 v, float footprint, inout float fineVoxels, out CloudSample sample)
    {
        sample = {};
        fineVoxels = 1.f;
        // Read up front even where a brick answers: fetching it only on the proxy paths made sunDepthAt's live march 5x slower (DXC),
        // and saved nothing measurable in the camera march.
        const float proxy = extinctionAtVoxel(v);
        // Timing oracle (cloudCostProbe), not an image: 1 stops here, so every sample costs what a proxy-resolved one costs, and
        // 2 walks the instance and the sparse hierarchy but never reads the atlas. The frame is wrong in both; the differences
        // decompose a sample into proxy fetch, virtual lookup and atlas filtering, which the all-sample cost fit cannot.
        if (SHIP(cloudCostProbe, 0u) == 1)
            return proxy;
        uint instanceIndex;
        float3 x;
        float footprintLevel;
        HSTRCloudInstance instance;
        if (!cloudInstanceAt(v, footprint, instanceIndex, x, footprintLevel, instance))
            return proxy;
        // 3 stops with the instance resolved and its transform applied, so mode 3 minus mode 1 is what cloudInstanceAt costs - the
        // sea tile modulo, the instance load and the domain-to-asset transform - and full minus 3 is the sparse walk proper.
        if (SHIP(cloudCostProbe, 0u) == 3)
            return proxy;
        const float scale = gScene.getGridVolume(0).data.densityScale * params.densityScale;
        // Thin cells: the optical depth along a level-0 cell's diagonal (2 sqrt(3) source voxels) at the bound is cloudThinDepth.
        const float thinDensity = params.cloudThinDepth / max(scale * 3.4641f * instance.sourceVoxelWorld, 1e-12f);
        const bool found = cloudAssetBrickAt(hstrCloudAssets[instance.asset], x, uint(max(footprintLevel, 0.f)), true, thinDensity, sample);
        sample.instance = instanceIndex;
        if (!found)
            return proxy;
        fineVoxels = float(1u << sample.level) * instance.sourceVoxelWorld / params.seaVoxelSize.x;
        if (sample.empty)
            return 0.f;
        // The hierarchy has been walked and sample is written, so the caller's skipping and stepping still depend on it and none of
        // that work folds away; only the atlas fetch and its filtering are missing.
        if (SHIP(cloudCostProbe, 0u) == 2)
            return proxy;
        const HSTRCloudBrick brick = hstrCloudBricks[sample.entry];
        const int3 i = int3(floor(x + 0.5f));
        float density = cloudBrickDensity(sample.entry, x, i) * scale;
        if (brick.fade < 1.f)
            density = lerp(brick.parent != kCloudRefNone ? cloudBrickDensity(brick.parent, x, i) * scale : proxy, density, brick.fade);
        return density;
    }

    /// Sun atlas slot holding a brick's bake for key ((sun generation << 4) | orientation class), kCloudRefNone if there is none.
    /// While a new sun direction bakes, the newest older bake of the class answers (cloudSunOldestGeneration: nothing but the direction
    /// has changed since that generation, and bakes rebake in their own slot most important first).
    uint cloudSunSlotOf(uint brickIndex, uint sunClass)
    {
        const uint base = brickIndex * 2u * kCloudSunBakesPerBrick;
        const uint generation = params.cloudSunGeneration;
        const uint current = (generation << 4) | sunClass;
        uint found = kCloudRefNone;
        uint best = kCloudRefNone;
        uint bestGeneration = 0;
        for (uint k = 0; k < kCloudSunBakesPerBrick; ++k)
        {
            const uint key = hstrCloudSunSlots[base + 2u * k];
            const uint slot = hstrCloudSunSlots[base + 2u * k + 1u];
            const uint keyGeneration = key >> 4;
            found = key == current ? slot : found;
            const bool older = key != kCloudRefNone && (key & 15u) == sunClass && keyGeneration >= params.cloudSunOldestGeneration &&
                               keyGeneration < generation && (best == kCloudRefNone || keyGeneration > bestGeneration);
            best = older ? slot : best;
            bestGeneration = older ? keyGeneration : bestGeneration;
        }
        return found != kCloudRefNone ? found : best;
    }

    /// Sun atlas slot of the bake a sample reading brick entry uses for key: the brick's own, else its nearest baked ancestor's (up to
    /// kCloudSunAncestors levels coarser; the sun slot pool keeps the most important bakes, so evicted bricks answer from coarser
    /// density rather than the live march). entry returns the brick whose bake it is.
    uint cloudSunSlotUp(inout uint entry, uint sunClass)
    {
        uint sunSlot = cloudSunSlotOf(entry, sunClass);
        for (uint k = 0; k < params.cloudSunAncestors && sunSlot == kCloudRefNone && hstrCloudBricks[entry].parent != kCloudRefNone; ++k)
        {
            entry = hstrCloudBricks[entry].parent;
            sunSlot = cloudSunSlotOf(entry, sunClass);
        }
        return sunSlot;
    }

    /// Baked sun optical depth of a brick at source position x from its bake in sun atlas slot sunSlot, in source voxel lengths.
    float cloudBrickSun(uint brickIndex, uint sunSlot, float3 x) { return cloudBrickSunAt(hstrCloudBricks[brickIndex].info & 15u, sunSlot, x); }

    /// cloudBrickSun for a baked brick of known level.
    float cloudBrickSunAt(uint level, uint sunSlot, float3 x)
    {
        const int3 i = int3(floor(x + 0.5f));
        const float3 levelPosition = (x + 0.5f) / float(1u << level) - 0.5f;
        const float3 local = clamp(levelPosition - float3(((i >> int(level)) >> 3) * 8), -0.5f, 7.5f);
        const uint shift = params.cloudAtlasShift;
        const uint mask = (1u << shift) - 1u;
        const uint3 slot = uint3(sunSlot & mask, (sunSlot >> shift) & mask, sunSlot >> (2u * shift));
        const float3 texel = float3(slot * kCloudBrickEdge) + 1.5f + local;
        return hstrCloudSunAtlas.SampleLevel(hstrLinearClampSampler, texel * params.cloudSunAtlasInvSize, 0.f);
    }

    float cameraExtinctionAtVoxel(float3 v, float footprint)
    {
        float fineVoxels;
        return cameraExtinctionAtVoxel(v, footprint, fineVoxels);
    }

    /// Voxel displacement per world unit along a world direction.
    float3 voxelDirection(float3 direction)
    {
        if (params.cloudDomain != 0)
            return direction / params.seaVoxelSize;
        return mul(gScene.getGridVolume(0).data.invTransform, float4(direction, 0.f)).xyz;
    }

    /// Whether voxel position v lies inside the medium's grid (the sea is unbounded in X/Z).
    bool insideExtinctionGrid(float3 v)
    {
        if (params.seaMode != 0)
            return v.y >= -1.f && v.y <= float(params.hstrExtinctionDims.y);
        return all(v >= -1.f) && all(v <= float3(params.hstrExtinctionDims));
    }

    /// Wraps a sea position's X/Z into the domain so that long photon paths keep their precision.
    float3 wrapSeaPosition(float3 v) { return params.seaMode != 0 ? seaVoxel(v) : v; }

    float extinctionAt(float3 leafPosition) { return extinctionAtVoxel(leafPosition * float(params.hstrCellWidth)); }

    /// Upper bound on the trilinear extinction within one majorant block of v in any direction.
    float majorantAt(float3 v)
    {
        v = seaVoxel(v);
        const int3 block = clamp(int3(floor(v * 0.25f)), int3(0), int3(params.hstrMajorantDims) - 1);
        return hstrMajorant.Load(int4(block, 0));
    }

    /// Upper bound on the trilinear extinction inside v's majorant block (for delta and ratio tracking, which re-read it at
    /// every block exit).
    float tightMajorantAt(float3 v)
    {
        v = seaVoxel(v);
        const int3 block = clamp(int3(floor(v * 0.25f)), int3(0), int3(params.hstrMajorantDims) - 1);
        return hstrTightMajorant.Load(int4(block, 0));
    }

    float3 cameraLightingAt(float3 leafPosition)
    {
        return hstrCameraLighting.SampleLevel(hstrLinearSampler, saturate(leafPosition / float3(params.hstrLeafDims)), 0.f).rgb;
    }

    /// Camera-basis source radiance towards the camera (before albedo).
    float3 sourceAtVoxel(float3 v) { return cameraLightingAt(v / float(params.hstrCellWidth)); }

    /// Camera factor of the sun pages for one ray direction: octave n contributes a^n p(theta; c^n g).
    float4 sunOctaveWeights(float3 direction)
    {
        const float cosine = dot(normalize(params.sunDirection), normalize(direction));
        const float a = params.octaveEnergy;
        const float c = params.octavePhase;
        const float g = params.anisotropy;
        return float4(phaseHG(cosine, g), a * phaseHG(cosine, c * g), a * a * phaseHG(cosine, c * c * g), a * a * a * phaseHG(cosine, c * c * c * g));
    }

    /// Sun optical depth from voxel position v to the grid boundary. The voxel-centre field alone steps at voxel scale, where
    /// a voxel of cloud is many optical depths: the first sunNearVoxels voxels towards the sun are marched on the trilinear
    /// density (majorant-bounded steps of at most stepOpticalDepth), so lit skins, shadow edges and terminators follow the
    /// density below voxel scale, and the field is read only where the march leaves off. Past kSunDepthOpaque nothing is lit.
    /// footprint >= 0 (camera samples): the near march reads the camera's fine density with that pixel footprint.
    /// Sun depth of a camera sample that read the proxy (no fine brick there at its footprint): the proxy is the density the
    /// voxel-resolution sun field integrates, so a near march over it would only re-integrate that field. Half a voxel of proxy
    /// extinction towards the sun, then the field.
    float sunDepthProxy(float3 v)
    {
        v = seaVoxel(v);
        const float3 direction = params.sunVoxelDirection;
        return hstrFineSun.SampleLevel(hstrLinearSampler, (seaVoxel(v + direction) + 0.5f) / float3(params.hstrFineDims), 0.f) +
               extinctionAtVoxel(v + 0.5f * direction) * params.sunVoxelWorldLength;
    }

    float sunDepthAt(float3 v, float footprint = -1.f)
    {
        CloudSample sample = {};
        uint instanceIndex;
        float3 x;
        float footprintLevel;
        HSTRCloudInstance instance;
        if (footprint >= 0.f && cloudInstanceAt(seaVoxel(v), footprint, instanceIndex, x, footprintLevel, instance))
        {
            cloudAssetBrickAt(hstrCloudAssets[instance.asset], x, uint(max(footprintLevel, 0.f)), true, sample);
            sample.instance = instanceIndex;
        }
        return sunDepthAt(v, footprint, sample);
    }

    /// sunDepthAt for a camera sample whose density lookup at v returned sample. Samples of the fine bricks march a reach fixed in
    /// the asset's source voxels (sunNearVoxels at instance scale 1), so every instance of an orientation class computes the same sun
    /// depth within an asset brick, and a brick baked for the sample's class and level (bakeCloudSun) answers with one lookup.
    ///
    /// MEASURED (4K, per-pixel view, settled, 2026-09-15) against this function's live march: near 348 -> 41 ms, farside 197 -> 37 ms,
    /// sea 218 -> 27 ms; beam view near 409 -> 18 ms, farside 93 -> 10 ms, sea 311 -> 12 ms. Log error mean 1.0e-4 / 8.7e-4 / 1.5e-4
    /// with p99.9 0.013 / 0.036 / 0.022 (farside, which reads the coarsest bricks, is the worst). Every query is answered from a bake
    /// in all three views (96% / 95% / 77% the brick's own, the rest an ancestor's); the live march below is the fallback for bricks
    /// whose bake is missing or stale, not a steady-state path.
    float sunDepthAt(float3 v, float footprint, CloudSample sample)
    {
        if (footprint >= 0.f && !sample.brick && params.cloudVirtual != 0 && SHIP(cloudSunCache, 1u) != 0)
            return sunDepthProxy(v);
        v = seaVoxel(v);
        static const float kSunDepthOpaque = kCloudSunDepthOpaque;
        const float3 direction = params.sunVoxelDirection;
        float near = params.sunNearVoxels;
        if (near > 0.f && footprint >= 0.f && sample.brick)
        {
            const HSTRCloudInstance instance = hstrCloudInstances[sample.instance];
            near = params.sunNearVoxels * instance.sourceVoxelWorld / hstrCloudAssets[instance.asset].sourceVoxelWorld;
            // A brick coarser than the footprint asks for (not streamed finer here) answers with its own level's bake: the density the
            // sample itself read. The live march would read finer bricks further towards the sun where those are mapped.
            uint bakedEntry = sample.entry;
            const uint sunSlot = SHIP(cloudSunCache, 1u) != 0 && !sample.empty && sample.level >= sample.desired
                                     ? cloudSunSlotUp(bakedEntry, instance.sunClass)
                                     : kCloudRefNone;
            if (sunSlot != kCloudRefNone)
            {
                const float depth = cloudBrickSun(bakedEntry, sunSlot, sample.x) * instance.sourceVoxelWorld;
                if (depth > kSunDepthOpaque)
                    return depth;
                return depth + hstrFineSun.SampleLevel(hstrLinearSampler, (seaVoxel(v + near * direction) + 0.5f) / float3(params.hstrFineDims), 0.f);
            }
        }
        if (near <= 0.f)
        {
            const float3 midpoint = v + 0.5f * direction;
            const float sigma = footprint >= 0.f ? cameraExtinctionAtVoxel(midpoint, footprint) : extinctionAtVoxel(midpoint);
            return hstrFineSun.SampleLevel(hstrLinearSampler, (seaVoxel(v + direction) + 0.5f) / float3(params.hstrFineDims), 0.f) +
                   sigma * params.sunVoxelWorldLength;
        }
#if !HSTR_SUN_LIVE
        // Compiled without the live march (the camera program): a sample with no bake of its own or of an ancestor - a brick whose
        // bake is still queued, or whose neighbourhood just changed - reads the voxel sun field, as the proxy regime does.
        return sunDepthProxy(v);
#else
        float depth = 0.f;
        float t = 0.f;
        float fineVoxels = 1.f;
        [loop]
        for (uint i = 0; i < 64 && t < near; ++i)
        {
            const float3 p = v + t * direction;
            const float majorant = majorantAt(p);
            const float minimum = footprint >= 0.f ? clamp(fineVoxels, params.cloudFineMinVoxels, 0.0625f) : 0.0625f;
            const float dt = min(clamp(params.stepOpticalDepth / max(1e-6f, majorant * params.sunVoxelWorldLength), minimum, 1.f), near - t);
            if (majorant > 0.f)
            {
                const float3 midpoint = p + 0.5f * dt * direction;
                const float sigma = footprint >= 0.f ? cameraExtinctionAtVoxel(midpoint, footprint, fineVoxels) : extinctionAtVoxel(midpoint);
                depth += sigma * dt * params.sunVoxelWorldLength;
            }
            t += dt;
            if (depth > kSunDepthOpaque)
                return depth;
        }
        return depth + hstrFineSun.SampleLevel(hstrLinearSampler, (seaVoxel(v + t * direction) + 0.5f) / float3(params.hstrFineDims), 0.f);
#endif
    }

    // A brick context experiment lived here (2026-09-15): a camera ray resolved its brick once - instance, directory walk, atlas
    // address, cell codes, sun slot - and later samples inside that brick, level and tile read it straight from registers until the
    // ray left it. It was exact (6e-9 log error against the full lookup) and removed ~80% of the lookups: 4.2 brick visits per pixel
    // with 5.6 samples each on the near view at 4K. It was also SLOWER - 46.5 -> 52.9 ms with lighting, 29.2 -> 32.8 ms without - and
    // merely carrying its code raised the march's cost when switched off. The context's registers cost more occupancy than the
    // lookups cost latency: the directory, node and brick reads are cache hits, and this march is bound by fetch latency, occupancy
    // and divergence, not by address arithmetic. Fewer samples (minStepVoxels) and fewer fetches per sample are the levers that pay.

    /// Rasterized sun residual at v (before albedo): the sun optical depth attenuated per octave.
    float3 residualAtVoxel(float3 v, float4 octaves, float footprint = -1.f) { return residualAtDepth(v, sunDepthAt(v, footprint), octaves); }

    /// residualAtVoxel for an already computed sun optical depth at v.
    float3 residualAtDepth(float3 v, float depth, float4 octaves)
    {
        v = seaVoxel(v);
        if (all(octaves.yzw == 0.f))
            return params.residualStrength * params.sunRadiance * octaves.x * exp(-depth);
        // Octaves 1-3 are multiply scattered: they arrive from the scatterers around v, not along v's own sun ray.
        const float4 spread = hstrSunOctaves.SampleLevel(hstrLinearSampler, (v + 0.5f) / (2.f * float3(params.hstrOctaveDims)), 0.f);
        const float b = params.octaveExtinction;
        const float3 local = exp(-depth * float3(b, b * b, b * b * b));
        const float3 multiple = (spread.rgb + 1e-3f * local) / (spread.a + 1e-3f);
        const float4 transmittance = float4(exp(-depth), multiple);
        return params.residualStrength * params.sunRadiance * dot(octaves, transmittance);
    }

    /// Scatterer-weighted sun arrival of octaves 1-3 over one 2-voxel cell: sum of sigma * exp(-b^n tau) and sum of sigma.
    void gatherSunOctaves(uint3 cell)
    {
        if (any(cell >= params.hstrOctaveDims))
            return;
        const float b = params.octaveExtinction;
        float4 sum = 0.f;
        for (uint z = 0; z < 2; ++z)
            for (uint y = 0; y < 2; ++y)
                for (uint x = 0; x < 2; ++x)
                {
                    const uint3 voxel = 2u * cell + uint3(x, y, z);
                    if (any(voxel >= params.hstrFineDims))
                        continue;
                    const float extinction = hstrExtinction.Load(int4(voxel, 0));
                    const float depth = hstrFineSun[voxel];
                    sum += extinction * float4(exp(-depth * float3(b, b * b, b * b * b)), 1.f);
                }
        hstrSunOctavesOutput[cell] = sum;
    }

    /// One axis of the separable Gaussian that spreads multiply scattered sun light between scatterers.
    void blurSunOctaves(uint3 cell)
    {
        if (any(cell >= params.hstrOctaveDims))
            return;
        const int radius = int(ceil(2.5f * params.octaveBlurSigma));
        int3 axis = 0;
        axis[params.octaveBlurAxis] = 1;
        float4 sum = 0.f;
        for (int i = -radius; i <= radius; ++i)
        {
            int3 p = int3(cell) + i * axis;
            if (params.seaMode != 0)
            {
                const int3 dims = int3(params.hstrOctaveDims);
                p.x = (p.x % dims.x + dims.x) % dims.x;
                p.z = (p.z % dims.z + dims.z) % dims.z;
            }
            if (any(p < 0) || any(p >= int3(params.hstrOctaveDims)))
                continue;
            sum += exp(-0.5f * float(i * i) / (params.octaveBlurSigma * params.octaveBlurSigma)) * hstrSunOctaves[uint3(p)];
        }
        hstrSunOctavesOutput[cell] = sum;
    }

    bool leafResidualResident(uint3 leaf) { return asfloat(hstrLeafResidual[leafIndex(leaf)]) > params.residualTolerance; }

    // ---------------------------------------------------------------------------------------------
    // Camera-side basis update: evaluated on camera motion only.

    void writeCameraLighting(uint3 samplePosition)
    {
        if (any(samplePosition >= params.hstrCameraLightingDims) || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        // Two samples per leaf axis resolve the 2x2 spatial modes of every retained face trace.
        const float3 leafPosition = (float3(samplePosition) + 0.5f) * float3(params.hstrLeafDims) / float3(params.hstrCameraLightingDims);
        const float3 indexPosition = float3(params.hstrGridMin) + float(params.hstrCellWidth) * leafPosition;
        const float3 worldPosition = mul(volume.data.transform, float4(indexPosition, 1.f)).xyz;
        hstrCameraLightingOutput[samplePosition] = float4(hstrLighting(indexPosition, worldPosition - gScene.camera.data.posW), 0.f);
    }

    float3 cutNodeWorldPosition(uint3 leafPosition, GridVolume volume)
    {
        // Flat domain cut: node coordinates are window-local leaf cells, and the sea's index space is the domain voxel grid
        // (voxel i centred at seaOrigin + (i + 0.5) size), not the scene grid's transform.
        if (params.hstrCutFlat != 0)
        {
            const float3 voxel = float3(int3(leafPosition) + params.hstrCutOriginLeaf) * float(params.hstrCellWidth);
            return params.seaOrigin + (voxel + 0.5f) * params.seaVoxelSize;
        }
        const float3 indexPosition = float3(params.hstrGridMin) + float(params.hstrCellWidth) * float3(leafPosition);
        return mul(volume.data.transform, float4(indexPosition, 1.f)).xyz;
    }

    /// Window-local leaf index of a flat-cut node, and the camera in the same coordinates.
    uint3 domainCutIndex(uint nodeID)
    {
        const uint3 dims = params.hstrLeafDims;
        return uint3(nodeID % dims.x, (nodeID / dims.x) % dims.y, nodeID / (dims.x * dims.y));
    }

    float3 domainCutCameraPosition(GridVolume volume)
    {
        return worldToVoxel(gScene.camera.data.posW, volume) / float(params.hstrCellWidth) - float3(params.hstrCutOriginLeaf);
    }

    /// Front-to-back key of a cell of the regular domain cut. The window grid is a binary space partition in its own right: pad
    /// each axis to a power of two and split the longest remaining axis in half, most significant bit first. A bit is 0 while the
    /// cell is on the camera's side of that split, so the key orders every pair of disjoint cells exactly as the painter would,
    /// for every ray leaving the eye - the same guarantee the Schur tree's ancestor path gives, without a tree to walk.
    uint domainCutOrderKey(uint nodeID)
    {
        const uint3 index = domainCutIndex(nodeID);
        const float3 camera = domainCutCameraPosition(gScene.getGridVolume(0));
        uint3 size = 1u;
        [unroll]
        for (uint axis = 0; axis < 3; ++axis)
            [loop] while (size[axis] < params.hstrLeafDims[axis])
                size[axis] <<= 1;
        uint3 low = 0u;
        uint key = 0;
        uint depth = 0;
        [loop]
        while (any(size > 1u) && depth < 30u)
        {
            const uint axis = size.x >= size.y && size.x >= size.z ? 0u : (size.y >= size.z ? 1u : 2u);
            const uint half = size[axis] >> 1u;
            const uint middle = low[axis] + half;
            const bool upper = index[axis] >= middle;
            const bool cameraUpper = camera[axis] >= float(middle);
            key = (key << 1u) | (upper == cameraUpper ? 0u : 1u);
            low[axis] = upper ? middle : low[axis];
            size[axis] = half;
            ++depth;
        }
        return depth == 0 ? 0u : key << (32u - depth);
    }

    void projectCutNode(uint nodeID)
    {
        if (nodeID >= params.hstrNodeCount || gScene.getGridVolumeCount() == 0)
            return;
        const HSTRCutNode node = hstrCutNodes[nodeID];
        const GridVolume volume = gScene.getGridVolume(0);
        float2 minimumPixel = float2(params.frameDim);
        float2 maximumPixel = 0.f;
        float minimumDepth = 1e30f;
        float maximumDepth = -1e30f;
        bool crossesCamera = false;
        [unroll]
        for (uint corner = 0; corner < 8; ++corner)
        {
            const uint3 leafPosition = uint3(
                (corner & 1u) ? node.maxRight.x : node.minLeft.x,
                (corner & 2u) ? node.maxRight.y : node.minLeft.y,
                (corner & 4u) ? node.maxRight.z : node.minLeft.z
            );
            const float3 worldPosition = cutNodeWorldPosition(leafPosition, volume);
            const float4 clip = mul(gScene.camera.data.viewProjMatNoJitter, float4(worldPosition, 1.f));
            crossesCamera |= clip.w <= 1e-5f;
            const float2 ndc = clip.xy / max(1e-5f, clip.w);
            const float2 pixel = (ndc * float2(0.5f, -0.5f) + 0.5f) * float2(params.frameDim);
            minimumPixel = min(minimumPixel, pixel);
            maximumPixel = max(maximumPixel, pixel);
            minimumDepth = min(minimumDepth, dot(worldPosition - gScene.camera.data.posW, gScene.camera.data.cameraW));
            maximumDepth = max(maximumDepth, dot(worldPosition - gScene.camera.data.posW, gScene.camera.data.cameraW));
        }
        // A node the camera sits among has no finite screen box, so it claims the whole frame. A node entirely BEHIND the camera
        // has no screen box either and must claim nothing: the cloud sea puts the camera inside its window, so half the cut is
        // behind the eye, and treating those as full-screen would bin a thousand cells into every tile of the frame and overflow
        // every tile list there is.
        if (maximumDepth <= 0.f)
        {
            minimumPixel = 0.f;
            maximumPixel = 0.f;
        }
        else if (crossesCamera)
        {
            minimumPixel = 0.f;
            maximumPixel = float2(params.frameDim);
        }
        hstrNodeProjection[nodeID] = float4(clamp(minimumPixel, 0.f, float2(params.frameDim)), clamp(maximumPixel, 0.f, float2(params.frameDim)));
        hstrNodeDepth[nodeID] = float2(minimumDepth, maximumDepth);
        hstrNodeOrder[nodeID] = bspOrderKey(nodeID, volume);
    }

    /// Front-to-back key of a node in the HST's binary space partition seen from the camera: one bit per ancestor split,
    /// set when the node lies in the far child, most significant bit at the root. For disjoint nodes the key order is the
    /// painter's order of every ray leaving the eye, so a tile's cut sorted by it composites correctly for all its pixels.
    uint bspOrderKey(uint nodeID, GridVolume volume)
    {
        if (params.hstrCutFlat != 0)
            return domainCutOrderKey(nodeID);
        const float3 cameraIndex = mul(volume.data.invTransform, float4(gScene.camera.data.posW, 1.f)).xyz;
        const float3 camera = (cameraIndex - float3(params.hstrGridMin)) / float(params.hstrCellWidth);
        uint path = 0;
        uint depth = 0;
        uint child = nodeID;
        uint parent = hstrCutNodeParents[nodeID];
        [loop]
        while (parent != ~0u && depth < 31)
        {
            const HSTRCutNode node = hstrCutNodes[parent];
            const HSTRCutNode left = hstrCutNodes[node.minLeft.w];
            // The children differ along the split axis only; the left child is the lower half.
            const uint3 lowerMax = left.maxRight.xyz;
            const uint axis = lowerMax.x != node.maxRight.x ? 0u : (lowerMax.y != node.maxRight.y ? 1u : 2u);
            const bool leftNear = camera[axis] < float(lowerMax[axis]);
            const bool isLeft = child == node.minLeft.w;
            if (isLeft != leftNear)
                path |= 1u << depth;
            ++depth;
            child = parent;
            parent = hstrCutNodeParents[parent];
        }
        // path holds the deepest split in bit 0 and the root split in bit depth - 1. Left-aligning puts the root split in the
        // most significant bit, so nodes at different depths compare lexicographically.
        return depth == 0 ? 0u : path << (32u - depth);
    }

    bool acceptTileCutNode(HSTRCutNode node, uint nodeID, GridVolume volume)
    {
        if (node.minLeft.w == ~0u)
            return true;
        const float3 leafExtent = float3(node.maxRight.xyz - node.minLeft.xyz) * float(params.hstrCellWidth);
        const float3 axisX = mul(volume.data.transform, float4(leafExtent.x, 0.f, 0.f, 0.f)).xyz;
        const float3 axisY = mul(volume.data.transform, float4(0.f, leafExtent.y, 0.f, 0.f)).xyz;
        const float3 axisZ = mul(volume.data.transform, float4(0.f, 0.f, leafExtent.z, 0.f)).xyz;
        const float maximumWorldLength = length(axisX) + length(axisY) + length(axisZ);
        const float pageError = 1.f - exp(-node.transmittancePage.x * maximumWorldLength);
        if (pageError <= params.cutTransmittanceTolerance)
            return true;
        const float variation = (node.statistics.y - node.statistics.x) / max(1e-6f, node.statistics.y);
        const float4 bounds = hstrNodeProjection[nodeID];
        const float projectedPixels = max(bounds.z - bounds.x, bounds.w - bounds.y);
        // Split/merge hysteresis: a node that was accepted keeps its page until it grows past the upper band, and a split node
        // only merges once it shrinks below the lower band, so an orbiting camera does not flicker between cuts.
        const bool wasAccepted = hstrCutStatePrevious[nodeID] != 0;
        const float band = wasAccepted ? 1.f + params.cutHysteresis : 1.f - params.cutHysteresis;
        return variation <= params.mixedFidelityThreshold && node.statistics.z <= params.mixedFidelityThreshold &&
               projectedPixels <= 12.f * band;
    }

    void binCutNode(uint nodeID)
    {
        if (nodeID >= params.hstrNodeCount || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        const HSTRCutNode node = hstrCutNodes[nodeID];
        const bool accepted = acceptTileCutNode(node, nodeID, volume);
        hstrCutState[nodeID] = accepted ? 1u : 0u;
        if (node.statistics.y <= 1e-7f || !accepted)
            return;

        // Every node tests its ancestor chain, producing one scene-wide antichain
        // without repeating a hierarchy traversal for every screen tile.
        uint parent = hstrCutNodeParents[nodeID];
        [loop]
        while (parent != ~0u)
        {
            if (acceptTileCutNode(hstrCutNodes[parent], parent, volume))
                return;
            parent = hstrCutNodeParents[parent];
        }

        const float4 bounds = hstrNodeProjection[nodeID];
        const uint2 tileMinimum = min(uint2(floor(bounds.xy / float(params.hstrTileSize))), params.hstrTileDims);
        const uint2 tileMaximum = min(uint2(ceil(bounds.zw / float(params.hstrTileSize))), params.hstrTileDims);
        [loop]
        for (uint y = tileMinimum.y; y < tileMaximum.y; ++y)
            [loop]
            for (uint x = tileMinimum.x; x < tileMaximum.x; ++x)
            {
                const uint tileID = x + params.hstrTileDims.x * y;
                uint slot;
                InterlockedAdd(hstrTileNodeCounts[tileID], 1u, slot);
                if (slot < params.hstrMaxTileNodes)
                    hstrTileNodes[tileID * params.hstrMaxTileNodes + slot] = nodeID;
            }
    }

    /// Trilinear page of a node evaluated at a point of its unit cube.
    float domainPageAt(float4 corners0, float4 corners1, float3 local)
    {
        const float x00 = lerp(corners0.x, corners0.y, local.x);
        const float x10 = lerp(corners0.z, corners0.w, local.x);
        const float x01 = lerp(corners1.x, corners1.y, local.x);
        const float x11 = lerp(corners1.z, corners1.w, local.x);
        return lerp(lerp(x00, x10, local.y), lerp(x01, x11, local.y), local.z);
    }

    /// One cell of the cloud sea's transport cut.
    ///
    /// The sea has no CPU Schur tree - its content is rasterized on worker threads into a toroidal domain proxy - so the cut is
    /// built here instead: the regular grid of 16^3-voxel domain cells, placed as the camera-centred window that tiles the
    /// reachable space (seaViewDistance is under half the domain, so each domain cell needs exactly one periodic image). Each cell
    /// carries the same page the CPU builder writes for an HST leaf: the eight shared corner extinctions of its trilinear page,
    /// the extinction range, and the page residual that decides per ray whether the page stands in for the cell or the cell is
    /// marched. Cells the domain holds nothing in get a zero range and binCutNode drops them, so empty sky never reaches a tile
    /// list at all.
    void buildDomainCutNode(uint nodeID)
    {
        const uint3 dims = params.hstrLeafDims;
        if (params.hstrCutFlat == 0 || nodeID >= dims.x * dims.y * dims.z || gScene.getGridVolumeCount() == 0)
            return;
        const uint3 index = domainCutIndex(nodeID);
        const float cellWidth = float(params.hstrCellWidth);
        // Domain voxel corner of this cell. X/Z may lie outside the domain: every lookup below wraps into the torus.
        const float3 base = float3(int3(index) + params.hstrCutOriginLeaf) * cellWidth;

        HSTRCutNode node;
        node.minLeft = uint4(index, ~0u);       ///< Leaf: nothing below it to refine to.
        node.maxRight = uint4(index + 1u, ~0u);
        float4 corners0;
        float4 corners1;
        {
            // Sampled where neighbouring cells share them, so the page is continuous across the cut.
            const float4 x = float4(base.x, base.x + cellWidth, base.x, base.x + cellWidth);
            const float4 y = float4(base.y, base.y, base.y + cellWidth, base.y + cellWidth);
            corners0 = float4(
                extinctionAtVoxel(float3(x.x, y.x, base.z)), extinctionAtVoxel(float3(x.y, y.y, base.z)),
                extinctionAtVoxel(float3(x.z, y.z, base.z)), extinctionAtVoxel(float3(x.w, y.w, base.z))
            );
            corners1 = float4(
                extinctionAtVoxel(float3(x.x, y.x, base.z + cellWidth)), extinctionAtVoxel(float3(x.y, y.y, base.z + cellWidth)),
                extinctionAtVoxel(float3(x.z, y.z, base.z + cellWidth)), extinctionAtVoxel(float3(x.w, y.w, base.z + cellWidth))
            );
        }
        // The cell's extinction range and the page's worst error over it. The majorant blocks bound the trilinear extinction the
        // march would read anywhere inside the block, which is what the page has to stand for; the interior lattice adds the
        // signed error, which the block maxima alone cannot see (a page above the density is as wrong as one below it).
        const uint blocks = params.hstrCellWidth / kDomainBlock;
        float maximum = 0.f;
        float minimum = 1e30f;
        float residual = 0.f;
        [loop]
        for (uint b = 0; b < blocks * blocks * blocks; ++b)
        {
            const uint3 block = uint3(b % blocks, (b / blocks) % blocks, b / (blocks * blocks));
            const float3 centre = base + (float3(block) + 0.5f) * float(kDomainBlock);
            const float bound = majorantAt(centre);
            maximum = max(maximum, bound);
            if (bound <= 0.f)
                continue;
            // The page over a sub-box is bounded by the page at that box's own corners (it is trilinear, hence monotone in each
            // axis between them), so this is a true bound on the page-minus-density error over the block.
            float pageLow = 1e30f;
            [unroll]
            for (uint corner = 0; corner < 8; ++corner)
            {
                const float3 local =
                    (float3(block) + float3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u)) * float(kDomainBlock) / cellWidth;
                pageLow = min(pageLow, domainPageAt(corners0, corners1, local));
            }
            residual = max(residual, bound - pageLow);
        }
        [loop]
        for (uint s = 0; s < 125; ++s)
        {
            const uint3 lattice = uint3(s % 5u, (s / 5u) % 5u, s / 25u);
            const float3 local = float3(lattice) * 0.25f;
            const float density = extinctionAtVoxel(base + local * cellWidth);
            minimum = min(minimum, density);
            residual = max(residual, abs(density - domainPageAt(corners0, corners1, local)));
        }
        node.statistics = float4(maximum > 0.f ? minimum : 0.f, maximum, 0.f, 0.f);
        // x is the page residual traverseCut turns into an optical-depth error per ray; y marks a leaf that holds samples, which is
        // what the CPU builder writes for an HST leaf with content.
        node.transmittancePage = float4(maximum > 0.f ? residual : 0.f, maximum > 0.f ? 1.f : 0.f, 0.f, 0.f);
        node.transmittanceCorners0 = maximum > 0.f ? corners0 : 0.f;
        node.transmittanceCorners1 = maximum > 0.f ? corners1 : 0.f;
        hstrDomainCutNodes[nodeID] = node;
    }

    bool intersectCutNode(HSTRCutNode node, float3 origin, float3 direction, out float2 interval)
    {
        const bool3 parallel = abs(direction) < 1e-8f;
        if (any(parallel && (origin < float3(node.minLeft.xyz) || origin >= float3(node.maxRight.xyz))))
        {
            interval = 0.f;
            return false;
        }
        const float3 inverseDirection = 1.f / select(parallel, float3(1.f), direction);
        const float3 a = (float3(node.minLeft.xyz) - origin) * inverseDirection;
        const float3 b = (float3(node.maxRight.xyz) - origin) * inverseDirection;
        const float3 lo = select(parallel, float3(-1e30f), min(a, b));
        const float3 hi = select(parallel, float3(1e30f), max(a, b));
        interval = float2(max(lo.x, max(lo.y, lo.z)), min(hi.x, min(hi.y, hi.z)));
        interval = clamp(interval, 0.f, 1.f);
        return interval.x < interval.y;
    }

    /// Octahedral map, +z at the zenith. Continuous across z = 0, no transcendentals, and its inverse is exact, so a world
    /// direction has one texel and keeps it: this is what removes re-anchoring. World up is y, so the frame swaps y and z.
    /// The direction in the octahedral frame: its z is the axis beamOctAxis names, and the kink circle is perpendicular to it.
    float3 octFrame(float3 w)
    {
        if (params.beamOctAxis == 1)
            return float3(w.y, w.z, w.x);
        if (params.beamOctAxis == 2)
            return float3(w.x, w.y, w.z);
        return float3(w.x, w.z, w.y);
    }

    float3 octWorld(float3 n)
    {
        if (params.beamOctAxis == 1)
            return float3(n.z, n.x, n.y);
        if (params.beamOctAxis == 2)
            return float3(n.x, n.y, n.z);
        return float3(n.x, n.z, n.y);
    }

    float2 octEncode(float3 direction)
    {
        const float3 d = octFrame(direction);
        const float3 n = d / (abs(d.x) + abs(d.y) + abs(d.z));
        float2 p = n.xy;
        if (n.z < 0.f)
            p = float2((1.f - abs(n.y)) * (n.x >= 0.f ? 1.f : -1.f), (1.f - abs(n.x)) * (n.y >= 0.f ? 1.f : -1.f));
        return p * 0.5f + 0.5f;
    }

    float3 octDecode(float2 uv)
    {
        const float2 p = uv * 2.f - 1.f;
        float3 n = float3(p.x, p.y, 1.f - abs(p.x) - abs(p.y));
        if (n.z < 0.f)
            n.xy = float2((1.f - abs(n.y)) * (n.x >= 0.f ? 1.f : -1.f), (1.f - abs(n.x)) * (n.y >= 0.f ? 1.f : -1.f));
        return normalize(octWorld(n));
    }

    float3 cameraRayDirection(float2 pixelPosition)
    {
        const float2 p = pixelPosition / float2(params.frameDim);
        const float2 ndc = float2(2.f, -2.f) * p + float2(-1.f, 1.f);
        return normalize(ndc.x * gScene.camera.data.cameraU + ndc.y * gScene.camera.data.cameraV + gScene.camera.data.cameraW);
    }

    /// World direction of a position in the beam basis image. With beamRefFrame off this is the screen, so the beam build is
    /// unchanged; with it on the image is the anchored reference frame, which does not turn with the camera.
    float3 beamRayDirection(float2 beamPosition)
    {
        if (params.beamRefFrame == 0)
            return cameraRayDirection(beamPosition);
        if (params.beamOct != 0)
            return octDecode(beamPosition / float2(params.beamFrameDim));
        if (params.beamRefAligned != 0)
        {
            const float2 screenPosition =
                beamPosition - 0.5f * float2(params.beamFrameDim) + 0.5f * float2(params.frameDim);
            return cameraRayDirection(screenPosition);
        }
        // Form the same projection about the image centre. This avoids losing the low bits of an anchor-frame screen position by
        // first dividing its larger, margin-offset coordinate and then subtracting one.
        precise float2 centred = beamPosition - 0.5f * float2(params.beamFrameDim);
        precise float2 scale = float2(2.f, -2.f) / float2(params.beamFrameDim);
        precise float3 direction = centred.x * scale.x * params.beamRefU + centred.y * scale.y * params.beamRefV + params.beamRefW;
        return normalize(direction);
    }

    /// The beam-image position a world direction lands at, and whether it lands in front of the frame at all. The inverse of
    /// beamRayDirection, up to the positive scale the direction was normalised by.
    bool beamImagePosition(float3 w, out float2 beamPosition)
    {
        beamPosition = 0.f;
        if (params.beamRefFrame == 0)
            return false;
        if (params.beamOct != 0)
        {
            // Every direction lands, which is exactly why the image never needs re-anchoring.
            beamPosition = octEncode(w) * float2(params.beamFrameDim);
            return true;
        }
        const float3 c = float3(dot(params.beamRefInverse0, w), dot(params.beamRefInverse1, w), dot(params.beamRefInverse2, w));
        if (c.z <= 1e-6f)
            return false;
        const float2 ndc = c.xy / c.z;
        beamPosition = (ndc * float2(0.5f, -0.5f) + 0.5f) * float2(params.beamFrameDim);
        return true;
    }

    /// The beam-image position of a screen pixel: the same point of the same ray, expressed where the basis lives.
    bool beamPixelPosition(float2 pixelPosition, out float2 beamPosition)
    {
        if (params.beamRefFrame == 0)
        {
            beamPosition = pixelPosition;
            return true;
        }
        if (params.beamRefAligned != 0 && params.beamOct == 0)
        {
            beamPosition = pixelPosition + 0.5f * (float2(params.beamFrameDim) - float2(params.frameDim));
            return true;
        }
        return beamImagePosition(cameraRayDirection(pixelPosition), beamPosition);
    }

    /// Voxel position of a world position: the cloud sea domain (voxel i centred at seaOrigin + (i + 0.5) size), or the scene grid.
    float3 worldToVoxel(float3 world, GridVolume volume)
    {
        if (params.cloudDomain != 0)
            return (world - params.seaOrigin) / params.seaVoxelSize - 0.5f;
        return mul(volume.data.invTransform, float4(world, 1.f)).xyz;
    }

    /// Ray parameters inside the medium: the cloud layer's slab (capped at seaViewDistance) for the sea, else the grid bounds.
    bool mediumInterval(float3 origin, float3 direction, GridVolume volume, out float2 interval)
    {
        if (params.seaMode != 0)
        {
            // The occupied band rather than the grid's floor and ceiling (cloudSlabClamp): outside it there is no density, so the
            // interval it removes is empty sky the march would otherwise cross a majorant block at a time.
            // Branch-free on purpose. The same clamp written as a ternary on a parameter cost 19% of the frame uniformly across every
            // scope, for a bit-identical image and strictly less marching - a codegen cliff, not work. seaContentY carries the whole
            // grid's height when the clamp is off, so the toggle lives on the CPU and this stays two instructions.
            const float bottom = max(params.seaOrigin.y, params.seaContentY.x);
            const float top = min(params.seaOrigin.y + float(params.hstrExtinctionDims.y) * params.seaVoxelSize.y, params.seaContentY.y);
            const float speed = length(direction);
            if (abs(direction.y) < 1e-8f)
            {
                interval = float2(0.f, params.seaViewDistance / speed);
                return origin.y >= bottom && origin.y <= top;
            }
            const float2 y = float2(bottom - origin.y, top - origin.y) / direction.y;
            interval = float2(min(y.x, y.y), max(y.x, y.y));
            // The view distance is measured from the CAMERA, not from where the ray enters the slab. It was written as
            // max(0, interval.x) + seaViewDistance, which was the same thing only while the slab was the whole grid and the camera
            // sat inside it, so interval.x was negative. Clamping the slab to the content band made interval.x the positive distance
            // to the band, and every ray then marched a view distance BEYOND it - a longer interval, not a shorter one, which is why
            // the clamp first measured 19% slower for a bit-identical image.
            interval.y = min(interval.y, params.seaViewDistance / speed);
            return interval.y > max(0.f, interval.x);
        }
        const AABB bounds = volume.getBounds();
        return intersectRayAABB(origin, direction, bounds.minPoint, bounds.maxPoint, interval);
    }

    bool makeCutRay(float3 origin, float3 direction, GridVolume volume, out CutRay ray)
    {
        ray.position = 0.f;
        ray.direction = 0.f;
        ray.worldLength = 0.f;
        ray.entryDistance = 0.f;
        ray.sunOctaves = 0.f;
        ray.viewDirection = normalize(direction);
        float2 interval;
        if (!mediumInterval(origin, direction, volume, interval))
            return false;
        const float entry = max(0.f, interval.x);
        if (interval.y <= entry)
            return false;
        const float3 entryIndex = worldToVoxel(origin + entry * direction, volume);
        const float3 exitIndex = worldToVoxel(origin + interval.y * direction, volume);
        ray.position = (entryIndex - float3(params.hstrGridMin)) / float(params.hstrCellWidth);
        ray.direction = (exitIndex - entryIndex) / float(params.hstrCellWidth);
        ray.worldLength = length(direction) * (interval.y - entry);
        ray.entryDistance = length(direction) * entry;
        const float single = (params.hstComponents & kComponentSunSingle) != 0 ? 1.f : 0.f;
        const float multiple = (params.hstComponents & kComponentSunMultiple) != 0 ? 1.f : 0.f;
        ray.sunOctaves = sunOctaveWeights(direction) * float4(single, multiple, multiple, multiple);
        return true;
    }

    // ---------------------------------------------------------------------------------------------
    // Transmittance and source integration along the projected cut.

    // ---------------------------------------------------------------------------------------------
    // Adaptive cache march. The cache frame's sources are exp(-b tau_sun) times a smooth factor: the cache modulation
    // (b = worldCacheModulation) times the stored cache R, and single scattering (b = 1) times a constant. Over a segment
    // with constant extinction, linear tau_sun and linear R, sigma T(s) exp(-b tau_sun(s)) R(s) integrates in closed form, so
    // the lit skin needs no small steps. Each segment is compared with its two halves; it is bisected only while that
    // difference, projected through the ray's transmittance, exceeds marchTolerance of the pixel.
    //
    // Measured and switched off by default (adaptiveMarch 0): 8x8-block log error against the 4K path tracer / GPU ms of the
    // per-pixel view 8 and the beam view 9. Sunbehind: fixed 1-voxel steps 0.0252 (6.1 / 3.3 ms), fixed 0.1-voxel steps
    // 0.0243 (~100 / 36 ms), adaptive 0.0236 (17 / 6.2 ms). Side: fixed 0.0264 (6.6 / 4.1 ms), adaptive 0.0256 (17 / 5.7 ms).
    // After the beam optimizations (beam_truth h16x3 full .05): fixed ~1.0 ms, adaptive ~2.0 ms for ~0.001-0.002 less
    // error. marchTolerance 0 (analytic integration, no bisection) still costs 1.8-2.0 ms: every sample does the full cache
    // lookup. The remaining error is in the cache, not the camera quadrature.

    /// Sample at ray parameter t. The stored cache R is smooth at cell scale, so it is looked up only at coarse segment ends
    /// (withCache); samples inside a coarse segment interpolate it and resolve only the extinction and the sun depths.
    /// Pixel footprint in world units at ray parameter t.
    float rayFootprint(CutRay ray, float t) { return (ray.entryDistance + t * ray.worldLength) * params.cloudPixelAngle; }

    MarchSample marchSampleAt(CutRay ray, float t, bool withSource, bool withSingle, bool withCache = true)
    {
        const float3 v = (ray.position + t * ray.direction) * float(params.hstrCellWidth);
        const float footprint = rayFootprint(ray, t);
        MarchSample s;
        s.t = t;
        s.sigma = cameraExtinctionAtVoxel(v, footprint);
        s.sunCache = 0.f;
        s.sunSingle = 0.f;
        s.cache = 0.f;
        if (withSource)
        {
            if (params.worldCacheModulation > 0.f)
                s.sunCache = min(hstrFineSun.SampleLevel(hstrLinearSampler, (seaVoxel(v) + 0.5f) / float3(params.hstrFineDims), 0.f),
                                 params.worldCacheModulationDepth);
            if (withCache)
                s.cache = worldCacheRadianceStored(v, ray.viewDirection);
        }
        if (withSingle)
            s.sunSingle = sunDepthAt(v, footprint);
        return s;
    }

    /// E0 = int_0^h exp(-K s) ds and E1 = int_0^h s exp(-K s) ds, with series near K h = 0.
    float2 exponentialMoments(float K, float h)
    {
        const float x = K * h;
        if (abs(x) < 1e-2f)
            return float2(h * (1.f - x / 2.f + x * x / 6.f), h * h * (0.5f - x / 3.f + x * x / 8.f));
        const float e = exp(-x);
        return float2((1.f - e) / K, (1.f - e * (1.f + x)) / (K * K));
    }

    /// Light of segment [a, b] of world length h from unit transmittance: extinction the endpoint mean, sun depths and cache
    /// linear between the endpoints.
    SegmentLight segmentLight(MarchSample a, MarchSample b, float h, float3 singleScale)
    {
        SegmentLight light;
        const float sigma = 0.5f * (a.sigma + b.sigma);
        light.depth = sigma * h;
        const float bc = params.worldCacheModulation > 0.f ? params.worldCacheModulation : 0.f;
        const float2 cacheMoments = exponentialMoments(sigma + bc * (b.sunCache - a.sunCache) / h, h);
        light.cache = sigma * exp(-bc * a.sunCache) * (a.cache * cacheMoments.x + (b.cache - a.cache) * (cacheMoments.y / h));
        const float2 singleMoments = exponentialMoments(sigma + (b.sunSingle - a.sunSingle) / h, h);
        light.single = sigma * exp(-a.sunSingle) * singleMoments.x * singleScale;
        return light;
    }

    void integrateCachedSegment(
        CutRay ray,
        float2 segment,
        bool withSource,
        bool withSingle,
        inout float3 radiance,
        inout float3 singleRadiance,
        inout float transmittance,
        inout uint steps
    )
    {
        const float voxelsPerUnit = length(ray.direction) * float(params.hstrCellWidth);
        if (voxelsPerUnit <= 1e-6f)
            return;
        const float worldPerVoxel = ray.worldLength / voxelsPerUnit;
        const float3 singleScale = withSingle ? params.residualStrength * params.sunRadiance * ray.sunOctaves.x : 0.f;
        const float skyLevel = dot(params.skyRadiance, 1.f / 3.f);
        const float minUnits = params.marchMinVoxels / voxelsPerUnit;

        MarchSample a;
        MarchSample b;
        bool haveA = false;
        bool haveB = false;
        MarchSample pending[kMarchStack];
        uint depth = 0;
        float t = segment.x;
        [loop]
        while (t < segment.y && transmittance > kMinTransmittance && steps < params.maxMarchSteps)
        {
            if (!haveB)
            {
                // A new coarse segment: majorant-bounded, and empty blocks are crossed without samples.
                const float3 v = (ray.position + t * ray.direction) * float(params.hstrCellWidth);
                const float majorant = majorantAt(v);
                const float stepVoxels = clamp(
                    params.stepOpticalDepth / max(1e-6f, majorant * worldPerVoxel),
                    params.marchCoarseVoxels,
                    max(params.maxStepVoxels, params.marchCoarseVoxels)
                );
                const float dt = min(stepVoxels / voxelsPerUnit, segment.y - t);
                if (majorant <= 0.f)
                {
                    t += dt;
                    haveA = false;
                    ++steps;
                    continue;
                }
                if (!haveA)
                {
                    a = marchSampleAt(ray, t, withSource, withSingle);
                    haveA = true;
                }
                b = marchSampleAt(ray, t + dt, withSource, withSingle);
                haveB = true;
            }
            const float h = (b.t - a.t) * ray.worldLength;
            if (params.marchTolerance <= 0.f)
            {
                // No refinement: one sample per step, as the fixed march, integrated analytically between the step's ends.
                const SegmentLight whole = segmentLight(a, b, h, singleScale);
                radiance += transmittance * whole.cache;
                singleRadiance += transmittance * whole.single;
                transmittance *= exp(-whole.depth);
                a = b;
                t = b.t;
                haveB = false;
                ++steps;
                continue;
            }
            MarchSample m = marchSampleAt(ray, 0.5f * (a.t + b.t), withSource, withSingle, false);
            m.cache = 0.5f * (a.cache + b.cache);
            ++steps;
            const SegmentLight whole = segmentLight(a, b, h, singleScale);
            const SegmentLight first = segmentLight(a, m, 0.5f * h, singleScale);
            const SegmentLight second = segmentLight(m, b, 0.5f * h, singleScale);
            const float firstTransmittance = exp(-first.depth);
            const float3 halves = first.cache + first.single + firstTransmittance * (second.cache + second.single);
            const float halvesDepth = first.depth + second.depth;
            // Error of the whole-segment model: its light, and its transmittance times the light behind (at most about the
            // sources here plus the sky), relative to 1 + the pixel's radiance so far and what may still arrive.
            const float behind = max(max(maxComponent(m.cache * exp(-params.worldCacheModulation * m.sunCache)), maxComponent(singleScale) * exp(-m.sunSingle)), skyLevel);
            const float error = transmittance * (maxComponent(abs(whole.cache + whole.single - halves)) + abs(exp(-whole.depth) - exp(-halvesDepth)) * behind);
            const float scale = 1.f + maxComponent(radiance + singleRadiance) + transmittance * behind;
            if (error <= params.marchTolerance * scale || b.t - a.t <= minUnits || depth >= kMarchStack)
            {
                radiance += transmittance * (first.cache + firstTransmittance * second.cache);
                singleRadiance += transmittance * (first.single + firstTransmittance * second.single);
                transmittance *= exp(-halvesDepth);
                a = b;
                t = b.t;
                haveB = depth > 0;
                if (haveB)
                    b = pending[--depth];
                // Once all the light that can still arrive is within the tolerance, the rest of the ray is closed off as
                // an opaque medium lit like this sample.
                const float3 cacheHere = a.cache * exp(-params.worldCacheModulation * a.sunCache);
                const float3 singleHere = singleScale * exp(-a.sunSingle);
                const float remaining = transmittance * max(maxComponent(cacheHere + singleHere), skyLevel);
                if (a.sigma > 0.f && remaining <= params.marchTolerance * (1.f + maxComponent(radiance + singleRadiance)))
                {
                    radiance += transmittance * cacheHere;
                    singleRadiance += transmittance * singleHere;
                    transmittance = 0.f;
                    break;
                }
            }
            else
            {
                pending[depth++] = b;
                b = m;
            }
        }
    }

    /// The fixed march's voxel "drip" artefacts against the path tracer come from its 1-voxel minimum step (a voxel of cloud is
    /// ~10 optical depths, so a sample sits half a voxel inside the lit skin), not from the cache; finer steps fix them at
    /// 10x the cost (see the adaptive march above). Background and single scatter are exact with fine steps (log error
    /// 0.0002 / 0.0006), so all remaining error is the cache.
    /// Majorant-bounded march of [segment.x, segment.y]: every step spans at most stepOpticalDepth at the local extinction
    /// bound, so thick shells are resolved at sub-voxel spacing while empty blocks are crossed in one step.
    /// radiance accumulates the camera-basis source when withSource is set; residualRadiance accumulates the rasterized residual
    /// whenever the segment lies in a resident page's footprint.
    void marchSegment(
        CutRay ray,
        float2 segment,
        bool withSource,
        bool residual,
        inout float3 radiance,
        inout float3 residualRadiance,
        inout float transmittance,
        inout uint steps
    )
    {
        float opacityDistance = 0.f;
        marchSegmentMoment(ray, segment, withSource, residual, radiance, residualRadiance, transmittance, steps, opacityDistance);
    }

    /// marchSegment that also accumulates the world distance from the camera weighted by each step's share of the opacity
    /// (divided by the opacity, the distance the ray's light comes from).
    void marchSegmentMoment(
        CutRay ray,
        float2 segment,
        bool withSource,
        bool residual,
        inout float3 radiance,
        inout float3 residualRadiance,
        inout float transmittance,
        inout uint steps,
        inout float opacityDistance
    )
    {
        float fineVoxels = 1.f;
        marchSegmentMoment(ray, segment, withSource, residual, radiance, residualRadiance, transmittance, steps, opacityDistance, fineVoxels);
    }

    /// marchSegmentMoment continuing from fineVoxels: the sampled voxel edge the march last read, which bounds its next step. A
    /// march resumed mid-ray passes the one it stopped with; starting at 1 makes the first step cross a whole domain voxel on one
    /// sample. Found splitting rays at cells for the cell views (51c2dace), where every resumed chord's first step was that long.
    void marchSegmentMoment(
        CutRay ray,
        float2 segment,
        bool withSource,
        bool residual,
        inout float3 radiance,
        inout float3 residualRadiance,
        inout float transmittance,
        inout uint steps,
        inout float opacityDistance,
        inout float fineVoxels
    )
    {
        const float cellWidth = float(params.hstrCellWidth);
        const float voxelsPerUnit = length(ray.direction) * cellWidth;
        if (voxelsPerUnit <= 1e-6f)
            return;
        const float worldPerVoxel = ray.worldLength / voxelsPerUnit;
        const bool cached = params.debugView == kWorldCacheView || params.debugView == kBeamView;
        if (cached && SHIP(adaptiveMarch, 0u) != 0)
        {
            integrateCachedSegment(ray, segment, withSource, residual, radiance, residualRadiance, transmittance, steps);
            return;
        }
        float3 held = 0.f;
        float3 heldAt = 0.f;
        float heldSun = 0.f;
        float3 sunAt = float3(1e30f);
        float3 heldSource = 0.f;   ///< Source at the previous step's end, which is this step's start (cloudSourceLinear).
        float3 heldResidual = 0.f; ///< The same for the single-scattered residual.
        bool heldEnds = false;     ///< Whether those two hold a value from a step that ended where this one starts.
        uint lastEntry = kCloudRefNone; ///< Brick the previous density sample read, so a change of it is one brick crossing.
        float heldSigma = -1.f; ///< The previous step's end extinction, or negative where the march jumped (cloudTrapezoid).
        float heldStep = -1.f;  ///< Step the previous sample's density bounds allow (cloudLocalStep), or negative for the majorant rule.
        uint sinceLighting = ~0u;
        float t = segment.x;
        // fineVoxels: voxel edge (domain voxels) of the density the last sample read: fine bricks shorten the minimum step to theirs.
        [loop]
        while (t < segment.y && transmittance > params.cloudMinTransmittance && steps < params.maxMarchSteps)
        {
            const float3 v = (ray.position + t * ray.direction) * cellWidth;
            const float majorant = majorantAt(v);
            const float minimumStep = clamp(fineVoxels, params.cloudFineMinVoxels, 1.f) * params.minStepVoxels;
            // The sea's majorant is the maximum over a dilated 4-voxel block, so near any cloud it pins every step to the minimum:
            // where the last sample's own bounds allow more (cloudLocalStep), they set the step instead.
            const float stepVoxels = clamp(params.stepOpticalDepth / max(1e-6f, majorant * worldPerVoxel), minimumStep, params.maxStepVoxels);
            const float dt = heldStep > 0.f ? min(clamp(heldStep, minimumStep / voxelsPerUnit, params.maxStepVoxels / voxelsPerUnit), segment.y - t)
                                            : min(stepVoxels / voxelsPerUnit, segment.y - t);
            CloudSample sample = {};
            float tSample = t + 0.5f * dt; ///< Where the step's density sample sits: cloudEmptyExit measures its box from there.
            if (majorant > 0.f)
            {
                // Midpoint quadrature (cloudTrapezoid 0), or the trapezoid rule on the step's ends: the previous step's end is this
                // step's start, so both cost one density sample per step, but the trapezoid is exact wherever the extinction is
                // linear and lets the steps grow (the sample count is what the march costs).
                // Local stepping samples the step's end so the bounds it reads are the ones the next step runs through.
                // cloudSourceLinear needs the sample, and so the source, at the step's end, which is the next step's start.
                const bool ends = SHIP(cloudTrapezoid, 0u) != 0 || SHIP(cloudLocalStep, 0u) != 0 || SHIP(cloudSourceLinear, 0u) != 0;
                const bool trapezoid = ends && heldSigma >= 0.f;
                tSample = ends ? t + dt : t + 0.5f * dt;
                float3 midpoint = (ray.position + tSample * ray.direction) * cellWidth;
                // The sample stands for the whole step, so its footprint is the wider of the pixel's and the step's (cloudStepFootprint).
                const float footprint = SHIP(cloudStepFootprint, 0.f) > 0.f
                                            ? max(rayFootprint(ray, tSample), SHIP(cloudStepFootprint, 0.f) * dt * ray.worldLength)
                                            : rayFootprint(ray, tSample);
                // Higher-order quadrature of the step (cloudQuadrature): the density is minValue + range * t^2 on a trilinear t, so
                // inside one interpolation cell it is a polynomial of degree at most six in the ray parameter, which Gauss-Legendre
                // integrates far better than the midpoint does. The nodes replace the step's own subdivision rather than adding to
                // it - the caller scales minStepVoxels with the node count to hold the taps per voxel fixed - so this asks whether
                // the coarse-step error is quadrature error or aliasing. The centre-most node stands for the step downstream: the
                // lighting, the sun lookup and the empty-cell skip all vary smoothly across one step.
                // Provably zero without asking the bricks (cloudTightReject): the step's sample lies inside a majorant block whose
                // undilated maximum is zero, and far enough inside that its trilinear support cannot reach the neighbour the dilated
                // majorant was covering. Left off for the quadrature and oracle paths, which take their own taps at their own places.
                // Mode 2 does the load and the test and then throws the answer away, which separates the two halves of the cost: the
                // extra fetch every sample pays against the brick walks the rejections save. Packing hstrTightMajorant into
                // hstrMajorant as a second channel would buy exactly the difference.
                bool tightZero = false;
                if (SHIP(cloudTightReject, 0u) != 0 && SHIP(cloudQuadrature, 1u) <= 1 && SHIP(cloudLongitudinalOracle, 0u) <= 1)
                {
                    const float3 block = seaVoxel(midpoint) * 0.25f;
                    const float3 frac = block - floor(block);
                    tightZero = all(frac > 0.125f) && all(frac < 0.875f) && tightMajorantAt(midpoint) <= 0.f &&
                                SHIP(cloudTightReject, 0u) == 1;
                }
                float sigma;
                if (tightZero)
                {
                    // sample stays zero-initialised, so it claims neither a brick nor an empty box: no cell skip, no transfer record,
                    // and the step advances by dt as it would have. fineVoxels goes back to the coarsest, the block holding nothing
                    // that could bound the next step shorter.
                    sigma = 0.f;
                    fineVoxels = 1.f;
                }
                else if (SHIP(cloudQuadrature, 1u) > 1)
                {
                    // Gauss-Legendre nodes and weights on [0, 1] for two and three points.
                    const float3 nodes = SHIP(cloudQuadrature, 1u) == 2 ? float3(0.21132487f, 0.78867513f, 0.f)
                                                                    : float3(0.11270167f, 0.5f, 0.88729833f);
                    const float3 weights = SHIP(cloudQuadrature, 1u) == 2 ? float3(0.5f, 0.5f, 0.f)
                                                                       : float3(0.27777778f, 0.44444444f, 0.27777778f);
                    const uint centre = SHIP(cloudQuadrature, 1u) / 2u;
                    sigma = 0.f;
                    for (uint q = 0; q < SHIP(cloudQuadrature, 1u); ++q)
                    {
                        const float tq = t + nodes[q] * dt;
                        const float3 vq = (ray.position + tq * ray.direction) * cellWidth;
                        CloudSample node = {};
                        float nodeVoxels = 1.f;
                        sigma += weights[q] * cameraExtinctionAtVoxel(vq, rayFootprint(ray, tq), nodeVoxels, node);
                        if (q == centre)
                        {
                            sample = node;
                            fineVoxels = nodeVoxels;
                            tSample = tq;
                            midpoint = vq;
                        }
                    }
                }
                else
                    sigma = cameraExtinctionAtVoxel(midpoint, footprint, fineVoxels, sample);
                // Longitudinal oracle: resolve the step's interior at the pixel's own footprint, and with it the optical depth the
                // step really carries and the place its extinction deposits the light. M0 accumulates the sub-step opacities under
                // their running transmittance and M1 the same weighted by distance along the step, so M1 / M0 is the opacity
                // centroid. The centre observation keeps the sample the lighting and the skip logic read, as the quadrature does.
                float oracleCentroid = 0.5f;
                if (SHIP(cloudLongitudinalOracle, 0u) > 1)
                {
                    const uint taps = SHIP(cloudLongitudinalOracle, 0u);
                    const float sub = dt / float(taps);
                    const uint centre = taps / 2u;
                    float oracleDepth = 0.f;
                    float moment0 = 0.f;
                    float moment1 = 0.f;
                    float running = 1.f;
                    for (uint o = 0; o < taps; ++o)
                    {
                        const float to = t + (float(o) + 0.5f) * sub;
                        const float3 vo = (ray.position + to * ray.direction) * cellWidth;
                        CloudSample tap = {};
                        float tapVoxels = 1.f;
                        const float sigmaO = cameraExtinctionAtVoxel(vo, rayFootprint(ray, to), tapVoxels, tap);
                        const float depthO = sigmaO * sub * ray.worldLength;
                        const float alphaO = 1.f - exp(-depthO);
                        moment0 += running * alphaO;
                        moment1 += running * alphaO * (float(o) + 0.5f) / float(taps);
                        running *= 1.f - alphaO;
                        oracleDepth += depthO;
                        if (o == centre)
                        {
                            sample = tap;
                            fineVoxels = tapVoxels;
                        }
                    }
                    sigma = oracleDepth / max(dt * ray.worldLength, 1e-12f);
                    oracleCentroid = moment0 > 1e-9f ? moment1 / moment0 : 0.5f;
                }
                const float depth = (trapezoid ? 0.5f * (heldSigma + sigma) : sigma) * dt * ray.worldLength;
                // Where the step's one source evaluation sits, and the sample that describes that place. These are kept apart from
                // tSample/midpoint/sample - the observation the step's density came from - because three consumers read a sample's
                // metadata against the position it was taken at, and the centroid moves the source by up to half a step:
                //   sample.t   is the ray parameter cloudLocalStep measures the next step from;
                //   cloudEmptyExit takes its box from sample.x/lo/hi and returns the exit as an offset from the t it is passed, so a
                //     centroid ahead of the observation would skip past where density returns (and the case where they part company
                //     furthest is precisely an empty observation among dense taps, which is the case that skips);
                //   sunDepthAt reads the near field from sample.entry/x and continues the far field from the position given.
                // The centroid's own sample is re-read rather than borrowed. That costs a density sample, which this oracle is free
                // to spend - it measures what the quantity would be worth if it were free, not a way to make it free.
                float3 sourcePoint = midpoint;
                CloudSample sourceSample = sample;
                if (SHIP(cloudLongitudinalOracle, 0u) > 1 && SHIP(cloudOracleCentroid, 0u) != 0)
                {
                    const float tSource = t + oracleCentroid * dt;
                    sourcePoint = (ray.position + tSource * ray.direction) * cellWidth;
                    float sourceVoxels = 1.f;
                    cameraExtinctionAtVoxel(sourcePoint, rayFootprint(ray, tSource), sourceVoxels, sourceSample);
                    sourceSample.t = tSource;
                }
                heldSigma = ends ? sigma : -1.f;
                sample.t = tSample;
                if (SHIP(cloudTransferClasses, 0u) > 0 && sample.brick && !sample.empty && sample.entry != lastEntry)
                {
                    recordTransferCrossing(ray, sample);
                    lastEntry = sample.entry;
                }
                heldStep = SHIP(cloudLocalStep, 0u) != 0 && sample.brick && !sample.empty
                               ? cloudLocalStep(ray, sample, params.stepOpticalDepth, gScene.getGridVolume(0).data.densityScale * params.densityScale)
                               : -1.f;
                const float alpha = 1.f - exp(-depth);
                // Extinction-weighted linear source across the step (cloudSourceLinear). Compositing radiance += T * alpha * S(centre)
                // treats one source value as standing for the whole interval, and that - not the optical depth - is what a long step
                // gets wrong (see cloudQuadrature). With S linear from the step's start to its end and sigma constant across it,
                //   C = S0 * (1 - A) + (S1 - S0) * (1 - A - tau A) / tau,   A = exp(-tau),
                // which is exact and costs nothing extra: the sample already sits at the step's end, so this step's S1 is the next
                // step's S0 and there is still one source evaluation per step. weightEnd is that second term's coefficient, rising to
                // alpha/2 for a thin step, where the two ends weigh equally, and falling towards zero for a thick one, where the
                // front of the step is all that is seen.
                const float weightEnd = SHIP(cloudSourceLinear, 0u) != 0
                                            ? (depth > 1e-4f ? (alpha - depth * (1.f - alpha)) / depth : 0.5f * depth)
                                            : 0.f;
                const float weightStart = alpha - weightEnd;
                if ((withSource || residual) && alpha > 1e-6f)
                {
                    if (withSource)
                    {
                        // The cache is smooth at cell scale: it is re-evaluated every lightingStride contributing steps, and
                        // whenever the march has moved a cache cell since, so a held value never crosses a gap between puffs.
                        const float3 moved = sourcePoint - heldAt;
                        const float cellVoxels = float(params.worldCacheCellVoxels);
                        if (!cached || sinceLighting >= params.lightingStride || dot(moved, moved) > cellVoxels * cellVoxels)
                        {
                            held = cached ? worldCacheRadiance(sourcePoint, ray.viewDirection) : sourceAtVoxel(sourcePoint);
                            heldAt = sourcePoint;
                            sinceLighting = 0;
                        }
                        ++sinceLighting;
                        if (SHIP(cloudSourceLinear, 0u) != 0)
                        {
                            radiance += transmittance * ((heldEnds ? heldSource : held) * weightStart + held * weightEnd);
                            heldSource = held;
                        }
                        else
                            radiance += transmittance * alpha * held;
                    }
                    if (residual)
                    {
                        // The sun depth is smooth along the ray: cloudSunReuse holds one for the samples within that distance. A step
                        // whose end sample is empty still carries the light of its start (trapezoid), and an empty sample has no bake
                        // to read - it would take the live march - so it keeps the last depth instead.
                        const float3 moved = sourcePoint - sunAt;
                        const bool stale = SHIP(cloudSunReuse, 0.f) <= 0.f || dot(moved, moved) > SHIP(cloudSunReuse, 0.f) * SHIP(cloudSunReuse, 0.f);
                        if (stale && !(sourceSample.empty && sunAt.x < 1e29f))
                        {
                            heldSun = sunDepthAt(sourcePoint, footprint, sourceSample);
                            sunAt = sourcePoint;
                        }
                        const float sunDepth = heldSun;
                        const float3 residualEnd = residualAtDepth(sourcePoint, sunDepth, ray.sunOctaves);
                        if (SHIP(cloudSourceLinear, 0u) != 0)
                        {
                            residualRadiance +=
                                transmittance * ((heldEnds ? heldResidual : residualEnd) * weightStart + residualEnd * weightEnd);
                            heldResidual = residualEnd;
                        }
                        else
                            residualRadiance += transmittance * alpha * residualEnd;
                    }
                    heldEnds = SHIP(cloudSourceLinear, 0u) != 0;
                }
                opacityDistance += transmittance * alpha * (ray.entryDistance + tSample * ray.worldLength);
                transmittance *= 1.f - alpha;
            }
            // Past an exactly empty cell or octant, or a zero 16-voxel block, in one step: the next step starts where density can return.
            const float next = majorant <= 0.f
                                   ? min(max(t + dt, majorantZeroExit(ray, t)), segment.y)
                                   : (sample.empty && SHIP(cloudEmptySkip, 1u) != 0 ? min(max(t + dt, cloudEmptyExit(ray, tSample, sample)), segment.y)
                                                                                 : t + dt);
            // A step that took no sample at its end, or that jumped, leaves no start value for the next trapezoid, local step or
            // source basis: the next step begins somewhere else, so its start source is its own.
            if (majorant <= 0.f || next > t + dt)
            {
                heldSigma = -1.f;
                heldStep = -1.f;
                heldEnds = false;
            }
            t = next;
            ++steps;
        }
    }

    /// Counts along marchSegment's camera march (marchProbe; scripts read the frame mean through an exact comparison with a black
    /// stored frame). Per sun query (alpha > 1e-6): 1 all, 2 own bake, 3 proxy, 4 live because the brick is finer than asked, 5 live
    /// because no bake (own or ancestor) matches, 6 live on an empty sample, 8 unbaked on bricks with no bake of any class, 9 unbaked
    /// on bricks coarser than asked, 18 an ancestor's bake. Per density sample: 7 all, 10 at the minimum step, 11 at the maximum step,
    /// 12 non-empty with alpha <= 1e-6, 13 exactly empty, 14 alpha > 0.5, 15 step length in sampled voxels. Per step: 17 majorant
    /// zero. Per density sample: 20 on a different brick than the last brick sample (brick visits), 21 read a brick. Per pixel: 16
    /// terminated by transmittance, 22 more than 32 steps, 23 more than 64 steps, 24 in a block whose undilated majorant is zero,
    /// 25 the same and far enough inside the block that a tight test could reject them, 26 brick samples that change the pixel,
    /// 27 the distinct brick entries holding at least one of those (26 over 27 is the contributing run length).
    /// The trilinear page a brick-level transport node would carry, evaluated at asset source position x: the brick's own eight
    /// shared corner densities (the apron texels, so neighbouring bricks agree on them) interpolated over the brick. This is the
    /// same representation HSTRCutNode.transmittanceCorners holds for an HST leaf, one level of the resident pyramid at a time.
    float cloudBrickPage(uint entry, float3 x)
    {
        const HSTRCloudBrick brick = hstrCloudBricks[entry];
        if ((brick.info & kCloudBrickConstant) != 0)
            return brick.minValue;
        const uint level = brick.info & 15u;
        const float3 levelPosition = (x + 0.5f) / float(1u << level) - 0.5f;
        const int3 i = int3(floor(x + 0.5f));
        const int3 brickCoord = (i >> int(level)) >> 3;
        const float3 local = clamp(levelPosition - float3(brickCoord * 8), -0.5f, 7.5f);
        const uint shift = params.cloudAtlasShift;
        const uint mask = (1u << shift) - 1u;
        const uint3 slot = uint3(brick.slot & mask, (brick.slot >> shift) & mask, brick.slot >> (2u * shift));
        const float3 f = (local + 0.5f) / 8.f;
        float page = 0.f;
        [unroll]
        for (uint corner = 0; corner < 8; ++corner)
        {
            const float3 c = float3(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
            const float3 texel = float3(slot * kCloudBrickEdge) + 1.5f + (c * 8.f - 0.5f);
            const float t = hstrCloudAtlas.SampleLevel(hstrLinearClampSampler, texel * params.cloudAtlasInvSize, 0.f);
            const float3 weights = lerp(1.f - f, f, c);
            page += (brick.minValue + brick.range * t * t) * weights.x * weights.y * weights.z;
        }
        return page;
    }

    float probeMarch(CutRay ray, uint mode)
    {
        const float cellWidth = float(params.hstrCellWidth);
        const float voxelsPerUnit = length(ray.direction) * cellWidth;
        if (voxelsPerUnit <= 1e-6f)
            return 0.f;
        const float worldPerVoxel = ray.worldLength / voxelsPerUnit;
        float t = 0.f;
        float fineVoxels = 1.f;
        float transmittance = 1.f;
        float result = 0.f;
        uint steps = 0;
        uint lastEntry = kCloudRefNone - 1u;
        uint lastInstance = 0;
        uint lastLiveEntry = kCloudRefNone - 1u;
        uint lastLiveInstance = 0;
        [loop]
        while (t < 1.f && transmittance > kMinTransmittance && steps < params.maxMarchSteps)
        {
            const float3 v = (ray.position + t * ray.direction) * cellWidth;
            const float majorant = majorantAt(v);
            const float minimumStep = clamp(fineVoxels, params.cloudFineMinVoxels, 1.f) * params.minStepVoxels;
            const float stepVoxels = clamp(params.stepOpticalDepth / max(1e-6f, majorant * worldPerVoxel), minimumStep, params.maxStepVoxels);
            const float dt = min(stepVoxels / voxelsPerUnit, 1.f - t);
            CloudSample sample = {};
            if (majorant > 0.f)
            {
                const float3 midpoint = (ray.position + (t + 0.5f * dt) * ray.direction) * cellWidth;
                const float footprint = rayFootprint(ray, t + 0.5f * dt);
                const float sigma = cameraExtinctionAtVoxel(midpoint, footprint, fineVoxels, sample);
                const float alpha = 1.f - exp(-sigma * dt * ray.worldLength);
                // Whether a brick-level transport page could have stood in for this density sample, which is the prerequisite for a
                // transport hierarchy over the resident pyramid: if a page cannot reproduce the samples the camera already takes at
                // the level residency already chose, no adaptive cut over that pyramid can make integratePage() the common case.
                // 30 counts the samples that read a brick, 31 those a page reproduces to 10% of the brick's peak density, 34 to 2%,
                // and 32 sums the relative page error so 32 over 30 is its mean. The page is exactly HSTRCutNode's representation:
                // the brick's eight shared corner densities, trilinearly interpolated (cloudBrickPage).
                if (mode >= 30 && mode <= 34 && sample.brick && !sample.empty && sample.entry != kCloudRefNone)
                {
                    const HSTRCloudBrick brick = hstrCloudBricks[sample.entry];
                    const float peak = max(brick.minValue + brick.range, 1e-12f);
                    // sigma carries the instance's density scale and the distance fade; the page does not, so compare stored values.
                    const float stored = cloudBrickDensity(sample.entry, sample.x, int3(floor(sample.x + 0.5f)));
                    const float error = abs(cloudBrickPage(sample.entry, sample.x) - stored) / peak;
                    result += mode == 30 ? 1.f : 0.f;
                    result += mode == 31 && error <= 0.1f ? 1.f : 0.f;
                    result += mode == 34 && error <= 0.02f ? 1.f : 0.f;
                    result += mode == 32 ? error : 0.f;
                    result += mode == 33 && alpha > 1e-6f ? 1.f : 0.f;
                }
                result += mode == 7 ? 1.f : 0.f;
                result += mode == 10 && stepVoxels <= minimumStep ? 1.f : 0.f;
                result += mode == 11 && stepVoxels >= params.maxStepVoxels ? 1.f : 0.f;
                result += mode == 12 && !sample.empty && alpha <= 1e-6f ? 1.f : 0.f;
                result += mode == 13 && sample.empty ? 1.f : 0.f;
                result += mode == 14 && alpha > 0.5f ? 1.f : 0.f;
                result += mode == 15 ? dt * voxelsPerUnit / max(fineVoxels, 1e-3f) : 0.f;
                // How much of the density sampling a tight majorant could reject outright. Both majorants sit on the same 4-voxel
                // block grid, so where the undilated maximum is zero the block's own texels are all zero, and a sample at least half
                // a voxel inside every face reads only those texels - its trilinear support cannot leave the block, so its density is
                // exactly zero and the brick walk that found it out was wasted. Mode 24 counts the samples in a tight-zero block
                // (the whole pool the dilation costs us), 25 the interior ones a conservative test could actually reject.
                if (mode == 24 || mode == 25)
                {
                    const float3 bv = seaVoxel(midpoint) * 0.25f;
                    const float3 frac = bv - floor(bv);
                    const bool interior = all(frac > 0.125f) && all(frac < 0.875f);
                    result += tightMajorantAt(midpoint) <= 0.f && (mode == 24 || interior) ? 1.f : 0.f;
                }
                result += mode == 20 && sample.brick && (sample.entry != lastEntry || sample.instance != lastInstance) ? 1.f : 0.f;
                result += mode == 21 && sample.brick ? 1.f : 0.f;
                if (sample.brick)
                {
                    lastEntry = sample.entry;
                    lastInstance = sample.instance;
                }
                // What a compact interval representation would actually replace. Samples per brick visit (21 over 20) counts every
                // sample in the visit, and most of those are empty or too faint to change the pixel - both now measured to be nearly
                // free (cloudTightReject, cloudThinDepth), so replacing them buys nothing. The run that matters is the CONTRIBUTING
                // one: 26 counts brick samples that change the pixel, 27 the distinct brick entries that hold at least one. Their
                // ratio is how many expensive events one interval operation could stand for, and if it is near 1 no interval
                // representation can pay however good its encoding.
                // How far the parent climb in cloudAssetBrickAt actually walks, which is what an ancestor table keyed on
                // (entry, desired level) would replace. sample.level is where the climb ended and sample.desired what was asked, so
                // their difference bounds the iterations: zero means the mapped brick already answered and the loop never ran.
                result += mode == 28 && sample.brick && sample.level > sample.mappedLevel ? float(sample.level - sample.mappedLevel) : 0.f;
                result += mode == 29 && sample.brick && sample.level > sample.mappedLevel ? 1.f : 0.f;
                const bool contributes = sample.brick && !sample.empty && alpha > 1e-6f;
                result += mode == 26 && contributes ? 1.f : 0.f;
                result += mode == 27 && contributes && (sample.entry != lastLiveEntry || sample.instance != lastLiveInstance) ? 1.f : 0.f;
                if (contributes)
                {
                    lastLiveEntry = sample.entry;
                    lastLiveInstance = sample.instance;
                }
                if (alpha > 1e-6f)
                {
                    const bool keyed = sample.brick && !sample.empty &&
                                       cloudSunSlotOf(sample.entry, hstrCloudInstances[sample.instance].sunClass) != kCloudRefNone;
                    result += mode == 1 ? 1.f : 0.f;
                    result += mode == 2 && keyed && sample.level >= sample.desired ? 1.f : 0.f;
                    result += mode == 3 && !sample.brick ? 1.f : 0.f;
                    result += mode == 4 && sample.brick && !sample.empty && sample.level < sample.desired ? 1.f : 0.f;
                    uint bakedEntry = sample.entry;
                    const bool keyedUp = sample.brick && !sample.empty &&
                                         cloudSunSlotUp(bakedEntry, hstrCloudInstances[sample.instance].sunClass) != kCloudRefNone;
                    result += mode == 5 && sample.brick && !sample.empty && !keyedUp ? 1.f : 0.f;
                    result += mode == 18 && keyedUp && !keyed && sample.level >= sample.desired ? 1.f : 0.f;
                    result += mode == 6 && sample.brick && sample.empty ? 1.f : 0.f;
                    if ((mode == 8 || mode == 9) && sample.brick && !sample.empty && !keyed)
                    {
                        bool anyClass = false;
                        for (uint c = 0; c < 16; ++c)
                            anyClass = anyClass || cloudSunSlotOf(sample.entry, c) != kCloudRefNone;
                        result += mode == 8 && !anyClass ? 1.f : 0.f;
                        result += mode == 9 && sample.level > sample.desired ? 1.f : 0.f;
                    }
                }
                transmittance *= 1.f - alpha;
            }
            result += mode == 17 && majorant <= 0.f ? 1.f : 0.f;
            if (majorant <= 0.f)
                t = min(max(t + dt, majorantZeroExit(ray, t)), 1.f);
            else
                t = sample.empty && SHIP(cloudEmptySkip, 1u) != 0 ? min(max(t + dt, cloudEmptyExit(ray, t + 0.5f * dt, sample)), 1.f) : t + dt;
            ++steps;
        }
        if (mode == 16)
            return transmittance <= kMinTransmittance ? 1.f : 0.f;
        if (mode == 22 || mode == 23)
            return steps > (mode == 22 ? 32u : 64u) ? 1.f : 0.f;
        return result;
    }

    float transmittancePageAt(HSTRCutNode node, float3 leafPosition)
    {
        const float3 local = saturate((leafPosition - float3(node.minLeft.xyz)) / float3(node.maxRight.xyz - node.minLeft.xyz));
        const float x00 = lerp(node.transmittanceCorners0.x, node.transmittanceCorners0.y, local.x);
        const float x10 = lerp(node.transmittanceCorners0.z, node.transmittanceCorners0.w, local.x);
        const float x01 = lerp(node.transmittanceCorners1.x, node.transmittanceCorners1.y, local.x);
        const float x11 = lerp(node.transmittanceCorners1.z, node.transmittanceCorners1.w, local.x);
        return lerp(lerp(x00, x10, local.y), lerp(x01, x11, local.y), local.z);
    }

    /// Accepted HST node: two-point Gauss quadrature of its trilinear transmittance page.
    void integratePage(HSTRCutNode node, CutRay ray, float2 segment, bool withSource, inout float3 radiance, inout float transmittance)
    {
        static const float kGaussOffset = 0.2886751345948129f;
        const float midpoint = 0.5f * (segment.x + segment.y);
        const float segmentLength = segment.y - segment.x;
        const float3 p0 = ray.position + (midpoint - kGaussOffset * segmentLength) * ray.direction;
        const float3 p1 = ray.position + (midpoint + kGaussOffset * segmentLength) * ray.direction;
        const float extinction0 = transmittancePageAt(node, p0);
        const float extinction1 = transmittancePageAt(node, p1);
        const float opticalDepth = 0.5f * ray.worldLength * segmentLength * (extinction0 + extinction1);
        if (opticalDepth <= 1e-8f)
            return;
        const float segmentTransmittance = exp(-opticalDepth);
        if (withSource)
        {
            const float weight0 = extinction0 * exp(-opticalDepth * (0.5f - kGaussOffset));
            const float weight1 = extinction1 * exp(-opticalDepth * (0.5f + kGaussOffset));
            const float3 source = (weight0 * cameraLightingAt(p0) + weight1 * cameraLightingAt(p1)) / max(1e-8f, weight0 + weight1);
            radiance += transmittance * (1.f - segmentTransmittance) * source;
        }
        transmittance *= segmentTransmittance;
    }

    /// Composites one tile's front-to-back sorted cut. Leaves whose page error exceeds the tolerance, or that hold a
    /// resident residual page, are marched; accepted nodes use their transmittance pages. Returns radiance before albedo.
    float3 traverseCut(
        CutRay ray,
        uint tileID,
        bool withSource,
        bool withResidual,
        inout float3 residualRadiance,
        inout float transmittance,
        inout uint visited,
        inout uint steps
    )
    {
        float3 radiance = 0.f;
        const uint count = hstrTileNodeCounts[tileID];
        if (count > params.hstrMaxTileNodes)
        {
            // The tile's cut overflowed its list: march the whole volume segment on the majorant grid.
            marchSegment(ray, float2(0.f, 1.f), withSource, withResidual, radiance, residualRadiance, transmittance, steps);
            visited = count;
            return radiance;
        }
        const uint base = tileID * params.hstrMaxTileNodes;
        [loop]
        for (uint i = 0; i < count && transmittance > kMinTransmittance; ++i)
        {
            const HSTRCutNode node = hstrCutNodes[hstrTileNodes[base + i]];
            float2 segment;
            if (!intersectCutNode(node, ray.position, ray.direction, segment))
                continue;
            ++visited;
            const bool leaf = node.minLeft.w == ~0u;
            const float pageError = transmittance * (1.f - exp(-node.transmittancePage.x * ray.worldLength * (segment.y - segment.x)));
            // Resident residual pages are the leaves whose footprint this tile's cut list rasterized; their segments are marched
            // in every pass so that transmittance stays identical between camera queries and pixels.
            const bool resident = leaf && leafResidualResident(node.minLeft.xyz);
            if (leaf && (resident || pageError > params.cutTransmittanceTolerance))
                marchSegment(ray, segment, withSource, withResidual && resident, radiance, residualRadiance, transmittance, steps);
            else
                integratePage(node, ray, segment, withSource, radiance, transmittance);
        }
        return radiance;
    }

    /// Premultiplied cloud radiance (rgb, background excluded) and opacity (a) along one camera ray.
    float4 cameraQuery(float2 pixelPosition, uint tileID)
    {
        const GridVolume volume = gScene.getGridVolume(0);
        CutRay ray;
        if (!makeCutRay(gScene.camera.data.posW, cameraRayDirection(pixelPosition), volume, ray))
            return 0.f;
        float transmittance = 1.f;
        uint visited = 0;
        uint steps = 0;
        float3 residual = 0.f;
        // The tile basis carries only the low-frequency camera basis; residuals are rasterized per pixel.
        const float3 radiance = traverseCut(ray, tileID, true, false, residual, transmittance, visited, steps);
        return float4(radiance * volume.data.albedo, 1.f - transmittance);
    }

    /// Tile-corner lattice plus one centre query per tile. Deterministic, and only rebuilt when the camera, cut or lighting change.
    void writeCameraQuery(uint2 query)
    {
        if (any(query >= params.hstrLatticeDims) || gScene.getGridVolumeCount() == 0)
            return;
        const uint2 tile = min(query, params.hstrTileDims - 1u);
        const uint tileID = tile.x + params.hstrTileDims.x * tile.y;
        const float tileSize = float(params.hstrTileSize);
        hstrCameraQueryOutput[query] = cameraQuery(float2(query) * tileSize, tileID);
        if (all(query < params.hstrTileDims))
            hstrTileCenterOutput[query] = cameraQuery((float2(query) + 0.5f) * tileSize, tileID);
    }

    float maxComponent(float3 v) { return max(v.x, max(v.y, v.z)); }

    /// Fits the tile camera basis and decides, with split/merge hysteresis, whether the tile descends to per-pixel integration.
    void buildTileBasis(uint2 tile)
    {
        if (any(tile >= params.hstrTileDims) || gScene.getGridVolumeCount() == 0)
            return;
        const uint tileID = tile.x + params.hstrTileDims.x * tile.y;
        const float4 q00 = hstrCameraQueries[tile];
        const float4 q10 = hstrCameraQueries[tile + uint2(1, 0)];
        const float4 q01 = hstrCameraQueries[tile + uint2(0, 1)];
        const float4 q11 = hstrCameraQueries[tile + uint2(1, 1)];
        const float4 qc = hstrTileCenters[tile];
        HSTRTileBasis basis;
        basis.c0 = q00;
        basis.cx = q10 - q00;
        basis.cy = q01 - q00;
        basis.cxy = q11 - q10 - q01 + q00;
        const float4 bilinearCentre = 0.25f * (q00 + q10 + q01 + q11);
        // Hierarchical surplus: the centre query's departure from the bilinear corner basis. It becomes the coefficient of a
        // bubble function that vanishes on the tile boundary, and is the tile's a-posteriori error indicator.
        basis.bubble = qc - bilinearCentre;
        const float3 exactSource = qc.rgb / max(qc.a, 1e-4f);
        const float3 predictedSource = bilinearCentre.rgb / max(bilinearCentre.a, 1e-4f);
        float surplus = qc.a * maxComponent(abs(exactSource - predictedSource)) / (maxComponent(exactSource) + 1e-2f);
        if ((qc.a > 1e-3f) != (bilinearCentre.a > 1e-3f))
            surplus = 1.f;

        const bool wasRefined = hstrTileState[tileID] != 0;
        const bool refine = surplus > params.basisTolerance * (wasRefined ? params.basisHysteresis : 1.f);
        hstrTileState[tileID] = refine ? 1u : 0u;
        basis.info = float4(surplus, refine ? 1.f : 0.f, min(min(q00.a, q10.a), min(q01.a, q11.a)), 0.f);
        hstrTileBasis[tileID] = basis;
    }

    float4 evaluateTileBasis(HSTRTileBasis basis, float2 f)
    {
        return basis.c0 + basis.cx * f.x + basis.cy * f.y + basis.cxy * (f.x * f.y) +
               basis.bubble * (16.f * f.x * (1.f - f.x) * f.y * (1.f - f.y));
    }

    // ---------------------------------------------------------------------------------------------
    // World-space residual pages: rebuilt when the sun or the density changes, never on camera motion.

    float voxelWorldLength(GridVolume volume)
    {
        if (params.cloudDomain != 0)
            return dot(params.seaVoxelSize, 1.f / 3.f);
        return (length(mul(volume.data.transform, float4(1.f, 0.f, 0.f, 0.f)).xyz) +
                length(mul(volume.data.transform, float4(0.f, 1.f, 0.f, 0.f)).xyz) +
                length(mul(volume.data.transform, float4(0.f, 0.f, 1.f, 0.f)).xyz)) /
               3.f;
    }

    float3 fineCellVoxel(uint3 cell) { return 2.f * float3(cell) + 1.f; }

    /// Sun optical depth from the centre of every voxel to the grid boundary, and page activation per leaf.
    void computeFineSun(uint3 voxel)
    {
        if (any(voxel >= params.hstrFineDims) || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        const float3 sunIndex = voxelDirection(normalize(params.sunDirection));
        const float voxelsPerWorld = length(sunIndex);
        const float3 direction = sunIndex / voxelsPerWorld;
        const float3 origin = float3(voxel);
        const float3 inverseDirection = 1.f / select(abs(direction) < 1e-8f, float3(1e-8f), direction);
        const float3 a = (-0.5f - origin) * inverseDirection;
        const float3 b = (float3(params.hstrExtinctionDims) - 0.5f - origin) * inverseDirection;
        const float3 far = max(a, b);
        // The sea only ends at the top and bottom of the cloud layer: neighbouring tiles shadow each other.
        const float exitDistance = max(0.f, params.seaMode != 0 ? far.y : min(far.x, min(far.y, far.z)));
        float opticalDepth = 0.f;
        float t = 0.f;
        uint steps = 0;
        [loop]
        while (t < exitDistance && opticalDepth < 60.f && steps < 4096)
        {
            const float3 v = origin + t * direction;
            const float majorant = majorantAt(v);
            const float stepVoxels = min(
                clamp(params.stepOpticalDepth * voxelsPerWorld / max(1e-6f, majorant), params.minStepVoxels * 0.5f, params.maxStepVoxels),
                exitDistance - t
            );
            if (majorant > 0.f)
                opticalDepth += extinctionAtVoxel(v + 0.5f * stepVoxels * direction) * stepVoxels / voxelsPerWorld;
            t += stepVoxels;
            ++steps;
        }
        opticalDepth = min(opticalDepth, 60.f);
        hstrFineSunOutput[voxel] = opticalDepth;
        // Residency: the most penetrating octave's sun arrival, weighted by the voxel's opacity.
        const float bb = params.octaveExtinction;
        const float opacity = 1.f - exp(-extinctionAtVoxel(origin) * voxelWorldLength(volume));
        const uint3 leaf = min(voxel / params.hstrCellWidth, params.hstrLeafDims - 1u);
        InterlockedMax(hstrLeafResidual[leafIndex(leaf)], asuint(opacity * exp(-bb * bb * bb * opticalDepth)));
    }

    /// Pages below the activation tolerance are not resident: their cells contribute nothing.
    void maskResidual(uint3 cell)
    {
        if (any(cell >= params.hstrFineDims))
            return;
        const uint3 leaf = min(cell / params.hstrCellWidth, params.hstrLeafDims - 1u);
        if (!leafResidualResident(leaf))
            hstrFineSunOutput[cell] = 60.f;
    }

    // ---------------------------------------------------------------------------------------------
    // Inside-cloud path: the local high-resolution working set.

    float3 deterministicResidual(GridVolume volume, Grid grid, float3 indexPosition, float3 viewDirection, inout Grid::Accessor accessor)
    {
        const float h = 1.5f;
        const float3 gradient = float3(
            grid.lookupLinearIndex(indexPosition + float3(h, 0.f, 0.f), accessor) -
                grid.lookupLinearIndex(indexPosition - float3(h, 0.f, 0.f), accessor),
            grid.lookupLinearIndex(indexPosition + float3(0.f, h, 0.f), accessor) -
                grid.lookupLinearIndex(indexPosition - float3(0.f, h, 0.f), accessor),
            grid.lookupLinearIndex(indexPosition + float3(0.f, 0.f, h), accessor) -
                grid.lookupLinearIndex(indexPosition - float3(0.f, 0.f, h), accessor)
        );
        const float density = max(0.f, grid.lookupLinearIndex(indexPosition, accessor));
        const float edge = saturate(2.f * length(gradient) / max(0.02f, density));
        if (edge <= 0.f)
            return 0.f;

        const float3 outwardIndex = -normalize(gradient + 1e-6f);
        const float3 sunIndex = normalize(mul(volume.data.invTransform, float4(normalize(params.sunDirection), 0.f)).xyz);
        const float facing = saturate(dot(outwardIndex, sunIndex));
        const float silver = facing * facing * edge;
        const float phase = 4.f * kPi * phaseHG(dot(normalize(params.sunDirection), viewDirection));
        return params.sunRadiance * (0.01f + 0.075f * silver) * (0.25f + phase);
    }

    float pixelJitter(uint2 pixel)
    {
        const uint seed = pixel.x * 1973u + pixel.y * 9277u + params.frameIndex * 26699u + 911u;
        uint value = seed;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        return (value & 0x00ffffffu) / 16777216.f;
    }

    float4 integrateVolume(Ray ray, GridVolume volume, uint stepCount, uint2 pixel, out float residualError)
    {
        residualError = 0.f;
        const float3 sky = params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * ray.dir.y));
        float2 interval;
        AABB bounds = volume.getBounds();
        if (!intersectRayAABB(ray.origin, ray.dir, bounds.minPoint, bounds.maxPoint, interval))
            return float4(sky, 1.f);
        interval.x = max(interval.x, 0.f);
        if (interval.x >= interval.y || !volume.hasDensityGrid())
            return float4(sky, 1.f);

        Grid grid;
        gScene.getGrid(volume.getDensityGrid(), grid);
        Grid::Accessor accessor = grid.createAccessor();

        const float stepLength = (interval.y - interval.x) / stepCount;
        float3 radiance = 0.f;
        float transmittance = 1.f;
        const float jitter = pixelJitter(pixel + uint2(stepCount, stepCount * 7));
        for (uint step = 0; step < stepCount && transmittance > 1e-3f; ++step)
        {
            const float t = interval.x + (step + jitter) * stepLength;
            const float3 position = ray.origin + t * ray.dir;
            const float extinction = densityAt(volume, grid, position, accessor);
            if (extinction <= 0.f)
                continue;

            const float alpha = 1.f - exp(-extinction * stepLength);
            const float3 indexPosition = mul(volume.data.invTransform, float4(position, 1.f)).xyz;
            const float3 base = hstrLighting(indexPosition, ray.dir);
            const int3 leafPosition = int3((indexPosition - float3(params.hstrGridMin)) / float(params.hstrCellWidth));
            const int3 windowOffset = abs(leafPosition - int3(params.schurWindowCenter));
            const bool insideWindow = all(windowOffset <= int(params.schurWindowRadius));
            const float3 correction = insideWindow ? deterministicResidual(volume, grid, indexPosition, ray.dir, accessor) : 0.f;
            const float3 source = volume.data.albedo * (0.72f * base + params.residualBlend * correction);
            radiance += transmittance * alpha * source;
            residualError += transmittance * alpha * hstrResidualBound(indexPosition);
            transmittance *= 1.f - alpha;
        }
        return float4(radiance + transmittance * sky, transmittance);
    }

    // ---------------------------------------------------------------------------------------------
    // Reference: unbiased volumetric path tracer on the same extinction field, sun and sky.

    float3 skyRadianceAt(float3 direction) { return params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * direction.y)); }

    /// Distance (world units along the ray) to the exit of v's majorant block (edge 4 voxels, or 16 for the occupancy
    /// blocks); d is the voxel displacement per world unit.
    float majorantBlockExit(float3 v, float3 d, float edge = 4.f)
    {
        const float3 lower = floor(v / edge) * edge;
        const float3 inverse = 1.f / select(abs(d) < 1e-12f, float3(1e-12f), d);
        const float3 t0 = (lower - v) * inverse;
        const float3 t1 = (lower + edge - v) * inverse;
        // A ray lying on a block face with a tiny direction component cannot leave that face in float precision; ignore
        // zero-length exits (the dilated majorant still bounds the neighbouring block) so delta tracking cannot stall.
        const float3 far = select(max(t0, t1) > 1e-5f, max(t0, t1), float3(1e30f));
        return min(far.x, min(far.y, far.z));
    }

    /// Delta tracking with the piecewise-constant (undilated) block majorant, in the medium with extinction scaled by
    /// extinctionScale. Returns true at a real collision (v moved there).
    /// Delta tracking in the transport medium (the proxy), or with fine set in the camera's medium (resident fine bricks at the
    /// finest mapped level, bounded by the dilated majorant, which is built from the maximum source density).
    bool sampleCollision(inout float3 v, float3 d, inout uint rng, float extinctionScale = 1.f, bool fine = false)
    {
        const float epsilon = 1e-3f;
        for (uint i = 0; i < 16384; ++i)
        {
            if (!insideExtinctionGrid(v))
                return false;
            // Empty 16-voxel blocks are crossed in one jump.
            const int3 group = clamp(int3(floor(seaVoxel(v) / 16.f)), int3(0), int3((params.hstrMajorantDims + 3u) / 4u) - 1);
            if (hstrOccupancy.Load(int4(group, 0)) == 0)
            {
                v += (majorantBlockExit(v, d, 16.f) + epsilon) * d;
                continue;
            }
            const float majorant = extinctionScale * (fine ? majorantAt(v) : tightMajorantAt(v));
            const float exit = majorantBlockExit(v, d);
            if (majorant <= 0.f)
            {
                v += (exit + epsilon) * d;
                continue;
            }
            const float flight = -log(1.f - randomFloat(rng)) / majorant;
            if (flight >= exit)
            {
                v += (exit + epsilon) * d;
                continue;
            }
            v += flight * d;
            if (randomFloat(rng) * majorant < extinctionScale * (fine ? cameraExtinctionAtVoxel(v, 0.f) : extinctionAtVoxel(v)))
                return true;
        }
        return false;
    }

    /// Ratio-tracking transmittance estimate from v to the grid boundary.
    float ratioTrack(float3 v, float3 d, inout uint rng, bool fine = false)
    {
        const float epsilon = 1e-3f;
        float transmittance = 1.f;
        for (uint i = 0; i < 16384 && transmittance > 1e-4f; ++i)
        {
            if (!insideExtinctionGrid(v))
                break;
            const float majorant = fine ? majorantAt(v) : tightMajorantAt(v);
            const float exit = majorantBlockExit(v, d);
            if (majorant <= 0.f)
            {
                v += (exit + epsilon) * d;
                continue;
            }
            const float flight = -log(1.f - randomFloat(rng)) / majorant;
            if (flight >= exit)
            {
                v += (exit + epsilon) * d;
                continue;
            }
            v += flight * d;
            transmittance *= max(0.f, 1.f - (fine ? cameraExtinctionAtVoxel(v, 0.f) : extinctionAtVoxel(v)) / majorant);
        }
        return transmittance;
    }

    uint hashUint(uint value)
    {
        const uint state = value * 747796405u + 2891336453u;
        const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    float randomFloat(inout uint rng)
    {
        rng = hashUint(rng);
        return float(rng >> 8) / 16777216.f;
    }

    /// Henyey-Greenstein sample around the propagation direction.
    float3 sampleHG(float3 forward, float2 u)
    {
        const float g = params.anisotropy;
        float cosTheta;
        if (abs(g) < 1e-3f)
            cosTheta = 1.f - 2.f * u.x;
        else
        {
            const float s = (1.f - g * g) / (1.f - g + 2.f * g * u.x);
            cosTheta = clamp((1.f + g * g - s * s) / (2.f * g), -1.f, 1.f);
        }
        const float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        const float phi = 2.f * kPi * u.y;
        const float3 up = abs(forward.y) < 0.999f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        const float3 tangent = normalize(cross(up, forward));
        const float3 bitangent = cross(forward, tangent);
        return normalize(sinTheta * cos(phi) * tangent + sinTheta * sin(phi) * bitangent + cosTheta * forward);
    }

    /// One reference path, split into the kComponent* radiance components.
    ReferenceSample referencePath(float3 origin, float3 direction, GridVolume volume, inout uint rng)
    {
        float2 interval;
        if (!mediumInterval(origin, direction, volume, interval) || interval.y <= max(0.f, interval.x))
        {
            ReferenceSample miss = {};
            miss.background = skyRadianceAt(direction);
            return miss;
        }
        const float3 entryIndex = worldToVoxel(origin + max(0.f, interval.x) * direction, volume);
        return tracePath(entryIndex - float3(params.hstrGridMin), direction, volume, rng, params.cloudDomain != 0);
    }

    /// Radiance arriving back along -direction at voxel position v, split into components by the number of scattering events
    /// after v. Direct sunlight at v itself is not included.
    /// fine: the reference sees the camera's medium (resident fine bricks); the world cache gather sees the transport proxy.
    ReferenceSample tracePath(float3 v, float3 direction, GridVolume volume, inout uint rng, bool fine = false)
    {
        ReferenceSample result = {};
        const float3 sunDirection = normalize(params.sunDirection);
        const float3 sunVoxel = voxelDirection(sunDirection);
        float3 throughput = 1.f;
        for (uint bounce = 0; bounce < 4096; ++bounce)
        {
            const float3 d = voxelDirection(direction);
            v = wrapSeaPosition(v);
            if (!sampleCollision(v, d, rng, 1.f, fine))
            {
                // Escaping after `bounce` scattering events: unscattered background or scattered sky.
                const float3 sky = throughput * skyRadianceAt(direction);
                if (bounce == 0)
                    result.background += sky;
                else
                    result.sky += sky;
                break;
            }
            throughput *= volume.data.albedo;
            // Sun: directional irradiance through the ratio-tracked transmittance; this is scattering order bounce + 1.
            const float3 sun = throughput * phaseHG(dot(sunDirection, direction)) * params.sunRadiance * ratioTrack(v, sunVoxel, rng, fine);
            if (bounce == 0)
                result.sunSingle += sun;
            else
                result.sunMultiple += sun;
            direction = sampleHG(direction, float2(randomFloat(rng), randomFloat(rng)));
            // Russian roulette keeps the weight at or below one. A survival probability capped below the albedo (the former
            // min(0.95, throughput)) grows the weight by albedo / cap per bounce and gives infinite variance at albedo 0.99.
            if (bounce >= 8)
            {
                const float survive = min(1.f, maxComponent(throughput));
                if (randomFloat(rng) >= survive)
                    break;
                throughput /= survive;
            }
        }
        return result;
    }

    // ---------------------------------------------------------------------------------------------
    // World-space radiance cache (experiment): per cell, order-2 SH of all incident radiance except direct sunlight,
    // gathered by paths started inside the cell. Scattered towards the camera it is the sun-multiple plus scattered-sky term.

    /// Real SH basis function k (0-8, bands 0-2) at unit direction d.
    float shBasis(uint k, float3 d)
    {
        switch (k)
        {
        case 0:
            return 0.282095f;
        case 1:
            return 0.488603f * d.y;
        case 2:
            return 0.488603f * d.z;
        case 3:
            return 0.488603f * d.x;
        case 4:
            return 1.092548f * d.x * d.y;
        case 5:
            return 1.092548f * d.y * d.z;
        case 6:
            return 0.315392f * (3.f * d.z * d.z - 1.f);
        case 7:
            return 1.092548f * d.x * d.z;
        default:
            return 0.546274f * (d.x * d.x - d.y * d.y);
        }
    }

    /// Real SH basis function m (0-6) of band 3 at unit direction d.
    float shBasis3(uint m, float3 d)
    {
        switch (m)
        {
        case 0:
            return 0.590044f * d.y * (3.f * d.x * d.x - d.y * d.y);
        case 1:
            return 2.890611f * d.x * d.y * d.z;
        case 2:
            return 0.457046f * d.y * (5.f * d.z * d.z - 1.f);
        case 3:
            return 0.373176f * d.z * (5.f * d.z * d.z - 3.f);
        case 4:
            return 0.457046f * d.x * (5.f * d.z * d.z - 1.f);
        case 5:
            return 1.445306f * d.z * (d.x * d.x - d.y * d.y);
        default:
            return 0.590044f * d.x * (d.x * d.x - 3.f * d.y * d.y);
        }
    }

    /// Sun extension slots in use: all of band 3, or zonal bands 3-worldCacheZonalBands about the sun direction.
    uint worldCacheSunExtension()
    {
        if (params.worldCacheSunOrder >= 3)
            return 7u;
        return params.worldCacheZonalBands > 2 ? min(params.worldCacheZonalBands, 8u) - 2u : 0u;
    }

    /// Basis of sun extension slot j at unit direction d: band 3 function j, or the zonal Y_l0 of band l = 3 + j about the sun,
    /// sqrt((2 l + 1) / (4 pi)) P_l(d . sun).
    float worldCacheExtensionBasis(uint j, float3 d)
    {
        if (params.worldCacheSunOrder >= 3)
            return shBasis3(j, d);
        const float mu = dot(d, normalize(params.sunDirection));
        float previous = 1.f;
        float current = mu;
        for (uint l = 2; l <= j + 3; ++l)
        {
            const float next = ((2.f * l - 1.f) * mu * current - (l - 1.f) * previous) / float(l);
            previous = current;
            current = next;
        }
        return sqrt((2.f * (j + 3) + 1.f) / (4.f * kPi)) * current;
    }

    /// Hanning window of SH band l for a truncation at band order (1 without windowing).
    float shWindow(uint l, uint order)
    {
        return params.worldCacheWindow != 0 ? 0.5f + 0.5f * cos(kPi * float(l) / float(order + 1)) : 1.f;
    }

    uint worldCacheIndex(uint3 cell) { return cell.x + params.worldCacheDims.x * (cell.y + params.worldCacheDims.y * cell.z); }

    /// Whether a cell holds an estimate. The gather skips cells with no density within a majorant block of their centre;
    /// light tracing deposits into every cell a photon crosses.
    bool worldCacheCellValid(uint3 cell)
    {
        if (params.worldCacheEstimator != 0 || params.worldCacheCellVoxels > 4)
            return true;
        return majorantAt((float3(cell) + 0.5f) * float(params.worldCacheCellVoxels)) > 0.f;
    }

    DepositAccumulator makeDepositAccumulator(uint field)
    {
        DepositAccumulator acc;
        acc.cell = int3(-1);
        acc.field = field;
        [unroll]
        for (uint k = 0; k < 16; ++k)
            acc.sums[k] = 0.f;
        return acc;
    }

    /// Adds the accumulated cell's sums to the cache, with stochastic rounding to keep the fixed-point sums unbiased.
    void flushDeposits(inout DepositAccumulator acc, inout uint rng)
    {
        if (acc.cell.x >= 0)
        {
            const uint index = kWorldCacheSlots * worldCacheIndex(uint3(acc.cell));
            const uint coefficients = params.worldCacheBands >= 2 ? 9u : 4u;
            const uint extension = acc.field == 0 ? worldCacheSunExtension() : 0u;
            [unroll]
            for (uint k = 0; k < 16; ++k)
            {
                const bool used = k < 9 ? k < coefficients : k - 9 < extension;
                if (used)
                {
                    const int quantized = int(floor(acc.sums[k] + randomFloat(rng)));
                    if (quantized != 0)
                        InterlockedAdd(hstrWorldCacheDeposit[index + (k < 9 ? 9 * acc.field + k : 18 + k - 9)], quantized);
                }
                acc.sums[k] = 0.f;
            }
        }
        acc.cell = int3(-1);
    }

    /// Light-tracing deposit of one straight flight from v0 to v1 (voxel coordinates): track length per crossed cell times
    /// the photon weight, projected on SH bands 0-worldCacheBands of the direction the light arrives from, plus the sun
    /// extension slots for the sun field, into the photon's accumulator.
    ///
    /// Tried and removed (sunbehind, 8x8-block log error): density-weighted collision deposits (0.07-0.37, far noisier than
    /// track length); skin-weighted deposits, sigma times the visibility from outside (best 0.0246, equal to track length
    /// within run noise); skipping empty 16-voxel blocks in this cell walk (1.75 vs 1.78 ms per frame, no gain: the
    /// per-cell majorant test already rejects them cheaply).
    void depositSegment(float3 v0, float3 v1, float3 look, float weight, inout DepositAccumulator acc, inout uint rng)
    {
        const uint coefficients = params.worldCacheBands >= 2 ? 9u : 4u;
        const uint extension = acc.field == 0 ? worldCacheSunExtension() : 0u;
        float basis[16];
        [unroll]
        for (uint k = 0; k < 16; ++k)
            basis[k] = k < 9 ? (k < coefficients ? shBasis(k, look) : 0.f) : (k - 9 < extension ? worldCacheExtensionBasis(k - 9, look) : 0.f);
        const float cellVoxels = float(params.worldCacheCellVoxels);
        const float worldLength = length((v1 - v0) * params.worldVoxelSize);
        if (worldLength <= 0.f || weight <= 0.f)
            return;
        const float cellVolume = cellVoxels * cellVoxels * cellVoxels * params.worldVoxelSize.x * params.worldVoxelSize.y *
                                 params.worldVoxelSize.z;
        // Fixed-point deposit per unit of segment parameter.
        const float perUnit = weight * worldLength / cellVolume * params.worldCacheDepositScale;

        const float3 p0 = v0 / cellVoxels;
        const float3 delta = (v1 - v0) / cellVoxels;
        int3 cell = int3(floor(p0));
        const int3 stepDirection = int3(sign(delta));
        const float3 inverse = 1.f / max(abs(delta), 1e-12f);
        float3 tMax = float3(
            delta.x > 0.f ? (floor(p0.x) + 1.f - p0.x) * inverse.x : (delta.x < 0.f ? (p0.x - floor(p0.x)) * inverse.x : 1e30f),
            delta.y > 0.f ? (floor(p0.y) + 1.f - p0.y) * inverse.y : (delta.y < 0.f ? (p0.y - floor(p0.y)) * inverse.y : 1e30f),
            delta.z > 0.f ? (floor(p0.z) + 1.f - p0.z) * inverse.z : (delta.z < 0.f ? (p0.z - floor(p0.z)) * inverse.z : 1e30f)
        );
        const float3 tDelta = float3(delta.x != 0.f ? inverse.x : 1e30f, delta.y != 0.f ? inverse.y : 1e30f, delta.z != 0.f ? inverse.z : 1e30f);
        float t = 0.f;
        for (uint i = 0; i < 1024; ++i)
        {
            const float tNext = min(min(tMax.x, tMax.y), min(tMax.z, 1.f));
            const float fraction = tNext - t;
            // Cells with no medium within a majorant block of their centre are never read by the camera (its trilinear
            // lookups at medium points reach one cell): escaping photons would otherwise deposit all the way to the grid box.
            int3 storedCell = cell;
            if (params.seaMode != 0)
            {
                const int3 dims = int3(params.worldCacheDims);
                storedCell.x = (cell.x % dims.x + dims.x) % dims.x;
                storedCell.z = (cell.z % dims.z + dims.z) % dims.z;
            }
            if (fraction > 0.f && all(storedCell >= 0) && all(storedCell < int3(params.worldCacheDims)) &&
                majorantAt((float3(storedCell) + 0.5f) * cellVoxels) > 0.f)
            {
                if (any(storedCell != acc.cell))
                {
                    flushDeposits(acc, rng);
                    acc.cell = storedCell;
                }
                // Deposits are divided by the modulation at the middle of this cell's part of the flight.
                const float inverseModulation = 1.f / worldCacheModulation((p0 + (t + 0.5f * fraction) * delta) * cellVoxels);
                const float scale = perUnit * fraction * inverseModulation;
                [unroll]
                for (uint k = 0; k < 16; ++k)
                    acc.sums[k] += basis[k] * scale;
            }
            if (tNext >= 1.f)
                break;
            t = tNext;
            if (tMax.x <= tMax.y && tMax.x <= tMax.z)
            {
                cell.x += stepDirection.x;
                tMax.x += tDelta.x;
            }
            else if (tMax.y <= tMax.z)
            {
                cell.y += stepDirection.y;
                tMax.y += tDelta.y;
            }
            else
            {
                cell.z += stepDirection.z;
                tMax.z += tDelta.z;
            }
        }
    }

    /// Emitted power of photon source i: the sun through the side facing it on axis i (0-2), or the unit-colour sky
    /// L(d) = 0.675 + 0.325 d.y through side i - 3 (outward normal n), whose inward flux is A (0.675 pi + 0.325 (2 pi / 3) n.y).
    float photonSourcePower(uint i, float3 sun, float3 faceArea, float sunMean, float skyMean)
    {
        // The sea's X/Z sides are periodic: light only enters through the top and bottom of the cloud layer.
        if (params.seaMode != 0 && ((i < 3u) ? i : ((i - 3u) >> 1)) != 1u)
            return 0.f;
        if (i < 3)
            return sunMean * abs(sun[i]) * faceArea[i];
        const uint axis = (i - 3) >> 1;
        const float ny = axis == 1 ? (((i - 3) & 1) != 0 ? 1.f : -1.f) : 0.f;
        return skyMean * faceArea[axis] * (0.675f * kPi + 0.325f * 2.f * kPi / 3.f * ny);
    }

    /// One light-tracing photon: emitted from the sun through a sun-facing side of the grid box, or from the unit-colour sky
    /// through any side, chosen by power. Every flight segment after the first sun segment (direct sunlight) is deposited.
    ///
    /// Photon cost history (65536 photons per update, 4K laptop GPU): 5.8-6.4 ms -> 1.8-1.9 ms from the similarity relation
    /// (photonSimilar), deposits skipping cells without medium, the undilated majorant with 16-voxel empty skipping
    /// (sampleCollision) and register deposit accumulation (DepositAccumulator). Equal-time error, 8x8 blocks against the
    /// path tracer: side 200 ms 0.0371 -> 0.0300, 800 ms 0.0290 -> 0.0265; sunbehind 200 ms 0.0383 -> 0.0279, 800 ms
    /// 0.0286 -> 0.0242, reaching the exact tracer's floor (0.0230 vs 0.0231 at 134M photons): no measurable bias.
    /// Tried and removed: Russian roulette of the tail after 32 (or 8/16) bounces saves ~0.1 ms per update but raises the
    /// error 0.0241 -> 0.0273, so the roulette stays weight-bounded from bounce 8.
    void tracePhoton(uint index)
    {
        if (index >= params.worldCachePhotons || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        uint rng = hashUint(index * 2654435761u + 1013u) ^ hashUint(params.worldCacheSamples * 9781u + 6271u);
        Photon photon = spawnPhoton(rng);
        float3 v = photon.positionWeight.xyz;
        const float weight = photon.positionWeight.w;
        float3 direction = photon.directionThroughput.xyz;
        const bool fromSun = (photon.state & kPhotonSky) == 0;

        // The medium is grey: one scalar albedo.
        const float albedo = dot(volume.data.albedo, 1.f / 3.f);
        float throughput = 1.f;
        DepositAccumulator acc = makeDepositAccumulator(fromSun ? 0u : 1u);
        for (uint bounce = 0; bounce < 4096; ++bounce)
        {
            const float3 d = voxelDirection(direction);
            v = wrapSeaPosition(v);
            const float3 start = v;
            const bool similar = photonSimilar(bounce);
            const bool collided = sampleCollision(v, d, rng, similar ? 1.f - albedo * params.anisotropy : 1.f);
            if (!fromSun || bounce > 0)
                depositSegment(start, v, -direction, weight * throughput, acc, rng);
            if (!collided)
                break;
            photonScatter(similar, albedo, direction, throughput, rng);
            // Weight-bounded Russian roulette, as in tracePath.
            if (bounce >= 8)
            {
                const float survive = min(1.f, throughput);
                if (randomFloat(rng) >= survive)
                    break;
                throughput /= survive;
            }
        }
        flushDeposits(acc, rng);
    }

    /// Whether a photon's flight after `bounce` scatterings runs in the similar medium: after worldCacheSimilarity exact
    /// scatterings the light is diffuse, and the isotropic medium with extinction sigma (1 - a g) and albedo a (1 - g) / (1 - a g)
    /// carries it with the same diffusion and absorption lengths in about (1 - a g) of the scatterings.
    bool photonSimilar(uint bounce) { return params.worldCacheSimilarity > 0 && bounce >= params.worldCacheSimilarity; }

    /// Scatters a photon at a collision: HG with the medium's albedo, or isotropic with the similar albedo.
    void photonScatter(bool similar, float albedo, inout float3 direction, inout float throughput, inout uint rng)
    {
        const float g = params.anisotropy;
        if (similar)
        {
            throughput *= albedo * (1.f - g) / (1.f - albedo * g);
            const float z = 1.f - 2.f * randomFloat(rng);
            const float phi = 2.f * kPi * randomFloat(rng);
            const float r = sqrt(max(0.f, 1.f - z * z));
            direction = float3(r * cos(phi), r * sin(phi), z);
        }
        else
        {
            throughput *= albedo;
            direction = sampleHG(direction, float2(randomFloat(rng), randomFloat(rng)));
        }
    }

    /// Persistent pool: advances one photon by worldCacheSegments flight segments, respawning it whenever it dies, so every
    /// update costs the same and every path is still deposited in full across updates. Counts the photons emitted.
    void advancePhoton(uint slot)
    {
        if (slot >= params.worldCachePhotons || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        const float albedo = dot(volume.data.albedo, 1.f / 3.f);
        Photon photon = hstrPhotonPool[slot];
        uint rng = photon.rng != 0 ? photon.rng : hashUint(slot * 2654435761u + 1013u) ^ hashUint(params.worldCacheSamples * 9781u + 7919u);
        uint emitted = 0;
        DepositAccumulator acc = makeDepositAccumulator((photon.state & kPhotonSky) != 0 ? 1u : 0u);
        for (uint segment = 0; segment < params.worldCacheSegments; ++segment)
        {
            if ((photon.state & kPhotonAlive) == 0)
            {
                flushDeposits(acc, rng);
                photon = spawnPhoton(rng);
                acc.field = (photon.state & kPhotonSky) != 0 ? 1u : 0u;
                ++emitted;
            }
            float3 v = photon.positionWeight.xyz;
            float3 direction = photon.directionThroughput.xyz;
            float throughput = photon.directionThroughput.w;
            const uint bounce = photon.state & kPhotonBounceMask;
            const bool fromSun = (photon.state & kPhotonSky) == 0;
            const float3 d = voxelDirection(direction);
            v = wrapSeaPosition(v);
            const float3 start = v;
            const bool similar = photonSimilar(bounce);
            const bool collided = sampleCollision(v, d, rng, similar ? 1.f - albedo * params.anisotropy : 1.f);
            if (!fromSun || bounce > 0)
                depositSegment(start, v, -direction, photon.positionWeight.w * throughput, acc, rng);
            bool alive = collided;
            if (collided)
            {
                photonScatter(similar, albedo, direction, throughput, rng);
                if (bounce >= 8)
                {
                    const float survive = min(1.f, throughput);
                    alive = randomFloat(rng) < survive;
                    throughput /= survive;
                }
            }
            photon.positionWeight.xyz = v;
            photon.directionThroughput = float4(direction, throughput);
            photon.state = (photon.state & ~(kPhotonAlive | kPhotonBounceMask)) | (alive ? kPhotonAlive : 0u) | min(bounce + 1, kPhotonBounceMask);
        }
        flushDeposits(acc, rng);
        photon.rng = rng;
        hstrPhotonPool[slot] = photon;
        const uint waveEmitted = WaveActiveSum(emitted);
        if (WaveIsFirstLane() && waveEmitted > 0)
            InterlockedAdd(hstrPhotonEmitted[0], waveEmitted);
    }

    /// A new photon at the grid box boundary (see tracePhoton), with its weight normalised by worldCachePhotons.
    Photon spawnPhoton(inout uint rng)
    {
        const float3 dims = float3(params.hstrExtinctionDims);
        const float3 extent = dims * params.worldVoxelSize;
        const float3 faceArea = float3(extent.y * extent.z, extent.x * extent.z, extent.x * extent.y);
        const float3 sun = normalize(params.sunDirection);
        const float sunMean = dot(params.sunRadiance, 1.f / 3.f);
        const float skyMean = dot(params.skyRadiance, 1.f / 3.f);

        // Sources by power: three sun-facing sides (0-2), then six sky sides (3-8: -x, +x, -y, +y, -z, +z).
        float total = 0.f;
        for (uint i = 0; i < 9; ++i)
            total += photonSourcePower(i, sun, faceArea, sunMean, skyMean);
        float u = randomFloat(rng) * total;
        uint choice = 8;
        for (uint i = 0; i < 9; ++i)
        {
            const float power = photonSourcePower(i, sun, faceArea, sunMean, skyMean);
            if (u < power)
            {
                choice = i;
                break;
            }
            u -= power;
        }

        const bool fromSun = choice < 3;
        const uint axis = fromSun ? choice : (choice - 3) >> 1;
        // Outward normal of the entry side: the sun side faces the sun; sky sides alternate -/+.
        const float side = fromSun ? (sun[axis] >= 0.f ? 1.f : -1.f) : (((choice - 3) & 1) != 0 ? 1.f : -1.f);
        float3 normal = 0.f;
        normal[axis] = side;
        float3 v = float3(randomFloat(rng), randomFloat(rng), randomFloat(rng)) * dims;
        v[axis] = side > 0.f ? dims[axis] - 1e-3f : 1e-3f;

        float3 direction;
        float weight;
        if (fromSun)
        {
            direction = -sun;
            weight = total / (sunMean * float(params.worldCachePhotons));
        }
        else
        {
            // Cosine-weighted direction d towards the sky about the outward normal; the photon travels along -d.
            const float r = sqrt(randomFloat(rng));
            const float phi = 2.f * kPi * randomFloat(rng);
            const float3 up = abs(normal.y) < 0.999f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
            const float3 tangent = normalize(cross(up, normal));
            const float3 bitangent = cross(normal, tangent);
            const float3 d = normalize(r * cos(phi) * tangent + r * sin(phi) * bitangent + sqrt(max(0.f, 1.f - r * r)) * normal);
            direction = -d;
            // f / pdf with pdf = (power / total) (1 / area) (cos / pi); power / (skyMean area) is the side's unit flux.
            const float unitFlux = photonSourcePower(choice, sun, faceArea, sunMean, skyMean) / (skyMean * faceArea[axis]);
            weight = (0.675f + 0.325f * d.y) * kPi * total / (skyMean * unitFlux * float(params.worldCachePhotons));
        }

        Photon photon;
        photon.positionWeight = float4(v, weight);
        photon.directionThroughput = float4(direction, 1.f);
        photon.rng = rng;
        photon.state = kPhotonAlive | (fromSun ? 0u : kPhotonSky);
        photon.pad = 0;
        return photon;
    }

    /// Adds one light-tracing batch to the running sums: deposits times the sun and sky colours.
    void resolveWorldCacheCell(uint3 cell)
    {
        if (any(cell >= params.worldCacheDims))
            return;
        const uint index = worldCacheIndex(cell);
        const bool restart = params.worldCacheSamples == 0;
        const float scale = 1.f / params.worldCacheDepositScale;
        [unroll]
        for (uint k = 0; k < 9; ++k)
        {
            const float3 value = scale * (float(hstrWorldCacheDeposit[kWorldCacheSlots * index + k]) * params.sunRadiance +
                                          float(hstrWorldCacheDeposit[kWorldCacheSlots * index + 9 + k]) * params.skyRadiance);
            [unroll]
            for (uint c = 0; c < 3; ++c)
            {
                const uint slot = 27 * index + 3 * k + c;
                hstrWorldCache[slot] = (restart ? 0.f : hstrWorldCache[slot]) + value[c];
            }
        }
    }

    /// One gather pass: a single path per cell, from a uniform point in the cell along a uniform direction.
    void updateWorldCacheCell(uint3 cell)
    {
        if (any(cell >= params.worldCacheDims) || gScene.getGridVolumeCount() == 0)
            return;
        const uint index = worldCacheIndex(cell);
        const float cellVoxels = float(params.worldCacheCellVoxels);
        const bool needed = worldCacheCellValid(cell);
        float3 value = 0.f;
        float3 d = float3(0.f, 0.f, 1.f);
        if (needed)
        {
            uint rng = hashUint(index * 7919u + 17u) ^ hashUint(params.worldCacheSamples * 9781u + 6271u);
            const float3 v = (float3(cell) + float3(randomFloat(rng), randomFloat(rng), randomFloat(rng))) * cellVoxels;
            const float z = 1.f - 2.f * randomFloat(rng);
            const float phi = 2.f * kPi * randomFloat(rng);
            const float r = sqrt(max(0.f, 1.f - z * z));
            d = float3(r * cos(phi), r * sin(phi), z);
            const ReferenceSample path = tracePath(v, d, gScene.getGridVolume(0), rng);
            value = path.background + path.sunSingle + path.sunMultiple + path.sky;
            if (any(isnan(value) || isinf(value)))
                value = 0.f;
        }
        const bool restart = params.worldCacheSamples == 0;
        [unroll]
        for (uint k = 0; k < 9; ++k)
        {
            const float3 projected = 4.f * kPi * shBasis(k, d) * value;
            [unroll]
            for (uint c = 0; c < 3; ++c)
            {
                const uint slot = 27 * index + 3 * k + c;
                hstrWorldCache[slot] = (restart ? 0.f : hstrWorldCache[slot]) + projected[c];
            }
        }
    }

    /// Mean of scalar cache slot s of a cell (unit-colour sun and sky fields), straight from the light tracer's fixed-point
    /// deposits, which accumulate over every batch since the last restart.
    float worldCacheSlotMean(uint index, uint slot)
    {
        // Photon weights are normalised by worldCachePhotons; a pool has emitted hstrPhotonEmitted[0] of them in total. In the sea
        // every tile counts the batches since its cloud arrived.
        float batches = params.worldCacheSegments > 0 ? float(max(hstrPhotonEmitted[0], 1u)) / float(params.worldCachePhotons)
                                                      : float(max(params.worldCacheSamples, 1u));
        if (params.cloudDomain != 0)
            batches = max(hstrCloudTileBatches[worldCacheTile(index)], 1.f);
        return float(hstrWorldCacheDeposit[kWorldCacheSlots * index + slot]) / (batches * params.worldCacheDepositScale);
    }

    /// Sea tile slot of world cache cell index.
    uint worldCacheTile(uint index)
    {
        const uint3 dims = params.worldCacheDims;
        const uint2 cell = uint2(index % dims.x, index / (dims.x * dims.y));
        const uint2 tile = min(cell * params.worldCacheCellVoxels / params.cloudTileVoxels, params.cloudTiles - 1u);
        return tile.x + params.cloudTiles.x * tile.y;
    }

    /// Writes one cell's cache means into the packed camera textures (light tracing only; the gather is read untextured).
    /// Cells without medium within a majorant block never receive deposits (depositSegment) and keep their cleared zeros.
    void bakeWorldCacheCell(uint3 cell)
    {
        if (any(cell >= params.worldCacheDims) || majorantAt((float3(cell) + 0.5f) * float(params.worldCacheCellVoxels)) <= 0.f)
            return;
        const uint index = worldCacheIndex(cell);
        [unroll]
        for (uint t = 0; t < kWorldCacheTextures; ++t)
        {
            float4 texel = 0.f;
            [unroll]
            for (uint c = 0; c < 4; ++c)
            {
                const uint slot = 4 * t + c;
                if (slot < kWorldCacheSlots)
                    texel[c] = worldCacheSlotMean(index, slot);
            }
            hstrWorldCacheTextureOutput[t][cell] = texel;
        }
    }

    /// Textured form of worldCacheRadiance: one trilinear fetch per four slots, only for the evaluated bands. The unit sun and
    /// sky fields take their colours here. With worldCacheZonalBands, each sun band splits into its zonal part about the sun
    /// direction s, f_l(s) P_l(w . s), windowed at the zonal order and continued by the zonal extension bands, and the rest,
    /// windowed at order 2. With worldCacheSunOrder 3 the sun field is plain SH to band 3.
    ///
    /// Angular resolution is not the bottleneck (sunbehind, 8x8-block log error against the path tracer, reference noise
    /// 0.0093): at 33M photons L2 0.0252, zonal 4 0.0245, zonal 6 0.0248, zonal 8 0.0250, sun L3 0.0237, the zonal part
    /// rewindowed at order 8 only 0.0376; at 134M photons L2 0.0239, zonal 4 0.0230, sun L3 0.0223, 1-voxel cells 0.0217,
    /// no window 0.0388. Beyond L2 removes at most ~8%, so the extension is off by default. Unwindowed order 2 is worse than
    /// windowed order 1 (0.040 vs 0.028). A 3x3x3 tent filter of the stored cache plateaus at 0.037. The error maps show lit
    /// faces too dark and silhouettes/creases too bright at every g: sub-cell skin structure, not angular detail.
    float3 worldCacheRadianceTextured(float3 v, float3 w)
    {
        v = seaVoxel(v);
        const float3 uvw = v / (float(params.worldCacheCellVoxels) * float3(params.worldCacheDims));
        const float g = params.anisotropy;
        const uint order = params.worldCacheOrder;
        const uint extension = order == 2 ? worldCacheSunExtension() : 0u;
        const uint sunOrder = extension == 7 ? 3u : order;
        const bool zonal = order == 2 && params.worldCacheSunOrder < 3 && params.worldCacheZonalBands > 0;
        const uint zonalOrder = params.worldCacheZonalWindow != 0 ? params.worldCacheZonalWindow : max(params.worldCacheZonalBands, 2u);

        float4 texels[kWorldCacheTextures];
        // Sun slots 0-8 and sky slots 9-17 of bands 0, 0-1 or 0-2 lie in textures {0, 2}, {0, 2, 3} or 0-4; the sun extension
        // (slots 18-24) in textures 4-6.
        const uint textures = (order == 0 ? 0x5u : (order == 1 ? 0xDu : 0x1Fu)) | (extension > 0 ? 0x70u : 0u);
        [unroll]
        for (uint t = 0; t < kWorldCacheTextures; ++t)
            texels[t] = (textures & (1u << t)) != 0 ? hstrWorldCacheTexture[t].SampleLevel(hstrLinearSampler, uvw, 0.f) : 0.f;

        const float3 s = normalize(params.sunDirection);
        float sun = 0.f;
        float sky = 0.f;
        float3 bandAtView = 0.f; // Sun band l at w and at s, for the zonal split.
        float3 bandAtSun = 0.f;
        [unroll]
        for (uint k = 0; k < 9; ++k)
        {
            const uint l = k == 0 ? 0u : (k < 4 ? 1u : 2u);
            if (l > order)
                continue;
            const float gl = l == 0 ? 1.f : (l == 1 ? g : g * g);
            const float basis = shBasis(k, w);
            sky += shWindow(l, order) * gl * basis * texels[(9 + k) / 4][(9 + k) % 4];
            const float c = texels[k / 4][k % 4];
            if (zonal)
            {
                bandAtView[l] += c * basis;
                bandAtSun[l] += c * shBasis(k, s);
            }
            else
                sun += shWindow(l, sunOrder) * gl * basis * c;
        }
        if (zonal)
        {
            // Legendre recurrence in mu = w . s through band 8.
            const float mu = dot(w, s);
            float previous = 1.f;
            float current = mu;
            float gl = g;
            [unroll]
            for (uint l = 0; l <= 8; ++l)
            {
                const float legendre = l == 0 ? 1.f : current;
                const float bandPower = l == 0 ? 1.f : gl;
                if (l <= 2)
                {
                    const float zonalPart = bandAtSun[l] * legendre;
                    sun += bandPower * (shWindow(l, 2) * (bandAtView[l] - zonalPart) + shWindow(l, zonalOrder) * zonalPart);
                }
                else if (l - 3 < extension)
                    sun += bandPower * shWindow(l, zonalOrder) * sqrt((2.f * l + 1.f) / (4.f * kPi)) * legendre * texels[(15 + l) / 4][(15 + l) % 4];
                if (l >= 1)
                {
                    const float next = ((2.f * (l + 1) - 1.f) * mu * current - float(l) * previous) / float(l + 1);
                    previous = current;
                    current = next;
                    gl *= g;
                }
            }
        }
        else if (extension == 7)
        {
            [unroll]
            for (uint m = 0; m < 7; ++m)
                sun += shWindow(3, 3) * g * g * g * shBasis3(m, w) * texels[(18 + m) / 4][(18 + m) % 4];
        }
        return max(0.f, sun * params.sunRadiance + sky * params.skyRadiance);
    }

    /// In-scattered radiance towards the camera (before albedo) from the cache at voxel position v, for camera ray direction w:
    /// the HG phase convolves SH band l by g^l. Trilinear over valid cell centres (cells without an estimate get no weight
    /// rather than black light); bands above worldCacheOrder are dropped.
    /// The modulation reads the voxel-centre sun field exactly as the photon deposits do, so the stored ratio cancels.
    float3 worldCacheRadiance(float3 v, float3 w) { return worldCacheModulation(v) * worldCacheRadianceStored(v, w); }

    /// Voxel-resolution factor the cache is divided by when deposited and multiplied by when read: exp(-b tau_sun), with tau_sun
    /// the fine sun optical depth (capped so deep interiors do not inflate the fixed-point deposits). Multiple scattering falls
    /// off into the cloud with the sun depth, so the stored ratio is much smoother than the radiance at cell scale, and the
    /// camera recovers the voxel-scale structure near lit surfaces from the fine sun field. b = 0 disables it.
    ///
    /// Measured at 33M photons, 2-voxel cells: side 0.032 -> 0.026, sunbehind 0.038 -> 0.025 with the modulation, matching
    /// 1-voxel cells at an eighth of the memory. With it, 1-voxel cells no longer help (0.0233 vs 0.0231 at 134M photons).
    /// Tried and removed instead of it: per-cell fits against the sun transmittance (0.033), medium-weighted cells (0.040),
    /// exterior-weighted cells (0.053), blends of the two (best 0.024 at 60% medium).
    float worldCacheModulation(float3 v)
    {
        if (params.worldCacheModulation <= 0.f)
            return 1.f;
        const float depth = hstrFineSun.SampleLevel(hstrLinearSampler, (seaVoxel(v) + 0.5f) / float3(params.hstrFineDims), 0.f);
        return exp(-params.worldCacheModulation * min(depth, params.worldCacheModulationDepth));
    }

    /// The cache as stored (divided by worldCacheModulation). The float sums (gather, or untextured light tracing) carry bands
    /// 0-2 of the coloured total only.
    float3 worldCacheRadianceStored(float3 v, float3 w)
    {
        v = seaVoxel(v);
        if (params.worldCacheTextured != 0 && params.worldCacheEstimator != 0)
            return worldCacheRadianceTextured(v, w);
        const float3 p = v / float(params.worldCacheCellVoxels) - 0.5f;
        const float3 base = floor(p);
        const float3 f = p - base;
        const float g = params.anisotropy;
        const uint coefficients = (params.worldCacheOrder + 1) * (params.worldCacheOrder + 1);
        float3 result = 0.f;
        float weightSum = 0.f;
        [unroll]
        for (uint corner = 0; corner < 8; ++corner)
        {
            const uint3 offset = uint3(corner & 1, (corner >> 1) & 1, corner >> 2);
            const float3 weights = lerp(1.f - f, f, float3(offset));
            int3 unclamped = int3(base) + int3(offset);
            if (params.seaMode != 0)
            {
                const int3 dims = int3(params.worldCacheDims);
                unclamped.x = (unclamped.x % dims.x + dims.x) % dims.x;
                unclamped.z = (unclamped.z % dims.z + dims.z) % dims.z;
            }
            const uint3 cell = uint3(clamp(unclamped, int3(0), int3(params.worldCacheDims) - 1));
            const float weight = worldCacheCellValid(cell) ? weights.x * weights.y * weights.z : 0.f;
            weightSum += weight;
            const uint index = worldCacheIndex(cell);
            for (uint k = 0; k < coefficients; ++k)
            {
                const uint l = k == 0 ? 0u : (k < 4 ? 1u : 2u);
                const float window = params.worldCacheWindow != 0 ? 0.5f + 0.5f * cos(kPi * float(l) / float(params.worldCacheOrder + 1)) : 1.f;
                const float band = window * (l == 0 ? 1.f : (l == 1 ? g : g * g));
                const float3 c = float3(hstrWorldCache[27 * index + 3 * k], hstrWorldCache[27 * index + 3 * k + 1], hstrWorldCache[27 * index + 3 * k + 2]);
                result += weight * band * shBasis(k, w) * c;
            }
        }
        return max(0.f, result / (float(max(params.worldCacheSamples, 1u)) * max(weightSum, 1e-6f)));
    }

    /// Reference averages of the selected components: the even-frame half, the odd-frame half, and both.
    ReferenceHalves referenceHalves(uint2 pixel, uint mask)
    {
        float3 evenSum = 0.f;
        float3 oddSum = 0.f;
        [unroll]
        for (uint c = 0; c < kComponentCount; ++c)
        {
            const float weight = (mask & (1u << c)) != 0 ? 1.f : 0.f;
            evenSum += weight * hstrReferenceSum[uint3(pixel, 2 * c)].rgb;
            oddSum += weight * hstrReferenceSum[uint3(pixel, 2 * c + 1)].rgb;
        }
        const float evenCount = hstrReferenceSum[uint3(pixel, 0)].a;
        const float oddCount = hstrReferenceSum[uint3(pixel, 1)].a;
        ReferenceHalves result;
        result.first = evenSum / max(evenCount, 1.f);
        result.second = oddSum / max(oddCount, 1.f);
        result.both = (evenSum + oddSum) / max(evenCount + oddCount, 1.f);
        return result;
    }

    void referencePixel(uint2 pixel)
    {
        pixel.y += params.referenceRowOffset;
        if (any(pixel >= params.frameDim))
            return;
        // Seeded by the sample index, not frameIndex: property changes reset frameIndex and would repeat earlier samples.
        uint rng = hashUint(pixel.x + params.frameDim.x * pixel.y) ^ hashUint(params.referenceSamples * 9781u + 6271u);
        const float2 jitter = float2(randomFloat(rng), randomFloat(rng));
        const float3 direction = cameraRayDirection(float2(pixel) + jitter);
        ReferenceSample path = {};
        path.background = skyRadianceAt(direction);
        if (gScene.getGridVolumeCount() > 0)
            path = referencePath(gScene.camera.data.posW, direction, gScene.getGridVolume(0), rng);
        bool valid = true;
        for (uint c = 0; c < kComponentCount; ++c)
            valid = valid && !any(isnan(path.component(c)) || isinf(path.component(c)));
        // Frames alternate between two independent halves, whose disagreement estimates the reference's own noise.
        const uint halfIndex = params.referenceSamples & 1u;
        const bool restart = params.referenceSamples < 2;
        for (uint c = 0; c < kComponentCount; ++c)
        {
            const uint3 slice = uint3(pixel, 2 * c + halfIndex);
            const float4 previous = restart ? float4(0.f) : hstrReferenceSum[slice];
            const float3 value = valid ? path.component(c) : float3(0.f);
            hstrReferenceSum[slice] = previous + float4(value, 1.f);
        }
        // referenceShow bits 4-5 pick one half (1: even frames, 2: odd frames); 0 shows both.
        const ReferenceHalves shown = referenceHalves(pixel, params.referenceShow & 0xFu);
        const uint shownHalves = (params.referenceShow >> 4) & 3u;
        color[pixel] = float4(shownHalves == 1 ? shown.first : (shownHalves == 2 ? shown.second : shown.both), 1.f);
        transportError[pixel] = 0.f;
        cutStats[pixel] = 0.f;
    }

    /// Mean absolute difference of one image row against the reference average, in linear radiance (x) and log(1 + radiance) (y).
    /// The compared value is the HST frame plus the substituted reference components; the reference is the target components.
    /// zw hold the same measures between the two reference halves, halved: the expected noise of the full reference average.
    void compareReferenceRow(uint row)
    {
        if (params.compareBlock > 1)
        {
            compareReferenceBlockRow(row);
            return;
        }
        if (row >= params.frameDim.y)
            return;
        float4 error = 0.f;
        uint count = 0;
        if (params.compareExact != 0)
            for (uint b = 0; b < kCompareBins; ++b)
                hstrReferenceRowHistogram[row * kCompareBins + b] = 0.f;
        const uint columnLow = min(params.compareColumnLow, params.frameDim.x);
        const uint columnHigh = params.compareColumnHigh == 0 ? params.frameDim.x : min(params.compareColumnHigh, params.frameDim.x);
        for (uint x = columnLow; x < columnHigh; ++x)
        {
            const uint2 pixel = uint2(x, row);
            if (params.compareExact != 0)
            {
                // Against the stored exact frame: zw become the fractions of pixels whose log error exceeds 0.02 and 0.1.
                const float3 value = color[pixel].rgb;
                const float3 reference = hstrExactFrame[pixel].rgb;
                if (any(isnan(value) || isinf(value)))
                    continue;
                const float logError = dot(abs(log(1.f + max(value, 0.f)) - log(1.f + max(reference, 0.f))), 1.f / 3.f);
                error += float4(dot(abs(value - reference), 1.f / 3.f), logError, logError > 0.02f ? 1.f : 0.f, logError > 0.1f ? 1.f : 0.f);
                ++count;
                // Bin 0 up to kCompareBinFloor, then 4 bins per octave.
                const uint bin = logError <= kCompareBinFloor ? 0u : min(uint(4.f * log2(logError / kCompareBinFloor)) + 1u, kCompareBins - 2u);
                hstrReferenceRowHistogram[row * kCompareBins + bin] += 1.f;
                hstrReferenceRowHistogram[row * kCompareBins + kCompareBins - 1u] = max(hstrReferenceRowHistogram[row * kCompareBins + kCompareBins - 1u], logError);
                continue;
            }
            const ReferenceHalves target = referenceHalves(pixel, params.compareTarget);
            const float3 reference = target.both;
            const float3 value = color[pixel].rgb + referenceHalves(pixel, params.compareSubstitute).both;
            if (any(isnan(value) || isinf(value) || isnan(reference) || isinf(reference)))
                continue;
            const float3 halfDifference = abs(target.first - target.second);
            const float3 halfLogDifference = abs(log(1.f + max(target.first, 0.f)) - log(1.f + max(target.second, 0.f)));
            error += float4(
                dot(abs(value - reference), 1.f / 3.f),
                dot(abs(log(1.f + max(value, 0.f)) - log(1.f + max(reference, 0.f))), 1.f / 3.f),
                0.5f * dot(halfDifference, 1.f / 3.f),
                0.5f * dot(halfLogDifference, 1.f / 3.f)
            );
            ++count;
        }
        hstrReferenceRowError[row] = error / float(max(count, 1u));
    }

    // ---------------------------------------------------------------------------------------------
    // GPU sun bake scheduling (CloudResidencyDesc::gpuSun). The CPU pass it replaces walked every mapped brick (and every held bake
    // when the slot pool was short) every frame: 9-19 ms of a moving sea's frame. Here the GPU owns the slot table and the free
    // slots, and each frame runs release, stamp, scan, select, emit, evict and assign; the bake pass then runs over the jobs assign
    // wrote. Only the bake count is read back, asynchronously, to let the scheduler idle. What it decides is what CloudResidency::scheduleSunBakes decided, except that a brick takes at most one
    // bake per frame and the most important candidates are chosen by histogram bin rather than exact order.

    uint sunCandidateBin(float priority) { return uint(clamp(int(priority * 16.f), 0, int(kCloudSunBins) - 1)); }

    /// Holders: unwanted bakes of bricks outside the cut first, then of bricks in it, then wanted bakes by priority.
    uint sunHolderBin(bool wanted, bool inCut, float priority)
    {
        return wanted ? 2u + uint(clamp(int(priority * 16.f), 0, int(kCloudSunBins) - 3)) : (inCut ? 1u : 0u);
    }

    /// The scheduler passes read their frame from here, never from params (HSTRCloudSunFrame).
    HSTRCloudSunFrame sunFrame() { return hstrCloudSunFrameInfo[0]; }

    /// A GPU brick's scheduler entry: where it is, its flags, and its priority at bin resolution.
    SunSchedEntry sunSched(uint b)
    {
        const HSTRCloudSunSched where = hstrCloudSunSched[b];
        const uint state = hstrCloudSunState[b];
        SunSchedEntry s;
        s.priority = float(state >> kCloudSunBinShift) / 16.f;
        s.flags = state & ((1u << kCloudSunBinShift) - 1u);
        s.coord = where.coord;
        s.asset = where.asset;
        return s;
    }

    bool sunKeyWanted(uint key, uint flags)
    {
        return key != kCloudRefNone && (key >> 4) == sunFrame().generation && (flags & (1u << (key & 15u))) != 0;
    }

    void pushSunSlot(uint slot)
    {
        uint top;
        InterlockedAdd(hstrCloudSunFreeTop[0], 1u, top);
        hstrCloudSunFree[top] = slot;
    }

    /// A free sun slot, or kCloudRefNone when there is none (pushes only happen in earlier dispatches).
    uint popSunSlot()
    {
        uint previous;
        InterlockedAdd(hstrCloudSunFreeTop[0], 0xFFFFFFFFu, previous);
        if (previous == 0u || previous > sunFrame().capacity)
        {
            uint unused;
            InterlockedAdd(hstrCloudSunFreeTop[0], 1u, unused);
            return kCloudRefNone;
        }
        return hstrCloudSunFree[previous - 1u];
    }

    uint2 sunFieldTexel(uint entry) { return uint2(entry & (kCloudSunFieldWidth - 1u), entry / kCloudSunFieldWidth); }

    /// Brick positions of a level in an asset.
    uint3 sunLevelDims(uint assetID, uint level)
    {
        const uint size = 8u << level;
        return (hstrCloudAssets[assetID].dims + size - 1u) / size;
    }

    /// The directory node over one 32-voxel cell of an asset, which exists while fine (level 0-1) bricks are mapped there.
    uint sunFineNode(uint assetID, uint3 cell)
    {
        const HSTRCloudAsset asset = hstrCloudAssets[assetID];
        const uint3 chunk = cell >> 2;
        if (any(chunk >= asset.chunkDims))
            return kCloudRefNone;
        uint entry = hstrCloudDirectory[asset.directoryOffset + chunk.x + asset.chunkDims.x * (chunk.y + asset.chunkDims.y * chunk.z)];
        if (entry == kCloudRefNone || (entry & kCloudRefNode) == 0)
            return kCloudRefNone;
        const uint3 local = cell & 3u;
        entry = hstrCloudNodes[(entry & ~kCloudRefNode) * 64 + local.x + 4 * (local.y + 4 * local.z)];
        return entry != kCloudRefNone && (entry & kCloudRefNode) != 0 ? entry & ~kCloudRefNode : kCloudRefNone;
    }

    /// A fine brick position's entry: per class, the node's 8 level-1 positions, then its 64 level-0 positions.
    uint sunFineEntry(uint node, uint sunClass, uint level, uint3 brick)
    {
        const uint3 local = brick & (level == 0 ? 3u : 1u);
        const uint index = level == 0 ? 8u + local.x + 4u * (local.y + 4u * local.z) : local.x + 2u * (local.y + 2u * local.z);
        return sunFrame().fineBase + node * kCloudSunFineEntries + sunClass * 72u + index;
    }

    /// Whether a change reached an entry after a bake of bakeFrame (bakes kCloudSunAgeCap old always count as stale).
    bool sunEntryNewer(uint entry, uint bakeFrame)
    {
        const uint frame = sunFrame().frame;
        const uint bakeAge = frame - bakeFrame;
        const uint entryAge = (frame - hstrCloudSunField[sunFieldTexel(entry)]) & 0xFFFFu;
        return bakeAge >= kCloudSunAgeCap || entryAge < bakeAge;
    }

    /// The positions of a level whose bakes read any of the change cells in [lo, hi) voxels. A level-L brick at origin o reads the
    /// cells floor((o + a) / 32) to floor((o + b) / 32): a and b are its texels' trilinear support and the sweep towards the sun.
    void sunStampRange(float3 lo, float3 hi, float3 sweep, uint level, int3 dims, out int3 first, out int3 last)
    {
        const float s = float(1u << level);
        const float3 a = -2.5f * s - 0.5f + min(sweep, 0.f);
        const float3 b = 10.5f * s - 0.5f + max(sweep, 0.f);
        first = max(int3(ceil((lo - b) / (8.f * s) - 1e-4f)), 0);
        last = min(int3(ceil((hi - a) / (8.f * s) + 1e-4f)) - 1, dims - 1);
    }

    /// Stamp: this frame into every entry, of every class, whose bakes read the change's cells.
    void stampSunChange(uint entry)
    {
        const HSTRCloudSunFrame frame = sunFrame();
        if (entry >= frame.changeCount)
            return;
        const HSTRCloudSunChange change = hstrCloudSunChanges[entry];
        const HSTRCloudAsset asset = hstrCloudAssets[change.asset];
        const float3 lo = float3(uint3(change.first & 1023u, (change.first >> 10) & 1023u, change.first >> 20)) * 32.f;
        const float3 hi = (float3(uint3(change.last & 1023u, (change.last >> 10) & 1023u, change.last >> 20)) + 1.f) * 32.f;
        const float reach = sunReach(change.asset);
        const uint epoch = frame.frame & 0xFFFFu;
        const uint levels = min(asset.topLevel + 1u, kCloudSunFieldLevels);
        const uint classes = min(frame.classCount, kCloudSunClasses);
        // Levels >= 2: the blocks of the asset's pairs.
        for (uint c = 0; c < classes; ++c)
        {
            const uint block = hstrCloudSunFieldBlocks[change.asset * kCloudSunClasses + c];
            if (block == kCloudRefNone)
                continue;
            const float3 sweep = reach * hstrCloudSunDirections[c].xyz;
            for (uint level = 2; level < levels; ++level)
            {
                const int3 dims = int3(sunLevelDims(change.asset, level));
                int3 first, last;
                sunStampRange(lo, hi, sweep, level, dims, first, last);
                const uint offset = block + hstrCloudSunFieldLevels[change.asset * kCloudSunFieldLevels + level];
                for (int z = first.z; z <= last.z; ++z)
                    for (int y = first.y; y <= last.y; ++y)
                        for (int x = first.x; x <= last.x; ++x)
                            hstrCloudSunField[sunFieldTexel(offset + uint(x + dims.x * (y + dims.y * z)))] = epoch;
            }
        }
        // Levels 0-1: the positions of every class, in the nodes of the 32-voxel cells they lie in.
        int3 firsts[2 * kCloudSunClasses];
        int3 lasts[2 * kCloudSunClasses];
        int3 cellFirst = int3(0x3FFFFFFF);
        int3 cellLast = int3(-1);
        for (uint c = 0; c < classes; ++c)
        {
            const float3 sweep = reach * hstrCloudSunDirections[c].xyz;
            for (uint level = 0; level < 2; ++level)
            {
                const uint i = 2u * c + level;
                sunStampRange(lo, hi, sweep, level, int3(sunLevelDims(change.asset, level)), firsts[i], lasts[i]);
                if (all(firsts[i] <= lasts[i]))
                {
                    cellFirst = min(cellFirst, firsts[i] >> int(2u - level));
                    cellLast = max(cellLast, lasts[i] >> int(2u - level));
                }
            }
        }
        for (int cz = cellFirst.z; cz <= cellLast.z; ++cz)
            for (int cy = cellFirst.y; cy <= cellLast.y; ++cy)
                for (int cx = cellFirst.x; cx <= cellLast.x; ++cx)
                {
                    const int3 cell = int3(cx, cy, cz);
                    const uint node = sunFineNode(change.asset, uint3(cell));
                    if (node == kCloudRefNone)
                        continue;
                    for (uint c = 0; c < classes; ++c)
                        for (uint level = 0; level < 2; ++level)
                        {
                            const uint i = 2u * c + level;
                            const int shift = int(2u - level);
                            const int3 first = max(firsts[i], cell << shift);
                            const int3 last = min(lasts[i], ((cell + 1) << shift) - 1);
                            for (int z = first.z; z <= last.z; ++z)
                                for (int y = first.y; y <= last.y; ++y)
                                    for (int x = first.x; x <= last.x; ++x)
                                        hstrCloudSunField[sunFieldTexel(sunFineEntry(node, c, level, uint3(x, y, z)))] = epoch;
                        }
                }
    }

    /// Reset: restarts field ranges (new pair blocks, new directory nodes) at this frame, which drops every bake they cover.
    void resetSunField(uint2 thread, bool nodes)
    {
        const HSTRCloudSunFrame frame = sunFrame();
        if (thread.y >= (nodes ? frame.nodeResets : frame.blockResets) || thread.x >= (nodes ? kCloudSunFineEntries : frame.blockLength))
            return;
        const uint first = hstrCloudSunResets[(nodes ? frame.blockResets : 0u) + thread.y];
        hstrCloudSunField[sunFieldTexel(first + thread.x)] = frame.frame & 0xFFFFu;
    }

    /// Age: entries kCloudSunAgeCap old or older are set to exactly that age, a few rows a frame, so a 16-bit age never wraps. Every
    /// bake younger than the cap stays newer than them; older bakes are stale anyway.
    void ageSunField(uint2 thread)
    {
        const HSTRCloudSunFrame frame = sunFrame();
        if (thread.y >= frame.ageRows)
            return;
        const uint2 texel = uint2(thread.x, frame.ageRow + thread.y);
        if (((frame.frame - hstrCloudSunField[texel]) & 0xFFFFu) >= kCloudSunAgeCap)
            hstrCloudSunField[texel] = (frame.frame - kCloudSunAgeCap) & 0xFFFFu;
    }

    float sunReach(uint assetID) { return sunFrame().reachScale / hstrCloudAssets[assetID].sourceVoxelWorld; }

    /// Release: the bakes of unloaded bricks give their slots back.
    void releaseSunBakes(uint entry)
    {
        if (entry >= sunFrame().releaseCount)
            return;
        const uint base = hstrCloudSunRelease[entry] * kCloudSunBakesPerBrick;
        for (uint k = 0; k < kCloudSunBakesPerBrick; ++k)
        {
            const uint slot = hstrCloudSunSlotsOutput[2u * (base + k) + 1u];
            if (slot != kCloudRefNone)
                pushSunSlot(slot);
            hstrCloudSunSlotsOutput[2u * (base + k)] = kCloudRefNone;
            hstrCloudSunSlotsOutput[2u * (base + k) + 1u] = kCloudRefNone;
        }
    }

    /// Scan: a mapped brick in the cut drops its stale bakes (keeping their slots for the rebake) and counts the classes still
    /// needing one; every held bake is a holder.
    void scanSunBakes(uint b)
    {
        const HSTRCloudSunFrame frame = sunFrame();
        if (b >= frame.capacity)
            return;
        const SunSchedEntry s = sunSched(b);
        const uint base = b * kCloudSunBakesPerBrick;
        const bool inCut = (s.flags & kCloudSunInCut) != 0;
        const bool active = inCut && (s.flags & kCloudSunMapped) != 0;
        uint needs = 0;
        uint baked = 0;
        uint waiting = 0;
        if (active)
        {
            // One field entry per bake: the brick's position in its class's block (levels >= 2) or in the node over its 32-voxel cell
            // (levels 0-1). A missing block or node means nothing tracked the bake's neighbourhood: stale.
            const uint level = hstrCloudBricks[b].info & 15u;
            const uint3 brick = uint3(s.coord & 1023u, (s.coord >> 10) & 1023u, s.coord >> 20);
            const bool fine = level < 2;
            const uint fineNode = fine ? sunFineNode(s.asset, brick >> (2u - level)) : kCloudRefNone;
            const uint3 dims = sunLevelDims(s.asset, level);
            const uint position = hstrCloudSunFieldLevels[s.asset * kCloudSunFieldLevels + level] + brick.x + dims.x * (brick.y + dims.y * brick.z);
            uint stale = 0;
            for (uint k = 0; k < kCloudSunBakesPerBrick; ++k)
            {
                const uint held = hstrCloudSunSlotsOutput[2u * (base + k)];
                if (!sunKeyWanted(held, s.flags))
                    continue;
                const uint sunClass = held & 15u;
                const uint block = hstrCloudSunFieldBlocks[s.asset * kCloudSunClasses + sunClass];
                const bool missing = fine ? fineNode == kCloudRefNone : block == kCloudRefNone;
                const uint entry = fine ? sunFineEntry(fineNode, sunClass, level, brick) : block + position;
                if (missing || sunEntryNewer(entry, hstrCloudSunFrames[base + k]))
                {
                    hstrCloudSunSlotsOutput[2u * (base + k)] = kCloudRefNone;
                    ++stale;
                }
            }
            if (stale > 0)
                InterlockedAdd(hstrCloudSunCounters[5], stale);
            for (uint c = 0; c < frame.classCount; ++c)
            {
                if ((s.flags & (1u << c)) == 0)
                    continue;
                const uint key = (frame.generation << 4) | c;
                bool have = false;
                for (uint k = 0; k < kCloudSunBakesPerBrick; ++k)
                    have = have || hstrCloudSunSlotsOutput[2u * (base + k)] == key;
                if (have)
                    ++baked;
                else
                {
                    ++waiting;
                    needs |= 1u << c;
                }
            }
        }
        uint held = 0;
        for (uint k = 0; k < kCloudSunBakesPerBrick; ++k)
        {
            const uint key = hstrCloudSunSlotsOutput[2u * (base + k)];
            const bool wanted = active && sunKeyWanted(key, s.flags);
            held += wanted ? 1u : 0u;
            if (hstrCloudSunSlotsOutput[2u * (base + k) + 1u] != kCloudRefNone)
                InterlockedAdd(hstrCloudSunHistogram[kCloudSunBins + sunHolderBin(wanted, inCut, s.priority)], 1u);
        }
        if (needs != 0 && held < kCloudSunBakesPerBrick)
            InterlockedAdd(hstrCloudSunHistogram[sunCandidateBin(s.priority)], 1u);
        else
            needs = 0;
        hstrCloudSunNeeds[b] = needs;
        if (baked > 0)
            InterlockedAdd(hstrCloudSunCounters[3], baked);
        if (waiting > 0)
            InterlockedAdd(hstrCloudSunCounters[4], waiting);
    }

    /// Select: the candidate bins that fill this frame's bakes, and the holder bins that free the slots they lack (only holders
    /// less important than the best candidate).
    void selectSunBakes()
    {
        const uint capacity = sunFrame().bakeMax;
        uint total = 0;
        uint candidateBin = 0;
        uint candidateTake = 0;
        int top = -1;
        for (int bin = int(kCloudSunBins) - 1; bin >= 0 && total < capacity; --bin)
        {
            const uint count = hstrCloudSunHistogram[bin];
            if (count == 0)
                continue;
            top = max(top, bin);
            candidateBin = uint(bin);
            candidateTake = min(count, capacity - total);
            total += candidateTake;
        }
        const uint freeSlots = hstrCloudSunFreeTop[0];
        const uint needed = total > freeSlots ? total - freeSlots : 0u;
        const uint limit = top >= 0 ? min(uint(top) + 2u, kCloudSunBins) : 0u;
        uint holders = 0;
        uint holderBin = 0;
        uint holderTake = 0;
        for (uint bin = 0; bin < limit && holders < needed; ++bin)
        {
            const uint count = hstrCloudSunHistogram[kCloudSunBins + bin];
            if (count == 0)
                continue;
            holderBin = bin;
            holderTake = min(count, needed - holders);
            holders += holderTake;
        }
        hstrCloudSunSelect[0] = candidateBin;
        hstrCloudSunSelect[1] = candidateTake;
        hstrCloudSunSelect[2] = holderBin;
        hstrCloudSunSelect[3] = holderTake;
        hstrCloudSunSelect[4] = holders;
    }

    /// Emit: the selected candidates (one bake per brick per frame) and holders into their lists.
    void emitSunBakes(uint b)
    {
        const HSTRCloudSunFrame frame = sunFrame();
        if (b >= frame.capacity)
            return;
        const SunSchedEntry s = sunSched(b);
        const uint needs = hstrCloudSunNeeds[b];
        if (needs != 0)
        {
            const uint bin = sunCandidateBin(s.priority);
            bool take = bin > hstrCloudSunSelect[0];
            if (!take && bin == hstrCloudSunSelect[0])
            {
                uint reserved;
                InterlockedAdd(hstrCloudSunCounters[6], 1u, reserved);
                take = reserved < hstrCloudSunSelect[1];
            }
            if (take)
            {
                uint index;
                InterlockedAdd(hstrCloudSunCounters[0], 1u, index);
                if (index < frame.bakeMax)
                    hstrCloudSunCandidates[index] = float4(float(b), float(firstbitlow(needs)), s.priority, 0.f);
            }
        }
        const uint holders = hstrCloudSunSelect[4];
        if (holders == 0)
            return;
        const uint base = b * kCloudSunBakesPerBrick;
        const bool inCut = (s.flags & kCloudSunInCut) != 0;
        const bool active = inCut && (s.flags & kCloudSunMapped) != 0;
        for (uint k = 0; k < kCloudSunBakesPerBrick; ++k)
        {
            if (hstrCloudSunSlotsOutput[2u * (base + k) + 1u] == kCloudRefNone)
                continue;
            const bool wanted = active && sunKeyWanted(hstrCloudSunSlotsOutput[2u * (base + k)], s.flags);
            const uint bin = sunHolderBin(wanted, inCut, s.priority);
            bool take = bin < hstrCloudSunSelect[2];
            if (!take && bin == hstrCloudSunSelect[2])
            {
                uint reserved;
                InterlockedAdd(hstrCloudSunCounters[7], 1u, reserved);
                take = reserved < hstrCloudSunSelect[3];
            }
            if (!take)
                continue;
            uint index;
            InterlockedAdd(hstrCloudSunCounters[1], 1u, index);
            if (index < holders)
                hstrCloudSunHolders[index] = base + k;
        }
    }

    /// Evict: the holders' slots go back to the free list.
    void evictSunBakes(uint entry)
    {
        if (entry >= min(hstrCloudSunCounters[1], hstrCloudSunSelect[4]))
            return;
        const uint pair = hstrCloudSunHolders[entry];
        const uint slot = hstrCloudSunSlotsOutput[2u * pair + 1u];
        if (slot != kCloudRefNone)
            pushSunSlot(slot);
        hstrCloudSunSlotsOutput[2u * pair] = kCloudRefNone;
        hstrCloudSunSlotsOutput[2u * pair + 1u] = kCloudRefNone;
    }

    /// Assign: a candidate takes a pair holding a slot it may reuse, else an empty pair and a free slot, and writes its bake job.
    void assignSunBake(uint entry)
    {
        const HSTRCloudSunFrame frame = sunFrame();
        if (entry >= frame.bakeMax)
            return;
        HSTRCloudSunBake job = {};
        job.slot = kCloudRefNone;
        if (entry < min(hstrCloudSunCounters[0], frame.bakeMax))
        {
            const float4 candidate = hstrCloudSunCandidates[entry];
            const uint b = uint(candidate.x);
            const uint sunClass = uint(candidate.y);
            const SunSchedEntry s = sunSched(b);
            const uint base = b * kCloudSunBakesPerBrick;
            uint chosen = kCloudRefNone;
            uint slot = kCloudRefNone;
            for (uint k = 0; k < kCloudSunBakesPerBrick && chosen == kCloudRefNone; ++k)
            {
                const uint held = hstrCloudSunSlotsOutput[2u * (base + k) + 1u];
                if (held != kCloudRefNone && !sunKeyWanted(hstrCloudSunSlotsOutput[2u * (base + k)], s.flags))
                {
                    chosen = k;
                    slot = held;
                }
            }
            for (uint k = 0; k < kCloudSunBakesPerBrick && chosen == kCloudRefNone; ++k)
            {
                if (hstrCloudSunSlotsOutput[2u * (base + k) + 1u] != kCloudRefNone)
                    continue;
                slot = popSunSlot();
                if (slot != kCloudRefNone)
                    chosen = k;
                break;
            }
            if (chosen != kCloudRefNone)
            {
                hstrCloudSunSlotsOutput[2u * (base + chosen)] = (frame.generation << 4) | sunClass;
                hstrCloudSunSlotsOutput[2u * (base + chosen) + 1u] = slot;
                hstrCloudSunFrames[base + chosen] = frame.frame;
                const HSTRCloudAsset asset = hstrCloudAssets[s.asset];
                job.brick = b;
                job.slot = slot;
                job.asset = s.asset;
                job.level = hstrCloudBricks[b].info & 15u;
                job.origin = float3(uint3(s.coord & 1023u, (s.coord >> 10) & 1023u, s.coord >> 20) * 8u);
                job.reach = sunReach(s.asset);
                job.direction = hstrCloudSunDirections[sunClass].xyz;
                job.cap = kCloudSunDepthOpaque / (kCloudSeaMinScale * asset.sourceVoxelWorld);
                InterlockedAdd(hstrCloudSunCounters[2], 1u);
            }
        }
        hstrCloudSunBakesOutput[entry] = job;
    }

    // ---------------------------------------------------------------------------------------------
    // Beam view: the split frame's terms are marched exactly only at tile corners and centres. Pixels reconstruct them from a
    // bilinear-plus-bubble tile basis and apply their own phase, transmittance and background.

    /// Split-frame terms along the [segment.x, segment.y] part of one camera ray's volume interval, starting from unit
    /// transmittance. The phase is applied per pixel: octave 0 carries a unit weight.
    BeamSample marchBeam(float3 direction, float2 segment)
    {
        BeamSample result = {};
        CutRay ray;
        if (!makeCutRay(gScene.camera.data.posW, direction, gScene.getGridVolume(0), ray))
            return result;
#if HSTR_STRIP == 1
        // PROBE: the ray setup alone; its result depends on the ray so the setup is not compiled away.
        result.depth = ray.worldLength + ray.entryDistance;
        return result;
#endif
        ray.sunOctaves = float4(1.f, 0.f, 0.f, 0.f);
        float transmittance = 1.f;
        uint steps = 0;
        float opacityDistance = 0.f;
#if (HSTR_SHIP & 1024) != 0
        marchBeamLean(ray, segment, beamWithCache(), beamWithSingle(), result.cache, result.single, transmittance, steps, opacityDistance);
#elif (HSTR_SHIP & 512) != 0
        marchBeamShipping(ray, segment, beamWithCache(), beamWithSingle(), result.cache, result.single, transmittance, steps, opacityDistance);
#else
        marchSegmentMoment(ray, segment, beamWithCache(), beamWithSingle(), result.cache, result.single, transmittance, steps, opacityDistance);
#endif
        result.depth = -log(max(transmittance, 1e-12f));
        result.steps = float(steps);
        result.distance = transmittance < 1.f - 1e-4f ? opacityDistance / (1.f - transmittance) : 0.f;
        return result;
    }

    // ---------------------------------------------------------------------------------------------
    // Cell views: a world-space transport cache for the rays translation invalidates.
    //
    // A camera ray's split-frame terms compose exactly over any partition of the ray into disjoint intervals: C = sum T_<k C_k,
    // T = prod T_k. The partition used here is the domain's voxel cells (cell i spans [i - 0.5, i + 0.5)^3, voxel i centred at i),
    // and each cell with content caches a VIEW of its own transfer: the cell's (cache, single, transmittance, opacity centroid) for
    // every ray of a cellViewEdge^2 grid over its projection, marched from the camera that built it. A ray from another camera
    // position crosses the cell along a chord through some point m; the build ray through m's projection crosses it along a chord
    // through the same point, whose ends are displaced sideways by (camera shift across the ray) / (distance) * (half chord). That
    // misregistration is what cellViewDrift bounds. It grows with the camera's movement over the cell's distance, so a cell survives
    // far more movement than the lattice guard allows, which is a texel of the IMAGE rather than a fraction of a voxel. A cell with
    // no usable view is marched exactly along its chord, in place, and queued for a build that the next frame reads. The phase is
    // applied per pixel (single carries none), so a view holds for every direction through the cell.

    /// The plane a view is laid out on: through the cell centre, facing the build camera, with axes u and v.
    void cellViewAxes(float3 centre, float3 camera, out float3 n, out float3 u, out float3 v)
    {
        n = normalize(centre - camera);
        const float3 up = abs(n.y) < 0.99f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        u = normalize(cross(up, n));
        v = cross(n, u);
    }

    /// Half-extent of a view on that plane: the cell's circumradius, widened for perspective (its near corners project outward).
    float cellViewRadius(float distance)
    {
        const float r = 0.8660254f;
        return r * distance / max(distance - r, 1e-3f);
    }

    /// Texel coordinates (0 to cellViewEdge) of voxel position p in the view of cell centre, built from camera.
    float2 cellViewTexel(float3 p, float3 centre, float3 camera)
    {
        float3 n, u, v;
        cellViewAxes(centre, camera, n, u, v);
        const float along = length(centre - camera);
        const float3 q = p - camera;
        const float3 onPlane = camera + q * (along / max(dot(q, n), 1e-6f));
        const float2 xy = float2(dot(onPlane - centre, u), dot(onPlane - centre, v)) / cellViewRadius(along);
        return (xy * 0.5f + 0.5f) * float(params.cellViewEdge);
    }

    /// Atlas texture coordinate of a texel position in slot's view, kept half a texel inside it so the filter never reads a neighbour.
    float2 cellViewAtlasUv(uint slot, float2 texel)
    {
        const float edge = float(params.cellViewEdge);
        const float2 origin = float2(uint2(slot % params.cellViewSlotsPerRow, slot / params.cellViewSlotsPerRow)) * edge;
        return (origin + clamp(texel, 0.5f, edge - 0.5f)) * float2(params.cellViewAtlasInvWidth, params.cellViewAtlasInvHeight);
    }

    /// The domain cell of an unwrapped cell, as the map indexes it (x and z wrap, y does not).
    uint3 cellViewKey(float3 cell)
    {
        const float3 dims = float3(params.hstrExtinctionDims);
        return uint3(uint(cell.x - floor(cell.x / dims.x) * dims.x), uint(max(cell.y, 0.f)), uint(cell.z - floor(cell.z / dims.z) * dims.z));
    }

    /// The sea's distance fade (cameraExtinctionAtVoxel) at a camera distance in voxels.
    float cellViewFade(float distance)
    {
        if (params.seaMode == 0 || params.cloudPixelAngle <= 0.f)
            return 1.f;
        return saturate((1.f - distance * params.seaVoxelSize.x / params.seaViewDistance) / 0.3f);
    }

    /// Camera distances (voxels, to the cell centre) over which the density a cell's samples read stays at the level they read
    /// from distance: the footprint level of the instance at the centre, and the proxy past it. Narrowed by the circumradius so that
    /// every sample of the cell stays inside it; a cell straddling a level boundary gets a quarter of a voxel either way.
    float2 cellViewBand(float3 centre, float distance)
    {
        float2 band = float2(0.f, 1e30f);
        const float k = params.seaVoxelSize.x * params.cloudPixelAngle; // Footprint (world) per voxel of distance.
        if (params.cloudVirtual != 0 && k > 0.f)
        {
            uint instanceIndex;
            float3 x;
            float unused;
            HSTRCloudInstance instance;
            cloudInstanceAt(seaVoxel(centre), distance * k, instanceIndex, x, unused, instance);
            if (instance.scale > 0.f)
            {
                const float sv = instance.sourceVoxelWorld;
                const float level = log2(max(distance * k / sv, 1.f)) + params.cloudLodBias;
                const float low = level >= instance.proxyLevel ? instance.proxyLevel : floor(max(level, 0.f));
                const float high = level >= instance.proxyLevel ? 1e30f : min(low + 1.f, instance.proxyLevel);
                band.x = low <= params.cloudLodBias ? 0.f : sv * exp2(low - params.cloudLodBias) / k;
                band.y = high >= 1e29f ? 1e30f : sv * exp2(high - params.cloudLodBias) / k;
            }
        }
        band += float2(0.8660254f, -0.8660254f);
        if (band.x > distance || band.y < distance)
            band = float2(distance - 0.25f, distance + 0.25f);
        return band;
    }

    /// The finest voxel edge (domain voxels) a sample in cell can read: its instance's source voxel, else the proxy's. A march that
    /// starts inside cloud begins at that step, as a continuous march reaching it from cloud would.
    float cellViewFineVoxels(float3 centre)
    {
        if (params.cloudVirtual == 0)
            return 1.f;
        uint instanceIndex;
        float3 x;
        float unused;
        HSTRCloudInstance instance;
        cloudInstanceAt(seaVoxel(centre), 0.f, instanceIndex, x, unused, instance);
        return instance.scale > 0.f ? min(instance.sourceVoxelWorld / params.seaVoxelSize.x, 1.f) : 1.f;
    }

    /// Ray parameters where the ray origin + t dv enters and leaves the box [lo, lo + 1].
    float2 cellViewSlab(float3 origin, float3 dv, float3 lo)
    {
        float enter = -1e30f;
        float exit = 1e30f;
        [unroll]
        for (uint a = 0; a < 3; ++a)
        {
            if (abs(dv[a]) > 1e-12f)
            {
                const float t0 = (lo[a] - origin[a]) / dv[a];
                const float t1 = (lo[a] + 1.f - origin[a]) / dv[a];
                enter = max(enter, min(t0, t1));
                exit = min(exit, max(t0, t1));
            }
            else if (origin[a] < lo[a] || origin[a] >= lo[a] + 1.f)
                exit = -1e30f;
        }
        return float2(enter, exit);
    }

    /// REMOVED (51c2dace, 4K walk, cvdebug5): composeBeam walked each dirty ray cell by cell, reading these views in place of
    /// density - 10.05 ms against 7.57 marched, 0.297% over 0.02 against 0.137%, ~46 cells visited and ~10 occupied per ray, ~58k
    /// distinct occupied cells a frame. Per-ray traversal cost what the march did: the views are not the problem, walking them
    /// from every ray is (README red flag). Their consumer is to be projected cell x BeamTile work (beamPushProbe), not a ray.

    /// Starts the build of cell's view: takes the ring's next slot, unmaps the cell that held it, and writes the header. ~0 when the
    /// camera is too close to the cell for a view (it is then marched every time).
    uint beginCellView(int3 cell)
    {
        const float3 camera = worldToVoxel(gScene.camera.data.posW, gScene.getGridVolume(0)) - float3(params.hstrGridMin);
        const float3 centre = float3(cell);
        const float distance = length(centre - camera);
        if (distance < 4.f || params.cellViewCapacity == 0u)
            return ~0u;
        uint cursor;
        InterlockedAdd(hstrCellCursor[0], 1u, cursor);
        const uint slot = cursor % params.cellViewCapacity;
        const CellView old = hstrCellViews[slot];
        if (old.frame != 0u)
        {
            uint previous;
            InterlockedCompareExchange(hstrCellMap[cellViewKey(float3(old.cell))], slot + 1u, 0u, previous);
        }
        CellView view;
        view.camera = camera;
        const float2 band = cellViewBand(centre, distance);
        view.bandLow = band.x;
        view.bandHigh = band.y;
        view.cell = cell;
        view.flags = 0u;
        view.frame = params.cellViewFrame + 1u;
        view.fade = cellViewFade(distance);
        view.pad = 0u;
        hstrCellViews[slot] = view;
        return slot;
    }

    /// Whether domain voxel neighbour (of the 27 around cell) can hold density a sample inside the cell reads (its conservative maximum).
    bool cellViewOccupied(int3 cell, uint neighbour)
    {
        const int3 offset = int3(neighbour % 3u, (neighbour / 3u) % 3u, neighbour / 9u) - 1;
        const float3 at = float3(cell + offset);
        if (at.y < 0.f || at.y >= float(params.hstrExtinctionDims.y))
            return false;
        return hstrDomainVolume.Load(int4(int3(cellViewKey(at)), 0)).y > 0.f;
    }

    /// One texel of cell's view in slot: the build camera's ray through it, marched along its chord of the cell.
    bool buildCellViewTexel(int3 cell, uint slot, uint2 texel)
    {
        const GridVolume volume = gScene.getGridVolume(0);
        const float3 cameraWorld = gScene.camera.data.posW;
        const float3 camera = worldToVoxel(cameraWorld, volume) - float3(params.hstrGridMin);
        const float3 centre = float3(cell);
        float3 n, u, v;
        cellViewAxes(centre, camera, n, u, v);
        const float2 xy = ((float2(texel) + 0.5f) / float(params.cellViewEdge) * 2.f - 1.f) * cellViewRadius(length(centre - camera));
        const float3 target = centre + xy.x * u + xy.y * v;
        float4 a = float4(0.f, 0.f, 0.f, 1.f);
        float4 b = float4(0.f, 0.f, 0.f, 0.5f);
        bool content = false;
        CutRay ray;
        if (makeCutRay(cameraWorld, (target - camera) * params.seaVoxelSize, volume, ray))
        {
            ray.sunOctaves = float4(1.f, 0.f, 0.f, 0.f);
            const float cellWidth = float(params.hstrCellWidth);
            const float2 slab = cellViewSlab(ray.position * cellWidth, ray.direction * cellWidth, centre - 0.5f);
            const float2 chord = float2(max(slab.x, 0.f), min(slab.y, 1.f));
            if (chord.y > chord.x)
            {
                float3 cache = 0.f;
                float3 single = 0.f;
                float transmittance = 1.f;
                uint steps = 0;
                float opacityDistance = 0.f;
                float fineVoxels = cellViewFineVoxels(centre);
                marchSegmentMoment(ray, chord, beamWithCache(), beamWithSingle(), cache, single, transmittance, steps, opacityDistance, fineVoxels);
                const float d0 = ray.entryDistance + chord.x * ray.worldLength;
                const float d1 = ray.entryDistance + chord.y * ray.worldLength;
                const float centroid = transmittance < 1.f - 1e-4f
                                           ? saturate((opacityDistance / (1.f - transmittance) - d0) / max(d1 - d0, 1e-6f))
                                           : 0.5f;
                a = float4(cache, transmittance);
                b = float4(single, centroid);
                content = transmittance < 1.f - 1e-5f || any(cache > 0.f) || any(single > 0.f);
            }
        }
        const uint2 at = uint2(slot % params.cellViewSlotsPerRow, slot / params.cellViewSlotsPerRow) * params.cellViewEdge + texel;
        hstrCellTexelsOutput[at] = a;
        hstrCellTexelsSingleOutput[at] = b;
        return content;
    }

    /// Finishes a build: an empty view is flagged so that lookups skip its texels, and the cell maps to the slot.
    void endCellView(int3 cell, uint slot, bool content)
    {
        if (!content)
            hstrCellViews[slot].flags = kCellViewEmpty;
        uint previous;
        InterlockedExchange(hstrCellMap[cellViewKey(float3(cell))], slot + 1u, previous);
        uint unused;
        InterlockedAdd(hstrBeamCounts[kCellViewBuilt], 1u, unused);
    }

    /// Unmaps a view whose cell lies in a changed region (hstrBeamInvalidations, beamInvalidateCount of them).
    void invalidateCellView(uint slot)
    {
        if (slot >= params.cellViewCapacity)
            return;
        const CellView view = hstrCellViews[slot];
        if (view.frame == 0u)
            return;
        const float3 world = params.seaOrigin + (float3(view.cell) + float3(params.hstrGridMin) + 0.5f) * params.seaVoxelSize;
        const float reach = 0.8660254f * params.seaVoxelSize.x;
        bool hit = false;
        for (uint k = 0; k < params.beamInvalidateCount; ++k)
        {
            const float4 sphere = hstrBeamInvalidations[k];
            hit = hit || length(world - sphere.xyz) <= sphere.w + reach;
        }
        if (!hit)
            return;
        uint previous;
        InterlockedCompareExchange(hstrCellMap[cellViewKey(float3(view.cell))], slot + 1u, 0u, previous);
        hstrCellViews[slot].frame = 0u;
    }

    /// beamPushProbe: the push architecture's work count, before any transfer is evaluated. The README's work unit is projected
    /// transport node x BeamTile, so this starts from the world cells, not from rays: each occupied domain cell is culled, projected
    /// once into the octahedral beam image and appended to the list of every tile its projection overlaps that the screen sees.
    /// Its counters (kPush*) answer how many cells, overlaps and cells per tile that work is at park, walk and sprint.

    /// Shells: shell k tiles the image at pushTileSize << k texels, each shell's grid after the previous one's in the tile index.
    uint pushShellCount() { return clamp(params.pushShells, 1u, kPushMaxShells); }
    uint pushShellEdge(uint shell) { return params.pushTileSize << shell; }
    uint2 pushShellDims(uint shell) { return (params.beamFrameDim + pushShellEdge(shell) - 1u) / pushShellEdge(shell); }
    uint pushShellBase(uint shell)
    {
        uint base = 0;
        for (uint k = 0; k < shell; ++k)
        {
            const uint2 dims = pushShellDims(k);
            base += dims.x * dims.y;
        }
        return base;
    }

    /// The shell of a cell at a camera distance (voxels): the one whose tiles its projection spans about two of.
    uint pushShellOf(float distance)
    {
        if (params.pushShells == 0u)
            return 0u;
        return uint(clamp(ceil(log2(params.pushShellDistance / max(distance, 1e-3f))), 0.f, float(pushShellCount() - 1u)));
    }

    /// The screen's box of the beam image (beamScreenBounds) in a shell's tiles, [low, high).
    void pushTileBox(uint shell, out uint2 low, out uint2 high)
    {
        const float s = float(pushShellEdge(shell));
        low = uint2(max(floor(params.beamScreenBounds.xy / s), 0.f));
        high = min(uint2(ceil(params.beamScreenBounds.zw / s)), pushShellDims(shell));
    }

    /// Whether the screen sees a tile of a shell (beamBoxOnScreen).
    bool pushTileOnScreen(uint shell, uint2 tile)
    {
        const float s = float(pushShellEdge(shell));
        return beamBoxOnScreen(float2(tile) * s, float2(tile + 1u) * s);
    }

    /// Adds per-thread counts to hstrPushCounts once per wave.
    void pushCount(uint slot, uint value)
    {
        const uint total = WaveActiveSum(value);
        if (WaveIsFirstLane() && total != 0u)
        {
            uint unused;
            InterlockedAdd(hstrPushCounts[slot], total, unused);
        }
    }

    /// Histogram bin of a count: 0 for none, else floor(log2) + 1.
    uint pushBin(uint value) { return value == 0u ? 0u : min(firstbithigh(value) + 1u, 31u); }

    /// Lists an occupied domain cell (27-voxel occupancy, hstrCellOccupancy) as a candidate. Runs when the domain changed only.
    void listPushCandidate(uint3 key)
    {
        const uint3 dims = params.hstrExtinctionDims;
        if (any(key >= dims))
            return;
        // pushDilate: also list cells within that many cells of an occupied one (x and z wrap), for the density the march's coarse
        // levels spread past the occupancy (see countPushShare).
        bool occupied = false;
        const int dilate = int(params.pushDilate);
        for (int dz = -dilate; dz <= dilate && !occupied; ++dz)
            for (int dy = -dilate; dy <= dilate && !occupied; ++dy)
                for (int dx = -dilate; dx <= dilate && !occupied; ++dx)
                {
                    const int3 n = int3(key) + int3(dx, dy, dz);
                    if (n.y < 0 || n.y >= int(dims.y))
                        continue;
                    const int3 wrapped = int3((n.x + int(dims.x)) % int(dims.x), n.y, (n.z + int(dims.z)) % int(dims.z));
                    occupied = hstrCellOccupancy.Load(int4(wrapped, 0)) != 0u;
                }
        const uint n = WaveActiveCountBits(occupied);
        uint base = 0;
        if (WaveIsFirstLane() && n != 0u)
            InterlockedAdd(hstrPushCandidateCount[0], n, base);
        base = WaveReadLaneFirst(base);
        if (occupied)
        {
            const uint index = base + WavePrefixCountBits(occupied);
            if (index < params.pushCandidateCapacity)
                hstrPushCandidates[index] = key.x + dims.x * (key.y + dims.y * key.z);
        }
        const bool own = occupied && hstrDomainVolume.Load(int4(int3(key), 0)).y > 0.f;
        const uint owned = WaveActiveCountBits(own);
        if (WaveIsFirstLane() && owned != 0u)
        {
            uint unused;
            InterlockedAdd(hstrPushCandidateCount[1], owned, unused);
        }
    }

    /// A packed candidate's centre relative to the camera, in voxels, at its periodic image nearest the camera (x and z wrap).
    float3 pushCellOffset(uint linear)
    {
        const uint3 dims = params.hstrExtinctionDims;
        const float3 camera = worldToVoxel(gScene.camera.data.posW, gScene.getGridVolume(0)) - float3(params.hstrGridMin);
        float3 cell = float3(uint3(linear % dims.x, (linear / dims.x) % dims.y, linear / (dims.x * dims.y)));
        cell.x += float(dims.x) * round((camera.x - cell.x) / float(dims.x));
        cell.z += float(dims.z) * round((camera.z - cell.z) / float(dims.z));
        return cell - camera;
    }

    /// Projects one candidate cell and counts it into the tiles it overlaps.
    ///
    /// The cell is the domain voxel box [key - 0.5, key + 0.5) at the periodic image nearest the camera (x and z wrap). Inside one
    /// octant of directions the octahedral map is a central projection onto the octant's face followed by an affine map, so the
    /// image of a box is the convex hull of its corners' images and their bounding box is conservative. A box spanning one of the
    /// camera's axis planes is therefore split at it into up to eight pieces, each projected alone - which is also what keeps a cell
    /// straddling the map's seam (x = 0 or y = 0 below the horizon) from claiming the whole image between its two halves. A tile
    /// several pieces overlap is counted once. A cell the camera is inside covers every direction: it claims the screen's box.
    ///
    /// With pushShells the tile size follows the cell's distance (pushShellOf). MEASURED with one tile size (push3, 4K, camera in
    /// a cloud's occupancy at (0, 40, 0)): 3.3M overlaps against 0.95M from clear air, 66% of them from the 7% of visible cells
    /// nearer than 32 voxels, one cell spanning all 29,230 on-screen tiles; each cell's tiles are one thread's loop, so the probe took
    /// 62-117 ms (project 36-71, scatter 25-45). A cell's projected area grows as 1 / distance^2; its tiles must grow with it.
    /// MEASURED with shells (push4, same run, pushShells 10 against 0): near park 3.33M overlaps / 170 ms -> 464k / 0.47 ms, at most
    /// 12 tiles a cell; clear-air walk 951k / 0.62 ms -> 388k / 0.43 ms (project 0.22, scatter 0.04, sort 0.14). pushShellDistance
    /// (~202 voxels at 4K) is conservative, so shell 0 held nothing there and the far field sat in 32-texel tiles.
    ///
    /// MEASURED and REJECTED, offline: pushing the fine density itself object-order (shear-warp slices front to back into the beam
    /// image, every frame, no temporal reuse). The work is not fewer contributions, only cheaper ones. This walk's 104k visible cells
    /// (25k at 64-128 voxels, 79k at 128-256) hold ~584M fine voxels at 4K pixel footprint; 52% non-empty in occupied cells, 13-22%
    /// of those in front of T = 1e-3 (the bench cloud) - ~50M contributions a frame at perfect occlusion, plausibly up to ~150M.
    /// scripts/HSTR/bench/shear_warp_rate.py (NVIDIA Warp, RTX 4080 Laptop, real density as a dense float16 array, slice-major):
    /// on the cloud's core (38-47% of samples non-empty) shear-warp composites 10-11 G contributions/s, a per-pixel perspective
    /// march 8.7 G/s - coherence buys ~1.2x once samples do work (the whole cloud's 4.6 vs 0.5 G/s is only cheap empty space, which
    /// these lists already skip). The shipping dirty march runs ~7.3 G steps/s with virtual addressing and lighting. A full frame
    /// is ~5-15 ms against a ~0.9 ms dirty budget, before LOD, sectors, virtual bricks or a real per-voxel source.
    void projectPushCell(uint index)
    {
        if (index == 0u)
        {
            hstrPushCounts[kPushCandidates] = hstrPushCandidateCount[0];
            hstrPushCounts[kPushCandidatesOwn] = hstrPushCandidateCount[1];
        }
        if (index >= min(hstrPushCandidateCount[0], params.pushCandidateCapacity) || gScene.getGridVolumeCount() == 0)
            return;
        const uint linear = hstrPushCandidates[index];
        const float3 d = pushCellOffset(linear);
        const float3 scale = params.seaVoxelSize;
        const float3 w = d * scale;
        const float radius = 0.8660254f * scale.x;
        // Past the view distance the sea's fade (cellViewFade) is zero, so nothing in the cell reaches the camera.
        if (params.seaMode != 0 && length(w) - radius >= params.seaViewDistance)
        {
            pushCount(kPushCulledDistance, 1u);
            return;
        }
        // The frustum's planes through the eye, from the camera basis inverse (ndc = c.xy / c.z, c = beamRefInverse * w), against
        // the cell's bounding sphere.
        const float3 r0 = params.beamRefInverse0;
        const float3 r1 = params.beamRefInverse1;
        const float3 r2 = params.beamRefInverse2;
        bool outside = dot(-r2, w) > radius * length(r2);
        outside = outside || dot(r0 - r2, w) > radius * length(r0 - r2) || dot(-r0 - r2, w) > radius * length(r0 + r2);
        outside = outside || dot(r1 - r2, w) > radius * length(r1 - r2) || dot(-r1 - r2, w) > radius * length(r1 + r2);
        if (outside)
        {
            pushCount(kPushCulledFrustum, 1u);
            return;
        }

        const uint shell = pushShellOf(length(d));
        uint2 boxLow, boxHigh;
        pushTileBox(shell, boxLow, boxHigh);
        const uint tileBase = pushShellBase(shell);
        const uint tileRow = pushShellDims(shell).x;
        const float2 image = float2(params.beamFrameDim);
        const float s = float(pushShellEdge(shell));
        const float4 bounds = params.beamScreenBounds;
        const bool near = all(abs(d) <= 0.5f + 1e-3f);
        uint4 rects[8];
        uint pieces = 0;
        uint rectTiles[4] = { 0u, 0u, 0u, 0u };
        if (near)
        {
            rects[0] = uint4(boxLow, boxHigh);
            pieces = 1;
            [unroll]
            for (uint k = 0; k < 4; ++k)
            {
                const float edge = float(4u << k);
                const uint2 span = uint2(floor(bounds.zw / edge) - floor(bounds.xy / edge)) + 1u;
                rectTiles[k] = span.x * span.y;
            }
        }
        else
        {
            const float3 lo = d - 0.5f;
            const float3 hi = d + 0.5f;
            const float eps = 1e-4f; // Keeps a corner on an axis plane on its piece's side of the map's seam.
            [loop]
            for (uint p = 0; p < 8; ++p)
            {
                float3 plo = lo;
                float3 phi = hi;
                bool valid = true;
                [unroll]
                for (uint a = 0; a < 3; ++a)
                {
                    const bool upper = ((p >> a) & 1u) != 0u;
                    if (lo[a] < 0.f && hi[a] > 0.f)
                    {
                        plo[a] = upper ? eps : lo[a];
                        phi[a] = upper ? hi[a] : -eps;
                    }
                    else
                    {
                        valid = valid && !upper;
                        if (lo[a] >= 0.f)
                            plo[a] = max(lo[a], eps);
                        else
                            phi[a] = min(hi[a], -eps);
                    }
                }
                if (!valid)
                    continue;
                float2 mn = image;
                float2 mx = 0.f;
                [unroll]
                for (uint c = 0; c < 8; ++c)
                {
                    const float3 corner = float3((c & 1u) ? phi.x : plo.x, (c & 2u) ? phi.y : plo.y, (c & 4u) ? phi.z : plo.z);
                    const float2 q = octEncode(corner * scale) * image;
                    mn = min(mn, q);
                    mx = max(mx, q);
                }
                const float2 cmn = max(mn, bounds.xy);
                const float2 cmx = min(mx, bounds.zw);
                if (any(cmn > cmx))
                    continue;
                [unroll]
                for (uint k = 0; k < 4; ++k)
                {
                    const float edge = float(4u << k);
                    const uint2 span = uint2(floor(cmx / edge) - floor(cmn / edge)) + 1u;
                    rectTiles[k] += span.x * span.y;
                }
                const uint2 tl = max(uint2(floor(mn / s)), boxLow);
                const uint2 th = min(uint2(floor(mx / s)) + 1u, boxHigh);
                if (all(tl < th))
                    rects[pieces++] = uint4(tl, th);
            }
        }

        uint overlaps = 0;
        for (uint i = 0; i < pieces; ++i)
            for (uint y = rects[i].y; y < rects[i].w; ++y)
                for (uint x = rects[i].x; x < rects[i].z; ++x)
                {
                    const uint2 tile = uint2(x, y);
                    bool seen = false;
                    for (uint j = 0; j < i; ++j)
                        seen = seen || (all(tile >= rects[j].xy) && all(tile < rects[j].zw));
                    if (seen || !pushTileOnScreen(shell, tile))
                        continue;
                    uint unused;
                    InterlockedAdd(hstrPushTileCounts[tileBase + x + tileRow * y], 1u, unused);
                    ++overlaps;
                }
        // Only a cell that reaches a tile the screen sees is a piece of work. It takes a slot, which is what the lists hold: the
        // slot addresses its key and its operator. A cell past the slots is counted and left out of the lists entirely.
        uint first = 0;
        uint slot = ~0u;
        if (overlaps != 0u)
        {
            InterlockedAdd(hstrPushCounts[kPushCountSlots + 2u], 1u, slot);
            if (slot < params.pushSlotCapacity)
            {
                hstrPushSlotKeys[slot] = linear;
                InterlockedAdd(hstrPushCounts[kPushCountSlots], pieces, first);
                for (uint i = 0; i < pieces; ++i)
                    if (first + i < params.pushPieceCapacity)
                        hstrPushPieces[first + i] =
                            uint4(slot, rects[i].x | (rects[i].y << 16), rects[i].z | (rects[i].w << 16), first | (shell << 27));
            }
            uint unused;
            if (slot >= params.pushSlotCapacity)
                InterlockedAdd(hstrPushCounts[kPushSlotsDropped], 1u, unused);
            InterlockedAdd(hstrPushCounts[kPushShellCells + shell], 1u, unused);
            InterlockedAdd(hstrPushCounts[kPushShellOverlaps + shell], overlaps, unused);
            InterlockedMax(hstrPushCounts[kPushMaxPerCell], overlaps, unused);
            InterlockedAdd(hstrPushCounts[kPushHistCell + pushBin(overlaps)], 1u, unused);
            InterlockedAdd(hstrPushCounts[kPushHistDistance + min(pushBin(uint(length(d))), 15u)], 1u, unused);
            InterlockedMax(hstrPushCounts[kPushMinDistance], ~asuint(length(d)), unused);
        }
        const float voxels = length(d);
        pushCount(kPushVisible, overlaps != 0u ? 1u : 0u);
        pushCount(kPushNear, near && overlaps != 0u ? 1u : 0u);
        pushCount(kPushPieces, overlaps != 0u ? pieces : 0u);
        pushCount(kPushOverlaps, overlaps);
        pushCount(kPushNearOverlaps, voxels < 8.f ? overlaps : 0u);
        pushCount(kPushMidOverlaps, voxels >= 8.f && voxels < 32.f ? overlaps : 0u);
        [unroll]
        for (uint k = 0; k < 4; ++k)
            pushCount(kPushRectTiles + k, overlaps != 0u ? rectTiles[k] : 0u);
        pushCount(kPushDropped, overlaps != 0u && first + pieces > params.pushPieceCapacity ? 1u : 0u);
    }

    /// Gives a tile its list: an offset for its count, and its count back to zero as the scatter's cursor; a tile of two or more
    /// cells is listed for the sort. The first thread also writes the scatter's arguments, now that the pieces are counted.
    void allocatePushTile(uint id)
    {
        if (id == 0u)
        {
            const uint pieces = min(hstrPushCounts[kPushCountSlots], params.pushPieceCapacity);
            hstrPushArgs[3] = (pieces + 63u) / 64u;
            hstrPushArgs[4] = 1u;
            hstrPushArgs[5] = 1u;
            // The operators' build: 27 threads per visible cell, in rows of 32768 groups of 64.
            const uint threads = min(hstrPushCounts[kPushCountSlots + 2u], params.pushSlotCapacity) * 27u;
            const uint groups = (threads + 63u) / 64u;
            hstrPushArgs[9] = min(groups, 32768u);
            hstrPushArgs[10] = (groups + 32767u) / 32768u;
            hstrPushArgs[11] = 1u;
        }
        if (id >= params.pushTileTotal)
            return;
        uint shell = 0;
        uint base = 0;
        uint2 dims = pushShellDims(0);
        while (shell + 1u < pushShellCount() && id >= base + dims.x * dims.y)
        {
            base += dims.x * dims.y;
            dims = pushShellDims(++shell);
        }
        const uint2 tile = uint2((id - base) % dims.x, (id - base) / dims.x);
        uint2 low, high;
        pushTileBox(shell, low, high);
        const bool inBox = all(tile >= low) && all(tile < high);
        const uint count = inBox ? hstrPushTileCounts[id] : 0u;
        const uint total = WaveActiveSum(count);
        uint offset = 0;
        if (WaveIsFirstLane() && total != 0u)
            InterlockedAdd(hstrPushCounts[kPushCountSlots + 1u], total, offset);
        offset = WaveReadLaneFirst(offset);
        hstrPushTileOffsets[id] = offset + WavePrefixSum(count);
        hstrPushTileCounts[id] = 0u;
        pushCount(kPushTilesNonEmpty, count != 0u ? 1u : 0u);
        if (count != 0u)
        {
            uint unused;
            InterlockedMax(hstrPushCounts[kPushMaxPerTile], count, unused);
        }
        if (count >= 2u)
        {
            uint slot;
            InterlockedAdd(hstrPushCounts[kPushSortTiles], 1u, slot);
            hstrPushSortTiles[slot] = id;
        }
        // Tiles by cells: the finest shell only, where the screen sees them (empty ones included).
        if (shell == 0u && inBox && pushTileOnScreen(0u, tile))
        {
            uint unused;
            InterlockedAdd(hstrPushCounts[kPushHistTile + pushBin(count)], 1u, unused);
            InterlockedAdd(hstrPushCounts[kPushTilesOnScreen], 1u, unused);
        }
    }

    /// Appends a piece's cell to the lists of the tiles it overlaps, skipping the tiles an earlier piece of its cell took.
    void scatterPushPiece(uint index)
    {
        if (index >= min(hstrPushCounts[kPushCountSlots], params.pushPieceCapacity))
            return;
        const uint4 piece = hstrPushPieces[index];
        const uint2 low = uint2(piece.y & 0xFFFFu, piece.y >> 16);
        const uint2 high = uint2(piece.z & 0xFFFFu, piece.z >> 16);
        const uint shell = piece.w >> 27;
        const uint first = piece.w & 0x7FFFFFFu;
        const uint tileBase = pushShellBase(shell);
        const uint tileRow = pushShellDims(shell).x;
        uint dropped = 0;
        for (uint y = low.y; y < high.y; ++y)
            for (uint x = low.x; x < high.x; ++x)
            {
                const uint2 tile = uint2(x, y);
                bool seen = false;
                for (uint j = first; j < index; ++j)
                {
                    const uint4 other = hstrPushPieces[j];
                    seen = seen || (all(tile >= uint2(other.y & 0xFFFFu, other.y >> 16)) && all(tile < uint2(other.z & 0xFFFFu, other.z >> 16)));
                }
                if (seen || !pushTileOnScreen(shell, tile))
                    continue;
                const uint id = tileBase + x + tileRow * y;
                uint slot;
                InterlockedAdd(hstrPushTileCounts[id], 1u, slot);
                const uint at = hstrPushTileOffsets[id] + slot;
                if (at < params.pushListCapacity)
                    hstrPushList[at] = piece.x;
                else
                    ++dropped;
            }
        pushCount(kPushDropped, dropped);
    }

    /// The cell operator: a camera-independent description of what the march integrates across a visible cell, built once per cell
    /// rather than resampled per ray. It holds 27 points, 3 x 3 x 3 at -0.5, 0 and +0.5 voxels about the centre, of the extinction
    /// (cameraExtinctionAtVoxel), the sun emission sigma * residualAtDepth(sunDepthAt) and the cache emission sigma * worldCacheRadiance
    /// toward the cell - emission rather than source, so a point without density needs no sun march and the interpolation carries
    /// no bias from it. The domain's proxy is trilinear between voxel centres, so inside each octant of the cell (between its centre
    /// and a corner) the extinction the proxy gives is exactly the trilinear function of that octant's 8 points: along a chord of
    /// the octant it is a cubic, and Simpson's rule integrates it exactly. Fine bricks under the proxy are what it cannot hold.
    /// PROBE: built every frame for every visible cell; a real one would be cached in world space and rebuilt with its content.
    ///
    /// MEASURED and REJECTED (eval2, 4K walk, the same 68k samples scored against their exact march; built 0.44 ms for 104k cells):
    /// transmittance off by more than 0.02 in 3.65% of samples in EVERY operator arm - Simpson or sub-stepped alike - so the
    /// extinction the march reads is not the octants' trilinear function: at these distances cameraExtinctionAtVoxel reads fine
    /// bricks below the proxy, and 27 points a voxel cannot hold them. Radiance over 0.02: tau + the chord's exact source 17.5%
    /// (exact emission behind a wrong transmittance), constant source 10.2%, linear 7.0%, trilinear emission 6.3%; views 2.7%;
    /// the gate is 0.137%. Evaluating is also 2.5-6.8 ms against 1.09 for the exact march of the same directions. A node at domain-
    /// voxel resolution is too coarse for the density this renderer shows; a node fine enough is the bricks themselves.
    ///
    /// MEASURED and REJECTED, offline (scripts/HSTR/bench/spectral_fit.py): a node carrying spectral extinction (a few oriented
    /// harmonics / Gabor-like modes over a macro-node, integrated analytically along the chord) instead of sampled points. On the
    /// bench cloud's true level-0 density (cumulus_1_size_1, 22 source voxels a domain voxel), 160 occupied nodes x 96 chords per
    /// shape, best-M terms kept, every chord within |dT| tol: an 8^3 node (512 dof, 0.7-1.4 domain voxels) needs a median 128-256
    /// DCT / 64-128 Haar terms at 0.02 and 256-512 at 0.005; 16^3 and larger (the macro-nodes meant to coarsen the push) fail even at
    /// 2048 terms; plane waves (the proposed cos(w.x + phi) mode) do worse than both. Per crossing, 8 terms barely beat 1 (a
    /// constant): L1 8^3 over 0.02 28.8% at M = 1, 20.9% at 8, 8.0% at 64. The fine density is broadband inside a node; no choice of
    /// basis makes it few coefficients. Only the analytic integral itself holds (Epanechnikov x cosine closed form, 1e-12).
    void buildPushOperator(uint thread)
    {
        const uint slot = thread / 27u;
        const uint g = thread % 27u;
        if (slot >= min(hstrPushCounts[kPushCountSlots + 2u], params.pushSlotCapacity) || gScene.getGridVolumeCount() == 0)
            return;
        const float3 camera = worldToVoxel(gScene.camera.data.posW, gScene.getGridVolume(0)) - float3(params.hstrGridMin);
        const float3 centre = camera + pushCellOffset(hstrPushSlotKeys[slot]);
        const float3 p = centre + (float3(uint3(g % 3u, (g / 3u) % 3u, g / 9u)) - 1.f) * 0.5f;
        const float footprint = length((p - camera) * params.seaVoxelSize) * params.cloudPixelAngle;
        float fineVoxels = 1.f;
        CloudSample sample = {};
        const float sigma = cameraExtinctionAtVoxel(p, footprint, fineVoxels, sample);
        float3 sun = 0.f;
        float3 cached = 0.f;
        if (sigma > 0.f)
        {
            if (beamWithSingle())
                sun = sigma * residualAtDepth(p, sunDepthAt(p, footprint, sample), float4(1.f, 0.f, 0.f, 0.f));
            if (beamWithCache())
                cached = sigma * worldCacheRadiance(p, normalize((centre - camera) * params.seaVoxelSize));
        }
        hstrPushOps[slot * 27u + g] = uint4(
            f32tof16(sigma) | (f32tof16(sun.r) << 16), f32tof16(sun.g) | (f32tof16(sun.b) << 16),
            f32tof16(cached.r) | (f32tof16(cached.g) << 16), f32tof16(cached.b)
        );
    }

    /// Trilinear value of an octant's 8 points (x fastest) at local coordinates u in [0, 1]^3.
    float pushTrilinear(float v[8], float3 u)
    {
        const float4 x = lerp(float4(v[0], v[2], v[4], v[6]), float4(v[1], v[3], v[5], v[7]), u.x);
        const float2 y = lerp(x.xz, x.yw, u.y);
        return lerp(y.x, y.y, u.z);
    }

    /// Composites a chord of a cell from its operator (pushEvalMode 1-4): the chord splits at the cell's centre planes into
    /// segments inside one octant each, whose extinction is the octant's trilinear function.
    ///   1: the operator's optical depth (Simpson, exact on the octant's cubic) and the chord's source from the exact march of it,
    ///      alone - an oracle, so any error is the extinction's;
    ///   2: one source per cell, the extinction-weighted mean of the emission over the 27 points;
    ///   3: that mean plus an extinction-weighted gradient, axis by axis;
    ///   4: emission trilinear per octant, like the extinction.
    /// 2-4 integrate each segment in four sub-steps with the exact exponential of each.
    void pushOperatorChord(
        CutRay ray,
        float3 origin,
        float3 dv,
        float3 cell,
        uint slot,
        float2 chord,
        bool withCache,
        bool withSingle,
        inout float3 cache,
        inout float3 single,
        inout float transmittance
    )
    {
        float sigma[27];
        float3 sun[27];
        float3 cached[27];
        for (uint g = 0; g < 27u; ++g)
        {
            const uint4 packed = hstrPushOps[slot * 27u + g];
            sigma[g] = f16tof32(packed.x & 0xFFFFu);
            sun[g] = float3(f16tof32(packed.x >> 16), f16tof32(packed.y & 0xFFFFu), f16tof32(packed.y >> 16));
            cached[g] = float3(f16tof32(packed.z & 0xFFFFu), f16tof32(packed.z >> 16), f16tof32(packed.w & 0xFFFFu));
        }
        // Constant and linear sources: extinction-weighted mean and per-axis gradient of source = emission / extinction.
        float3 meanSun = 0.f;
        float3 meanCache = 0.f;
        float3 gradSun[3] = { float3(0.f), float3(0.f), float3(0.f) };
        float3 gradCache[3] = { float3(0.f), float3(0.f), float3(0.f) };
        if (params.pushEvalMode == 2u || params.pushEvalMode == 3u)
        {
            float weight = 0.f;
            float3 centroid = 0.f;
            for (uint g = 0; g < 27u; ++g)
            {
                weight += sigma[g];
                meanSun += sun[g];
                meanCache += cached[g];
                centroid += sigma[g] * (float3(uint3(g % 3u, (g / 3u) % 3u, g / 9u)) - 1.f) * 0.5f;
            }
            const float inverse = weight > 1e-12f ? 1.f / weight : 0.f;
            meanSun *= inverse;
            meanCache *= inverse;
            centroid *= inverse;
            if (params.pushEvalMode == 3u)
            {
                float3 spread = 0.f;
                for (uint g = 0; g < 27u; ++g)
                {
                    const float3 x = (float3(uint3(g % 3u, (g / 3u) % 3u, g / 9u)) - 1.f) * 0.5f - centroid;
                    const float3 sourceSun = sigma[g] > 1e-12f ? sun[g] / sigma[g] : meanSun;
                    const float3 sourceCache = sigma[g] > 1e-12f ? cached[g] / sigma[g] : meanCache;
                    [unroll]
                    for (uint a = 0; a < 3; ++a)
                    {
                        gradSun[a] += sigma[g] * (sourceSun - meanSun) * x[a];
                        gradCache[a] += sigma[g] * (sourceCache - meanCache) * x[a];
                    }
                    spread += sigma[g] * x * x;
                }
                [unroll]
                for (uint a = 0; a < 3; ++a)
                {
                    gradSun[a] = spread[a] > 1e-12f ? gradSun[a] / spread[a] : 0.f;
                    gradCache[a] = spread[a] > 1e-12f ? gradCache[a] / spread[a] : 0.f;
                }
                // The gradient is about the centroid; the evaluation below measures from the centre.
                meanSun -= gradSun[0] * centroid.x + gradSun[1] * centroid.y + gradSun[2] * centroid.z;
                meanCache -= gradCache[0] * centroid.x + gradCache[1] * centroid.y + gradCache[2] * centroid.z;
            }
        }
        // The chord's segment ends: its own ends and the centre planes it crosses, in order.
        float cuts[3];
        [unroll]
        for (uint a = 0; a < 3; ++a)
        {
            const float tc = abs(dv[a]) > 1e-12f ? (cell[a] - origin[a]) / dv[a] : -1.f;
            cuts[a] = tc > chord.x && tc < chord.y ? tc : chord.y;
        }
        if (cuts[0] > cuts[1]) { const float s = cuts[0]; cuts[0] = cuts[1]; cuts[1] = s; }
        if (cuts[1] > cuts[2]) { const float s = cuts[1]; cuts[1] = cuts[2]; cuts[2] = s; }
        if (cuts[0] > cuts[1]) { const float s = cuts[0]; cuts[0] = cuts[1]; cuts[1] = s; }
        float operatorDepth = 0.f;
        float3 localCache = 0.f;
        float3 localSingle = 0.f;
        float localTransmittance = 1.f;
        float start = chord.x;
        for (uint k = 0; k < 4u; ++k)
        {
            const float end = k < 3u ? cuts[k] : chord.y;
            if (end <= start)
                continue;
            // The segment's octant, and its 8 points of the 27 (x fastest).
            const float3 middle = origin + 0.5f * (start + end) * dv;
            const uint3 octant = uint3(middle >= cell);
            const float3 low = cell - 0.5f + 0.5f * float3(octant);
            float s8[8];
            float3 sun8[8];
            float3 cache8[8];
            [unroll]
            for (uint c = 0; c < 8; ++c)
            {
                const uint3 at = octant + uint3(c & 1u, (c >> 1) & 1u, (c >> 2) & 1u);
                const uint g = at.x + 3u * at.y + 9u * at.z;
                s8[c] = sigma[g];
                sun8[c] = sun[g];
                cache8[c] = cached[g];
            }
            const float worldLength = (end - start) * ray.worldLength;
            if (params.pushEvalMode == 1u)
            {
                const float s0 = pushTrilinear(s8, (origin + start * dv - low) * 2.f);
                const float sm = pushTrilinear(s8, (middle - low) * 2.f);
                const float s1 = pushTrilinear(s8, (origin + end * dv - low) * 2.f);
                operatorDepth += worldLength / 6.f * (s0 + 4.f * sm + s1);
            }
            else
            {
                const float dt = 0.25f * (end - start);
                for (uint q = 0; q < 4u; ++q)
                {
                    const float3 p = origin + (start + (float(q) + 0.5f) * dt) * dv;
                    const float3 u = (p - low) * 2.f;
                    const float s = max(pushTrilinear(s8, u), 0.f);
                    const float alpha = 1.f - exp(-s * 0.25f * worldLength);
                    float3 sourceSun;
                    float3 sourceCache;
                    if (params.pushEvalMode == 4u)
                    {
                        float r[8];
                        float gr[8];
                        float b[8];
                        [unroll]
                        for (uint c = 0; c < 8; ++c) { r[c] = sun8[c].r; gr[c] = sun8[c].g; b[c] = sun8[c].b; }
                        const float3 emitSun = float3(pushTrilinear(r, u), pushTrilinear(gr, u), pushTrilinear(b, u));
                        [unroll]
                        for (uint c = 0; c < 8; ++c) { r[c] = cache8[c].r; gr[c] = cache8[c].g; b[c] = cache8[c].b; }
                        const float3 emitCache = float3(pushTrilinear(r, u), pushTrilinear(gr, u), pushTrilinear(b, u));
                        sourceSun = s > 1e-12f ? emitSun / s : 0.f;
                        sourceCache = s > 1e-12f ? emitCache / s : 0.f;
                    }
                    else
                    {
                        const float3 x = p - cell;
                        sourceSun = meanSun + gradSun[0] * x.x + gradSun[1] * x.y + gradSun[2] * x.z;
                        sourceCache = meanCache + gradCache[0] * x.x + gradCache[1] * x.y + gradCache[2] * x.z;
                    }
                    localSingle += localTransmittance * alpha * max(sourceSun, 0.f);
                    localCache += localTransmittance * alpha * max(sourceCache, 0.f);
                    localTransmittance *= 1.f - alpha;
                }
            }
            start = end;
        }
        if (params.pushEvalMode == 1u)
        {
            // The oracle: the chord's own source, exactly, from unit transmittance; the operator's transmittance for what follows.
            uint oracleSteps = 0;
            float oracleDistance = 0.f;
            float oracleTransmittance = 1.f;
            float fineVoxels = cellViewFineVoxels(cell);
            marchSegmentMoment(ray, chord, withCache, withSingle, localCache, localSingle, oracleTransmittance, oracleSteps, oracleDistance, fineVoxels);
            localTransmittance = exp(-operatorDepth);
        }
        else if (params.pushEvalExactDepth != 0u)
        {
            // The converse oracle: the chord's exact transmittance, the operator's source. The operator's emission is rescaled to
            // the exact opacity, i.e. its opacity-weighted source per unit opacity is kept - exact for a constant source.
            // MEASURED (certify2, 68k samples against their exact march, walk / sprint): transmittance off by over 0.02 in 0.15% of
            // samples (the extinction is right), but log radiance over 0.02 in 7.8 / 3.7% with a constant source and 2.1 / 2.4%
            // with trilinear emission, against a 0.137% frame gate. The source varies inside a domain cell too (sun shadowing at
            // fine-voxel scale): a compact per-node source fails 15x over even with exact extinction.
            float3 exactCache = 0.f;
            float3 exactSingle = 0.f;
            uint exactSteps = 0;
            float exactDistance = 0.f;
            float exactTransmittance = 1.f;
            float fineVoxels = cellViewFineVoxels(cell);
            marchSegmentMoment(ray, chord, withCache, withSingle, exactCache, exactSingle, exactTransmittance, exactSteps, exactDistance, fineVoxels);
            const float scale = (1.f - exactTransmittance) / max(1.f - localTransmittance, 1e-6f);
            localCache *= scale;
            localSingle *= scale;
            localTransmittance = exactTransmittance;
        }
        cache += transmittance * localCache;
        single += transmittance * localSingle;
        transmittance *= localTransmittance;
    }

    /// pushEval sample index -> beam image position, or false where the grid runs past its dimensions.
    bool pushSamplePosition(uint2 index, out float2 position)
    {
        position = (float2(params.pushSampleOriginX, params.pushSampleOriginY) + float2(index) + 0.5f) * float(params.pushSampleStep);
        return all(index < uint2(params.pushSampleDimsX, params.pushSampleDimsY));
    }

    /// pushEval, Step 5 of the handoff as a MEASUREMENT: what the push lists cost to consume and how good cell views are through
    /// them. A sample direction takes the sorted list of the tile containing it in every shell, nearest shell first (shells are
    /// ordered by distance, so this is one front-to-back order), and composites each listed cell its ray crosses: from the cell's
    /// view where one is usable (the validity composeBeam used: same cell, distance band, drift, age, fade), else by marching only
    /// that cell's chord - the residual - and queueing the view's build. It stops once opaque. Nothing here walks the volume: the
    /// lists are the work. Samples are one per pushSampleStep^2 texels, so the counts are the measurement, not the time.
    ///
    /// MEASURED (eval1, 4K, shells, views 8 x 8, 131k capacity, 16k builds a frame, same run, same 68k samples; drift .04 / .1
    /// nearly identical): walk - 60 list entries read per sample, 6.6 cells crossed, views answer 87% of crossings (65% empty
    /// views), residual chords 12.7% at 4.6% of the exact march's steps; but 3.0% of samples over 0.02 against the exact march
    /// (the frame gate is 0.137%) and 2.05 ms to evaluate them against 1.09 ms to march them exactly (30 against 16 ns a sample).
    /// Sprint - views answer only 41% (59% residual), 4.9 ms against 1.2. Near walk (camera in cloud) - every sample goes opaque in
    /// cells nearer than 4 voxels, all residual, exact. So reading an 8 x 8 camera-conditioned view per crossed cell costs more
    /// than marching the cell and is far less accurate: the view payload fails Question B, whatever consumes it.
    void evaluatePushSample(uint2 index)
    {
        float2 position;
        if (!pushSamplePosition(index, position))
            return;
        const uint sampleId = index.x + params.pushSampleDimsX * index.y;
        hstrPushSamples[sampleId] = float4(0.f, 0.f, 0.f, -1.f);
        if (!beamBoxOnScreen(position - 0.5f, position + 0.5f) || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        const float3 cameraWorld = gScene.camera.data.posW;
        const float3 w = beamRayDirection(position);
        CutRay ray;
        if (!makeCutRay(cameraWorld, w, volume, ray))
            return;
        ray.sunOctaves = float4(1.f, 0.f, 0.f, 0.f);
        const float cellWidth = float(params.hstrCellWidth);
        const float3 origin = ray.position * cellWidth;
        const float3 dv = ray.direction * cellWidth;
        const float voxelsPerUnit = length(dv);
        const float3 camera = worldToVoxel(cameraWorld, volume) - float3(params.hstrGridMin);
        const float3 dims = float3(params.hstrExtinctionDims);
        const bool withCache = beamWithCache();
        const bool withSingle = beamWithSingle();
        const uint frameStamp = params.cellViewFrame + 1u;
        float3 cache = 0.f;
        float3 single = 0.f;
        float transmittance = 1.f;
        float opacityDistance = 0.f;
        uint steps = 0;
        uint entries = 0;
        uint crossed = 0;
        uint viewHits = 0;
        uint viewEmpty = 0;
        uint residual = 0;
        uint requests = 0;
        for (int shell = int(pushShellCount()) - 1; shell >= 0 && transmittance > params.cloudMinTransmittance; --shell)
        {
            const uint2 tile = uint2(position / float(pushShellEdge(uint(shell))));
            const uint2 shellDims = pushShellDims(uint(shell));
            if (any(tile >= shellDims))
                continue;
            const uint id = pushShellBase(uint(shell)) + tile.x + shellDims.x * tile.y;
            const uint base = hstrPushTileOffsets[id];
            const uint count = base < params.pushListCapacity ? min(hstrPushTileCounts[id], params.pushListCapacity - base) : 0u;
            for (uint i = 0; i < count && transmittance > params.cloudMinTransmittance; ++i)
            {
                ++entries;
                const uint cellSlot = hstrPushList[base + i];
                const float3 cell = camera + pushCellOffset(hstrPushSlotKeys[cellSlot]);
                const float2 slab = cellViewSlab(origin, dv, cell - 0.5f);
                const float2 chord = float2(max(slab.x, 0.f), min(slab.y, 1.f));
                if (!(chord.y > chord.x))
                    continue;
                ++crossed;
                const uint3 key = cellViewKey(cell);
                const float distance = length(cell - camera);
                const bool keyed = cell.y >= 0.f && cell.y < dims.y && distance >= 4.f;
                bool usable = false;
                bool request = keyed && params.pushEvalMode == 0u;
                // The cell operator: far enough out that it stands for the cell (nearer cells stay residual, as the views did).
                if (params.pushEvalMode != 0u && keyed)
                {
                    pushOperatorChord(ray, origin, dv, cell, cellSlot, chord, withCache, withSingle, cache, single, transmittance);
                    usable = true;
                    ++viewHits;
                }
                else if (keyed && params.pushEvalMode == 0u)
                {
                    const uint entry = hstrCellMap[key];
                    if (entry != 0u)
                    {
                        const uint slot = entry - 1u;
                        const CellView view = hstrCellViews[slot];
                        // How far the build ray through the chord's midpoint strays from this chord at the chord's ends.
                        const float3 m = origin + 0.5f * (chord.x + chord.y) * dv;
                        const float3 toMid = normalize(m - camera);
                        const float3 moved = camera - view.camera;
                        const float3 across = moved - dot(moved, toMid) * toMid;
                        const float drift = length(across) / max(length(m - view.camera), 1e-3f) * 0.5f * (chord.y - chord.x) * voxelsPerUnit;
                        const uint age = frameStamp - view.frame;
                        const float fadeNow = cellViewFade(distance);
                        const float fadeRatio = view.fade > 1e-4f ? fadeNow / view.fade : (fadeNow <= 1e-4f ? 1.f : 1e30f);
                        const bool same = view.frame != 0u && all(view.cell == int3(cell));
                        usable = same && distance >= view.bandLow && distance <= view.bandHigh && drift <= params.cellViewDrift &&
                                 age <= 2u * params.cellViewMaxAge && fadeRatio >= 0.9f && fadeRatio <= 1.1f;
                        request = !usable || drift > 0.5f * params.cellViewDrift || age > params.cellViewMaxAge ||
                                  fadeRatio < 0.95f || fadeRatio > 1.05f;
                        if (usable && (view.flags & kCellViewEmpty) != 0u)
                            ++viewEmpty;
                        else if (usable)
                        {
                            float2 uv = cellViewAtlasUv(slot, cellViewTexel(m, cell, view.camera));
                            float4 a = hstrCellTexels.SampleLevel(hstrLinearClampSampler, uv, 0.f);
                            float4 b = hstrCellTexelsSingle.SampleLevel(hstrLinearClampSampler, uv, 0.f);
                            // An opaque skin sits at the chord's opacity centroid, and the build ray through THAT point saw it.
                            if (params.cellViewRefine != 0 && a.a < 0.98f)
                            {
                                const float3 c = origin + (chord.x + saturate(b.a) * (chord.y - chord.x)) * dv;
                                uv = cellViewAtlasUv(slot, cellViewTexel(c, cell, view.camera));
                                a = hstrCellTexels.SampleLevel(hstrLinearClampSampler, uv, 0.f);
                                b = hstrCellTexelsSingle.SampleLevel(hstrLinearClampSampler, uv, 0.f);
                            }
                            // The fade scales the cell's extinction: exact for the transmittance, and for the light where the
                            // source is constant along the chord.
                            float T = saturate(a.a);
                            float scale = 1.f;
                            if (abs(fadeRatio - 1.f) > 1e-4f)
                            {
                                const float faded = pow(max(T, 1e-12f), fadeRatio);
                                scale = T < 0.9999f ? (1.f - faded) / (1.f - T) : fadeRatio;
                                T = faded;
                            }
                            cache += transmittance * scale * a.rgb;
                            single += transmittance * scale * b.rgb;
                            const float d0 = ray.entryDistance + chord.x * ray.worldLength;
                            const float d1 = ray.entryDistance + chord.y * ray.worldLength;
                            opacityDistance += transmittance * (1.f - T) * (d0 + saturate(b.a) * (d1 - d0));
                            transmittance *= T;
                            ++viewHits;
                        }
                    }
                }
                if (!usable)
                {
                    uint residualSteps = 0;
                    float fineVoxels = cellViewFineVoxels(cell);
                    marchSegmentMoment(ray, chord, withCache, withSingle, cache, single, transmittance, residualSteps, opacityDistance, fineVoxels);
                    steps += residualSteps;
                    ++residual;
                }
                if (request && hstrCellRequest[key] != frameStamp)
                {
                    uint previous;
                    InterlockedExchange(hstrCellRequest[key], frameStamp, previous);
                    if (previous != frameStamp)
                    {
                        uint slot;
                        InterlockedAdd(hstrCellCounters[0], 1u, slot);
                        if (slot < params.cellViewBuildCapacity)
                            hstrCellBuild[slot] = int4(int3(cell), 0);
                        ++requests;
                    }
                }
            }
        }
        const float depth = -log(max(transmittance, 1e-12f));
        hstrPushSamples[sampleId] = float4(beamRadiance(cache, single, depth, w, volume.data.albedo), transmittance);
        pushCount(kPushEvalSamples, 1u);
        pushCount(kPushEvalEntries, entries);
        pushCount(kPushEvalCrossed, crossed);
        pushCount(kPushEvalViewHits, viewHits);
        pushCount(kPushEvalViewEmpty, viewEmpty);
        pushCount(kPushEvalResidual, residual);
        pushCount(kPushEvalResidualSteps, steps);
        pushCount(kPushEvalTerminated, transmittance <= params.cloudMinTransmittance ? 1u : 0u);
        pushCount(kPushEvalRequests, requests);
        pushCount(kPushEvalResidualSamples, residual != 0u ? 1u : 0u);
    }

    /// pushEval: the exact march of a sample's direction, and the composited sample's error against it.
    void comparePushSample(uint2 index)
    {
        float2 position;
        if (!pushSamplePosition(index, position))
            return;
        const float4 composed = hstrPushSamples[index.x + params.pushSampleDimsX * index.y];
        if (composed.a < 0.f)
            return;
        const float3 w = beamRayDirection(position);
        const BeamSample exact = marchBeam(w, float2(0.f, 1.f));
        const float3 radiance = beamRadiance(exact.cache, exact.single, exact.depth, w, gScene.getGridVolume(0).data.albedo);
        const float3 difference = abs(log(1.f + max(composed.rgb, 0.f)) - log(1.f + max(radiance, 0.f)));
        const float error = max(difference.r, max(difference.g, difference.b));
        pushCount(kPushEvalOver02, error > 0.02f ? 1u : 0u);
        pushCount(kPushEvalOverT, abs(composed.a - exp(-exact.depth)) > 0.02f ? 1u : 0u);
        pushCount(kPushEvalErrorSum, uint(min(error, 100.f) * 1e4f));
        pushCount(kPushEvalExactSteps, uint(exact.steps));
        uint unused;
        InterlockedMax(hstrPushCounts[kPushEvalErrorMax], asuint(error), unused);
    }

    /// pushShare, experiment 1 as a MEASUREMENT: how many of this frame's dirty rays (every lattice point the dirty query marched,
    /// every unit the dirty march marched) cross each cell x tile interaction of the push lists, and the march steps they spend
    /// inside it. That is the most work one shared evaluation per interaction could replace. A ray walks its tile's lists front
    /// to back as evaluatePushSample does, marching each crossed cell's chord (the shipping march, so the steps are the real
    /// cost), and stops once opaque as the march does. Its exact march's steps are counted too: the crossings should account for
    /// them, or the lists miss work.
    ///
    /// MEASURED (share1, 4K, shipping frame, shells 10, last scored step): walk - 742k dirty rays (456k query points, 295k units),
    /// 32% opaque; 10.44M crossings (14.1 a ray) of 161k interactions, so 65 rays and 166 chord steps an interaction, 2.6 steps a
    /// crossing; 93% of the chord steps are in interactions 32 or more rays cross, 61% in those 128 or more. Sprint - 803k rays,
    /// 9.49M crossings, 172k interactions (55 rays, 140 steps each), 90% of steps at 32+ rays. But the chords hold only 57% (walk,
    /// 26.8M of 47.0M) and 52% (sprint) of the same rays' exact march steps: nearly half the march is outside the listed cells.
    /// MEASURED (share2, walk, same counts as share1): the whole ray with the chords' marcher takes exactly the shipping march's
    /// 47.02M steps, and the 20.2M (43%) outside the chords hold 3.8% of its optical depth - the march spends 43% of its steps
    /// crossing empty space. But that 3.8% is density the lists miss: 1.16% of the rays differ by over 0.02 in transmittance,
    /// 0.53% in log radiance.
    /// MEASURED (share3, walk): those misses are on screen (331 of 8,609 are not) and sit 64-256 voxels out, around the view
    /// distance (192), where the march reads the coarse levels - filtered density reaching past the occupancy's one-cell dilation.
    /// Payloads, rays over 0.02 against the chords' composite: each ray's own chords marched alone 1 of 742k (composition is
    /// exact); a lattice per tile interpolating each cell's transfer as is: 2^2 49%, 3^2 46%, 5^2 34%, 9^2 17.5% (mean log error
    /// 0.128, 0.063, 0.029, 0.012), the 9^2 costing 81 x 2.25 = 182 steps an interaction against the rays' own 166.
    /// MEASURED (share4, walk): normalised by the cell's geometry (extinction per length and source per opacity from the corners
    /// that meet the cell, along the ray's own chord), 2^2 25% (54% of its crossings meet no corner), 3^2 35%, 5^2 25%, 9^2 13%.
    /// So the cube's silhouette is not what fails: the transfer has structure at the scale of the image's texels. And listing every
    /// cell within one cell of an occupied one (pushDilate 1) added 3.2M crossings and no density: the same 8,604 rays miss.
    /// MEASURED (share5, walk): nor do they miss - the chords hold 28.2k (x 1e3) optical depth beyond the whole march as well as
    /// 46.0k short of it, and of the 8,609 rays over 0.02 in transmittance 4,202 are the more transparent, 4,407 the more opaque:
    /// a march restarted at every cell face places its steps differently, which moves 1.2% of rays either way. The lists are
    /// complete. Experiment 3, the oracle: interactions whose every crossing a payload answers within 0.005 / 0.01 / 0.02 -
    /// 3^2: 82 / 84 / 86% of the 161k, holding 49 / 51 / 55% of the chord steps; 5^2: the same to 1%. The easy interactions are easy
    /// at any resolution and the hard 14% (45% of the steps) are hard at all of them: at domain-cell nodes a payload can take at most
    /// about half of the in-cell march, i.e. with the empty space the lists skip, 47.0M steps become ~12M residual + ~10M crossings
    /// to compose + the payloads' builds - about 2x, not the 6-8x the budget needs.
    /// pushShare's per-crossing error, as experiments 3 and 4 score it: a crossing's light and transmittance against a reference's,
    /// seen through what lies in front of it (before).
    float pushShareLocal(float3 c, float3 s, float T, float3 cRef, float3 sRef, float TRef, float before, float3 w, float3 albedo)
    {
        const float3 a = beamRadiance(c, s, -log(max(T, 1e-12f)), w, albedo);
        const float3 b = beamRadiance(cRef, sRef, -log(max(TRef, 1e-12f)), w, albedo);
        const float3 e3 = abs(log(1.f + before * max(a, 0.f)) - log(1.f + before * max(b, 0.f)));
        return max(max(e3.r, max(e3.g, e3.b)), before * abs(T - TRef));
    }

    /// The larger error of a value moved by +-bound (light up and transmittance down, and the reverse) against the value itself.
    float pushShareBound(float3 c, float3 s, float T, float3 bc, float3 bs, float bT, float before, float3 w, float3 albedo)
    {
        const float up = pushShareLocal(c + bc, s + bs, saturate(T - bT), c, s, T, before, w, albedo);
        const float down = pushShareLocal(max(c - bc, 0.f), max(s - bs, 0.f), saturate(T + bT), c, s, T, before, w, albedo);
        return max(up, down);
    }

    /// Experiment 6: one hstrPushBrickVisits record for a run of a chord's samples in one atlas brick.
    void pushBrickVisit(uint tile, uint entry, uint samples)
    {
        if (entry == kCloudRefNone || samples == 0u)
            return;
        uint capacity, stride;
        hstrPushBrickVisits.GetDimensions(capacity, stride);
        uint slot;
        InterlockedAdd(hstrPushCounts[kPushShareVisits], 1u, slot);
        if (slot < capacity)
            hstrPushBrickVisits[slot] = uint2(tile, entry | (min(samples, 4095u) << 20));
        else
            InterlockedAdd(hstrPushCounts[kPushShareVisitsDropped], 1u, slot);
    }

    /// Experiment 6: re-walks a dirty chord at the march's step rule (majorant step, fine-voxel minimum, midpoint sample, early out
    /// at the march's opacity) and records the atlas bricks its samples read, in runs. What a tile workgroup staging each brick
    /// once in shared memory would amortise its load over is the samples per (tile, brick) pair across all the tile's rays.
    /// MEASURED and REJECTED (brickshare1, 4K, walk / sprint): of 26.6M / 23.3M chord steps, 10.2M / 8.4M read a fine brick; they
    /// fall in 265k / 261k (tile, brick) pairs over 92k distinct bricks - 38 / 32 samples a pair (20 / 17 ray visits), 111 / 92 a
    /// brick. Staging a 4 KB brick (fp16 density + sun, 10^3) per pair is 104 / 124 bytes a sample, 1.06 GB a frame (~2.5 ms of
    /// DRAM); even staged once per distinct brick, 36 B a sample and 0.37 GB a frame. The dirty work is broad and thin across the
    /// sea, not concentrated. And the cooperative kernel itself (scripts/HSTR/bench/tile_transport_rate.py, Warp, RTX 4080
    /// Laptop, real bricks, persistent tile workgroups of 64-256 rays): 5.6-11 G steps/s staged (153-232 GB/s of staging at
    /// 96-282 steps a stage, a denser case than this), 17-30 G/s with the brick left resident (the inner loop's ceiling) -
    /// against the ~7.3 G/s shipping march and the 50-70 G/s a 2 ms frame needs. Brick x tile staging does not pay.
    void pushBrickWalk(CutRay ray, float2 chord, float transmittance, uint tile, float fineVoxels)
    {
        const float cellWidth = float(params.hstrCellWidth);
        const float voxelsPerUnit = length(ray.direction) * cellWidth;
        if (voxelsPerUnit <= 1e-6f)
            return;
        const float worldPerVoxel = ray.worldLength / voxelsPerUnit;
        uint entry = kCloudRefNone;
        uint samples = 0;
        uint total = 0;
        uint guard = 0;
        uint air = 0, empty = 0, proxy = 0, proxyZero = 0;
        float t = chord.x;
        [loop]
        while (t < chord.y && transmittance > params.cloudMinTransmittance && guard < params.maxMarchSteps)
        {
            ++guard;
            const float3 v = (ray.position + t * ray.direction) * cellWidth;
            const float majorant = majorantAt(v);
            const float minimumStep = clamp(fineVoxels, params.cloudFineMinVoxels, 1.f) * params.minStepVoxels;
            const float stepVoxels = clamp(params.stepOpticalDepth / max(1e-6f, majorant * worldPerVoxel), minimumStep, params.maxStepVoxels);
            const float dt = min(stepVoxels / voxelsPerUnit, chord.y - t);
            air += majorant > 0.f ? 0u : 1u;
            if (majorant > 0.f)
            {
                CloudSample sample = {};
                const float tm = t + 0.5f * dt;
                const float sigma = cameraExtinctionAtVoxel((ray.position + tm * ray.direction) * cellWidth, rayFootprint(ray, tm), fineVoxels, sample);
                transmittance *= exp(-sigma * dt * ray.worldLength);
                empty += sample.brick && sample.empty ? 1u : 0u;
                proxy += !sample.brick && sigma > 0.f ? 1u : 0u;
                proxyZero += !sample.brick && sigma <= 0.f ? 1u : 0u;
                if (sample.brick && !sample.empty)
                {
                    if (sample.entry != entry)
                    {
                        pushBrickVisit(tile, entry, samples);
                        entry = sample.entry;
                        samples = 0;
                    }
                    ++samples;
                    ++total;
                }
            }
            t += dt;
        }
        pushBrickVisit(tile, entry, samples);
        pushCount(kPushShareVisitSamples, total);
        pushCount(kPushShareWalkSteps, guard);
        pushCount(kPushShareWalkAir, air);
        pushCount(kPushShareWalkEmpty, empty);
        pushCount(kPushShareWalkProxy, proxy);
        pushCount(kPushShareWalkProxyZero, proxyZero);
    }

    void countPushShare(uint thread)
    {
        uint listed, units, stride;
        hstrPushShareRays.GetDimensions(listed, stride);
        hstrBeamDirtyUnits.GetDimensions(units, stride);
        const uint queries = min(hstrPushCounts[kPushShareQueries], listed);
        units = min(hstrBeamDirtyCount[1], units);
        if (thread == 0u)
            hstrPushCounts[kPushShareUnits] = units;
        if (thread >= queries + units || gScene.getGridVolumeCount() == 0)
            return;
        // A lattice point's ray goes through its integer position, a unit's through its texel centre (see marchBeamUnitExact).
        const uint packed = thread < queries ? hstrPushShareRays[thread] : hstrBeamDirtyUnits[thread - queries];
        const float2 position = float2(packed & 0xFFFFu, packed >> 16) + (thread < queries ? 0.f : 0.5f);
        const GridVolume volume = gScene.getGridVolume(0);
        const float3 cameraWorld = gScene.camera.data.posW;
        const float3 w = beamRayDirection(position);
        CutRay ray;
        if (!makeCutRay(cameraWorld, w, volume, ray))
            return;
        ray.sunOctaves = float4(1.f, 0.f, 0.f, 0.f);
        const float cellWidth = float(params.hstrCellWidth);
        const float3 origin = ray.position * cellWidth;
        const float3 dv = ray.direction * cellWidth;
        const float3 camera = worldToVoxel(cameraWorld, volume) - float3(params.hstrGridMin);
        const bool withCache = beamWithCache();
        const bool withSingle = beamWithSingle();
        float3 cache = 0.f;
        float3 single = 0.f;
        float transmittance = 1.f;
        float opacityDistance = 0.f;
        uint crossings = 0;
        uint steps = 0;
        float3 payCache[kPushSharePayloads];
        float3 paySingle[kPushSharePayloads];
        float payT[kPushSharePayloads];
        uint paySteps[kPushSharePayloads];
        [unroll]
        for (uint r = 0; r < kPushSharePayloads; ++r)
        {
            payCache[r] = 0.f;
            paySingle[r] = 0.f;
            payT[r] = 1.f;
            paySteps[r] = 0u;
        }
        float3 ageCache[3];
        float3 ageSingle[3];
        float ageT[3];
        [unroll]
        for (uint k = 0; k < 3; ++k)
        {
            ageCache[k] = 0.f;
            ageSingle[k] = 0.f;
            ageT[k] = 1.f;
        }
        // Experiment 5: chord steps per tolerance (see kPushShareCertify), and per guard and tolerance (.005, .02) the ray's composite
        // taking what the guard accepts and the exact chord for the rest.
        uint certify[24];
        [unroll]
        for (uint k = 0; k < 24; ++k)
            certify[k] = 0u;
        float3 certCache[6];
        float3 certSingle[6];
        float certT[6];
        [unroll]
        for (uint k = 0; k < 6; ++k)
        {
            certCache[k] = 0.f;
            certSingle[k] = 0.f;
            certT[k] = 1.f;
        }
        uint unkeyedSteps = 0;
        // Whole waves certify or not: one lane in eight would still hold its wave for all nine marches.
        const bool certifying = (params.pushShareMode & 4u) != 0u && (thread / 64u) % kPushShareCertifyStride == 0u;
        uint entryCapacity;
        hstrPushShareEntries.GetDimensions(entryCapacity, stride);
        for (int shell = int(pushShellCount()) - 1; shell >= 0 && transmittance > params.cloudMinTransmittance; --shell)
        {
            const uint2 tile = uint2(position / float(pushShellEdge(uint(shell))));
            const uint2 shellDims = pushShellDims(uint(shell));
            if (any(tile >= shellDims))
                continue;
            const uint id = pushShellBase(uint(shell)) + tile.x + shellDims.x * tile.y;
            const uint base = hstrPushTileOffsets[id];
            const uint count = base < params.pushListCapacity ? min(hstrPushTileCounts[id], params.pushListCapacity - base) : 0u;
            for (uint i = 0; i < count && transmittance > params.cloudMinTransmittance; ++i)
            {
                const float3 cell = camera + pushCellOffset(hstrPushSlotKeys[hstrPushList[base + i]]);
                const float2 slab = cellViewSlab(origin, dv, cell - 0.5f);
                const float2 chord = float2(max(slab.x, 0.f), min(slab.y, 1.f));
                if (!(chord.y > chord.x))
                    continue;
                uint chordSteps = 0;
                float fineVoxels = cellViewFineVoxels(cell);
                const float before = transmittance;
                marchSegmentMoment(ray, chord, withCache, withSingle, cache, single, transmittance, chordSteps, opacityDistance, fineVoxels);
                uint unused;
                if (base + i < entryCapacity)
                {
                    InterlockedAdd(hstrPushShareEntries[base + i].x, 1u, unused);
                    InterlockedAdd(hstrPushShareEntries[base + i].y, chordSteps, unused);
                }
                else
                    pushCount(kPushShareEntriesDropped, 1u);
                ++crossings;
                steps += chordSteps;
                if ((params.pushShareMode & 8u) != 0u)
                    pushBrickWalk(ray, chord, before, id, cellViewFineVoxels(cell));
                // Experiment 2: the cell's shared payload over this tile. Variant 0 is the ray's own chord marched alone (what
                // composing separately marched chords costs in accuracy); variant r > 0 is a lattice of (2^(r-1) + 1)^2 chords
                // over the tile, the payload one evaluation per interaction would store, read bilinearly at the ray.
                float3 c0, s0;
                uint payloadSteps;
                float ownLength;
                const float T0 = pushChordAt(position, cell, cameraWorld, volume, c0, s0, payloadSteps, ownLength);
                // Experiment 4, the temporal oracle: this interaction's exact result as a cache would hold it from frames ago - the
                // same cell, from where the camera was then - against its result now. Nothing is approximated: a hit reuses a
                // result that was exact when it was marched.
                // MEASURED (share6, 4K), slot 0 as the same texel's direction 1 / 4 / 16 frames old, at 0.02: walk - 83% of the
                // interactions reusable but only 53 / 51 / 52% of the chord steps (77 / 69 / 69% counted per crossing), 24.5 / 46 /
                // 51% of rays over 0.02 reusing all; sprint (1 frame) 53% of steps, 40% of rays. The dense interactions change.
                // MEASURED (share7), the midpoint ray, 1 / 4 frames old, interactions reusable and their chord steps at 0.005 /
                // 0.01 / 0.02: walk - 1 frame 93 / 96 / 99% of interactions, 73 / 84 / 94% of steps (per crossing 93 / 96 / 99%),
                // 3.7% of rays over 0.02 reusing every crossing; 4 frames 77% of steps at 0.02, 8.0% of rays. Sprint - 1 frame
                // (20 units) 90% of interactions, 66% of steps at 0.02 (per crossing 89%), 8.5% of rays; 4 frames 57%, 14%. This
                // is a DENSE cache: the old ray through exactly this chord's midpoint, which the old frame never marched.
                // MEASURED (share8, same settle, anchor reproduced at 94%): snapped to the centre of the old beam texel, as a cache
                // of one result per texel must be, walk keeps 50 / 53 / 58% of the steps at 0.005 / 0.01 / 0.02 (1 frame; 56% at 4
                // frames), 6.9% of rays over 0.02 reusing all; sprint 57% (1 frame), 56% (4). The snap moves the ray a median of
                // under 0.5 fine voxels from the midpoint (p90 under 1, p99 under 1.41, max 1.6-1.7): a beam texel is about a fine
                // voxel here, and half of one loses 36 of the 94 points. Temporal reuse fails for the reason the payloads did.
                // MEASURED offline (scripts/HSTR/bench/ray_jet_fit.py, whole rays through the bench cloud at mip 1, texel = 1 fine
                // voxel, oracle routing): a texel storing its ray's transfer plus 4D ray-space derivatives ("ray jet"). Of the steps
                // the snapped texel loses, sprint recovers at 0.02 / 0.005 - Jacobian 69 / 51% (D 300) and 87 / 73% (D 800), bilinear
                // 57 / 27% and 70 / 41%, Hessian 26 / 20% and 86 / 70% (finite differences across voxel-edge kinks: worse than J
                // near). Predicting tau (the Lie-algebra log of [[T, E], [0, 1]]) instead of T, E gains nothing. On share8's
                // population that projects to ~87% (near) / ~94% (far) of dirty steps at 0.02, 79 / 89% at 0.005 - under the 90%
                // sprint kill line near, and that is with oracle routing: a real cache needs a bound to trust a jet unmarched, and
                // every miss marches 4 tangents in a register-bound kernel. Random frustum rays are easy (nearest keeps 92-96%), so
                // only the recovered share transfers; confirming it needs the jet predictor in this probe on the real dirty rays.
                if ((params.pushShareMode & 2u) != 0u)
                {
                    const float3 motion = float3(params.pushShareMotionX, params.pushShareMotionY, params.pushShareMotionZ);
                    const float3 now = beamRadiance(c0, s0, -log(max(T0, 1e-12f)), w, volume.data.albedo);
                    // A world-keyed cache: the old camera's ray through the same world point, the chord's midpoint - its ends stray
                    // only by the sideways step x half the chord / the distance. Slot 0 is that exact ray 1 frame ago (the dense
                    // cache, the anchor); slots 1 and 2 snap it to the centre of the beam texel it lands in, 1 and 4 frames ago - the
                    // ray a cache holding one result per texel would have. (share6's slot 0, the texel's direction from the old
                    // camera, is MEASURED below: translation moves that ray sideways by the whole step.)
                    const float3 midpoint = cameraWorld + w * (ray.entryDistance + 0.5f * (chord.x + chord.y) * ray.worldLength);
                    [unroll]
                    for (uint k = 0; k < 3; ++k)
                    {
                        const float3 old = cameraWorld - (k == 2u ? 4.f : 1.f) * motion;
                        float3 direction = normalize(midpoint - old);
                        float2 texel;
                        if (k != 0u && beamImagePosition(direction, texel))
                        {
                            direction = beamRayDirection(floor(texel) + 0.5f);
                            // How far the snapped ray passes from the midpoint, in the cell's fine voxels: half-octave bins.
                            const float offset = length(cross(midpoint - old, direction)) / params.seaVoxelSize.x / max(cellViewFineVoxels(cell), 1e-3f);
                            const uint bin = uint(clamp(floor(2.f * log2(max(offset, 1e-12f))) + 32.f, 0.f, 47.f));
                            InterlockedAdd(hstrPushCounts[kPushShareSnapOffsets + 48u * (k - 1u) + bin], 1u, unused);
                            InterlockedMax(hstrPushCounts[kPushShareSnapMax + k - 1u], asuint(offset), unused);
                        }
                        float3 c, s;
                        float length;
                        uint oldSteps;
                        const float T = pushChordAlong(direction, cell, old, volume, c, s, oldSteps, length);
                        const float3 then = beamRadiance(c, s, -log(max(T, 1e-12f)), w, volume.data.albedo);
                        const float3 e3 = abs(log(1.f + before * max(then, 0.f)) - log(1.f + before * max(now, 0.f)));
                        const float local = max(max(e3.r, max(e3.g, e3.b)), before * abs(T - T0));
                        if (base + i < entryCapacity && k == 0u)
                            InterlockedMax(hstrPushShareAges[base + i].x, asuint(local), unused);
                        else if (base + i < entryCapacity && k == 1u)
                            InterlockedMax(hstrPushShareAges[base + i].y, asuint(local), unused);
                        else if (base + i < entryCapacity)
                            InterlockedMax(hstrPushShareAges[base + i].z, asuint(local), unused);
                        [unroll]
                        for (uint t = 0; t < 3; ++t)
                        {
                            const float tolerance = t == 0u ? 0.005f : (t == 1u ? 0.01f : 0.02f);
                            pushCount(kPushShareAgeCrossingSteps + 3u * k + t, local <= tolerance ? chordSteps : 0u);
                        }
                        ageCache[k] += ageT[k] * c;
                        ageSingle[k] += ageT[k] * s;
                        ageT[k] *= T;
                    }
                }
                // Experiment 5, certification: share8's snapped 1-frame cache, which half the time is already right, is only worth
                // having if something can tell WHICH half without marching. A central-difference Jacobian of the snapped chord in the
                // 4 ray coordinates (offset across the ray at the plane through its point nearest the midpoint, and tilt about that
                // point) predicts the new crossing and gives three guards: its predicted change accepts the stale value; the L1 sum
                // of |J_i dr_i| (what one scalar sensitivity per offset / tilt pair would store) accepts it; the second differences
                // estimate the jet's own error and accept its prediction. Each is scored against the exact chord now.
                // MEASURED and REJECTED (certify2, 4K, one settle, 1 in 8 blocks of 64 dirty rays, ~90k rays; on battery, so no
                // timings), chord steps per crossing at 0.02, walk / sprint: snapped value right 91 / 86%, jet 96 / 89%. Guards
                // (accepted, of which wrong): J change 92 / 93% (2.8 / 8.1%), L1 scalar 90 / 89% (2.2 / 7.0%), jet self 95 / 94%
                // (1.7 / 8.0%); at 0.005 walk 79-87% accepted with 5-7% wrong. After each guard 3.7-4.0% (walk) and 5.7-6.6%
                // (sprint) of the dirty rays are over 0.02 against the chords - the frame's gate is 0.137% of pixels. The wrong
                // accepts are as large as the headroom: a local derivative cannot see the voxel-edge kinks it would cross, so it
                // cannot certify. The jet's own ceiling at sprint (89% at 0.02, oracle routing) is under the 93% that pays for it.
                if (certifying)
                {
                    const float3 albedo = volume.data.albedo;
                    const float3 motion = float3(params.pushShareMotionX, params.pushShareMotionY, params.pushShareMotionZ);
                    const float3 old = cameraWorld - motion;
                    const float3 midpoint = cameraWorld + w * (ray.entryDistance + 0.5f * (chord.x + chord.y) * ray.worldLength);
                    float2 texel;
                    bool keyed = beamImagePosition(normalize(midpoint - old), texel);
                    const float3 d0 = keyed ? beamRayDirection(floor(texel) + 0.5f) : w;
                    const float3 e1 = normalize(cross(d0, abs(d0.y) < 0.9f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f)));
                    const float3 e2 = cross(d0, e1);
                    const float span = dot(midpoint - old, d0);
                    const float3 pivot = old + span * d0;
                    const float along = dot(w, d0);
                    keyed = keyed && along > 1e-3f && span > 1e-3f;
                    float3 cS = c0;
                    float3 sS = s0;
                    float TS = T0;
                    float3 cJ = c0;
                    float3 sJ = s0;
                    float TJ = T0;
                    float eNear = 1e30f;
                    float eJet = 1e30f;
                    float estChange = 1e30f;
                    float estL1 = 1e30f;
                    float estSelf = 1e30f;
                    if (keyed)
                    {
                        const float tNew = dot(pivot - cameraWorld, d0) / along;
                        const float3 x = cameraWorld + tNew * w - pivot;
                        const float4 dr = float4(dot(x, e1), dot(x, e2), dot(w, e1) / along, dot(w, e2) / along);
                        // Steps: a quarter of the cell's fine voxel across, and a tilt moving the chord's ends by as much.
                        const float h = 0.25f * max(cellViewFineVoxels(cell), 1e-3f) * params.seaVoxelSize.x;
                        const float tilt = 2.f * h / max((chord.y - chord.x) * ray.worldLength, 1e-3f);
                        const float4 step = float4(h, h, tilt, tilt);
                        uint unusedSteps;
                        float unusedLength;
                        TS = pushChordAlong(d0, cell, old, volume, cS, sS, unusedSteps, unusedLength);
                        float3 jC = 0.f, jS = 0.f, aC = 0.f, aS = 0.f, kC = 0.f, kS = 0.f;
                        float jT = 0.f, aT = 0.f, kT = 0.f;
                        [unroll]
                        for (uint a = 0; a < 4; ++a)
                        {
                            const float3 axis = (a & 1u) == 0u ? e1 : e2;
                            float3 cp, sp, cm, sm;
                            float Tp, Tm;
                            if (a < 2u)
                            {
                                Tp = pushChordAlong(d0, cell, old + step[a] * axis, volume, cp, sp, unusedSteps, unusedLength);
                                Tm = pushChordAlong(d0, cell, old - step[a] * axis, volume, cm, sm, unusedSteps, unusedLength);
                            }
                            else
                            {
                                const float3 dp = normalize(d0 + step[a] * axis);
                                const float3 dm = normalize(d0 - step[a] * axis);
                                Tp = pushChordAlong(dp, cell, pivot - span * dp, volume, cp, sp, unusedSteps, unusedLength);
                                Tm = pushChordAlong(dm, cell, pivot - span * dm, volume, cm, sm, unusedSteps, unusedLength);
                            }
                            const float u = dr[a] / step[a];
                            jC += 0.5f * (cp - cm) * u;
                            jS += 0.5f * (sp - sm) * u;
                            jT += 0.5f * (Tp - Tm) * u;
                            aC += abs(0.5f * (cp - cm) * u);
                            aS += abs(0.5f * (sp - sm) * u);
                            aT += abs(0.5f * (Tp - Tm) * u);
                            kC += 0.5f * abs(cp + cm - 2.f * cS) * u * u;
                            kS += 0.5f * abs(sp + sm - 2.f * sS) * u * u;
                            kT += 0.5f * abs(Tp + Tm - 2.f * TS) * u * u;
                        }
                        cJ = max(cS + jC, 0.f);
                        sJ = max(sS + jS, 0.f);
                        TJ = saturate(TS + jT);
                        eNear = pushShareLocal(cS, sS, TS, c0, s0, T0, before, w, albedo);
                        eJet = pushShareLocal(cJ, sJ, TJ, c0, s0, T0, before, w, albedo);
                        estChange = pushShareLocal(cJ, sJ, TJ, cS, sS, TS, before, w, albedo);
                        estL1 = pushShareBound(cS, sS, TS, aC, aS, aT, before, w, albedo);
                        estSelf = pushShareBound(cJ, sJ, TJ, kC, kS, kT, before, w, albedo);
                    }
                    else
                        unkeyedSteps += chordSteps;
                    [unroll]
                    for (uint t = 0; t < 3; ++t)
                    {
                        const float tolerance = t == 0u ? 0.005f : (t == 1u ? 0.01f : 0.02f);
                        certify[8u * t + 0u] += eNear <= tolerance ? chordSteps : 0u;
                        certify[8u * t + 1u] += eJet <= tolerance ? chordSteps : 0u;
                        certify[8u * t + 2u] += estChange <= tolerance ? chordSteps : 0u;
                        certify[8u * t + 3u] += estChange <= tolerance && eNear > tolerance ? chordSteps : 0u;
                        certify[8u * t + 4u] += estL1 <= tolerance ? chordSteps : 0u;
                        certify[8u * t + 5u] += estL1 <= tolerance && eNear > tolerance ? chordSteps : 0u;
                        certify[8u * t + 6u] += estSelf <= tolerance ? chordSteps : 0u;
                        certify[8u * t + 7u] += estSelf <= tolerance && eJet > tolerance ? chordSteps : 0u;
                    }
                    [unroll]
                    for (uint g = 0; g < 6; ++g)
                    {
                        const float tolerance = g < 3u ? 0.005f : 0.02f;
                        const uint guard = g % 3u;
                        const float estimate = guard == 0u ? estChange : (guard == 1u ? estL1 : estSelf);
                        const bool accept = estimate <= tolerance;
                        const float3 c = accept ? (guard == 2u ? cJ : cS) : c0;
                        const float3 s = accept ? (guard == 2u ? sJ : sS) : s0;
                        const float T = accept ? (guard == 2u ? TJ : TS) : T0;
                        certCache[g] += certT[g] * c;
                        certSingle[g] += certT[g] * s;
                        certT[g] *= T;
                    }
                }
                if ((params.pushShareMode & 1u) == 0u)
                    continue; // The payloads (experiments 2 and 3) are measured; see above.
                payCache[0] += payT[0] * c0;
                paySingle[0] += payT[0] * s0;
                payT[0] *= T0;
                const float edge = float(pushShellEdge(uint(shell)));
                [unroll]
                for (uint r = 1; r < kPushSharePayloads; ++r)
                {
                    // Normalised by the cell's geometry, so the cube's silhouette is not interpolated: the corners' extinction per
                    // unit length and source per unit opacity, over the corners whose ray meets the cell, applied along this ray's
                    // own chord. With no corner in the cell the ray's own chord stands (the residual).
                    const float spacing = edge / float(1u << (r - 1u));
                    const float2 g = floor(position / spacing);
                    const float2 f = position / spacing - g;
                    float3 cc = 0.f;
                    float3 ss = 0.f;
                    float opacity = 0.f;
                    float sigma = 0.f;
                    float hit = 0.f;
                    [unroll]
                    for (uint k = 0; k < 4; ++k)
                    {
                        const float2 corner = float2(k & 1u, k >> 1u);
                        const float weight = (corner.x != 0.f ? f.x : 1.f - f.x) * (corner.y != 0.f ? f.y : 1.f - f.y);
                        float3 c, s;
                        float length;
                        const float T = pushChordAt((g + corner) * spacing, cell, cameraWorld, volume, c, s, payloadSteps, length);
                        paySteps[r] += payloadSteps;
                        if (length > 0.f)
                        {
                            cc += weight * c;
                            ss += weight * s;
                            opacity += weight * (1.f - T);
                            sigma += weight * -log(max(T, 1e-12f)) / length;
                            hit += weight;
                        }
                    }
                    float3 cellCache = c0;
                    float3 cellSingle = s0;
                    float cellT = T0;
                    if (hit > 1e-6f)
                    {
                        cellT = exp(-sigma / hit * ownLength);
                        const float perOpacity = opacity > 1e-6f ? (1.f - cellT) / opacity : 0.f;
                        cellCache = cc * perOpacity;
                        cellSingle = ss * perOpacity;
                    }
                    else
                        pushCount(kPushSharePayloadResidual + r, 1u);
                    // Experiment 3's oracle: this crossing's own error under the 3^2 and 5^2 payloads - the cell's light and
                    // transmittance against its own chord's, as seen through what lies in front - kept as the interaction's maximum.
                    if ((r == 2u || r == 3u) && base + i < entryCapacity)
                    {
                        const float3 mine = beamRadiance(c0, s0, -log(max(T0, 1e-12f)), w, volume.data.albedo);
                        const float3 theirs = beamRadiance(cellCache, cellSingle, -log(max(cellT, 1e-12f)), w, volume.data.albedo);
                        const float3 e3 = abs(log(1.f + before * max(theirs, 0.f)) - log(1.f + before * max(mine, 0.f)));
                        const float local = max(max(e3.r, max(e3.g, e3.b)), before * abs(cellT - T0));
                        if (r == 2u)
                            InterlockedMax(hstrPushShareEntries[base + i].z, asuint(local), unused);
                        else
                            InterlockedMax(hstrPushShareEntries[base + i].w, asuint(local), unused);
                    }
                    payCache[r] += payT[r] * cellCache;
                    paySingle[r] += payT[r] * cellSingle;
                    payT[r] *= cellT;
                }
            }
        }
        [unroll]
        for (uint r = 0; r < kPushSharePayloads; ++r)
        {
            const float3 payload = beamRadiance(payCache[r], paySingle[r], -log(max(payT[r], 1e-12f)), w, volume.data.albedo);
            const float3 chords = beamRadiance(cache, single, -log(max(transmittance, 1e-12f)), w, volume.data.albedo);
            const float3 error3 = abs(log(1.f + max(payload, 0.f)) - log(1.f + max(chords, 0.f)));
            const float error = max(error3.r, max(error3.g, error3.b));
            pushCount(kPushSharePayloadOver02 + r, error > 0.02f ? 1u : 0u);
            pushCount(kPushSharePayloadOverT + r, abs(payT[r] - transmittance) > 0.02f ? 1u : 0u);
            pushCount(kPushSharePayloadErrorSum + r, uint(min(error, 100.f) * 1e4f));
            pushCount(kPushSharePayloadSteps + r, paySteps[r]);
        }
        // The temporal oracle per ray: every crossing reused from k frames ago, composited, against the chords now.
        [unroll]
        for (uint k = 0; k < 3; ++k)
        {
            const float3 reused = beamRadiance(ageCache[k], ageSingle[k], -log(max(ageT[k], 1e-12f)), w, volume.data.albedo);
            const float3 chords = beamRadiance(cache, single, -log(max(transmittance, 1e-12f)), w, volume.data.albedo);
            const float3 error3 = abs(log(1.f + max(reused, 0.f)) - log(1.f + max(chords, 0.f)));
            pushCount(kPushShareAgeOver02 + k, max(error3.r, max(error3.g, error3.b)) > 0.02f ? 1u : 0u);
        }
        if ((params.pushShareMode & 4u) != 0u)
        {
            [unroll]
            for (uint k = 0; k < 24; ++k)
                pushCount(kPushShareCertify + k, certify[k]);
            pushCount(kPushShareCertifyUnkeyed, unkeyedSteps);
            pushCount(kPushShareCertifySteps, certifying ? steps : 0u);
            pushCount(kPushShareCertifyRays, certifying ? 1u : 0u);
            const float3 chords = beamRadiance(cache, single, -log(max(transmittance, 1e-12f)), w, volume.data.albedo);
            [unroll]
            for (uint g = 0; g < 6; ++g)
            {
                const float3 guarded = beamRadiance(certCache[g], certSingle[g], -log(max(certT[g], 1e-12f)), w, volume.data.albedo);
                const float3 error3 = abs(log(1.f + max(guarded, 0.f)) - log(1.f + max(chords, 0.f)));
                pushCount(kPushShareCertifyOver02 + g, certifying && max(error3.r, max(error3.g, error3.b)) > 0.02f ? 1u : 0u);
            }
        }
        const BeamSample exact = marchBeam(w, float2(0.f, 1.f));
        // The whole ray with the chords' own marcher: its steps less theirs are the march outside every listed cell, and its
        // optical depth and radiance against the chords' composite say whether that march met density the lists miss.
        float3 fullCache = 0.f;
        float3 fullSingle = 0.f;
        float fullTransmittance = 1.f;
        uint fullSteps = 0;
        float fullDistance = 0.f;
        marchSegmentMoment(ray, float2(0.f, 1.f), withCache, withSingle, fullCache, fullSingle, fullTransmittance, fullSteps, fullDistance);
        const float chordDepth = -log(max(transmittance, 1e-12f));
        const float fullDepth = -log(max(fullTransmittance, 1e-12f));
        const float3 chordRadiance = beamRadiance(cache, single, chordDepth, w, volume.data.albedo);
        const float3 fullRadiance = beamRadiance(fullCache, fullSingle, fullDepth, w, volume.data.albedo);
        const float3 difference = abs(log(1.f + max(chordRadiance, 0.f)) - log(1.f + max(fullRadiance, 0.f)));
        pushCount(kPushShareFullSteps, fullSteps);
        pushCount(kPushShareOverT, abs(transmittance - fullTransmittance) > 0.02f ? 1u : 0u);
        // Where the density the lists miss is: rays the screen cannot see (the guard's reach past its edge), and the rays that
        // miss by camera distance of the whole march's opacity centroid.
        const bool onScreen = beamBoxOnScreen(position - 0.5f, position + 0.5f);
        const bool missed = abs(transmittance - fullTransmittance) > 0.02f;
        pushCount(kPushShareOffscreen, onScreen ? 0u : 1u);
        pushCount(kPushShareOffscreenOverT, !onScreen && missed ? 1u : 0u);
        if (missed)
        {
            const float voxels = fullDistance / max(1.f - fullTransmittance, 1e-6f) / params.seaVoxelSize.x;
            uint unused;
            InterlockedAdd(hstrPushCounts[kPushShareMissDistance + min(pushBin(uint(max(voxels, 0.f))), 15u)], 1u, unused);
        }
        pushCount(kPushShareOver02, max(difference.r, max(difference.g, difference.b)) > 0.02f ? 1u : 0u);
        pushCount(kPushShareDepthMissing, uint(clamp(fullDepth - chordDepth, 0.f, 10.f) * 1000.f));
        // The other sign: if the chords also hold depth the whole march lacks, the difference is where the steps fall, not a miss.
        pushCount(kPushShareDepthExtra, uint(clamp(chordDepth - fullDepth, 0.f, 10.f) * 1000.f));
        pushCount(kPushShareOverTMore, transmittance - fullTransmittance > 0.02f ? 1u : 0u);
        pushCount(kPushShareDepthTotal, uint(min(fullDepth, 10.f) * 1000.f));
        pushCount(kPushShareRays, 1u);
        pushCount(kPushShareOpaque, transmittance <= params.cloudMinTransmittance ? 1u : 0u);
        pushCount(kPushShareCrossings, crossings);
        pushCount(kPushShareSteps, steps);
        pushCount(kPushShareExactSteps, uint(exact.steps));
    }

    /// pushShare's payload: a cell's transfer along the camera ray through beam position p, marched alone (cache and single
    /// premultiplied by its own transmittance, which it returns). Nothing where the ray misses the cell.
    float pushChordAt(float2 p, float3 cell, float3 cameraWorld, GridVolume volume, out float3 cache, out float3 single, out uint steps,
                      out float length)
    {
        return pushChordAlong(beamRayDirection(p), cell, cameraWorld, volume, cache, single, steps, length);
    }

    /// pushChordAt along direction w from cameraWorld; cell is in the voxel frame of the current camera (pushCellOffset).
    float pushChordAlong(float3 w, float3 cell, float3 cameraWorld, GridVolume volume, out float3 cache, out float3 single, out uint steps,
                         out float length)
    {
        cache = 0.f;
        single = 0.f;
        steps = 0u;
        length = 0.f;
        CutRay ray;
        if (!makeCutRay(cameraWorld, w, volume, ray))
            return 1.f;
        ray.sunOctaves = float4(1.f, 0.f, 0.f, 0.f);
        const float cellWidth = float(params.hstrCellWidth);
        const float2 slab = cellViewSlab(ray.position * cellWidth, ray.direction * cellWidth, cell - 0.5f);
        const float2 chord = float2(max(slab.x, 0.f), min(slab.y, 1.f));
        if (!(chord.y > chord.x))
            return 1.f;
        length = (chord.y - chord.x) * ray.worldLength;
        float transmittance = 1.f;
        float opacityDistance = 0.f;
        float fineVoxels = cellViewFineVoxels(cell);
        marchSegmentMoment(ray, chord, beamWithCache(), beamWithSingle(), cache, single, transmittance, steps, opacityDistance, fineVoxels);
        return transmittance;
    }

    /// pushShare: per tile, its interactions some dirty ray crossed, by shell and by how many rays crossed each.
    void histogramPushShare(uint id)
    {
        if (id >= params.pushTileTotal)
            return;
        uint shell = 0;
        uint next = pushShellDims(0).x * pushShellDims(0).y;
        while (id >= next && shell + 1u < pushShellCount())
        {
            ++shell;
            next += pushShellDims(shell).x * pushShellDims(shell).y;
        }
        const uint base = hstrPushTileOffsets[id];
        uint entryCapacity, stride;
        hstrPushShareEntries.GetDimensions(entryCapacity, stride);
        const uint count = base < entryCapacity ? min(hstrPushTileCounts[id], entryCapacity - base) : 0u;
        uint interactions = 0;
        uint crossings = 0;
        uint steps = 0;
        uint unused;
        for (uint i = 0; i < count; ++i)
        {
            const uint4 e = hstrPushShareEntries[base + i];
            if (e.x == 0u)
                continue;
            // Experiment 4: the interactions whose every crossing a cache k frames old answers within a tolerance.
            if ((params.pushShareMode & 2u) != 0u)
            {
                const uint4 ages = hstrPushShareAges[base + i];
                [unroll]
                for (uint k = 0; k < 3; ++k)
                {
                    const float worst = asfloat(k == 0u ? ages.x : (k == 1u ? ages.y : ages.z));
                    [unroll]
                    for (uint t = 0; t < 3; ++t)
                    {
                        const float tolerance = t == 0u ? 0.005f : (t == 1u ? 0.01f : 0.02f);
                        if (worst <= tolerance)
                        {
                            InterlockedAdd(hstrPushCounts[kPushShareAgeSafe + 6u * k + 2u * t], 1u, unused);
                            InterlockedAdd(hstrPushCounts[kPushShareAgeSafe + 6u * k + 2u * t + 1u], e.y, unused);
                        }
                    }
                }
            }
            ++interactions;
            crossings += e.x;
            steps += e.y;
            // Experiment 3: the interactions whose every crossing the payload answers within a tolerance, and their chord steps.
            [unroll]
            for (uint p = 0; p < 2; ++p)
            {
                const float worst = asfloat(p == 0u ? e.z : e.w);
                [unroll]
                for (uint k = 0; k < 3; ++k)
                {
                    const float tolerance = k == 0u ? 0.005f : (k == 1u ? 0.01f : 0.02f);
                    if (worst <= tolerance)
                    {
                        InterlockedAdd(hstrPushCounts[kPushShareSafe + 6u * p + 2u * k], 1u, unused);
                        InterlockedAdd(hstrPushCounts[kPushShareSafe + 6u * p + 2u * k + 1u], e.y, unused);
                    }
                }
            }
            const uint bin = min(firstbithigh(e.x), 15u);
            InterlockedAdd(hstrPushCounts[kPushShareHistCount + bin], 1u, unused);
            InterlockedAdd(hstrPushCounts[kPushShareHistSteps + bin], e.y, unused);
        }
        if (interactions == 0u)
            return;
        InterlockedAdd(hstrPushCounts[kPushShareShell + 3u * shell], interactions, unused);
        InterlockedAdd(hstrPushCounts[kPushShareShell + 3u * shell + 1u], crossings, unused);
        InterlockedAdd(hstrPushCounts[kPushShareShell + 3u * shell + 2u], steps, unused);
    }

    /// marchSegmentMoment in the beam view with every HSTR_SHIP switch at its shipping value (HSTR_SHIP bit 512): a midpoint march
    /// with the empty-cell skip, the held cache and the sun depth per contributing sample, and none of the experiments' state. The
    /// same frame, from a loop DXC sees without the dead branches.
    /// MEASURED and OFF (4K, 2026-09-17, same frame everywhere): 4-6% slower than the folded generic loop (mask 509) - near 12.34 ->
    /// 13.04 ms, farside 7.15 -> 7.58, sea 5.83 -> 5.98, flying 2 units 5.96 -> 6.12. DXC already drops what the folds leave dead.
    void marchBeamShipping(
        CutRay ray,
        float2 segment,
        bool withSource,
        bool residual,
        inout float3 radiance,
        inout float3 residualRadiance,
        inout float transmittance,
        inout uint steps,
        inout float opacityDistance
    )
    {
        const float cellWidth = float(params.hstrCellWidth);
        const float voxelsPerUnit = length(ray.direction) * cellWidth;
        if (voxelsPerUnit <= 1e-6f)
            return;
        const float worldPerVoxel = ray.worldLength / voxelsPerUnit;
        const float cellVoxels = float(params.worldCacheCellVoxels);
        float3 held = 0.f;
        float3 heldAt = 0.f;
        float heldSun = 0.f;
        float3 sunAt = float3(1e30f);
        uint sinceLighting = ~0u;
        float fineVoxels = 1.f;
        float t = segment.x;
        [loop]
        while (t < segment.y && transmittance > params.cloudMinTransmittance && steps < params.maxMarchSteps)
        {
            const float3 v = (ray.position + t * ray.direction) * cellWidth;
            const float majorant = majorantAt(v);
            const float minimumStep = clamp(fineVoxels, params.cloudFineMinVoxels, 1.f) * params.minStepVoxels;
            const float stepVoxels = clamp(params.stepOpticalDepth / max(1e-6f, majorant * worldPerVoxel), minimumStep, params.maxStepVoxels);
            const float dt = min(stepVoxels / voxelsPerUnit, segment.y - t);
            ++steps;
            if (majorant <= 0.f)
            {
                t = min(max(t + dt, majorantZeroExit(ray, t)), segment.y);
                continue;
            }
            const float tSample = t + 0.5f * dt;
            const float3 midpoint = (ray.position + tSample * ray.direction) * cellWidth;
            const float footprint = rayFootprint(ray, tSample);
            CloudSample sample = {};
            const float sigma = cameraExtinctionAtVoxel(midpoint, footprint, fineVoxels, sample);
            const float alpha = 1.f - exp(-(sigma * dt * ray.worldLength));
            if ((withSource || residual) && alpha > 1e-6f)
            {
                if (withSource)
                {
                    const float3 moved = midpoint - heldAt;
                    if (sinceLighting >= params.lightingStride || dot(moved, moved) > cellVoxels * cellVoxels)
                    {
                        held = worldCacheRadiance(midpoint, ray.viewDirection);
                        heldAt = midpoint;
                        sinceLighting = 0;
                    }
                    ++sinceLighting;
                    radiance += transmittance * alpha * held;
                }
                if (residual)
                {
                    if (!(sample.empty && sunAt.x < 1e29f))
                    {
                        heldSun = sunDepthAt(midpoint, footprint, sample);
                        sunAt = midpoint;
                    }
                    residualRadiance += transmittance * alpha * residualAtDepth(midpoint, heldSun, ray.sunOctaves);
                }
            }
            opacityDistance += transmittance * alpha * (ray.entryDistance + tSample * ray.worldLength);
            transmittance *= 1.f - alpha;
            t = sample.empty ? min(max(t + dt, cloudEmptyExit(ray, tSample, sample)), segment.y) : t + dt;
        }
    }

    /// cameraExtinctionAtVoxel for the lean march: the same value, but whatever the rest of the step needs from the lookup is taken
    /// out here - the empty box's exit (skipTo, negative where the sample is not an empty box) and the brick for the sun (hit) -
    /// so the CloudSample dies inside this function instead of staying live across the lighting.
    /// MEASURED and REMOVED (leanloads1, 4K walk / sprint, errors identical): two exact load cuts, both slower - the proxy fetched
    /// only where it answers, 6.04 -> 6.16 / 7.52 -> 7.68 ms (the up-front fetch is independent and overlaps the chain), and the
    /// sea tile instance cached in registers along the ray, reloaded only on a tile change, 6.04 -> 6.53 / 7.52 -> 8.12 ms (its ~20
    /// registers cost more occupancy than its L1-hit loads cost latency). Registers are the currency here, not load counts.
    float leanExtinction(CutRay ray, float3 v, float footprint, float tSample, inout float fineVoxels, out LeanHit hit, out float skipTo)
    {
        hit.entry = kCloudRefNone;
        hit.instance = 0;
        hit.levels = 0;
        hit.x = 0.f;
        hit.scale = 0.f;
        hit.near = 0.f;
        skipTo = -1.f;
        fineVoxels = 1.f;
        const float fade = params.seaMode != 0 && params.cloudPixelAngle > 0.f
                               ? saturate((1.f - footprint / (params.cloudPixelAngle * params.seaViewDistance)) / 0.3f)
                               : 1.f;
        const float proxy = extinctionAtVoxel(v);
        uint instanceIndex;
        float3 x;
        float footprintLevel;
        HSTRCloudInstance instance;
        if (!cloudInstanceAt(v, footprint, instanceIndex, x, footprintLevel, instance))
            return fade * proxy;
        const float scale = gScene.getGridVolume(0).data.densityScale * params.densityScale;
        const float thinDensity = params.cloudThinDepth / max(scale * 3.4641f * instance.sourceVoxelWorld, 1e-12f);
        CloudSample sample;
#if (HSTR_SHIP & 2048) != 0
        const bool found = cloudAssetBrickAtFlat(instance, x, uint(max(footprintLevel, 0.f)), true, thinDensity, sample);
#else
        const bool found = cloudAssetBrickAt(hstrCloudAssets[instance.asset], x, uint(max(footprintLevel, 0.f)), true, thinDensity, sample);
#endif
        sample.instance = instanceIndex;
        if (!found)
            return fade * proxy;
        fineVoxels = float(1u << sample.level) * instance.sourceVoxelWorld / params.seaVoxelSize.x;
        if (sample.empty)
        {
            skipTo = cloudEmptyExitOf(ray, tSample, sample, instance);
            return 0.f;
        }
        hit.entry = sample.entry;
        hit.instance = instanceIndex;
        hit.levels = sample.level | (sample.desired << 8) | (instance.sunClass << 16);
        hit.x = x;
        hit.scale = instance.sourceVoxelWorld;
        hit.near = instance.scale;
        const HSTRCloudBrick brick = hstrCloudBricks[sample.entry];
        const int3 i = int3(floor(x + 0.5f));
        float density = cloudBrickDensity(sample.entry, x, i) * scale;
        if (brick.fade < 1.f)
            density = lerp(brick.parent != kCloudRefNone ? cloudBrickDensity(brick.parent, x, i) * scale : proxy, density, brick.fade);
        return fade * density;
    }

    /// sunDepthAt for a lean sample, from the few words of it the lean march kept.
    float leanSunDepth(float3 v, float footprint, LeanHit hit)
    {
#if (HSTR_SHIP & 2048) != 0
        // The frame's resolved bake (resolveCloudSunSlots): sunDepthAt's baked branch without the key search. Anything else - no
        // bake of the brick or an ancestor, a brick coarser than the footprint asked for, no near reach - takes sunDepthAt below.
        // The hit carries what the bake needs from the instance (see LeanHit).
        if (hit.entry != kCloudRefNone && params.sunNearVoxels > 0.f && SHIP(cloudSunCache, 1u) != 0 &&
            (hit.levels & 255u) >= ((hit.levels >> 8) & 255u))
        {
            const float sourceVoxelWorld = hit.scale;
            const float near = params.sunNearVoxels * hit.near;
            const uint2 resolved = hstrCloudSunResolved[hit.entry * kCloudSunClasses + (hit.levels >> 16)];
            if (resolved.x != kCloudRefNone)
            {
                // The far field fetched beside the bake rather than after it: its address never depended on the bake, only the
                // opaque early-out did, which put one more memory round trip in series on every lit sample. Exact.
                // MEASURED (leanchain1, 4K, errors identical): walk 6.02 / 6.02 -> 5.97 ms, sprint 7.50 / 7.51 -> 7.46.
                const float3 far = seaVoxel(seaVoxel(v) + near * params.sunVoxelDirection);
                const float farDepth = hstrFineSun.SampleLevel(hstrLinearSampler, (far + 0.5f) / float3(params.hstrFineDims), 0.f);
                const float baked = cloudBrickSunAt(resolved.y >> 24, resolved.x, hit.x) * sourceVoxelWorld;
                return baked > kCloudSunDepthOpaque ? baked : baked + farDepth;
            }
        }
#endif
        CloudSample sample = {};
        sample.brick = hit.entry != kCloudRefNone;
        sample.entry = hit.entry;
        sample.instance = hit.instance;
        sample.level = hit.levels & 255u;
        sample.desired = (hit.levels >> 8) & 255u;
        sample.x = hit.x;
        return sunDepthAt(v, footprint, sample);
    }

    /// The shipping beam march (every HSTR_SHIP switch at its shipping value, lightingStride 1, cloudSunReuse 0) with as little
    /// state live across a step as it can have: the Nsight capture of the shipping walk frame showed its dirty passes at 22% / 32%
    /// warps in flight with 68% / 63% of the warp slots unallocated on active SMs and DRAM read at 31% / 22% - occupancy-bound, not
    /// bandwidth-bound, so every register the loop holds is latency the SM cannot hide. Same samples, same values: the step rule,
    /// the midpoint, the empty-box and zero-majorant skips, the baked sun and the cache at every contributing sample.
    /// MEASURED (lean1, 4K, one settle, against ship / ship again): walk 7.45 / 7.42 -> 6.63 ms (dirty query 2.90 -> 2.62, units
    /// 3.43 -> 2.90), sprint 9.18 / 9.18 -> 8.18 ms, with every error statistic identical (0.137% / 0.028% over 0.02, p99.9, max).
    /// MEASURED and REMOVED (leanbatch1): B = 2 / 4 / 8 steps in flight - planned ahead on the majorant, density, sun and cache
    /// fetched before any compositing, a batch cut where a skip or a new fineVoxels left its plan, so the samples stayed exact -
    /// took walk 6.68 -> 9.06 / 22.2 / 70.1 ms and sprint 8.25 -> 11.2 / 27.5 / 87.4. Unrolling this 8-10-load branchy lookup
    /// chain multiplies the code and the live state, and occupancy was already the limit. The offline gather's win
    /// (gather_path_rate.py) needs its two-load branch-free lookup first; batch only after the chain is flat.
    void marchBeamLean(
        CutRay ray,
        float2 segment,
        bool withSource,
        bool residual,
        inout float3 radiance,
        inout float3 residualRadiance,
        inout float transmittance,
        inout uint steps,
        inout float opacityDistance
    )
    {
        const float cellWidth = float(params.hstrCellWidth);
        const float voxelsPerUnit = length(ray.direction) * cellWidth;
        if (voxelsPerUnit <= 1e-6f)
            return;
        // One per-ray value where there were two (voxelsPerUnit, worldPerVoxel): the reciprocal, the world distance per voxel
        // taken from worldLength (live anyway) per step, and the step divides as multiplies. With the sun fields carried in the
        // hit (LeanHit): sunhit4 below.
        const float unitsPerVoxel = 1.f / voxelsPerUnit;
        const bool cached = params.debugView == kWorldCacheView || params.debugView == kBeamView;
        // The step bounds as set. MEASURED and REMOVED (sunhit3, 4K, errors identical): beamPolicy scaling them per build (a
        // scale read from a buffer, so both bounds became live registers instead of constant-buffer values) cost walk 0.02-0.03
        // and sprint 0.07 ms at scale 1 - and longer steps had already failed on jog's content (policy3).
        const float minStepVoxels = params.minStepVoxels;
        const float maxStepVoxels = params.maxStepVoxels;
        float fineVoxels = 1.f;
        float t = segment.x;
#if (HSTR_SHIP & 4096) != 0
        // Step classes (probe, pushShare's counters): all, majorant zero, empty box, proxy, contributing (alpha > 1e-6).
        uint countSteps = 0, countZero = 0, countEmpty = 0, countProxy = 0, countLit = 0;
        uint countFaint = 0, countThin = 0, countAtMax = 0, countAtMin = 0, countDeep = 0;
#endif
#if (HSTR_SHIP & 262144) != 0
        // Span probe: the density samples (neither majorant-zero nor an empty box) in runs of one brick (or the proxy) at one step,
        // each run evenly spaced from its first midpoint - what evaluateSpans integrates alone. Recording arm only: never timed.
        uint spanRay;
        InterlockedAdd(hstrSpanCounts[kSpanRays], 1u, spanRay);
        uint spanRays, spanStride;
        hstrSpanRays.GetDimensions(spanRays, spanStride);
        const bool spanKept = spanRay < spanRays / kSpanRayWords;
        uint spanCount = 0;
        bool spanOverflow = false;
        uint openEntry = 0, openInstance = 0, openSamples = 0;
        float openFirst = 0.f, openDt = 0.f;
#endif
        // MEASURED and REMOVED (leandefer1, 4K, errors identical): deferred lighting. The loop issues 81.4M lane steps for 61.5M
        // steps (76%, leancount3 walk), but only 28% of steps are lit, so the light body ran with ~7 of 32 lanes (Nsight: 14.3
        // threads per warp active). A lit sample parked its light (midpoint, T alpha, footprint, hit) and stopped stepping; the
        // wave lit the parked samples together once pending * K >= lanes in the loop. Walk 6.01 / 6.03 -> 9.34 / 6.57 / 6.37 ms at
        // K = 1 / 2 / 4, sprint 7.47 / 7.50 -> 11.53 / 8.31 / 8.03. The light waits on memory, not on lanes: fuller waves do not
        // hide its latency better, and the parked state and the stalled traversal cost more.
        const uint maxSteps = HSTR_STRIP == 2 ? 1u : params.maxMarchSteps; // PROBE: one step.
        [loop]
        while (t < segment.y && transmittance > params.cloudMinTransmittance && steps < maxSteps)
        {
            const float3 v = (ray.position + t * ray.direction) * cellWidth;
            const float majorant = majorantAt(v);
            const float minimumStep = clamp(fineVoxels, params.cloudFineMinVoxels, 1.f) * minStepVoxels;
            const float stepVoxels = clamp(params.stepOpticalDepth / max(1e-6f, majorant * ray.worldLength * unitsPerVoxel), minimumStep, maxStepVoxels);
#if (HSTR_SHIP & (16384 | 32768)) != 0
            // Transmittance-scaled steps (an approximation, judged against the path-traced reference): a step's error reaches the
            // pixel weighted by the transmittance in front of it, so behind transmittance T the step may grow by 1 / sqrt(T), up to
            // 2x (bit 16384) or 4x (bit 32768).
            // MEASURED and REMOVED (budgetjitter2, 4K, beamOctScale 0.5, 2x steps + bit 32768): a per-direction phase jitter of
            // the first step with density after every skip (skips snap t to block boundaries, which is what locks the phase to
            // the world grid and makes the long step's error structured). Walk 2.46 ms 0.760% -> 2.71 ms 0.639% over 0.02; 2.5x
            // steps 2.49 ms 1.650%, 3x 2.32 ms 2.481%; sprint 2.90 ms 0.283% -> 3.16 ms 0.238%. The error falls 16-25% but the
            // extra steps cost ~0.25 ms, and the unjittered shorter step dominates on both axes.
            const float boost = min((HSTR_SHIP & 32768) != 0 ? 4.f : 2.f, rsqrt(max(transmittance, 1e-6f)));
            const float dt = min(boost * stepVoxels * unitsPerVoxel, segment.y - t);
#else
            const float dt = min(stepVoxels * unitsPerVoxel, segment.y - t);
#endif
            float next = t + dt;
#if (HSTR_SHIP & 4096) != 0
            ++countSteps;
            countZero += majorant > 0.f ? 0u : 1u;
#endif
            if (majorant > 0.f)
            {
                const float tSample = t + 0.5f * dt;
                const float3 midpoint = (ray.position + tSample * ray.direction) * cellWidth;
                const float footprint = rayFootprint(ray, tSample);
                LeanHit hit;
                float skipTo;
                const float sigma = leanExtinction(ray, midpoint, footprint, tSample, fineVoxels, hit, skipTo);
                const float alpha = 1.f - exp(-(sigma * dt * ray.worldLength));
#if (HSTR_SHIP & 4096) != 0
                countEmpty += skipTo >= 0.f ? 1u : 0u;
                countProxy += hit.entry == kCloudRefNone && skipTo < 0.f ? 1u : 0u;
                countLit += alpha > 1e-6f ? 1u : 0u;
                if (alpha > 1e-6f)
                {
                    countFaint += alpha < 1e-3f ? 1u : 0u;
                    countThin += alpha >= 1e-3f && alpha < 1e-2f ? 1u : 0u;
                    countAtMax += stepVoxels >= maxStepVoxels ? 1u : 0u;
                    countAtMin += stepVoxels <= minimumStep ? 1u : 0u;
                    countDeep += transmittance < 0.25f ? 1u : 0u;
                }
#endif
#if (HSTR_SHIP & 262144) != 0
                if (skipTo < 0.f && spanKept)
                {
                    const uint entry = hit.entry; // kCloudRefNone: the proxy.
                    const bool continues = openSamples != 0u && entry == openEntry && hit.instance == openInstance && dt == openDt &&
                                           abs(tSample - (openFirst + float(openSamples) * openDt)) <= 1e-4f * dt;
                    if (continues)
                        ++openSamples;
                    else
                    {
                        if (openSamples != 0u)
                        {
                            if (spanCount < kSpanPerRay)
                            {
                                const uint at = 2u * (spanRay * kSpanPerRay + spanCount);
                                hstrSpanRecords[at] = uint4(asuint(openFirst), asuint(openDt), openSamples, openEntry);
                                hstrSpanRecords[at + 1u] = uint4(openInstance, 0u, 0u, 0u);
                                ++spanCount;
                            }
                            else
                                spanOverflow = true;
                        }
                        openEntry = entry;
                        openInstance = hit.instance;
                        openFirst = tSample;
                        openDt = dt;
                        openSamples = 1u;
                    }
                }
#endif
                if (alpha > 1e-6f)
                {
                    // PROBE (HSTR_STRIP 4 / 5 / 6): density alone, density + sun, density + cache.
                    if (residual && HSTR_STRIP != 4 && HSTR_STRIP != 6)
                        residualRadiance += transmittance * alpha * residualAtDepth(midpoint, leanSunDepth(midpoint, footprint, hit), ray.sunOctaves);
                    // The cache at every contributing sample: the shipping lightingStride 1, and the only stride this march takes.
                    // MEASURED and REMOVED (leanstride1, 4K walk / sprint): holding it over 2 / 4 / 8 samples (never across a cache
                    // cell) saved nothing - 6.39 -> 6.37 / 6.38 / 6.64 ms - for 0.137% -> 0.153% / 1.048% / 4.985% of walk over 0.02,
                    // and carrying the hold state cost ~0.3 ms even at stride 1 (6.06 in leanflat2). The cache is not what this waits on.
                    // MEASURED and REMOVED (leanchain1, 4K, errors identical): the cache evaluated ahead of the sun chain, so its
                    // independent fetches would not wait behind it - walk 6.02 -> 6.53 ms, sprint 7.50 -> 8.09. Codegen, not memory.
                    // MEASURED and REMOVED (sunhit3, errors identical): the view direction recomputed per lit sample, normalize(ray.direction),
                    // instead of held along the ray - no change (walk 2.074 -> 2.071 ms, sprint 2.400 -> 2.409).
                    if (withSource && HSTR_STRIP != 4 && HSTR_STRIP != 5)
                        radiance += transmittance * alpha * (cached ? worldCacheRadiance(midpoint, ray.viewDirection) : sourceAtVoxel(midpoint));
                }
                opacityDistance += transmittance * alpha * (ray.entryDistance + tSample * ray.worldLength);
                transmittance *= 1.f - alpha;
                if (skipTo >= 0.f)
                    next = min(max(next, skipTo), segment.y);
            }
            else
#if (HSTR_SHIP & 8192) != 0
                next = min(max(next, majorantZeroExitFine(ray, t)), segment.y);
#else
                next = min(max(next, majorantZeroExit(ray, t)), segment.y);
#endif
            t = next;
            ++steps;
        }
#if (HSTR_SHIP & 262144) != 0
        if (spanKept)
        {
            if (openSamples != 0u)
            {
                if (spanCount < kSpanPerRay)
                {
                    const uint at = 2u * (spanRay * kSpanPerRay + spanCount);
                    hstrSpanRecords[at] = uint4(asuint(openFirst), asuint(openDt), openSamples, openEntry);
                    hstrSpanRecords[at + 1u] = uint4(openInstance, 0u, 0u, 0u);
                    ++spanCount;
                }
                else
                    spanOverflow = true;
            }
            hstrSpanRays[kSpanRayWords * spanRay] = uint4(asuint(ray.position), asuint(ray.worldLength));
            hstrSpanRays[kSpanRayWords * spanRay + 1u] = uint4(asuint(ray.direction), asuint(ray.entryDistance));
            hstrSpanRays[kSpanRayWords * spanRay + 2u] =
                uint4(spanCount | (spanOverflow ? 0x80000000u : 0u), asuint(transmittance), asuint(opacityDistance), 0u);
            hstrSpanRays[kSpanRayWords * spanRay + 3u] = uint4(asuint(residualRadiance), 0u);
        }
#endif
#if (HSTR_SHIP & 4096) != 0
        pushCount(kPushShareWalkSteps, countSteps);
        pushCount(kPushShareWalkAir, countZero);
        pushCount(kPushShareWalkEmpty, countEmpty);
        pushCount(kPushShareWalkProxy, countProxy);
        pushCount(kPushShareWalkProxyZero, countLit);
        pushCount(kPushShareVisitSamples, 1u); // Rays.
        pushCount(kPushShareWalkLitFaint, countFaint);
        pushCount(kPushShareWalkLitThin, countThin);
        pushCount(kPushShareWalkLitAtMax, countAtMax);
        pushCount(kPushShareWalkLitAtMin, countAtMin);
        pushCount(kPushShareWalkLitDeep, countDeep);
        // Against the sums above: the share of issued lane slots that march a step (SIMD efficiency over the loop).
        const uint waveSteps = WaveActiveMax(countSteps) * WaveGetLaneCount();
        const uint waveLit = WaveActiveMax(countLit) * WaveGetLaneCount();
        if (WaveIsFirstLane())
        {
            uint unused;
            InterlockedAdd(hstrPushCounts[kPushShareWalkWaveSteps], waveSteps, unused);
            InterlockedAdd(hstrPushCounts[kPushShareWalkWaveLit], waveLit, unused);
        }
#endif
    }

    /// Span probe: one recorded ray integrated from its spans alone - the direct-span evaluator's ceiling on the real dirty rays,
    /// with span production free. Per span the brick, the instance and the resolved sun bake are loaded once; per sample only the
    /// density atlas, the sun atlas and the far field (or the proxy) are fetched: no majorant, no instance or page lookup, no
    /// empty-box or majorant-zero steps. The same samples as the march (its own runs, first midpoint and step), density and the
    /// single-scattered sun only - the arm runs hstComponents 1 | 2 - and the result is checked against the march's (kSpanMismatch*).
    /// MEASURED (leanspan1, 4K, one settle): walk - march query + units 3.65 ms (anchor 3.62), spans 1.77 ms for the 80% of the
    /// 673k rays within kSpanPerRay spans (20.5% overflowed and were skipped: ~2.2 ms for all); sprint 4.51 -> 2.24. 2.52M spans,
    /// 9.06M samples: 3.6 samples a span - a ray crosses a brick in a handful of samples, so the per-span setup is paid every 3-4 -
    /// at ~5.1 G samples/s (the offline direct arm's 15 was one packed texture, no far sun, no proxy). Exactness: 854 / 1,703 rays
    /// of ~535k differ in T / light (max |dT| 0.19; fading bricks or sample drift). With spans free and no traversal the direct
    /// evaluator buys ~1.7-2x on density + sun, not the ~5x the dirty passes need: gate A (<= 0.7 ms) FAILED, and span production,
    /// the far sun + cache and the overflow rays would all add to it. Kept as a probe (compiled out unless spanProbe).
    /// counting: the untimed second pass - the same integration, compared with the march and counted. The timed pass only writes
    /// its result (word 4 of the ray), as a real evaluator would; per-ray atomics in it would time themselves (see
    /// tile_transport_rate.py).
    void evaluateSpanRay(uint ray, bool counting)
    {
        uint capacity, stride;
        hstrSpanRays.GetDimensions(capacity, stride);
        if (ray >= min(hstrSpanCounts[kSpanRays], capacity / kSpanRayWords))
            return;
        const uint4 h2 = hstrSpanRays[kSpanRayWords * ray + 2u];
        if ((h2.x & 0x80000000u) != 0u)
        {
            if (counting)
            {
                uint unused;
                InterlockedAdd(hstrSpanCounts[kSpanOverflow], 1u, unused);
            }
            return;
        }
        const uint4 h0 = hstrSpanRays[kSpanRayWords * ray];
        const uint4 h1 = hstrSpanRays[kSpanRayWords * ray + 1u];
        const float3 position = asfloat(h0.xyz);
        const float worldLength = asfloat(h0.w);
        const float3 direction = asfloat(h1.xyz);
        const float entryDistance = asfloat(h1.w);
        const float cellWidth = float(params.hstrCellWidth);
        const float scale = gScene.getGridVolume(0).data.densityScale * params.densityScale;
        const float fadeScale = params.seaMode != 0 && params.cloudPixelAngle > 0.f ? 1.f / (params.cloudPixelAngle * params.seaViewDistance) : 0.f;
        const float3 sunColour = params.residualStrength * params.sunRadiance;
        float T = 1.f;
        float3 L = 0.f;
        uint samples = 0, baked = 0, fallback = 0;
        const uint spans = h2.x;
        for (uint k = 0; k < spans; ++k)
        {
            const uint4 a = hstrSpanRecords[2u * (ray * kSpanPerRay + k)];
            const float first = asfloat(a.x);
            const float dt = asfloat(a.y);
            const uint n = a.z;
            const uint entry = a.w;
            const float opticalScale = dt * worldLength;
            samples += n;
            if (entry == kCloudRefNone)
            {
                // The proxy: its density and its sun (sunDepthProxy), as leanExtinction and leanSunDepth answer there.
                for (uint j = 0; j < n; ++j)
                {
                    const float t = first + float(j) * dt;
                    const float3 v = (position + t * direction) * cellWidth;
                    const float footprint = (entryDistance + t * worldLength) * params.cloudPixelAngle;
                    const float fade = fadeScale > 0.f ? saturate((1.f - footprint * fadeScale) / 0.3f) : 1.f;
                    const float alpha = 1.f - exp(-(fade * extinctionAtVoxel(v) * opticalScale));
                    if (alpha > 1e-6f)
                        L += T * alpha * sunColour * exp(-sunDepthProxy(v));
                    T *= 1.f - alpha;
                }
                continue;
            }
            const uint instanceIndex = hstrSpanRecords[2u * (ray * kSpanPerRay + k) + 1u].x;
            const HSTRCloudInstance instance = hstrCloudInstances[instanceIndex];
            const HSTRCloudBrick brick = hstrCloudBricks[entry];
            // Source position along the span: cloudInstanceAt's transform, linear in t inside one sea tile.
            const float tileVoxels = float(params.cloudTileVoxels);
            const float3 v0 = (position + first * direction) * cellWidth;
            const float2 worldTile = floor((v0.xz + 0.5f) / tileVoxels);
            const float3 p0 = float3(v0.x - worldTile.x * tileVoxels, v0.y, v0.z - worldTile.y * tileVoxels);
            const float3 dv = direction * cellWidth;
            const float3 x0 = float3(dot(instance.row0.xyz, p0) + instance.row0.w, dot(instance.row1.xyz, p0) + instance.row1.w,
                                     dot(instance.row2.xyz, p0) + instance.row2.w);
            const float3 dx = float3(dot(instance.row0.xyz, dv), dot(instance.row1.xyz, dv), dot(instance.row2.xyz, dv));
            const uint level = brick.info & 15u;
            const uint2 resolved = hstrCloudSunResolved[entry * kCloudSunClasses + instance.sunClass];
            const bool bakeable = resolved.x != kCloudRefNone && params.sunNearVoxels > 0.f && SHIP(cloudSunCache, 1u) != 0;
            const float near = params.sunNearVoxels * instance.sourceVoxelWorld / hstrCloudAssets[instance.asset].sourceVoxelWorld;
            for (uint j = 0; j < n; ++j)
            {
                const float t = first + float(j) * dt;
                const float3 x = x0 + (t - first) * dx;
                const int3 i = int3(floor(x + 0.5f));
                const float footprint = (entryDistance + t * worldLength) * params.cloudPixelAngle;
                const float fade = fadeScale > 0.f ? saturate((1.f - footprint * fadeScale) / 0.3f) : 1.f;
                float density = cloudBrickDensity(entry, x, i) * scale;
                if (brick.fade < 1.f)
                {
                    const float3 v = (position + t * direction) * cellWidth;
                    density = lerp(brick.parent != kCloudRefNone ? cloudBrickDensity(brick.parent, x, i) * scale : extinctionAtVoxel(v),
                                   density, brick.fade);
                }
                const float alpha = 1.f - exp(-(fade * density * opticalScale));
                if (alpha > 1e-6f)
                {
                    const float3 v = (position + t * direction) * cellWidth;
                    const uint desired = uint(max(log2(max(footprint / instance.sourceVoxelWorld, 1.f)) + params.cloudLodBias, 0.f));
                    float depth;
                    if (bakeable && level >= desired)
                    {
                        const float bakedDepth = cloudBrickSunAt(resolved.y >> 24, resolved.x, x) * instance.sourceVoxelWorld;
                        const float3 far = seaVoxel(seaVoxel(v) + near * params.sunVoxelDirection);
                        const float farDepth = hstrFineSun.SampleLevel(hstrLinearSampler, (far + 0.5f) / float3(params.hstrFineDims), 0.f);
                        depth = bakedDepth > kCloudSunDepthOpaque ? bakedDepth : bakedDepth + farDepth;
                        ++baked;
                    }
                    else
                    {
                        // No bake the lean march would read: its own fallback, from the same few words.
                        LeanHit hit;
                        hit.entry = entry;
                        hit.instance = instanceIndex;
                        hit.levels = level | (desired << 8) | (instance.sunClass << 16);
                        hit.x = x;
                        hit.scale = instance.sourceVoxelWorld;
                        hit.near = instance.scale;
                        depth = leanSunDepth(v, footprint, hit);
                        ++fallback;
                    }
                    L += T * alpha * sunColour * exp(-depth);
                }
                T *= 1.f - alpha;
            }
        }
        if (!counting)
        {
            hstrSpanRays[kSpanRayWords * ray + 4u] = uint4(asuint(L), asuint(T));
            return;
        }
        // Against the march's own answer for this ray.
        const float marchT = asfloat(h2.y);
        const float3 marchL = asfloat(hstrSpanRays[kSpanRayWords * ray + 3u].xyz);
        const float dT = abs(T - marchT);
        const float dL = maxComponent(abs(L - marchL)) / max(maxComponent(marchL), 1e-3f);
        uint unused;
        InterlockedAdd(hstrSpanCounts[kSpanSpans], spans, unused);
        InterlockedAdd(hstrSpanCounts[kSpanEvaluated], samples, unused);
        InterlockedAdd(hstrSpanCounts[kSpanBaked], baked, unused);
        InterlockedAdd(hstrSpanCounts[kSpanFallback], fallback, unused);
        if (dT > 1e-4f)
            InterlockedAdd(hstrSpanCounts[kSpanMismatchT], 1u, unused);
        if (dL > 1e-3f)
            InterlockedAdd(hstrSpanCounts[kSpanMismatchL], 1u, unused);
        InterlockedMax(hstrSpanCounts[kSpanMaxDT], asuint(dT), unused);
        InterlockedMax(hstrSpanCounts[kSpanMaxDL], asuint(dL), unused);
    }

    bool beamWithCache() { return (params.hstComponents & (kComponentSunMultiple | kComponentSkyScattered)) != 0; }

    bool beamWithSingle() { return (params.hstComponents & kComponentSunSingle) != 0; }

    uint beamTileEdge(uint level) { return params.beamTileSize >> level; }

    /// Whether a box of the beam image is one the screen can see. With the reference frame off the beam image IS the screen and
    /// everything qualifies; with it on, the image reaches past the screen so that turning does not leave it, and that surplus is
    /// never built - a direction only earns a basis when it turns on screen, and then it has no history to carry anyway.
    bool beamBoxOnScreen(float2 low, float2 high)
    {
        if (params.beamRefFrame == 0 || params.beamBuildAll != 0)
            return true;
        if (params.beamOct != 0 && params.beamOctNdcPerTexel > 0.f)
        {
            // The octahedral footprint of the screen is curved, so a bounding box around it has to be padded and over-covers. The
            // direction itself does not: project the box's centre through the camera basis and widen the screen rectangle by the
            // box's own angular radius, which is what beamOctNdcPerTexel converts from texels into normalised device units.
            const float3 w = beamRayDirection(0.5f * (low + high));
            const float3 c =
                float3(dot(params.beamRefInverse0, w), dot(params.beamRefInverse1, w), dot(params.beamRefInverse2, w));
            if (c.z <= 1e-6f)
                return false; // Behind the camera, and a box of a few texels cannot reach around to the front of it.
            const float2 ndc = c.xy / c.z;
            const float radius = 0.5f * max(high.x - low.x, high.y - low.y) + 1.f;
            const float margin = 1.f + radius * params.beamOctNdcPerTexel;
            return all(abs(ndc) <= margin);
        }
        return all(high >= params.beamScreenBounds.xy) && all(low <= params.beamScreenBounds.zw);
    }

    /// With the translation guard, whether anything is built at all is decided per GUARD BLOCK, never per point, tile or unit.
    ///
    /// The guard certifies a block from the points of it that were verified, and from then on nobody asks about the block again
    /// until the camera moves. If a block straddling the screen edge had only its on-screen part built, the certificate vouched
    /// for points, tiles and units that were skipped - and when a turn brought them on screen nothing listed them, because the
    /// block was "done". Measured, 4K flick from a reset: dashed seams through the clouds along the edges of everything built
    /// earlier, which survived both the turn-driven dirty list and the tile apron. So a block is built whole or not at all:
    /// its tiles and units while the block is on screen, and its basis points while the block or any block whose tiles read them
    /// (beamTransmittanceRange reaches two tiles, rounded up to whole blocks) is.
    bool beamGuardBlocks() { return params.beamGuard != 0 && params.beamRefFrame != 0 && params.beamDirtyBlockEdge != 0; }

    bool beamGuardBlockOnScreen(uint2 cell, uint dilate)
    {
        const float edge = float(params.beamDirtyBlockEdge);
        return beamBoxOnScreen((float2(cell) - float(dilate)) * edge, (float2(cell) + 1.f + float(dilate)) * edge);
    }

    /// Guard blocks around a block whose tiles read its basis points.
    uint beamGuardReachBlocks()
    {
        return (2u * params.beamTileSize + params.beamDirtyBlockEdge - 1u) / max(params.beamDirtyBlockEdge, 1u);
    }

    bool beamUnitOnScreen(uint2 unit)
    {
        if (beamGuardBlocks())
            return beamGuardBlockOnScreen(unit / params.beamDirtyBlockEdge, 0);
        return beamBoxOnScreen(float2(unit), float2(unit) + 1.f);
    }

    bool beamTileOnScreen(uint level, uint2 tile)
    {
        const float s = float(beamTileEdge(level));
        if (level == 0 && beamGuardBlocks())
            return beamGuardBlockOnScreen(tile * beamTileEdge(0) / params.beamDirtyBlockEdge, 0);
        return beamBoxOnScreen(float2(tile) * s, (float2(tile) + 1.f) * s);
    }

    /// A basis point is kept while any tile that could read it is on screen. beamTransmittanceRange reads the corners and centres
    /// of a tile's eight neighbours at level 0, so the reach is two tiles, not one - at one tile the tests on the outermost built
    /// tiles read lattice this build never wrote. Widened for that reason alone: it did NOT move the 720p screen-edge error the
    /// reference frame shows (max log 7.88e-2 on about 0.002% of pixels, bit-identical before and after), which is still open.
    bool beamPointOnScreen(uint2 position)
    {
        if (beamGuardBlocks())
            return beamGuardBlockOnScreen(position / params.beamDirtyBlockEdge, beamGuardReachBlocks());
        const float s = 2.f * float(params.beamTileSize);
        return beamBoxOnScreen(float2(position) - s, float2(position) + s);
    }

    /// Queries covering a grid of lattice points in blocks of 8 x 4.
    uint beamBlockQueries(uint2 grid) { return ((grid.x + 7u) / 8u) * ((grid.y + 3u) / 4u) * 32u; }

    /// Start of the list of tiles refined at level - 1, whose children are processed at level (>= 1).
    uint beamListOffset(uint level)
    {
        const uint tiles = params.beamTileDims.x * params.beamTileDims.y;
        return tiles * (((1u << (2u * (level - 1u))) - 1u) / 3u);
    }

    uint packBeamTile(uint2 tile) { return tile.x | (tile.y << 16); }

    uint2 unpackBeamTile(uint value) { return uint2(value & 0xFFFFu, value >> 16); }

    /// Lattice texel of a pixel-corner position. With beamPageIndirect the position resolves through a page table first, which
    /// is what a sparse direction atlas would have to do on every one of these reads; the table holds the identity, so the texel
    /// it returns is the one the direct path would have used and only the cost differs.
    uint3 beamLatticeTexel(uint2 position, uint slice)
    {
        uint2 texel = position / params.beamLatticeStep;
        if (params.beamPageIndirect == 1)
        {
            const uint shift = params.beamPageShift;
            const uint2 page = min(texel >> shift, params.beamPageDims - 1u);
            const uint entry = hstrBeamPageTable[page.y * params.beamPageDims.x + page.x];
            const uint2 base = uint2(entry % params.beamPageDims.x, entry / params.beamPageDims.x) << shift;
            texel = base | (texel & ((1u << shift) - 1u));
        }
        return uint3(texel, slice);
    }

    /// One page resolution, without using its result to address anything. beamPageIndirect 2 calls this once per resolve pixel
    /// and then reads the lattice directly, which is what an apron-paged atlas does: a page stores its own one-texel border, so a
    /// tile's whole 2 x 2 neighbourhood is guaranteed in-page and thirteen reads share one lookup. Mode 1 looks up on every read
    /// instead, so the pair brackets the real design from above and below. The sentinel compare is never true with the identity
    /// table; it is there so the dependent read cannot be optimised away.
    bool beamPageMissing(uint2 position)
    {
        const uint shift = params.beamPageShift;
        const uint2 page = min((position / params.beamLatticeStep) >> shift, params.beamPageDims - 1u);
        return hstrBeamPageTable[page.y * params.beamPageDims.x + page.x] == 0xFFFFFFFFu;
    }

    /// Lattice query at a pixel-corner position (a multiple of beamLatticeStep).
    float4 beamLatticeAt(uint2 position, uint slice) { return hstrBeamLattice[beamLatticeTexel(position, slice)]; }

    /// Pixel-corner position of query q. Per level: the current pass level, over the lists refined this frame. Temporal: every
    /// level in one dispatch, level 0 first, then each finer level over the tiles refined last frame.
    bool beamQueryPosition(uint q, out uint2 position)
    {
        if (params.beamTemporal == 0)
            return beamQueryAt(params.beamPassLevel, q, false, position);
        const uint levelZero = beamBlockQueries(params.beamTileDims + 1u) + beamBlockQueries(params.beamTileDims);
        if (q < levelZero)
            return beamQueryAt(0, q, true, position);
        q -= levelZero;
        for (uint level = 1; level < params.beamLevels; ++level)
        {
            const uint queries = 8u * hstrBeamPreviousCounts[level];
            if (q < queries)
                return beamQueryAt(level, q, true, position);
            q -= queries;
        }
        position = 0;
        return false;
    }

    /// Query q of one level. Level 0 queries every coarsest corner, then every coarsest centre; a finer level queries eight
    /// points per tile refined at the level above (this frame's list, or last frame's when previous is set): its edge
    /// midpoints, then its children's centres (its corners and centre are already on the lattice).
    bool beamQueryAt(uint level, uint q, bool previous, out uint2 position)
    {
        position = 0;
        if (level == 0)
        {
            // Corners, then centres, each in blocks of 8 x 4 lattice points per 32 consecutive queries so that a wave's rays
            // are neighbours.
            const uint s = params.beamTileSize;
            const uint2 corners = params.beamTileDims + 1u;
            const uint cornerQueries = beamBlockQueries(corners);
            const bool centre = q >= cornerQueries;
            const uint2 grid = centre ? params.beamTileDims : corners;
            const uint local = centre ? q - cornerQueries : q;
            const uint blocksX = (grid.x + 7u) / 8u;
            const uint block = local / 32u;
            const uint lane = local % 32u;
            const uint2 point = uint2(block % blocksX, block / blocksX) * uint2(8u, 4u) + uint2(lane % 8u, lane / 8u);
            if (any(point >= grid))
                return false;
            position = point * s + (centre ? s / 2 : 0u);
            if (!beamPointOnScreen(position))
                return false;
            // Only the centre belongs to the root tile alone. Its corners are the outer corners of its children too - a refined
            // tile's own eight queries are the edge midpoints and the child centres, nothing else - so a skipped corner would
            // leave those children interpolating a stale lattice. Corners are always marched; a tile predicted to fail its own
            // level (beamRootDifficult) saves its centre ray and its test.
            if (centre && params.beamCentreless != 0)
                return false;
            if (params.beamAdaptiveRoot != 0 && params.beamLevels >= 2 && centre)
                return !beamRootDifficult(point);
            return true;
        }
        const uint entry = q / 8;
        if (entry >= (previous ? hstrBeamPreviousCounts[level] : hstrBeamCounts[level]))
            return false;
        const uint s = beamTileEdge(level - 1);
        const uint h = s / 2;
        const uint k = s / 4;
        uint2 offset;
        switch (q % 8)
        {
        case 0:
            offset = uint2(h, 0);
            break;
        case 1:
            offset = uint2(0, h);
            break;
        case 2:
            offset = uint2(s, h);
            break;
        case 3:
            offset = uint2(h, s);
            break;
        case 4:
            offset = uint2(k, k);
            break;
        case 5:
            offset = uint2(h + k, k);
            break;
        case 6:
            offset = uint2(k, h + k);
            break;
        default:
            offset = uint2(h + k, h + k);
            break;
        }
        const uint index = beamListOffset(level) + entry;
        position = unpackBeamTile(previous ? hstrBeamPreviousLists[index] : hstrBeamLists[index]) * s + offset;
        return true;
    }

    /// Writes a lattice query: slice 0 the cache and optical depth, 1 the single scatter and opacity-weighted distance, 2 the march
    /// steps it cost (beamQueue's prediction for the next build).
    void writeBeamQuery(uint2 position, float4 depthCache, float4 single, float steps = 0.f)
    {
        const uint2 texel = position / params.beamLatticeStep;
        hstrBeamLatticeOutput[uint3(texel, 0)] = depthCache;
        hstrBeamLatticeOutput[uint3(texel, 1)] = single;
        // .y stamps the build that wrote this point. In the reference frame the image reaches past the screen and only the part the
        // screen covers is ever built, so a point that has just turned ON screen holds whatever it held before it turned off - the
        // stamp is what tells a carry that the previous build actually marched this ray.
        // .y marks the point as holding a marched value. With the ping-ponged pair that had to be a build stamp, because the
        // image it was written into became the history exactly once; the persistent lattice is cleared when the frame re-anchors
        // and a mark then stands until then.
        hstrBeamLatticeOutput[uint3(texel, 2)] = float4(steps, 1.f, 0.f, 0.f);
    }

    uint beamSparseListOffset(uint level)
    {
        const uint roots = params.beamTileDims.x * params.beamTileDims.y;
        return roots * (((1u << (2u * level)) - 1u) / 3u);
    }

    /// A conservative no-medium proof for the unbounded sea slab. The y component
    /// of an unnormalised camera ray is affine over a screen tile, so four corners
    /// all pointing away from a slab on the other side proves every enclosed ray
    /// misses. Finite volumes currently return false and remain quality-neutral.
    bool beamRootProvablyEmpty(uint2 tile)
    {
        if (params.cloudDomain == 0 || params.seaMode == 0)
            return false;
        const float bottom = max(params.seaOrigin.y, params.seaContentY.x);
        const float top = min(params.seaOrigin.y + float(params.hstrExtinctionDims.y) * params.seaVoxelSize.y, params.seaContentY.y);
        if (bottom >= top)
            return true;
        const uint s = params.beamTileSize;
        const float2 p[4] = {
            float2(tile * s), float2(tile * s + uint2(s, 0)),
            float2(tile * s + uint2(0, s)), float2(tile * s + s),
        };
        float lo = 1e30f;
        float hi = -1e30f;
        for (uint i = 0; i < 4; ++i)
        {
            const float y = beamRayDirection(p[i]).y;
            lo = min(lo, y);
            hi = max(hi, y);
        }
        const float cameraY = gScene.camera.data.posW.y;
        return (cameraY < bottom && hi <= 0.f) || (cameraY > top && lo >= 0.f);
    }

    /// Cheap domain-metadata scheduling hint. It never accepts a tile: it only
    /// chooses the first level at which the exact radiometric verifier runs.
    uint beamMetadataStartLevel(uint2 root)
    {
        const uint last = params.beamLevels - 1u;
        if (last == 0)
            return 0;
        const uint s = params.beamTileSize;
        const float2 p[5] = {
            float2(root * s), float2(root * s + uint2(s, 0)), float2(root * s + uint2(0, s)),
            float2(root * s + s), float2(root * s + s / 2u),
        };
        uint covered = 0;
        float centreLo = 1e30f;
        float centreHi = -1e30f;
        float lengthLo = 1e30f;
        float lengthHi = 0.f;
        float majorantLo = 1e30f;
        float majorantHi = 0.f;
        const GridVolume volume = gScene.getGridVolume(0);
        for (uint i = 0; i < 5; ++i)
        {
            const float3 w = beamRayDirection(p[i]);
            float2 interval;
            if (!mediumInterval(gScene.camera.data.posW, w, volume, interval))
                continue;
            interval.x = max(interval.x, 0.f);
            if (interval.y <= interval.x)
                continue;
            ++covered;
            const float centre = 0.5f * (interval.x + interval.y);
            const float length = interval.y - interval.x;
            centreLo = min(centreLo, centre);
            centreHi = max(centreHi, centre);
            lengthLo = min(lengthLo, length);
            lengthHi = max(lengthHi, length);
            const float3 v = worldToVoxel(gScene.camera.data.posW + centre * w, volume);
            const float m = majorantAt(v);
            majorantLo = min(majorantLo, m);
            majorantHi = max(majorantHi, m);
        }
        // Mixed coverage is a silhouette candidate. Large depth or majorant spread
        // likewise begins finer; exact surplus/silhouette tests still decide acceptance.
        if (covered != 0 && covered != 5)
            return last;
        if (covered == 0)
            return 0;
        // Compare the interval geometry *between* the five rays. Using the union's
        // exit-entry distance here measured slab thickness instead and forced every
        // 4K near-view root to level 2 (8160/8160, 286155 unique queries).
        const float centreSpread = (centreHi - centreLo) / max(abs(0.5f * (centreHi + centreLo)), 1e-3f);
        const float lengthSpread = (lengthHi - lengthLo) / max(lengthHi, 1e-3f);
        const float depthSpread = max(centreSpread, lengthSpread);
        const float densitySpread = (majorantHi - majorantLo) / max(majorantHi, 1e-6f);
        if (depthSpread > 0.35f || densitySpread > 0.75f)
            return min(last, 2u);
        if (depthSpread > 0.12f || densitySpread > 0.35f)
            return min(last, 1u);
        return 0;
    }

    /// Reserves one contiguous run of a level's candidate list.
    ///
    /// The list is what the emit dispatch reads, five basis positions per entry, so neighbouring entries are what a wave marches
    /// together. Taking one slot per tile puts them in the order threads happened to finish, which scatters a wave's lanes across
    /// the frame: they then share no bricks, no sun bakes and no cache cells, and the wave waits for whichever of its unrelated
    /// rays is longest. Reserving a tile's whole family in one atomic keeps the family contiguous, so a wave stays inside one
    /// screen block. This is the same coherence the beamQueue experiment lost by construction (see beamQueue).
    uint reserveBeamSparseCandidates(uint level, uint count)
    {
        uint slot;
        InterlockedAdd(hstrBeamCounts[level], count, slot);
        uint unused;
        InterlockedAdd(hstrBeamCounts[kBeamSparseCandidates], count, unused);
        return slot;
    }

    void writeBeamSparseCandidate(uint level, uint slot, uint2 tile)
    {
        hstrBeamSparseCandidates[beamSparseListOffset(level) + slot] = packBeamTile(tile);
    }

    void appendBeamSparseCandidate(uint level, uint2 tile)
    {
        writeBeamSparseCandidate(level, reserveBeamSparseCandidates(level, 1u), tile);
    }

    /// Metadata phase: produce candidate leaves before any exact query is emitted.
    void classifyBeamRoot(uint thread)
    {
        const uint roots = params.beamTileDims.x * params.beamTileDims.y;
        if (thread >= roots)
            return;
        const uint2 root = uint2(thread % params.beamTileDims.x, thread / params.beamTileDims.x);
        if (!beamTileOnScreen(0, root))
            return;
        uint unused;
        InterlockedAdd(hstrBeamCounts[kBeamSparseRoots], 1u, unused);
        if (beamRootProvablyEmpty(root))
        {
            InterlockedAdd(hstrBeamCounts[kBeamSparseEmpty], 1u, unused);
            const uint finest = beamTileEdge(params.beamLevels - 1u);
            const uint cells = params.beamTileSize / finest;
            const uint2 first = root * cells;
            for (uint y = 0; y < cells; ++y)
                for (uint x = 0; x < cells; ++x)
                {
                    const uint2 cell = first + uint2(x, y);
                    if (all(cell < params.beamLevelDims))
                    {
                        hstrBeamLevelOutput[cell] = params.beamLevels + 1u;
                        hstrBeamTileMap[cell] = 0xfffffffeu;
                    }
                }
            return;
        }
        const uint level = max(min(uint(HSTR_BEAM_SPARSE_MIN_LEVEL), params.beamLevels - 1u), beamMetadataStartLevel(root));
        InterlockedAdd(hstrBeamCounts[kBeamSparseStarts + level], 1u, unused);
        const uint n = 1u << level;
        const uint slot = reserveBeamSparseCandidates(level, n * n);
        for (uint y = 0; y < n; ++y)
            for (uint x = 0; x < n; ++x)
                writeBeamSparseCandidate(level, slot + y * n + x, root * n + uint2(x, y));
    }

    /// Accepted cut cell: two-point Gauss quadrature of its trilinear extinction page, with the split frame's two source terms
    /// taken once at each node. This is integratePage's quadrature with the beam's terms: the cache comes from the world cache
    /// (the sea's lighting field) rather than the HST camera lighting texture, and single scattering is the proxy sun depth, which
    /// is the field the page's own extinction was derived from.
    void integrateDomainPage(
        HSTRCutNode node,
        CutRay ray,
        float3 cutPosition,
        float2 segment,
        bool withCache,
        bool withSingle,
        inout float3 cache,
        inout float3 single,
        inout float transmittance,
        inout float opacityDistance
    )
    {
        static const float kGaussOffset = 0.2886751345948129f;
        const float midpoint = 0.5f * (segment.x + segment.y);
        const float span = segment.y - segment.x;
        const float2 t = float2(midpoint - kGaussOffset * span, midpoint + kGaussOffset * span);
        const float3 local0 =
            (cutPosition + t.x * ray.direction - float3(node.minLeft.xyz)) / float3(node.maxRight.xyz - node.minLeft.xyz);
        const float3 local1 =
            (cutPosition + t.y * ray.direction - float3(node.minLeft.xyz)) / float3(node.maxRight.xyz - node.minLeft.xyz);
        // The sea fades out over the last 30% of the view distance so tiles entering the window never pop in; a paged cell has to
        // carry the same fade the marched cell beside it does (cameraExtinctionAtVoxel).
        const float2 fade = params.seaViewDistance > 0.f
                                ? saturate((1.f - (ray.entryDistance + t * ray.worldLength) / params.seaViewDistance) / 0.3f)
                                : float2(1.f);
        const float sigma0 = fade.x * domainPageAt(node.transmittanceCorners0, node.transmittanceCorners1, saturate(local0));
        const float sigma1 = fade.y * domainPageAt(node.transmittanceCorners0, node.transmittanceCorners1, saturate(local1));
        const float depth = 0.5f * ray.worldLength * span * (sigma0 + sigma1);
        if (depth <= 1e-8f)
            return;
        const float alpha = 1.f - exp(-depth);
        if (withCache || withSingle)
        {
            // Where the node's opacity actually sits along the segment: the two nodes weighted by their own extinction under the
            // transmittance they see, which is what integratePage's source weights are.
            const float weight0 = sigma0 * exp(-depth * (0.5f - kGaussOffset));
            const float weight1 = sigma1 * exp(-depth * (0.5f + kGaussOffset));
            const float split = weight1 / max(1e-8f, weight0 + weight1);
            const float tSource = lerp(t.x, t.y, split);
            const float3 v = (ray.position + tSource * ray.direction) * float(params.hstrCellWidth);
            if (withCache)
                cache += transmittance * alpha * worldCacheRadiance(v, ray.viewDirection);
            if (withSingle)
                single += transmittance * alpha * residualAtDepth(v, sunDepthProxy(v), ray.sunOctaves);
        }
        opacityDistance += transmittance * alpha * (ray.entryDistance + midpoint * ray.worldLength);
        transmittance *= 1.f - alpha;
    }

    /// Composites one basis ray through the cloud sea's projected transport cut. The tile's cut list is already the front-to-back
    /// antichain of non-empty domain cells covering this pixel, so the ray never sees empty sky and never re-derives an ordering:
    /// each cell is either taken from its page (cheap, constant work) or, where the page cannot carry its optical depth, marched
    /// over that cell's segment alone, starting from the transmittance the cells in front of it left.
    void traverseDomainCut(
        CutRay ray,
        uint tileID,
        inout float3 cache,
        inout float3 single,
        inout float transmittance,
        inout float opacityDistance,
        inout uint paged,
        inout uint marched,
        inout uint steps
    )
    {
        const bool withCache = beamWithCache();
        const bool withSingle = beamWithSingle();
        const uint count = hstrTileNodeCounts[tileID];
        if (count > params.hstrMaxTileNodes)
        {
            // The tile's cut overflowed its list: march the whole volume segment, as the HST camera path does.
            marchSegmentMoment(ray, float2(0.f, 1.f), withCache, withSingle, cache, single, transmittance, steps, opacityDistance);
            ++marched;
            return;
        }
        // Node boxes are window-local; the ray keeps its domain position so every density and lighting lookup still wraps.
        const float3 cutPosition = ray.position - float3(params.hstrCutOriginLeaf);
        const uint base = tileID * params.hstrMaxTileNodes;
        [loop]
        for (uint i = 0; i < count && transmittance > params.cloudMinTransmittance; ++i)
        {
            const HSTRCutNode node = hstrCutNodes[hstrTileNodes[base + i]];
            float2 segment;
            if (!intersectCutNode(node, cutPosition, ray.direction, segment))
                continue;
            const float pageError = transmittance * (1.f - exp(-node.transmittancePage.x * ray.worldLength * (segment.y - segment.x)));
            if (pageError > params.cutTransmittanceTolerance)
            {
                marchSegmentMoment(ray, segment, withCache, withSingle, cache, single, transmittance, steps, opacityDistance);
                ++marched;
            }
            else
            {
                integrateDomainPage(node, ray, cutPosition, segment, withCache, withSingle, cache, single, transmittance, opacityDistance);
                ++paged;
            }
        }
    }

    /// Evaluate a sparse basis position from the already projected, front-to-back transport cut instead of from a full-volume
    /// march. Cells the cut resolves cost a page evaluation; only the unresolved residual enters the marcher.
    BeamSample evaluateBeamCut(uint2 position, out uint paged, out uint marched, out uint steps)
    {
        BeamSample result = {};
        paged = 0u;
        marched = 0u;
        steps = 0u;
        const GridVolume volume = gScene.getGridVolume(0);
        CutRay ray;
        if (!makeCutRay(gScene.camera.data.posW, beamRayDirection(float2(position)), volume, ray))
            return result;
        // The beam applies the phase per pixel, so octave 0 carries a unit weight here as it does in marchBeam().
        ray.sunOctaves = float4(1.f, 0.f, 0.f, 0.f);
        const uint2 tile = min(position / params.hstrTileSize, params.hstrTileDims - 1u);
        const uint tileID = tile.x + params.hstrTileDims.x * tile.y;
        float transmittance = 1.f;
        float opacityDistance = 0.f;
        traverseDomainCut(ray, tileID, result.cache, result.single, transmittance, opacityDistance, paged, marched, steps);
        result.depth = -log(max(transmittance, 1e-12f));
        result.steps = float(steps);
        result.distance = transmittance < 1.f - 1e-4f ? opacityDistance / (1.f - transmittance) : 0.f;
        return result;
    }

    void emitBeamSparseQuery(uint thread)
    {
        const uint level = params.beamPassLevel;
        const uint entry = thread / 5u;
        if (entry >= hstrBeamCounts[level])
            return;
        const uint2 tile = unpackBeamTile(hstrBeamSparseCandidates[beamSparseListOffset(level) + entry]);
        const uint s = beamTileEdge(level);
        const uint2 offsets[5] = { uint2(0, 0), uint2(s, 0), uint2(0, s), uint2(s, s), uint2(s / 2u, s / 2u) };
        const uint2 position = tile * s + offsets[thread % 5u];
        const uint2 texel = position / params.beamLatticeStep;
        // The lattice coordinate is the durable virtual query identity. This avoids
        // shuffling coherent screen-space work through a globally atomic compact ID
        // queue (4K near: 78,783 queries but 12.2 ms in the compacted pipeline).
        const uint queryId = texel.x + params.beamLatticeDims.x * texel.y;
        uint prior;
        InterlockedCompareExchange(hstrBeamQueryMap[texel], 0xffffffffu, queryId, prior);
        if (prior != 0xffffffffu)
            return;
        uint paged = 0u;
        uint marched = 0u;
        uint steps = 0u;
#if HSTR_BEAM_SPARSE_CUT
        const BeamSample q = evaluateBeamCut(position, paged, marched, steps);
#else
        const BeamSample q = marchBeam(beamRayDirection(float2(position)), float2(0.f, 1.f));
        steps = uint(q.steps);
#endif
        const float4 depthCache = float4(q.cache, q.depth);
        const float4 single = float4(q.single, q.distance);
        const BeamResult result = makeBeamResult(depthCache, single);
        hstrBeamResults[queryId] = result;
        uint unused;
        InterlockedAdd(hstrBeamCounts[kBeamSparseUnique], 1u, unused);
        InterlockedAdd(hstrBeamCounts[kBeamSparseLevelUnique + level], 1u, unused);
#if HSTR_BEAM_SPARSE_CUT
        // What the cut left unresolved: the queries that still had to march at all, the steps they spent there, and the split of
        // the cells a ray actually hit between the page and the marcher. Accumulated per ray, so this is two atomics a query.
        if (steps > 0u)
            InterlockedAdd(hstrBeamCounts[kBeamSparseCutResidualQueries], 1u, unused);
        InterlockedAdd(hstrBeamCounts[kBeamSparseCutResidualSteps], steps, unused);
        InterlockedAdd(hstrBeamCounts[kBeamSparseCutPaged], paged, unused);
        InterlockedAdd(hstrBeamCounts[kBeamSparseCutMarched], marched, unused);
#endif
    }

    uint beamPackHalf2(float2 v) { return uint(f32tof16(v.x)) | (uint(f32tof16(v.y)) << 16); }

    float2 beamUnpackHalf2(uint v) { return float2(f16tof32(v & 0xffffu), f16tof32(v >> 16)); }

    BeamResult makeBeamResult(float4 depthCache, float4 single)
    {
        BeamResult result;
        result.depthCache01 = beamPackHalf2(depthCache.xy);
        result.depthCache23 = beamPackHalf2(depthCache.zw);
        result.single01 = beamPackHalf2(single.xy);
        result.single23 = beamPackHalf2(single.zw);
        return result;
    }

    float4 beamResultDepthCache(BeamResult result)
    {
        return float4(beamUnpackHalf2(result.depthCache01), beamUnpackHalf2(result.depthCache23));
    }

    float4 beamResultSingle(BeamResult result)
    {
        return float4(beamUnpackHalf2(result.single01), beamUnpackHalf2(result.single23));
    }

    uint beamSparseQueryId(uint2 position) { return hstrBeamQueryMap[position / params.beamLatticeStep]; }

    /// Sparse verifier/reconstruction source. Unclaimed coordinates occur only in
    /// conservatively empty neighbouring roots and represent transparent sky.
    float4 beamSparseAt(uint2 position, uint slice)
    {
        const uint id = beamSparseQueryId(position);
        if (id == 0xffffffffu || id == 0xfffffffeu)
            return 0.f;
        const BeamResult result = hstrBeamResults[id];
        return slice == 0 ? beamResultDepthCache(result) : beamResultSingle(result);
    }

    float4 beamVerifierAt(uint2 position, uint slice, bool sparse)
    {
        return sparse ? beamSparseAt(position, slice) : beamLatticeAt(position, slice);
    }

    void verifyBeamSparseTile(uint thread)
    {
        const uint level = params.beamPassLevel;
        if (thread >= hstrBeamCounts[level])
            return;
        const uint2 tile = unpackBeamTile(hstrBeamSparseCandidates[beamSparseListOffset(level) + thread]);
        if (any(tile * beamTileEdge(level) >= params.beamFrameDim))
            return;
        if (!beamTileRefines(level, tile, true))
        {
            const uint s = beamTileEdge(level);
            const uint2 origin = tile * s;
            BeamTile finalTile;
            finalTile.origin = origin;
            finalTile.level = level;
            finalTile.q00 = beamSparseQueryId(origin);
            finalTile.q10 = beamSparseQueryId(origin + uint2(s, 0));
            finalTile.q01 = beamSparseQueryId(origin + uint2(0, s));
            finalTile.q11 = beamSparseQueryId(origin + s);
            finalTile.qc = beamSparseQueryId(origin + s / 2u);
            uint slot;
            InterlockedAdd(hstrBeamCounts[kBeamSparseFinal], 1u, slot);
            uint unused;
            InterlockedAdd(hstrBeamCounts[kBeamSparseAccepted + level], 1u, unused);
            hstrBeamFinalTiles[slot] = finalTile;
            writeBeamLevel(level, tile, level);
            const uint finest = beamTileEdge(params.beamLevels - 1u);
            const uint cells = s / finest;
            const uint2 first = tile * cells;
            for (uint y = 0; y < cells; ++y)
                for (uint x = 0; x < cells; ++x)
                {
                    const uint2 cell = first + uint2(x, y);
                    if (all(cell < params.beamLevelDims))
                        hstrBeamTileMap[cell] = slot;
                }
            return;
        }
        if (level + 1u < params.beamLevels)
        {
            const uint slot = reserveBeamSparseCandidates(level + 1u, 4u);
            for (uint child = 0; child < 4; ++child)
                writeBeamSparseCandidate(level + 1u, slot + child, 2u * tile + uint2(child & 1u, child >> 1));
        }
        else
        {
            uint slot;
            InterlockedAdd(hstrBeamCounts[params.beamLevels], 1u, slot);
            hstrBeamLists[beamListOffset(params.beamLevels) + slot] = packBeamTile(tile);
            writeBeamLevel(level, tile, params.beamLevels);
            hstrBeamTileMap[tile] = 0xffffffffu;
        }
    }

    void writeBeamSparseArgs()
    {
        const uint count = hstrBeamCounts[params.beamPassLevel];
        hstrBeamSparseArgs[0] = (count * 5u + 63u) / 64u;
        hstrBeamSparseArgs[1] = 1u;
        hstrBeamSparseArgs[2] = 1u;
        hstrBeamSparseArgs[3] = (count + 63u) / 64u;
        hstrBeamSparseArgs[4] = 1u;
        hstrBeamSparseArgs[5] = 1u;
        hstrBeamSparseArgs[6] = 0u;
        hstrBeamSparseArgs[7] = 1u;
        hstrBeamSparseArgs[8] = 1u;
    }

    float3 beamSparseReconstruct(uint tileIndex, float2 beamPosition, float3 w)
    {
        const BeamTile tile = hstrBeamFinalTiles[tileIndex];
        const uint s = beamTileEdge(tile.level);
        const BeamResult r00 = hstrBeamResults[tile.q00];
        const BeamResult r10 = hstrBeamResults[tile.q10];
        const BeamResult r01 = hstrBeamResults[tile.q01];
        const BeamResult r11 = hstrBeamResults[tile.q11];
        const BeamResult rc = hstrBeamResults[tile.qc];
        const float2 f = saturate((beamPosition - float2(tile.origin)) / float(s));
        const float4 a00 = beamResultDepthCache(r00);
        const float4 a10 = beamResultDepthCache(r10);
        const float4 a01 = beamResultDepthCache(r01);
        const float4 a11 = beamResultDepthCache(r11);
        const float4 ac = beamResultDepthCache(rc);
        const float4 b00 = beamResultSingle(r00);
        const float4 b10 = beamResultSingle(r10);
        const float4 b01 = beamResultSingle(r01);
        const float4 b11 = beamResultSingle(r11);
        const float4 bc = beamResultSingle(rc);
        float4 a = bilinear(a00, a10, a01, a11, f);
        float4 b = bilinear(b00, b10, b01, b11, f);
        const float bubble = 16.f * f.x * (1.f - f.x) * f.y * (1.f - f.y);
        a += (ac - 0.25f * (a00 + a10 + a01 + a11)) * bubble;
        b += (bc - 0.25f * (b00 + b10 + b01 + b11)) * bubble;
        return beamRadiance(max(a.rgb, 0.f), max(b.rgb, 0.f), max(a.a, 0.f), w, gScene.getGridVolume(0).data.albedo);
    }

    void resolveBeamSparsePixel(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        const float3 w = cameraRayDirection(float2(pixel) + 0.5f);
        float2 beamPosition;
        if (!beamPixelPosition(float2(pixel) + 0.5f, beamPosition))
            return;
        const uint finest = beamTileEdge(params.beamLevels - 1u);
        const uint tile = hstrBeamTileMap[uint2(beamPosition) / finest];
        if (tile == 0xffffffffu)
            return;
        if (tile == 0xfffffffeu)
        {
            color[pixel] = float4(beamRadiance(0.f, 0.f, 0.f, w, gScene.getGridVolume(0).data.albedo), 1.f);
            return;
        }
        color[pixel] = float4(beamSparseReconstruct(tile, beamPosition, w), 1.f);
    }

    /// Queues a root query for the march pass of its predicted cost (beamQueue).
    void beamEnqueue(uint2 position)
    {
        const float steps = params.beamHistoryValid != 0 ? beamLatticePrevAt(position, 2).x : 0.f;
        const uint bucket = min(uint(steps / max(params.beamQueueSteps, 1e-3f)), kBeamQueueBuckets - 1u);
        uint slot;
        InterlockedAdd(hstrBeamQueueCounts[bucket], 1u, slot);
        if (slot < params.beamQueueCapacity)
            hstrBeamQueue[bucket * params.beamQueueCapacity + slot] = packBeamTile(position);
    }

    /// One queued root query of list beamQueueBucket.
    void marchBeamQueued(uint thread)
    {
        const uint bucket = params.beamQueueBucket;
        if (thread >= min(hstrBeamQueueCounts[bucket], params.beamQueueCapacity))
            return;
        const uint2 position = unpackBeamTile(hstrBeamQueue[bucket * params.beamQueueCapacity + thread]);
        const BeamSample q = marchBeam(beamRayDirection(float2(position)), float2(0.f, 1.f));
        writeBeamQuery(position, float4(q.cache, q.depth), float4(q.single, q.distance), q.steps);
    }

    /// Queues a finest tile of the per-pixel march list by the steps its corners cost (beamQueue, one level).
    void queueBeamTile(uint thread)
    {
        if (thread >= hstrBeamCounts[params.beamLevels])
            return;
        const uint2 tile = unpackBeamTile(hstrBeamLists[beamListOffset(params.beamLevels) + thread]);
        const uint s = beamTileEdge(params.beamLevels - 1);
        const uint2 origin = tile * s;
        const float steps = max(max(beamLatticeAt(origin, 2).x, beamLatticeAt(origin + uint2(s, 0), 2).x),
                                max(beamLatticeAt(origin + uint2(0, s), 2).x, beamLatticeAt(origin + uint2(s, s), 2).x));
        const uint bucket = min(uint(steps / max(params.beamQueueSteps, 1e-3f)), kBeamQueueBuckets - 1u);
        uint slot;
        InterlockedAdd(hstrBeamQueueCounts[bucket], 1u, slot);
        if (slot < params.beamQueueCapacity)
            hstrBeamQueue[bucket * params.beamQueueCapacity + slot] = packBeamTile(tile);
    }

    /// One pixel of a queued finest tile of list beamQueueBucket.
    void marchBeamQueuedPixel(uint thread)
    {
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint entry = thread / (finest * finest);
        const uint bucket = params.beamQueueBucket;
        if (entry >= min(hstrBeamQueueCounts[bucket], params.beamQueueCapacity))
            return;
        const uint local = thread % (finest * finest);
        const uint2 pixel = unpackBeamTile(hstrBeamQueue[bucket * params.beamQueueCapacity + entry]) * finest + uint2(local % finest, local / finest);
        if (any(pixel >= params.frameDim))
            return;
        marchBeamAt(pixel);
    }

    /// Dispatch arguments of the queued march passes, one triple per list.
    void writeBeamQueueArgs()
    {
        for (uint bucket = 0; bucket < kBeamQueueBuckets; ++bucket)
        {
            hstrBeamQueueArgs[3 * bucket] = (min(hstrBeamQueueCounts[bucket], params.beamQueueCapacity) * params.beamQueueThreads + 63u) / 64u;
            hstrBeamQueueArgs[3 * bucket + 1] = 1;
            hstrBeamQueueArgs[3 * bucket + 2] = 1;
        }
    }

    float4 bilinear(float4 q00, float4 q10, float4 q01, float4 q11, float2 f) { return lerp(lerp(q00, q10, f.x), lerp(q01, q11, f.x), f.y); }

    /// Split-frame radiance from its terms (before albedo) for view direction w.
    float3 beamRadiance(float3 cache, float3 single, float depth, float3 w, float3 albedo)
    {
        const float phase = phaseHG(dot(normalize(params.sunDirection), w));
        const float3 background = (params.hstComponents & kComponentBackground) != 0 ? exp(-depth) * skyRadianceAt(w) : 0.f;
        return albedo * (phase * single + cache) + background;
    }

    /// Transmittance range of the lattice queries around a tile. A silhouette can cross a tile between its own five queries, but
    /// then reaches one of these: at the coarsest level the corners and centres of the tile and its eight neighbours (all queried),
    /// at a finer level the thirteen queries of its parent (corners, edge midpoints, centre and the children's centres).
    float beamTransmittanceRange(uint level, uint2 tile, bool sparse)
    {
        float lo = 1.f;
        float hi = 0.f;
        if (level == 0)
        {
            const uint s = params.beamTileSize;
            const int2 first = max(int2(tile) - 1, 0);
            const int2 corners = int2(params.beamTileDims);
            for (int y = first.y; y <= min(int(tile.y) + 2, corners.y); ++y)
                for (int x = first.x; x <= min(int(tile.x) + 2, corners.x); ++x)
                {
                    const float t = exp(-beamVerifierAt(uint2(x, y) * s, 0, sparse).a);
                    lo = min(lo, t);
                    hi = max(hi, t);
                }
            for (int y = first.y; y <= min(int(tile.y) + 1, corners.y - 1) && params.beamCentreless == 0; ++y)
                for (int x = first.x; x <= min(int(tile.x) + 1, corners.x - 1); ++x)
                {
                    const float t = exp(-beamVerifierAt(uint2(x, y) * s + s / 2, 0, sparse).a);
                    lo = min(lo, t);
                    hi = max(hi, t);
                }
            return hi - lo;
        }
        const uint s = beamTileEdge(level - 1);
        const uint2 origin = (tile / 2u) * s;
        const uint h = s / 2;
        const uint k = s / 4;
        const uint2 offsets[13] = {
            uint2(0, 0), uint2(s, 0), uint2(0, s), uint2(s, s), uint2(h, 0), uint2(0, h), uint2(s, h),
            uint2(h, s), uint2(h, h), uint2(k, k), uint2(h + k, k), uint2(k, h + k), uint2(h + k, h + k),
        };
        for (uint i = 0; i < 13; ++i)
        {
            const float t = exp(-beamVerifierAt(origin + offsets[i], 0, sparse).a);
            lo = min(lo, t);
            hi = max(hi, t);
        }
        return hi - lo;
    }

    /// Guide features of the pixels around pixel-corner position q (the 2 x 2 pixels sharing it): transmittance, then log distance
    /// and the sun attenuations exp(-tau) and exp(-b tau) where the optical depth reaches one, averaged over the pixels that reach it
    /// (valid only if all of them do).
    struct BeamGuideFeatures
    {
        float transmittance;
        float3 front; ///< log distance, exp(-tau_sun), exp(-b tau_sun).
        bool valid;
    };

    BeamGuideFeatures beamGuideFeatures(float4 guide)
    {
        BeamGuideFeatures f;
        f.transmittance = exp(-guide.x);
        f.valid = guide.y > 0.f;
        const float b = max(params.worldCacheModulation, 0.f);
        f.front = f.valid ? float3(log(guide.y), exp(-guide.z), exp(-b * guide.z)) : 0.f;
        return f;
    }

    BeamGuideFeatures beamGuideCorner(uint2 q)
    {
        BeamGuideFeatures sum = {};
        sum.valid = true;
        for (uint k = 0; k < 4; ++k)
        {
            const uint2 pixel = min(uint2(max(int2(q) - 1 + int2(k & 1u, k >> 1), 0)), params.frameDim - 1u);
            const BeamGuideFeatures f = beamGuideFeatures(hstrBeamGuide[pixel]);
            sum.transmittance += 0.25f * f.transmittance;
            sum.front += 0.25f * f.front;
            sum.valid = sum.valid && f.valid;
        }
        return sum;
    }

    /// Guide test of one tile: the resolve's corner-and-centre basis fitted to the full-resolution guide must reproduce the guide at
    /// every pixel of the tile, so detail between the tile's queries (edges, billow fronts, sun shadows) cannot pass unseen.
    bool beamTileGuideFails(uint level, uint2 tile)
    {
        const uint s = beamTileEdge(level);
        const uint2 origin = tile * s;
        const BeamGuideFeatures c00 = beamGuideCorner(origin);
        const BeamGuideFeatures c10 = beamGuideCorner(origin + uint2(s, 0));
        const BeamGuideFeatures c01 = beamGuideCorner(origin + uint2(0, s));
        const BeamGuideFeatures c11 = beamGuideCorner(origin + uint2(s, s));
        const BeamGuideFeatures cc = beamGuideCorner(origin + s / 2);
        const float4 t = float4(c00.transmittance, c10.transmittance, c01.transmittance, c11.transmittance);
        const float tBubble = cc.transmittance - 0.25f * (t.x + t.y + t.z + t.w);
        const bool frontValid = c00.valid && c10.valid && c01.valid && c11.valid && cc.valid;
        const float3 frontBubble = cc.front - 0.25f * (c00.front + c10.front + c01.front + c11.front);
        const float3 frontTolerance = float3(params.beamGuideDepth, params.beamGuideSun, params.beamGuideSun);
        const uint2 last = min(origin + s, params.beamFrameDim);
        for (uint y = origin.y; y < last.y; ++y)
            for (uint x = origin.x; x < last.x; ++x)
            {
                const float2 f = (float2(uint2(x, y) - origin) + 0.5f) / float(s);
                const float bubble = 16.f * f.x * (1.f - f.x) * f.y * (1.f - f.y);
                const BeamGuideFeatures p = beamGuideFeatures(hstrBeamGuide[uint2(x, y)]);
                const float tPredicted = lerp(lerp(t.x, t.y, f.x), lerp(t.z, t.w, f.x), f.y) + tBubble * bubble;
                if (params.beamEdgeContrast > 0.f && abs(p.transmittance - tPredicted) > params.beamEdgeContrast)
                    return true;
                if (frontValid != p.valid)
                {
                    // A pixel reaches optical depth one where the corners do not, or the other way round: an edge of the
                    // opaque cloud, which matters only if the transmittance fit is off there too.
                    continue;
                }
                if (!p.valid)
                    continue;
                const float3 frontPredicted = lerp(lerp(c00.front, c10.front, f.x), lerp(c01.front, c11.front, f.x), f.y) + frontBubble * bubble;
                // Front deviations matter in proportion to the opacity in front of the transmitted background.
                if (any(abs(p.front - frontPredicted) * (1.f - p.transmittance) > frontTolerance))
                    return true;
            }
        return false;
    }

    /// Pixel of a tile reconstructed from its basis: bilinear corners plus the centre's bubble (none for a centreless root tile).
    float3 beamReconstruct(uint level, float2 beamPosition, float3 w, bool sparse)
    {
        const uint s = beamTileEdge(level);
        const uint2 origin = uint2(beamPosition) / s * s;
        const float4 a00 = beamVerifierAt(origin, 0, sparse);
        const float4 a10 = beamVerifierAt(origin + uint2(s, 0), 0, sparse);
        const float4 a01 = beamVerifierAt(origin + uint2(0, s), 0, sparse);
        const float4 a11 = beamVerifierAt(origin + uint2(s, s), 0, sparse);
        const float4 b00 = beamVerifierAt(origin, 1, sparse);
        const float4 b10 = beamVerifierAt(origin + uint2(s, 0), 1, sparse);
        const float4 b01 = beamVerifierAt(origin + uint2(0, s), 1, sparse);
        const float4 b11 = beamVerifierAt(origin + uint2(s, s), 1, sparse);
        const float2 f = saturate((beamPosition - float2(origin)) / float(s));
        float4 a = bilinear(a00, a10, a01, a11, f);
        float4 b = bilinear(b00, b10, b01, b11, f);
        if (!(level == 0 && params.beamCentreless != 0))
        {
            const float bubble = 16.f * f.x * (1.f - f.x) * f.y * (1.f - f.y);
            a += (beamVerifierAt(origin + s / 2, 0, sparse) - 0.25f * (a00 + a10 + a01 + a11)) * bubble;
            b += (beamVerifierAt(origin + s / 2, 1, sparse) - 0.25f * (b00 + b10 + b01 + b11)) * bubble;
        }
        // The phase and the background belong to the pixel's own ray, which the caller knows exactly; only the basis is read in
        // the beam image.
        return beamRadiance(max(a.rgb, 0.f), max(b.rgb, 0.f), max(a.a, 0.f), w, gScene.getGridVolume(0).data.albedo);
    }

    /// Oracle tile test (beamOracle): the tile's own reconstruction against the stored exact frame, in the compare metric.
    bool beamTileOracleFails(uint level, uint2 tile, bool sparse)
    {
        const uint s = beamTileEdge(level);
        const uint2 origin = tile * s;
        const uint2 last = min(origin + s, params.beamFrameDim);
        uint failing = 0;
        for (uint y = origin.y; y < last.y; ++y)
            for (uint x = origin.x; x < last.x; ++x)
            {
                const float2 point = float2(x, y) + 0.5f;
                const float3 value = beamReconstruct(level, point, beamRayDirection(point), sparse);
                const float3 reference = max(hstrExactFrame[uint2(x, y)].rgb, 0.f);
                if (dot(abs(log(1.f + value) - log(1.f + reference)), 1.f / 3.f) > params.beamOracleBar)
                    ++failing;
            }
        return failing >= params.beamOracle;
    }

    /// log(1 + radiance) of a coarsest corner, in corner units, clamped to the lattice.
    float3 beamCornerLog(int2 corner, bool sparse)
    {
        const uint2 position = uint2(clamp(corner, 0, int2(params.beamTileDims))) * params.beamTileSize;
        const float4 a = beamVerifierAt(position, 0, sparse);
        const float4 b = beamVerifierAt(position, 1, sparse);
        const float3 w = beamRayDirection(float2(position));
        return log(1.f + beamRadiance(a.rgb, b.rgb, a.a, w, gScene.getGridVolume(0).data.albedo));
    }

    /// Centreless root test: the largest second difference of log radiance along either axis at the tile's corners, which is what a
    /// bilinear basis cannot reproduce.
    float beamCornerCurvature(uint2 tile, bool sparse)
    {
        float curvature = 0.f;
        for (uint k = 0; k < 4; ++k)
        {
            const int2 c = int2(tile) + int2(k & 1u, k >> 1);
            const float3 centre = beamCornerLog(c, sparse);
            const float3 dxx = beamCornerLog(c - int2(1, 0), sparse) - 2.f * centre + beamCornerLog(c + int2(1, 0), sparse);
            const float3 dyy = beamCornerLog(c - int2(0, 1), sparse) - 2.f * centre + beamCornerLog(c + int2(0, 1), sparse);
            curvature = max(curvature, max(maxComponent(abs(dxx)), maxComponent(abs(dyy))));
        }
        return curvature;
    }

    /// Surplus test of one tile: the centre as the bilinear corner basis predicts it, against the exact centre query, in
    /// log(1 + radiance); and the silhouette test of beamTransmittanceRange against beamEdgeContrast.
    bool beamTileRefines(uint level, uint2 tile, bool sparse)
    {
        if (params.beamOracle != 0)
            return beamTileOracleFails(level, tile, sparse);
        const uint s = beamTileEdge(level);
        const uint2 origin = tile * s;
        const bool centreless = level == 0 && params.beamCentreless != 0;
        const float4 a00 = beamVerifierAt(origin, 0, sparse);
        const float4 a10 = beamVerifierAt(origin + uint2(s, 0), 0, sparse);
        const float4 a01 = beamVerifierAt(origin + uint2(0, s), 0, sparse);
        const float4 a11 = beamVerifierAt(origin + uint2(s, s), 0, sparse);
        const float4 ac = beamVerifierAt(origin + s / 2, 0, sparse);
        const float4 bp = 0.25f * (beamVerifierAt(origin, 1, sparse) + beamVerifierAt(origin + uint2(s, 0), 1, sparse) +
                                   beamVerifierAt(origin + uint2(0, s), 1, sparse) + beamVerifierAt(origin + uint2(s, s), 1, sparse));
        const float4 bc = beamVerifierAt(origin + s / 2, 1, sparse);
        const float4 ap = 0.25f * (a00 + a10 + a01 + a11);
        const float3 albedo = gScene.getGridVolume(0).data.albedo;
        const float3 wc = beamRayDirection(float2(origin + s / 2));
        const float3 exactCentre = beamRadiance(ac.rgb, bc.rgb, ac.a, wc, albedo);
        const float3 predictedCentre = beamRadiance(ap.rgb, bp.rgb, ap.a, wc, albedo);
        if (centreless)
        {
            bool refine = beamCornerCurvature(tile, sparse) > beamPolicyTolerance();
            if (!refine && params.beamEdgeContrast > 0.f)
                refine = beamTransmittanceRange(level, tile, sparse) > params.beamEdgeContrast;
            return refine;
        }
        bool refine = maxComponent(abs(log(1.f + max(predictedCentre, 0.f)) - log(1.f + exactCentre))) > beamPolicyTolerance();
        if (!refine && params.beamEdgeContrast > 0.f)
            refine = beamTransmittanceRange(level, tile, sparse) > params.beamEdgeContrast;
        if (!refine && (params.beamGuide == 1 || (params.beamGuide == 2 && level + 1 < params.beamLevels)))
            refine = beamTileGuideFails(level, tile);
        return refine;
    }

    /// Appends a refined tile to the list of the level below it; past the finest level, that list is the per-pixel march list.
    /// The tile's origin cell in the history map records the refinement for the next frame's temporal queries.
    void recordBeamRefinement(uint level, uint2 tile)
    {
        uint slot;
        InterlockedAdd(hstrBeamCounts[level + 1], 1u, slot);
        hstrBeamLists[beamListOffset(level + 1) + slot] = packBeamTile(tile);
        // The temporal queries read this, and so does the adaptive root level; neither pays for it when both are off.
        if (params.beamTemporal != 0 || params.beamAdaptiveRoot != 0)
        {
            const uint2 cell = tile * beamTileEdge(level) / beamTileEdge(params.beamLevels - 1);
            if (all(cell < params.beamLevelDims))
                InterlockedOr(hstrBeamHistoryOutput[cell], 1u << level);
        }
    }

    /// Whether a coarsest tile is predicted to fail its own level, from the finest-level refinements of the frame before: a cell
    /// carrying that bit was marched per pixel last frame. Both the query pass and the tile pass read this same function of the
    /// previous frame, which is what keeps them consistent - a tile predicted difficult never reads its own basis, so the lattice
    /// points only it would have used are never marched. A wrong prediction costs rays, never correctness: call a hard region smooth
    /// and its own surplus test still refines it this frame; call a smooth region hard and it merely spends the finer level's rays.
    bool beamRootDifficult(uint2 tile)
    {
        if (params.beamAdaptiveRoot == 0 || params.beamLevels < 2)
            return false;
        const uint finest = params.beamLevels - 1;
        const uint cells = beamTileEdge(0) / beamTileEdge(finest);
        const uint2 first = tile * cells;
        for (uint y = 0; y < cells; ++y)
            for (uint x = 0; x < cells; ++x)
            {
                const uint2 cell = first + uint2(x, y);
                if (all(cell < params.beamLevelDims) && (hstrBeamHistory[cell] & (1u << finest)) != 0)
                    return true;
            }
        return false;
    }

    /// Whether this frame queried the children of a level tile: it was refined last frame.
    bool beamChildrenQueried(uint level, uint2 tile)
    {
        const uint2 cell = tile * beamTileEdge(level) / beamTileEdge(params.beamLevels - 1);
        return all(cell < params.beamLevelDims) && (hstrBeamHistory[cell] & (1u << level)) != 0;
    }

    /// Writes the level that finalised a tile into all its level-map cells.
    void writeBeamLevel(uint level, uint2 tile, uint value)
    {
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint cells = beamTileEdge(level) / finest;
        const uint2 first = tile * cells;
        for (uint y = 0; y < cells; ++y)
            for (uint x = 0; x < cells; ++x)
            {
                const uint2 cell = first + uint2(x, y);
                if (all(cell < params.beamLevelDims))
                    hstrBeamLevelOutput[cell] = value;
            }
    }

    /// Tile test of the current pass level. Accepted tiles write their level into the level map. Refined tiles join the next
    /// level's list, or at the finest level are marked for per-pixel marching (level beamLevels).
    void testBeamTile(uint thread) { testBeamTile(uint2(thread, 0)); }

    void testBeamTile(uint2 gridTile)
    {
        const uint thread = gridTile.x;
        const uint level = params.beamPassLevel;
        uint2 tile;
        if (level == 0)
        {
            // Generated exactly as the query dispatch is (see beamGridBlocks): a tile can only change its classification where
            // one of its basis points was re-marched, so the pass covers those regions rather than every tile in the image.
            if (params.beamTileGenerated != 0)
            {
                const uint rb = max(params.beamRefreshBlock, 1u);
                tile = params.beamGridBlocks != 0
                           ? (params.beamGridOrigin + (gridTile / rb) * params.beamGridBlockStride) * rb + (gridTile % rb)
                           : gridTile + params.beamGridOrigin;
                if (any(tile >= params.beamTileDims))
                    return;
            }
            else
            {
                if (thread >= params.beamTileDims.x * params.beamTileDims.y)
                    return;
                tile = uint2(thread % params.beamTileDims.x, thread / params.beamTileDims.x);
            }
        }
        else
        {
            const uint entry = thread / 4;
            if (entry >= hstrBeamCounts[level])
                return;
            const uint child = thread % 4;
            tile = 2u * unpackBeamTile(hstrBeamLists[beamListOffset(level) + entry]) + uint2(child & 1u, child >> 1);
        }
        if (any(tile * beamTileEdge(level) >= params.beamFrameDim) || !beamTileOnScreen(level, tile))
            return;
        // A root tile the previous frame refined all the way down skips its own test: its lattice points were not marched.
        if (!(level == 0 && beamRootDifficult(tile)) && !beamTileRefines(level, tile, false))
        {
            writeBeamLevel(level, tile, level);
            return;
        }
        recordBeamRefinement(level, tile);
        if (level + 1 >= params.beamLevels)
            writeBeamLevel(level, tile, params.beamLevels);
    }

    /// The root tile test over the guard's dirty list: every tile of every listed block, after the dirty query has re-marched
    /// their basis. Without it a listed block got new lattice and a new residual but kept whatever classification it had - and
    /// a block no build had reached kept the level map's clear value, "failed", so every pixel of it went to the residual.
    ///
    /// Over the block AND the tiles around it whose test reads its points. A tile's test reads the corners of the tiles from one
    /// before it to two after it and the centres from one before to one after (beamTransmittanceRange), so new basis in a block
    /// changes the answer of tiles the block does not contain: from two before its first tile to one after its last. Tested
    /// over the block alone, the edge tiles of whatever had been built beside a never-built region kept a classification made
    /// against lattice nobody had written yet: after a flick from a reset, the outline of every earlier sliver stayed on screen
    /// as a dashed seam through the clouds.
    ///
    /// Each such tile is tested ONCE, by the listed block with the lowest slot among those whose points it reads
    /// (beamDirtyTileOwner). Listed blocks come in contiguous regions, and each tested every tile of its own two-tile apron, so a
    /// tile inside a region was tested by up to nine of them: 4K sprint, 0.73 ms of tile tests for ~118k blocks.
    ///
    /// A tile that fails lists its units that need marching (beamFailedUnitNeedsMarch) for the dirty unit march: those are the only
    /// units the build can have to march, since the rest belong to accepted tiles or to blocks whose certificate held. That replaced
    /// a pass of one thread per unit of every listed block and its apron - 576 per block for its 64, almost none of the apron ever
    /// marching - which cost 0.75 ms at 4K walk and 1.3 ms at sprint.
    void testBeamDirtyTile(uint thread)
    {
        const uint side = max(params.beamDirtyBlockEdge / max(params.beamTileSize, 1u), 1u);
        const uint padded = side + 2u * kBeamDirtyApron;
        const uint index = thread / (padded * padded);
        if (index >= min(hstrBeamDirtyCount[0], params.beamDirtyCapacity))
            return;
        const uint packed = hstrBeamDirty[index];
        const uint local = thread % (padded * padded);
        const int2 signedTile = int2(uint2(packed & 0xFFFFu, packed >> 16) * side + uint2(local % padded, local / padded)) -
                                int(kBeamDirtyApron);
        if (any(signedTile < 0))
            return;
        const uint2 tile = uint2(signedTile);
        if (any(tile >= params.beamTileDims) || any(tile * beamTileEdge(0) >= params.beamFrameDim) || !beamTileOnScreen(0, tile))
            return;
        if (beamDirtyTileOwner(tile, side) != index)
            return;
        if (!beamTileRefines(0, tile, false))
        {
            writeBeamLevel(0, tile, 0);
            return;
        }
        recordBeamRefinement(0, tile);
        if (params.beamLevels <= 1)
            writeBeamLevel(0, tile, params.beamLevels);
        listBeamFailedTileUnits(tile);
    }

    /// The listed block that tests this tile: the one with the lowest dirty-list slot among the blocks whose re-marched points the
    /// tile's test reads - per axis, a block b of side tiles marches the corners and centres side * b to side * b + side - 1, which
    /// tiles side * b - 2 to side * b + side read. ~0 when none of them is listed, and then the tile cannot have changed.
    uint beamDirtyTileOwner(uint2 tile, uint side)
    {
        const int2 t = int2(tile);
        const int2 lo = max((t + int(side) - 1) / int(side) - 1, 0);
        const int2 hi = min((t + 2) / int(side), int2(params.beamGuardDims) - 1);
        uint owner = 0xFFFFFFFFu;
        for (int y = lo.y; y <= hi.y; ++y)
            for (int x = lo.x; x <= hi.x; ++x)
                owner = min(owner, hstrBeamDirtyMark[uint2(x, y)]);
        return owner;
    }

    /// Appends the units of a tile that has just failed that need marching to hstrBeamDirtyUnits (count hstrBeamDirtyCount[1]): all
    /// of them where the tile's block is listed (its certificate now names this camera, which its units were not marched from),
    /// otherwise only those its certificate does not keep. A tile's units go in together, and a wave's tiles are neighbours, so a
    /// wave of the unit march marches neighbouring directions: appended a unit at a time across the wave instead - unit k of 64
    /// tiles, then unit k + 1 - the same 295k units took the 4K walk's unit march from ~4 to ~12 ms.
    void listBeamFailedTileUnits(uint2 tile)
    {
        const uint ts = max(params.beamTileSize, 1u);
        const uint2 block = tile * ts / max(params.beamDirtyBlockEdge, 1u); // A tile lies in one block.
        const bool own = all(block < params.beamGuardDims) && hstrBeamDirtyMark[block] != 0xFFFFFFFFu;
        // The block's certificate, once for the tile rather than once per unit: it walks the depth pyramid.
        const bool certified = !own && beamCellCertified(block);
        uint mask = 0;
        for (uint k = 0; k < ts * ts; ++k)
            if (beamFailedUnitNeedsMarch(tile * ts + uint2(k % ts, k / ts), own, true, certified))
                mask |= 1u << k;
        const uint count = countbits(mask);
        const uint total = WaveActiveSum(count);
        if (total == 0)
            return;
        const uint owned = WaveActiveSum(own ? count : 0u);
        uint base = 0;
        if (WaveIsFirstLane())
        {
            InterlockedAdd(hstrBeamDirtyCount[1], total, base);
            uint unused;
            if (owned != 0)
                InterlockedAdd(hstrBeamCounts[kBeamDirtyOwnMarched], owned, unused);
            if (total != owned)
                InterlockedAdd(hstrBeamCounts[kBeamDirtyApronMarched], total - owned, unused);
        }
        uint at = WaveReadLaneFirst(base) + WavePrefixSum(count);
        for (uint bits = mask; bits != 0; bits &= bits - 1u)
        {
            const uint k = firstbitlow(bits);
            const uint2 unit = tile * ts + uint2(k % ts, k / ts);
            hstrBeamDirtyUnits[at++] = unit.x | (unit.y << 16);
        }
    }

    /// Temporal tile tests: one thread walks one coarsest tile down through the levels whose queries exist this frame (children
    /// of tiles refined last frame). A tile that needs refining but whose children were not queried keeps its own basis for
    /// this frame; its refinement is recorded, so the next frame queries them.
    void testBeamTileTemporal(uint thread)
    {
        if (thread >= params.beamTileDims.x * params.beamTileDims.y)
            return;
        // Depth-first stack of (tile, level); each refined tile pushes four children, so it never exceeds 1 + 3 levels.
        uint stackTile[1 + 3 * kBeamMaxLevels];
        uint stackLevel[1 + 3 * kBeamMaxLevels];
        uint depth = 1;
        stackTile[0] = packBeamTile(uint2(thread % params.beamTileDims.x, thread / params.beamTileDims.x));
        stackLevel[0] = 0;
        while (depth > 0)
        {
            --depth;
            const uint2 tile = unpackBeamTile(stackTile[depth]);
            const uint level = stackLevel[depth];
            if (any(tile * beamTileEdge(level) >= params.beamFrameDim) || !beamTileOnScreen(level, tile))
                continue;
            if (!(level == 0 && beamRootDifficult(tile)) && !beamTileRefines(level, tile, false))
            {
                writeBeamLevel(level, tile, level);
                continue;
            }
            const bool childrenQueried = level + 1 < params.beamLevels && beamChildrenQueried(level, tile);
            recordBeamRefinement(level, tile);
            if (level + 1 >= params.beamLevels)
                writeBeamLevel(level, tile, params.beamLevels);
            else if (!childrenQueried)
                writeBeamLevel(level, tile, level);
            else
                for (uint child = 0; child < 4; ++child)
                {
                    stackTile[depth] = packBeamTile(2u * tile + uint2(child & 1u, child >> 1));
                    stackLevel[depth] = level + 1;
                    ++depth;
                }
        }
    }

    /// Indirect arguments of the query and tile passes of the current pass level, from its list count; at level beamLevels,
    /// of the per-pixel march pass. Temporal level 0: the single query pass over every level (see beamQueryPosition).
    void writeBeamArgs()
    {
        const uint level = params.beamPassLevel;
        if (params.beamTemporal != 0 && level == 0)
        {
            uint queries = beamBlockQueries(params.beamTileDims + 1u) + beamBlockQueries(params.beamTileDims);
            for (uint l = 1; l < params.beamLevels; ++l)
                queries += 8u * hstrBeamPreviousCounts[l];
            hstrBeamArgs[0] = (queries * params.beamSegments + 63u) / 64u;
            hstrBeamArgs[1] = 1;
            hstrBeamArgs[2] = 1;
            return;
        }
        const uint count = hstrBeamCounts[level];
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint threadsPerEntry = level < params.beamLevels ? 8u * params.beamSegments : finest * finest;
        hstrBeamArgs[6 * level + 0] = (count * threadsPerEntry + 63u) / 64u;
        hstrBeamArgs[6 * level + 1] = 1;
        hstrBeamArgs[6 * level + 2] = 1;
        hstrBeamArgs[6 * level + 3] = (count * 4u + 63u) / 64u;
        hstrBeamArgs[6 * level + 4] = 1;
        hstrBeamArgs[6 * level + 5] = 1;
    }

    /// The best value known for one beam unit's direction, which is what a failed tile's pixel interpolates between.
    ///
    /// The residual image only holds what the residual marched: units of FAILED tiles. A pixel near a failed tile's edge
    /// interpolates across into the next tile, whose units the residual never touched if that tile was accepted - so it read
    /// whatever the texel last held. Measured at 4K parked: resolving from the stored units scored 0.431% of pixels over 0.02
    /// (p99.9 0.086, max 0.61); marching those very same unit directions afresh in the resolve scored 0.244% (p99.9 0.030, max
    /// 0.18). Same sample positions, so the gap was never resampling - it was reading units nobody had written. An accepted
    /// tile's unit is instead taken from its own basis: the estimate that tile was accepted with, at no march. (Acceptance tests
    /// the centre surplus and the edge contrast, so this is the basis the resolve would use there anyway, not a bound on every
    /// unit of the tile.)
    float3 beamUnitRadiance(uint2 unit)
    {
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint level = hstrBeamLevel[unit / finest];
        const float2 p = float2(unit) + 0.5f;
        const float3 w = beamRayDirection(p);
        if (level == params.beamLevels + 1u)
            return beamRadiance(0.f, 0.f, 0.f, w, gScene.getGridVolume(0).data.albedo);
        if (level >= params.beamLevels)
        {
            const float4 held = hstrBeamPixels[unit];
            if (held.a >= 0.f)
                return held.rgb;
            // A failed tile's unit the residual has not reached - outside the on-screen box it marches. Its finest basis
            // failed the surplus test, but it is still an estimate of this direction; the texel is not.
            return beamReconstruct(params.beamLevels - 1u, p, w, false);
        }
        return beamReconstruct(level, p, w, false);
    }

    /// beamWarp: where the persistent image holds what the current camera sees along w. A value in the image is the radiance along
    /// a world direction AS SEEN FROM the camera its guard block was verified from, so from a camera that has moved since, the
    /// content along w was stored under the direction from that capture camera to it: X = eye + w z, read at normalize(X - Pc),
    /// with z the query depth (opacity-weighted distance) nearest the position read. Twice, the second time with the depth and
    /// capture camera found where the first landed. First order in |eye - Pc| / z: the stored depth stands in for the distance
    /// from the new eye. A block never verified is read as is.
    /// MEASURED (4K walk, 2x steps + tstep + beamOctScale 0.5, beamGuardParallax 8, in-flight scoring): 1.985% of pixels over
    /// 0.02 unwarped, 1.439% warped by each query's own depth (budgetwarp1), 0.960% with the sky fallback below (budgetwarp2).
    void beamWarpPosition(float3 w, inout float2 position, out float3 read)
    {
        read = w;
        const float3 eye = gScene.camera.data.posW;
        const uint step = max(params.beamLatticeStep, 1u);
        const uint2 last = (params.beamLatticeDims - 1u) * step;
        [unroll]
        for (uint k = 0; k < 2; ++k)
        {
            const uint2 lattice = min(uint2(round(max(position, 0.f) / float(step))) * step, last);
            const uint2 cell = beamGuardCell(lattice);
            const float4 held = hstrBeamGuardCamera[cell];
            if (held.w < 0.5f)
                return;
            float z = beamLatticeAt(lattice, 1).w;
            // Sky: no depth of its own, but a cloud may have slid into it since the capture. Warp by the nearest depth among the
            // guard blocks around; the second pass then reads what is really there. 91.9% of walk's pixels read a query that hit
            // nothing (budgetwarpdiag1), and warping only by their own depth left every cloud edge moving over sky unfollowed.
            if (!(z > 0.f))
            {
                float nearest = 3.4e38f;
                for (int y = -1; y <= 1; ++y)
                    for (int x = -1; x <= 1; ++x)
                    {
                        const int2 n = int2(cell) + int2(x, y);
                        if (all(n >= 0) && all(n < int2(params.beamGuardDims)))
                        {
                            const uint stored = hstrBeamGuardDepth[uint2(n)];
                            if (stored != 0xFFFFFFFFu)
                                nearest = min(nearest, asfloat(stored));
                        }
                    }
                if (!(nearest < 1e30f))
                    return;
                z = nearest;
            }
            const float3 d = normalize(eye + w * z - held.xyz);
            float2 warped;
            if (!beamImagePosition(d, warped))
                return;
            position = warped;
            read = d;
        }
    }

    /// The lattice box buildBeamWarpField covers: the on-screen box (beamScreenBounds) in lattice points, padded by two so the
    /// resolve's bilinear read of the field never reaches past what was written. The host computes the same box to dispatch it.
    void beamWarpFieldBox(out uint2 origin, out uint2 end)
    {
        const float step = float(max(params.beamLatticeStep, 1u));
        origin = uint2(max(floor(params.beamScreenBounds.xy / step) - 2.f, 0.f));
        end = min(uint2(ceil(params.beamScreenBounds.zw / step) + 2.f), params.beamLatticeDims);
    }

    /// beamWarp's field: once per on-screen lattice point, the offset from its position to where beamWarpPosition finds its
    /// content was captured. The warp is a function of direction that changes at lattice scale - it reads the nearest query's
    /// depth and its block's capture camera - so the resolve interpolates it instead of running it: per 4K pixel it cost the
    /// resolve 0.43 -> 0.76 ms (budgetwarp3), and at beamOctScale 0.5 a lattice point covers ~16 pixels.
    void buildBeamWarpFieldAt(uint2 thread)
    {
        uint2 origin, end;
        beamWarpFieldBox(origin, end);
        const uint2 texel = origin + thread;
        if (any(texel >= end))
            return;
        const float2 position = float2(texel * max(params.beamLatticeStep, 1u));
        float2 warped = position;
        float3 read;
        beamWarpPosition(beamRayDirection(position), warped, read);
        hstrBeamWarpFieldOutput[texel] = warped - position;
    }

    /// beamPolicy's tile tolerance for this build (decideBeamPolicy); off, the parameter as set.
    float beamPolicyTolerance() { return params.beamPolicy != 0 ? asfloat(hstrBeamWarpArgs[11]) : params.beamTolerance; }

    /// Where the resolve reads a beam-image position under beamWarp: the field's offset, interpolated between lattice points.
    float2 beamWarpedPosition(float2 position)
    {
        const float step = float(max(params.beamLatticeStep, 1u));
        const float2 uv = (position / step + 0.5f) / float2(params.beamLatticeDims);
        return position + hstrBeamWarpField.SampleLevel(hstrLinearClampSampler, uv, 0.f);
    }

    /// Beam view pixel reconstructed from the tile level that accepted it: bilinear corners plus the centre's bubble. No voxel
    /// is touched. Pixels of tiles that failed the finest level are left to marchBeamPixel. Returns whether the pixel is left to
    /// the reference frame's residual resolve instead.
    bool resolveBeamPixel(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return false;
        float3 w = cameraRayDirection(float2(pixel) + 0.5f);
        float2 beamPosition;
        if (!beamPixelPosition(float2(pixel) + 0.5f, beamPosition))
            return false;
#if HSTR_BEAM_WARP
        beamPosition = beamWarpedPosition(beamPosition);
        w = beamRayDirection(beamPosition);
#endif
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint level = hstrBeamLevel[uint2(beamPosition) / finest];
        if (level == params.beamLevels + 1u)
        {
            color[pixel] = float4(beamRadiance(0.f, 0.f, 0.f, w, gScene.getGridVolume(0).data.albedo), 1.f);
            return false;
        }
        if (level >= params.beamLevels)
        {
            // A tile the basis could not carry. In the reference frame its residual lives in the beam image beside the basis, so
            // the resolve reads it here instead of a separate screen-space march pass writing colour directly.
            if (params.beamScreenResidual != 0)
                return false; // The screen-space march already wrote this pixel, exactly, with no resampling.
            // In the reference frame the residual is resolved by resolveBeamResidualPixel, in its own dispatch: see there. This
            // pass has already found the pixel, so it says so, and the resolveBeam entry queues its 8 x 8 group with a mask.
            return params.beamRefFrame != 0;
        }
        color[pixel] = float4(beamReconstruct(level, beamPosition, w, false), 1.f);
        return false;
    }

    /// 8 x 8 resolve groups in the frame: the capacity of the residual queue.
    uint beamResidualGroups() { return ((params.frameDim.x + 7u) / 8u) * ((params.frameDim.y + 7u) / 8u); }

    /// A failed tile's pixel in the reference frame: the bilinear over the four units around it, each the best value known for
    /// its direction (beamUnitRadiance).
    ///
    /// A separate dispatch from resolveBeamPixel on purpose. Inlined there, the four basis reconstructions this may need made the
    /// resolve heavier for EVERY pixel, not just the ~6% it serves: 4K resolve 0.37 -> 0.81-0.96 ms in every arm, including the
    /// screen-residual arm that never takes the branch. That is register pressure and occupancy, not work.
    ///
    /// One 64-thread group per 8 x 8 resolve group that had a residual pixel, one thread per pixel, and only the pixels its mask
    /// names go on - so the pass costs the failed pixels, and its queue is three uints per screen group rather than one per pixel.
    /// A per-pixel list cost 31.6 MiB at 4K; this is 1.5 MiB. A full-frame dispatch testing every pixel cost 0.13-0.20 ms.
    void resolveBeamResidualPixel(uint entry, uint lane)
    {
        if (entry >= min(hstrBeamResidualArgs[0], beamResidualGroups()))
            return;
        const uint packed = hstrBeamResidualList[3 * entry];
        if ((hstrBeamResidualList[3 * entry + 1 + (lane >> 5)] & (1u << (lane & 31u))) == 0)
            return;
        const uint2 pixel = uint2(packed & 0xFFFFu, packed >> 16) * 8u + uint2(lane & 7u, lane >> 3);
        float2 beamPosition;
        if (!beamPixelPosition(float2(pixel) + 0.5f, beamPosition))
            return;
#if HSTR_BEAM_WARP
        beamPosition = beamWarpedPosition(beamPosition);
#endif
        const float2 texel = beamPosition - 0.5f;
        const int2 base = clamp(int2(floor(texel)), int2(0), int2(params.beamFrameDim) - 2);
        const float2 g = saturate(texel - float2(base));
        const float3 v = lerp(
            lerp(beamUnitRadiance(uint2(base)), beamUnitRadiance(uint2(base + int2(1, 0))), g.x),
            lerp(beamUnitRadiance(uint2(base + int2(0, 1))), beamUnitRadiance(uint2(base + int2(1, 1))), g.x), g.y
        );
        color[pixel] = float4(v, 1.f);
    }

    /// One pixel of a finest tile that failed its surplus test, marched exactly.
    void marchBeamPixel(uint thread)
    {
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint entry = thread / (finest * finest);
        if (entry >= hstrBeamCounts[params.beamLevels])
            return;
        const uint local = thread % (finest * finest);
        const uint2 pixel = unpackBeamTile(hstrBeamLists[beamListOffset(params.beamLevels) + entry]) * finest + uint2(local % finest, local / finest);
        if (any(pixel >= params.frameDim))
            return;
        marchBeamAt(pixel);
    }

    /// marchBeamPixel as a full-frame dispatch (beamGridDispatch): every pixel whose finest tile was not accepted.
    /// One beam unit of a finest tile the basis could not carry, in the rotation-invariant frame.
    ///
    /// This is the per-pixel fallback moved into the beam image, for the same reason the basis was: the residual is a function of
    /// world direction, so turning the camera neither invalidates one nor moves one relative to another. A unit that the previous
    /// build marched is read back at its own index - no reprojection, no search, no agreement test between neighbours - and only a
    /// unit that has turned on screen, whose tile has just started failing, or whose parallax the camera's translation broke, is
    /// marched again. The resolve samples this image where a pixel's tile failed.
    void marchBeamUnit(uint2 gridUnit)
    {
        uint2 unit = gridUnit;
        if (params.beamUnitGenerated != 0)
        {
            // Generated like the query and tile passes: the thread's unit is a tile inside one of this build's regions, plus
            // its offset within that tile. Everything outside them is carrying a residual it already holds.
            const uint ts = max(params.beamTileSize, 1u);
            const uint rb = max(params.beamRefreshBlock, 1u);
            const uint2 local = gridUnit / ts;
            const uint2 tile = params.beamGridBlocks != 0
                                   ? (params.beamGridOrigin + (local / rb) * params.beamGridBlockStride) * rb + (local % rb)
                                   : local + params.beamGridOrigin;
            if (any(tile >= params.beamTileDims))
                return;
            unit = tile * ts + (gridUnit % ts);
        }
        marchBeamUnitAt(unit);
    }

    /// The residual for one beam unit, once a dispatch has decided which unit this thread owns.
    /// Returns whether it marched, rather than carried, deferred to the basis or skipped.
    bool marchBeamUnitAt(uint2 unit, bool force = false)
    {
        if (!beamUnitNeedsMarch(unit, force))
            return false;
        marchBeamUnitExact(unit);
        return true;
    }

    /// Everything marchBeamUnitAt decides before it marches, including the carry it writes when it does not.
    bool beamUnitNeedsMarch(uint2 unit, bool force)
    {
        // MEASURED and REVERTED: bounding a SWEPT residual to the box the screen covers, on the theory that dispatching a thread
        // per texel of the octahedral sphere to have almost all of them fail the test below was the 1.28 ms of a 1.79 ms sprint
        // frame. It was not - sprint went 1.79 to 1.85 - and it cost coverage: the resolve reads a 2x2 neighbourhood, so units
        // just outside the box are read after all, and a parked octahedral frame went from 1.663% of pixels over 0.02 to 7.790%.
        // The residual's cost under translation is units genuinely re-marching, not threads arriving to be rejected.
        if (any(unit >= params.beamFrameDim) || !beamUnitOnScreen(unit))
            return false;
        // MEASURED and REMOVED: marching a one-unit apron around every failed tile, so the resolve's bilinear never reaches into
        // an accepted neighbour's unwritten units. Failed tiles are 4 units square and scattered, so a one-unit border is 2.25x
        // their area and the march went 0.59 -> 1.71 ms parked and 0.75 -> 3.39 under yaw - and it did not move the seam it was
        // aimed at (720p max log error 7.88e-2, bit-identical before and after), so the seam is something else.
        const uint finest = beamTileEdge(params.beamLevels - 1);
        if (hstrBeamLevel[unit / finest] < params.beamLevels)
            return false; // The basis carries this tile; the resolve reconstructs it.
        return beamFailedUnitNeedsMarch(unit, force);
    }

    /// beamUnitNeedsMarch for a unit of a tile known to have failed (the dirty tile pass, which has just tested it). certifiedKnown:
    /// the caller has already evaluated the unit's block certificate (certified), once for all the units of a tile.
    bool beamFailedUnitNeedsMarch(uint2 unit, bool force, bool certifiedKnown = false, bool certified = false)
    {
        if (any(unit >= params.beamFrameDim) || !beamUnitOnScreen(unit))
            return false;
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const float3 w = beamRayDirection(float2(unit) + 0.5f);
        // Carried where THIS unit already holds a value marched for this direction, and the camera has not moved past its
        // parallax. The residual image is persistent, so the question is about the unit itself rather than about what the
        // previous build happened to cover: alpha < 0 means never marched since the frame was anchored, and that is the only
        // thing that makes a unit new. That replaces two conditions that were really proxies for it - the previous level map
        // (did this tile fail last build) and beamPrevScreenBounds (did the previous build reach this far) - and it lets a unit
        // that leaves the screen and comes back carry, which neither proxy could express.
        // With the guard the unit is kept exactly when its block's certificate holds - it was marched from close enough to here -
        // and otherwise re-marched; nothing is carried in place with a moved distance (see beamCellCertified). force is set for
        // the units of a block the dirty list re-verified this build, whose certificate now names this camera although these
        // units were marched from the old one.
        if (params.beamGuard != 0 && params.beamRefFrame != 0)
        {
            const uint2 cell = unit / max(params.beamDirtyBlockEdge, 1u);
            const bool held = params.beamRefValid != 0 && !beamRefreshes(unit / finest) && hstrBeamPixelOutput[unit].a >= 0.f;
            if (!force && held && (certifiedKnown ? certified : beamCellCertified(cell)))
                return false;
        }
        else if (params.beamRefValid != 0 && params.beamRefresh != 0 && !beamRefreshes(unit / finest))
        {
            const float4 previous = hstrBeamPixelOutput[unit];
            if (previous.a >= 0.f)
            {
                const float d = previous.a > 0.f ? beamCurrentDistance(previous.a, w) : 0.f;
                if (previous.a == 0.f || (d > 0.f && beamParallaxAllows(d, w)))
                {
                    // A camera that has not translated cannot have moved this unit, so the carry writes nothing at all.
                    if (params.beamStationary == 0 && previous.a > 0.f)
                        hstrBeamPixelOutput[unit] = float4(previous.rgb, d);
                    uint unused;
                    InterlockedAdd(hstrBeamCounts[kBeamMaxLevels], 1u, unused);
                    return false;
                }
            }
        }
        return true;
    }

#if HSTR_BEAM_REPAIR_PROBE
    /// Largest per-channel difference in log(1 + radiance): the image gate's metric.
    float beamProbeError(float3 a, float3 b) { const float3 e = abs(log(1.f + max(a, 0.f)) - log(1.f + max(b, 0.f))); return max(e.x, max(e.y, e.z)); }

    /// Where the previous build saw what this camera sees along w, if it lies at the distance d0 the previous build stored for w:
    /// single-depth reprojection. False for a direction that stored no opacity (nothing there to move).
    bool beamProbeReproject(float3 w, float d0, out float2 position)
    {
        position = 0.f;
        if (!(d0 > 0.f))
            return false;
        const float3 camera = gScene.camera.data.posW;
        const float3 x = camera + beamCurrentDistance(d0, w) * w;
        return beamImagePosition(normalize(x - params.beamPrevCamera), position);
    }

    /// Scores what a repair could have predicted for a sample this build re-marched: the old value kept in place, and the old image
    /// reprojected by the old distance. Four counters from base: marched, in place within 0.02, reprojected within 0.02, either.
    /// The same four again from kBeamProbeSteps (units, then rays), weighted by the march steps the sample cost: what a prediction
    /// would save is the steps it avoids, not the samples.
    void beamProbeCount(uint base, bool scored, bool inPlace, bool reprojected, float steps)
    {
        const uint n = WaveActiveCountBits(scored);
        const uint a = WaveActiveCountBits(scored && inPlace);
        const uint b = WaveActiveCountBits(scored && reprojected);
        const uint c = WaveActiveCountBits(scored && (inPlace || reprojected));
        const uint s = scored ? uint(steps) : 0u;
        const uint sn = WaveActiveSum(s);
        const uint sa = WaveActiveSum(inPlace ? s : 0u);
        const uint sb = WaveActiveSum(reprojected ? s : 0u);
        const uint sc = WaveActiveSum(inPlace || reprojected ? s : 0u);
        if (WaveIsFirstLane() && n != 0)
        {
            uint unused;
            InterlockedAdd(hstrBeamCounts[base], n, unused);
            InterlockedAdd(hstrBeamCounts[base + 1], a, unused);
            InterlockedAdd(hstrBeamCounts[base + 2], b, unused);
            InterlockedAdd(hstrBeamCounts[base + 3], c, unused);
            const uint weighted = kBeamProbeSteps + (base == kBeamProbeUnits ? 0u : 4u);
            InterlockedAdd(hstrBeamCounts[weighted], sn, unused);
            InterlockedAdd(hstrBeamCounts[weighted + 1], sa, unused);
            InterlockedAdd(hstrBeamCounts[weighted + 2], sb, unused);
            InterlockedAdd(hstrBeamCounts[weighted + 3], sc, unused);
        }
    }

    void beamProbeUnit(uint2 unit, float3 w, float3 exact, float steps)
    {
        const float4 old = hstrBeamPixelsSnapshot[unit];
        const bool scored = old.a >= 0.f; // Never marched before has nothing to predict from.
        const bool inPlace = scored && beamProbeError(old.rgb, exact) <= 0.02f;
        bool reprojected = inPlace && !(old.a > 0.f); // Empty: nothing moves.
        float2 position;
        if (scored && beamProbeReproject(w, old.a, position))
        {
            const float4 moved = hstrBeamPixelsSnapshot.SampleLevel(hstrLinearClampSampler, position / float2(params.beamFrameDim), 0.f);
            reprojected = beamProbeError(moved.rgb, exact) <= 0.02f;
        }
        beamProbeCount(kBeamProbeUnits, scored, inPlace, reprojected, steps);
        // The residual of the blocks a witness policy would have carried: of their units' steps, how many were within 0.02 in place,
        // and how many kept or reprojected. Units of blocks the query did not list read no acceptance (cleared each build).
        {
            const uint2 cell = unit / max(params.beamDirtyBlockEdge, 1u);
            const uint accepted = all(cell < params.beamGuardDims) ? hstrBeamProbeAccept[cell] : 0u;
            const uint unitSteps = scored ? uint(steps) : 0u;
            if (accepted != 0u && unitSteps != 0u)
            {
                for (uint b = 0; b < 3u * kBeamProbePolicies; ++b)
                {
                    if ((accepted & (1u << b)) == 0u)
                        continue;
                    uint unused;
                    InterlockedAdd(hstrBeamCounts[kBeamProbeUnitAccept + 3u * b], unitSteps, unused);
                    if (inPlace)
                        InterlockedAdd(hstrBeamCounts[kBeamProbeUnitAccept + 3u * b + 1u], unitSteps, unused);
                    if (inPlace || reprojected)
                        InterlockedAdd(hstrBeamCounts[kBeamProbeUnitAccept + 3u * b + 2u], unitSteps, unused);
                }
            }
        }
        // MEASURED and REMOVED: what a runtime check would accept of that reprojection. The content's depth found in two steps (the
        // unit's own old depth, then the source's), the previous image read bilinearly there, accepted only if every unit the read
        // weighs was marched, belongs to a failed tile and agrees with the depth to 5%: 4K walk accepted 62.5% of the re-marched units
        // and 19% of those were over 0.02 (12% of all of them); sprint 21.9%, 42.5% bad. Requiring the sources to agree within 0.02
        // too: walk 27.7% accepted, 14.5% of those bad; sprint 9.0%, 29.6%. What changes is what the new viewpoint sees, not the
        // resampling, and no check on the old image can see that.
        beamProbeDivergence(kBeamProbeDivergence, steps);
    }

    /// How much of a wave's march the lanes spent waiting: the steps its lanes took, and 64 times its longest (two counters from
    /// base). Their ratio is the wave's efficiency - what sorting marches by cost could at most recover.
    void beamProbeDivergence(uint base, float steps)
    {
        const uint s = uint(steps);
        const uint taken = WaveActiveSum(s);
        const uint paid = WaveActiveMax(s) * WaveGetLaneCount();
        if (WaveIsFirstLane())
        {
            uint unused;
            InterlockedAdd(hstrBeamCounts[base], taken, unused);
            InterlockedAdd(hstrBeamCounts[base + 1], paid, unused);
        }
    }

    /// The lattice's radiance at a beam position, bilinear over its cache, single and depth (then composed, as the basis does).
    float3 beamProbeLatticeRadiance(float2 position, float3 w, float3 albedo)
    {
        const float2 t = position / float(max(params.beamLatticeStep, 1u)); // Lattice texel k holds beam position k * step.
        const int2 base = int2(floor(t));
        const float2 f = t - float2(base);
        float4 depthCache = 0.f;
        float4 single = 0.f;
        for (uint k = 0; k < 4; ++k)
        {
            const int2 texel = max(base + int2(k & 1u, k >> 1), 0);
            const float weight = ((k & 1u) != 0 ? f.x : 1.f - f.x) * ((k >> 1) != 0 ? f.y : 1.f - f.y);
            const uint2 at = uint2(texel) * max(params.beamLatticeStep, 1u);
            depthCache += weight * hstrBeamLatticeSnapshot[beamLatticeTexel(at, 0)];
            single += weight * hstrBeamLatticeSnapshot[beamLatticeTexel(at, 1)];
        }
        return beamRadiance(depthCache.rgb, single.rgb, depthCache.a, w, albedo);
    }

    /// What a current-view witness would see along one ray, from one camera: an extinction march (the same stepping as the camera
    /// march, no lighting) giving the opacity, its centroid and the distances at which 25/50/75% of it has accumulated, then a
    /// transport fingerprint at those three places - the sun transmittance from the bakes and one world-cache luminance at d50.
    struct ProbeSignature
    {
        float opacity;
        float centroid;
        float3 d;
        float3 sun;
        float cache;
        float steps; ///< Steps of one extinction march: what the witness would cost.
    };

    float beamProbeExtinction(CutRay ray, float3 targets, out float3 tAt, out float centroid, out float steps)
    {
        tAt = -1.f;
        centroid = 0.f;
        steps = 0.f;
        const float cellWidth = float(params.hstrCellWidth);
        const float voxelsPerUnit = length(ray.direction) * cellWidth;
        if (voxelsPerUnit <= 1e-6f)
            return 0.f;
        const float worldPerVoxel = ray.worldLength / voxelsPerUnit;
        float transmittance = 1.f;
        float weighted = 0.f;
        float fineVoxels = 1.f;
        float t = 0.f;
        [loop]
        while (t < 1.f && transmittance > params.cloudMinTransmittance && steps < float(params.maxMarchSteps))
        {
            const float3 v = (ray.position + t * ray.direction) * cellWidth;
            const float majorant = majorantAt(v);
            const float minimumStep = clamp(fineVoxels, params.cloudFineMinVoxels, 1.f) * params.minStepVoxels;
            const float stepVoxels = clamp(params.stepOpticalDepth / max(1e-6f, majorant * worldPerVoxel), minimumStep, params.maxStepVoxels);
            const float dt = min(stepVoxels / voxelsPerUnit, 1.f - t);
            steps += 1.f;
            if (majorant <= 0.f)
            {
                t = min(max(t + dt, majorantZeroExit(ray, t)), 1.f);
                continue;
            }
            const float tSample = t + 0.5f * dt;
            const float3 midpoint = (ray.position + tSample * ray.direction) * cellWidth;
            CloudSample sample = {};
            const float sigma = cameraExtinctionAtVoxel(midpoint, rayFootprint(ray, tSample), fineVoxels, sample);
            const float alpha = 1.f - exp(-(sigma * dt * ray.worldLength));
            const float before = 1.f - transmittance;
            weighted += transmittance * alpha * (ray.entryDistance + tSample * ray.worldLength);
            transmittance *= 1.f - alpha;
            const float after = 1.f - transmittance;
            for (uint q = 0; q < 3; ++q)
                if (tAt[q] < 0.f && after >= targets[q])
                    tAt[q] = t + dt * saturate((targets[q] - before) / max(after - before, 1e-6f));
            t = sample.empty ? min(max(t + dt, cloudEmptyExit(ray, tSample, sample)), 1.f) : t + dt;
        }
        const float opacity = 1.f - transmittance;
        centroid = opacity > 1e-4f ? weighted / opacity : 0.f;
        return opacity;
    }

    ProbeSignature beamProbeSignature(float3 origin, float3 w)
    {
        ProbeSignature g = {};
        CutRay ray;
        if (!makeCutRay(origin, w, gScene.getGridVolume(0), ray))
            return g;
        float3 tAt;
        g.opacity = beamProbeExtinction(ray, float3(2.f), tAt, g.centroid, g.steps);
        if (g.opacity <= 0.01f)
            return g;
        float centroid, steps;
        beamProbeExtinction(ray, g.opacity * float3(0.25f, 0.5f, 0.75f), tAt, centroid, steps);
        const float cellWidth = float(params.hstrCellWidth);
        for (uint q = 0; q < 3; ++q)
        {
            const float tq = max(tAt[q], 0.f);
            g.d[q] = ray.entryDistance + tq * ray.worldLength;
            const float3 v = (ray.position + tq * ray.direction) * cellWidth;
            const float footprint = rayFootprint(ray, tq);
            CloudSample sample = {};
            float fineVoxels = 1.f;
            cameraExtinctionAtVoxel(v, footprint, fineVoxels, sample);
            g.sun[q] = exp(-sunDepthAt(v, footprint, sample));
            if (q == 1)
                g.cache = dot(worldCacheRadiance(v, ray.viewDirection), float3(0.2126f, 0.7152f, 0.0722f));
        }
        return g;
    }

    float beamProbeRelative(float a, float b) { return abs(a - b) / max(max(abs(a), abs(b)), 1e-3f); }

    /// Which signatures (bits, kBeamProbeSignatures) say the ray's transport is unchanged. Distances seen from the new camera are
    /// moved by its displacement along w, which is where the old ray's plane through the same content lies.
    uint beamProbeMatches(ProbeSignature o, ProbeSignature n, float shift)
    {
        const bool emptyBoth = o.opacity <= 0.01f && n.opacity <= 0.01f;
        const float dOpacity = abs(o.opacity - n.opacity);
        float centroid = 0.f;
        float3 d = 0.f;
        float3 sun = 0.f;
        float cache = 0.f;
        if (!emptyBoth)
        {
            centroid = beamProbeRelative(o.centroid, n.centroid + shift);
            for (uint q = 0; q < 3; ++q)
            {
                d[q] = beamProbeRelative(o.d[q], n.d[q] + shift);
                sun[q] = abs(o.sun[q] - n.sun[q]);
            }
            cache = beamProbeRelative(o.cache, n.cache);
        }
        const float dMax = max(d.x, max(d.y, d.z));
        const float sunMax = max(sun.x, max(sun.y, sun.z));
        const bool a = dOpacity <= 0.02f;
        const bool c = a && dMax <= 0.02f;
        const bool e = c && sunMax <= 0.02f && cache <= 0.02f;
        const bool cLoose = dOpacity <= 0.05f && dMax <= 0.05f;
        uint bits = 0;
        bits |= a ? 1u : 0u;
        bits |= (a && centroid <= 0.02f) ? 2u : 0u;
        bits |= c ? 4u : 0u;
        bits |= (c && sun.y <= 0.02f) ? 8u : 0u;
        bits |= e ? 16u : 0u;
        bits |= cLoose ? 32u : 0u;
        bits |= (cLoose && sunMax <= 0.05f && cache <= 0.05f) ? 64u : 0u;
        return bits;
    }

    /// The witness policies, as masks over a block's g lanes (per corners, then per centres): one centre, two opposite centres,
    /// every centre, every ray, and the one or two rays whose OLD values were nearest (the most parallax per unit of motion) -
    /// all of which a runtime knows before marching anything.
    uint beamProbePolicyMask(uint policy, uint per, uint g, uint nearest, uint second)
    {
        const uint centres = ((1u << per) - 1u) << per;
        if (policy == 0)
            return 1u << per;
        if (policy == 1)
            return (1u << per) | (1u << (2u * per - 1u));
        if (policy == 2)
            return centres;
        if (policy == 3)
            return (1u << g) - 1u;
        if (policy == 4)
            return nearest < g ? 1u << nearest : 0u;
        return (nearest < g ? 1u << nearest : 0u) | (second < g ? 1u << second : 0u);
    }

    /// Returns whether beamShadowCarry keeps this ray's old value (its block's policy accepted it).
    bool beamProbeQuery(uint2 position, BeamSample fresh, bool march)
    {
        const float3 w = beamRayDirection(float2(position));
        const float3 albedo = gScene.getGridVolume(0).data.albedo;
        const float3 exact = beamRadiance(fresh.cache, fresh.single, fresh.depth, w, albedo);
        const bool scored = march && hstrBeamLatticeSnapshot[beamLatticeTexel(position, 2)].y > 0.5f;
        const float4 depthCache = hstrBeamLatticeSnapshot[beamLatticeTexel(position, 0)];
        const float4 single = hstrBeamLatticeSnapshot[beamLatticeTexel(position, 1)];
        const float inPlaceError = beamProbeError(beamRadiance(depthCache.rgb, single.rgb, depthCache.a, w, albedo), exact);
        const bool inPlace = scored && inPlaceError <= 0.02f;
        bool reprojected = inPlace && !(single.a > 0.f);
        float2 moved;
        if (scored && beamProbeReproject(w, single.a, moved))
            reprojected = beamProbeError(beamProbeLatticeRadiance(moved, w, albedo), exact) <= 0.02f;
        beamProbeCount(kBeamProbeRays, scored, inPlace, reprojected, fresh.steps);
        beamProbeDivergence(kBeamProbeDivergence + 2, fresh.steps);
        // Block-coherent: would carrying the WHOLE block have been right, and would a geometry signature of this view have said so?
        // The signature is taken from this exact march, so it prices the classifier, not a validator march.
        const float opacityOld = 1.f - exp(-depthCache.a);
        const float opacityNew = 1.f - exp(-fresh.depth);
        const float dOpacity = abs(opacityNew - opacityOld);
        const float dDistance = (opacityOld > 0.01f || opacityNew > 0.01f)
                                    ? abs(fresh.distance - single.a) / max(max(fresh.distance, single.a), 1e-3f)
                                    : 0.f;
        const bool tight = scored && dOpacity <= 0.02f && dDistance <= 0.02f;
        const bool loose = scored && dOpacity <= 0.05f && dDistance <= 0.10f;
        beamProbeBlock(!scored || inPlace, tight || !scored, loose || !scored, scored, fresh.steps);
        return beamProbeWitnesses(position, w, march, scored, inPlace, scored && inPlaceError <= 0.01f, fresh.steps, single.a);
    }

    /// MEASURED and REMOVED: the runtime witnessed carry this matrix priced (beamWitness; 4K OCT_T001 parallax 1, one settle,
    /// witness6 / witness_split). A dirty block marched its witnesses first and kept every old value when all agreed within the
    /// tolerance, otherwise a compacted dependent dispatch marched its other rays. Walk base 7.46 ms, four centres at 0.01 8.82
    /// (query 2.92 -> 4.16 with the repair 1.67 of it), deferred repair 10.14 at 0.333% > 0.02 (base 0.136%): the rejected blocks'
    /// stale rays failed their tiles and the unit march grew 3.42 -> 4.94. Sprint 9.27 -> 10.88 / 16.05. Even carrying EVERY block
    /// (tolerance 1e9, a wrong image) only reached 6.10 ms: the witness pass alone cost 2.71 ms, so the query it attacks cannot fall
    /// much below 2 ms this way and the 3.4 ms of units are untouched. The ceiling, not the implementation, rejected it: predicting
    /// which old camera-ray results survive translation cannot reach the target; translation needs world-space transport.
    /// The witness matrix: for every signature and policy, would observing only the policy's rays have carried the block, and was
    /// the whole block within 0.02 in place (the exact oracle)? Counted in query steps the carry would have saved, with the steps
    /// the witnesses' extinction marches would have cost. Every lane of the wave is active here (buildBeamDirtyQueries).
    bool beamProbeWitnesses(uint2 position, float3 w, bool march, bool scored, bool inPlace, bool inPlaceTight, float steps, float oldDistance)
    {
        const uint rb = max(params.beamDirtyPointStride, 1u);
        const uint per = rb * rb;
        const uint g = per * 2u;
        if (g > 32u || (32u % g) != 0u || params.beamCentreless != 0)
            return false;
        bool shadow = false;
        uint matches = (1u << kBeamProbeSignatures) - 1u;
        float witnessSteps = 0.f;
        if (scored)
        {
            const uint2 cell = beamGuardCell(position);
            const float3 oldCamera = hstrBeamGuardCameraSnapshot[cell].xyz;
            const float3 camera = gScene.camera.data.posW;
            const ProbeSignature o = beamProbeSignature(oldCamera, w);
            const ProbeSignature n = beamProbeSignature(camera, w);
            matches = beamProbeMatches(o, n, dot(camera - oldCamera, w));
            // The witness marched in full and compared with its own old radiance, which the lattice already holds: the
            // signature a runtime gets for no storage, since a block it rejects needs that march anyway.
            matches |= inPlace ? (1u << 7) : 0u;
            matches |= inPlaceTight ? (1u << 8) : 0u;
            matches |= (inPlace && (matches & 16u) != 0) ? (1u << 9) : 0u;
            witnessSteps = n.steps;
        }
        // A ray this build marches for the first time has no old value: its block cannot be carried under any policy.
        const bool firstMarch = march && !scored;
        const uint lane = WaveGetLaneIndex();
        const uint start = lane - lane % g;
        const uint local = lane % g;
        const uint word = start / 32u;
        const uint shift = start % 32u;
        const uint groupMask = (g == 32u ? ~0u : ((1u << g) - 1u)) << shift;
        const bool blockOk = (WaveActiveBallot(scored && !inPlace)[word] & groupMask) == 0;
        const bool blockFresh = (WaveActiveBallot(firstMarch)[word] & groupMask) != 0;
        const bool blockScored = (WaveActiveBallot(scored)[word] & groupMask) != 0;
        // Nearest old values first; a ray that stored nothing (empty) goes after every ray that stored a distance, and a ray that
        // was not scored is never a witness.
        const float key = scored ? (oldDistance > 0.f ? oldDistance : 1e30f) : 3e38f;
        uint nearest = g;
        uint second = g;
        float best = 3e38f;
        float next = 3e38f;
        for (uint k = 0; k < g; ++k)
        {
            const float other = WaveReadLaneAt(key, start + k);
            if (other < best)
            {
                second = nearest;
                next = best;
                nearest = k;
                best = other;
            }
            else if (other < next)
            {
                second = k;
                next = other;
            }
        }
        const uint s = blockScored ? uint(steps) : 0u;
        uint acceptBits = 0;
        for (uint sig = 0; sig < kBeamProbeSignatures; ++sig)
        {
            const uint failing = (WaveActiveBallot(scored && (matches & (1u << sig)) == 0)[word] & groupMask) >> shift;
            for (uint policy = 0; policy < kBeamProbePolicies; ++policy)
            {
                const uint witnesses = beamProbePolicyMask(policy, per, g, nearest, second);
                const bool accept = blockScored && !blockFresh && witnesses != 0 && (failing & witnesses) == 0;
                const uint acceptSteps = WaveActiveSum(accept ? s : 0u);
                const uint badSteps = WaveActiveSum(accept && !blockOk ? s : 0u);
                if (WaveIsFirstLane())
                {
                    uint unused;
                    const uint slot = kBeamProbeMatrix + 2u * (sig * kBeamProbePolicies + policy);
                    InterlockedAdd(hstrBeamCounts[slot], acceptSteps, unused);
                    InterlockedAdd(hstrBeamCounts[slot + 1], badSteps, unused);
                }
                for (uint j = 0; j < 3; ++j)
                    if (sig == kBeamProbeAcceptSignatures[j] && accept)
                        acceptBits |= 1u << (j * kBeamProbePolicies + policy);
                // An accepted block keeps EVERY old value, its witnesses' too, even where they were marched in full: the next build
                // must compare a witness with the value from the camera the carried rays came from, or a slow drift passes every
                // per-frame check.
                if (params.beamShadowCarry == 1u + sig * kBeamProbePolicies + policy)
                    shadow = accept && scored;
            }
        }
        // What the witnesses themselves would have marched: the extinction steps of each policy's rays, from this camera.
        for (uint policy = 0; policy < kBeamProbePolicies; ++policy)
        {
            const bool witness = blockScored && (beamProbePolicyMask(policy, per, g, nearest, second) & (1u << local)) != 0;
            const uint cost = WaveActiveSum(witness ? uint(witnessSteps) : 0u);
            if (WaveIsFirstLane())
            {
                uint unused;
                InterlockedAdd(hstrBeamCounts[kBeamProbeWitnessSteps + policy], cost, unused);
            }
        }
        const uint marching = (WaveActiveBallot(march)[word] & groupMask) >> shift;
        if (march && local == firstbitlow(marching))
            hstrBeamProbeAccept[beamGuardCell(position)] = acceptBits;
        return shadow;
    }

    /// Groups the dirty query's lanes by block (beamDirtyRay: rb * rb * passes consecutive rays) and counts whole blocks.
    /// MEASURED (4K, OCT_T001, parallax 1, 16 frames, dirty_march_probe.py): blocks whose every re-marched ray was within 0.02 in
    /// place hold 71.1% of the query's steps at walk and 67.6% at sprint, the ceiling of a coherent block carry. A signature at
    /// opacity 0.02 / distance 2% accepts 75.3% of walk steps, 11.8% of them in blocks with a bad ray, and 59.5% of sprint's, 2.9%
    /// bad; at 0.05 / 10% it accepts 82.0% / 80.2% with 16.2% / 16.5% bad. So the signal separates, but a validator that marches
    /// every ray for it pays the extinction half of a step (march_components.py) to save at most ~60%: it must be sparse or coarse.
    void beamProbeBlock(bool ok, bool tight, bool loose, bool scored, float steps)
    {
        const uint rb = max(params.beamDirtyPointStride, 1u);
        const uint g = rb * rb * (params.beamCentreless != 0 ? 1u : 2u);
        if (g > 32u || (32u % g) != 0u)
            return;
        const uint lane = WaveGetLaneIndex();
        const uint start = lane - lane % g;
        const uint word = start / 32u;
        const uint shift = start % 32u;
        const uint groupMask = (g == 32u ? ~0u : ((1u << g) - 1u)) << shift;
        const uint active = WaveActiveBallot(true)[word] & groupMask;
        const bool blockOk = (WaveActiveBallot(!ok)[word] & groupMask) == 0;
        const bool blockTight = (WaveActiveBallot(!tight)[word] & groupMask) == 0;
        const bool blockLoose = (WaveActiveBallot(!loose)[word] & groupMask) == 0;
        const bool blockScored = (WaveActiveBallot(scored)[word] & groupMask) != 0;
        const bool lead = lane == firstbitlow(active);
        const uint s = blockScored ? uint(steps) : 0u;
        const uint blocks = WaveActiveCountBits(lead && blockScored);
        const uint okBlocks = WaveActiveCountBits(lead && blockScored && blockOk);
        const uint okSteps = WaveActiveSum(blockOk ? s : 0u);
        const uint allSteps = WaveActiveSum(s);
        const uint tightSteps = WaveActiveSum(blockTight ? s : 0u);
        const uint tightBad = WaveActiveSum(blockTight && !blockOk ? s : 0u);
        const uint looseSteps = WaveActiveSum(blockLoose ? s : 0u);
        const uint looseBad = WaveActiveSum(blockLoose && !blockOk ? s : 0u);
        if (WaveIsFirstLane())
        {
            uint unused;
            InterlockedAdd(hstrBeamCounts[kBeamProbeBlocks], blocks, unused);
            InterlockedAdd(hstrBeamCounts[kBeamProbeBlocks + 1], okBlocks, unused);
            InterlockedAdd(hstrBeamCounts[kBeamProbeBlocks + 2], okSteps, unused);
            InterlockedAdd(hstrBeamCounts[kBeamProbeBlocks + 3], allSteps, unused);
            InterlockedAdd(hstrBeamCounts[kBeamProbeBlocks + 4], tightSteps, unused);
            InterlockedAdd(hstrBeamCounts[kBeamProbeBlocks + 5], tightBad, unused);
            InterlockedAdd(hstrBeamCounts[kBeamProbeBlocks + 6], looseSteps, unused);
            InterlockedAdd(hstrBeamCounts[kBeamProbeBlocks + 7], looseBad, unused);
        }
    }
#endif

    /// The residual march itself, for a unit beamUnitNeedsMarch has passed.
    /// PROBE (HSTR_STRIP): keeps a stripped march's work without writing it - the condition cannot hold for a real sample, but
    /// the compiler cannot know that.
    void beamStripSink(BeamSample s)
    {
        if (s.depth + s.distance + s.steps + dot(s.cache + s.single, 1.f) == -1.f)
            InterlockedAdd(hstrBeamDirtyCount[2], 1u);
    }

    void marchBeamUnitExact(uint2 unit)
    {
        const float3 w = beamRayDirection(float2(unit) + 0.5f);
        const float3 albedo = gScene.getGridVolume(0).data.albedo;
        const BeamSample exact = marchBeam(w, float2(0.f, 1.f));
#if HSTR_STRIP != 0
        beamStripSink(exact);
        return;
#endif
        hstrBeamPixelOutput[unit] = float4(beamRadiance(exact.cache, exact.single, exact.depth, w, albedo), exact.distance);
#if HSTR_BEAM_REPAIR_PROBE
        beamProbeUnit(unit, w, hstrBeamPixelOutput[unit].rgb, exact.steps);
#endif
        // A failed tile's unit can see nearer cloud than any basis point of its block, and the block is only as safe as its
        // nearest sample.
        if (params.beamGuard != 0 && params.beamRefFrame != 0)
        {
            const uint2 cell = unit / max(params.beamDirtyBlockEdge, 1u);
            const uint depth = asuint(exact.distance > 0.f ? exact.distance : 3.4e38f);
            uint previous;
            InterlockedMin(hstrBeamGuardDepth[cell], depth, previous);
            if (depth < previous)
                beamGuardDropCoarse(cell);
        }
    }

    void marchBeamGridPixel(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        // The level map lives in the beam image; the march itself is the screen pixel's own ray, at full resolution. That keeps the
        // one part of the frame the basis cannot represent free of any resampling.
        float2 beamPosition;
        if (!beamPixelPosition(float2(pixel) + 0.5f, beamPosition))
            return;
        if (hstrBeamLevel[uint2(beamPosition) / beamTileEdge(params.beamLevels - 1)] < params.beamLevels)
            return;
        marchBeamAt(pixel);
    }

    /// Root lattice query at grid point p as a 2D dispatch (beamGridDispatch): a tile corner, or a tile centre with beamGridCentres.
    void buildBeamGridQuery(uint2 gridPoint)
    {
        const bool centre = params.beamGridCentres != 0;
        // Either the dispatch covers a region and a thread's point is its index plus the origin, or it is GENERATED and the
        // thread maps to one of the blocks this build's refresh phase selects (see beamGridBlocks). beamRefreshes still decides
        // in both cases; generating only avoids launching threads whose answer is already known to be "carry".
        uint2 point;
        if (params.beamGridBlocks != 0)
        {
            const uint rb = max(params.beamRefreshBlock, 1u);
            const uint2 block = params.beamGridOrigin + (gridPoint / rb) * params.beamGridBlockStride;
            point = block * rb + (gridPoint % rb);
        }
        else
            point = gridPoint + params.beamGridOrigin;
        buildBeamQueryPoint(point, centre);
    }

    /// The query itself, once a dispatch has decided which point this thread owns. Shared by the swept and generated grid
    /// dispatches and by the dirty-block dispatch, so all three ask the same question of a point.
    void buildBeamQueryPoint(uint2 point, bool centre) { buildBeamQueryPoint(point, centre, false); }

    /// trusted means the dispatch has already decided this point's region is on screen, so the point does not re-test it.
    ///
    /// The two tests are not the same test. A block asks whether a box of 32 beam texels is on screen; a point asks about a box
    /// of 16, centred somewhere else. Where they disagree the block is dispatched, every one of its points returns here, nothing
    /// reaches the code that vouches for the block, and the block is listed as invalidated again on the next frame and for ever
    /// after. Measured: 77,895 dirty threads ran and produced zero verifications, so all 2,434 blocks were re-listed every frame.
    /// Trusting the coarser test is the conservative direction - it can only do work that was not needed, never skip work that was.
    void buildBeamQueryPoint(uint2 point, bool centre, bool trusted)
    {
        uint2 position;
        if (!beamQueryNeedsMarch(point, centre, trusted, position))
            return;
        const BeamSample q = marchBeam(beamRayDirection(float2(position)), float2(0.f, 1.f));
        beamQueryMarched(position, q);
    }

    /// What a marched query writes: the point itself, and the guard entry it is now entitled to vouch for.
    void beamQueryMarched(uint2 position, BeamSample q)
    {
        writeBeamQuery(position, float4(q.cache, q.depth), float4(q.single, q.distance), q.steps);
        beamGuardVerified(position, q.distance);
#if HSTR_PUSH_SHARE
        uint listed, stride;
        hstrPushShareRays.GetDimensions(listed, stride);
        uint slot;
        InterlockedAdd(hstrPushCounts[kPushShareQueries], 1u, slot);
        if (slot < listed)
            hstrPushShareRays[slot] = position.x | (position.y << 16);
#endif
        if (params.beamRefreshDebug == 7)
        {
            uint unused;
            InterlockedAdd(hstrBeamCounts[kBeamMaxLevels + 1], uint(q.steps), unused);
        }
    }

    /// Everything a query does short of marching: the range and screen tests, the certified early-out and the reprojected carry,
    /// each with its own writes. Returns whether the point still has to be marched, and where. Kept apart from the march so a
    /// split-ray dispatch can run it once per ray and share the answer, rather than on every slice.
    bool beamQueryNeedsMarch(uint2 point, bool centre, bool trusted, out uint2 position)
    {
        position = 0;
        if (any(point >= (centre ? params.beamTileDims : params.beamTileDims + 1u)))
            return false;
        if (centre && params.beamAdaptiveRoot != 0 && params.beamLevels >= 2 && beamRootDifficult(point))
            return false;
        const uint s = params.beamTileSize;
        position = point * s + (centre ? s / 2 : 0u);
        if (!trusted && !beamPointOnScreen(position))
            return false;
        // A parked camera cannot move a persistent point: it already holds the value, measured from this very camera, so the
        // carry has nothing to read, nothing to test and nothing to write.
        const bool persistent = params.beamRefFrame != 0;
        // beamAssumeCarry answers "what if a translated point could be certified as surviving without testing it": it takes the
        // same early-out a parked camera takes. The result is not a correct image, only a floor on what the test costs.
        // beamGuard is that early-out earned rather than assumed - see beamBlockCertified.
        // With the guard, a parked camera keeps a point only while its block is certified: the block can also be invalidated by the
        // content changing under it (invalidateBeamGuardBlock), and a parked camera's certificate is then all that says so.
        const bool stationary = (params.beamStationary != 0 && params.beamGuard == 0) || params.beamAssumeCarry != 0 ||
                                beamBlockCertified(position);
        if (persistent && stationary && params.beamRefValid != 0 && !beamRefreshes(point, centre) &&
            beamLatticeValid(position))
        {
            uint held;
            InterlockedAdd(hstrBeamCounts[kBeamMaxLevels - 1], 1u, held);
            return false;
        }
        float4 depthCache;
        float4 single;
        if (!beamRefreshes(point, centre) && beamReprojectQuery(position, depthCache, single))
        {
            // The carried value keeps the cost last measured here, so the prediction survives until the point is marched again.
            const float carriedSteps =
                params.beamHistoryValid == 0 ? 0.f : (persistent ? beamLatticeHeldAt(position, 2).x : beamLatticePrevAt(position, 2).x);
            if (persistent)
                carryBeamQuery(position, single); // Only the distance moves; the radiance and the mark are already in place.
            else
                writeBeamQuery(position, depthCache, single, carriedSteps);
            // This point was tested in full against this camera, so it is entitled to vouch for its block.
            beamGuardVerified(position, single.a);
            uint unused;
            InterlockedAdd(hstrBeamCounts[kBeamMaxLevels - 1], 1u, unused);
            if (params.beamRefreshDebug == 7)
                InterlockedAdd(hstrBeamCounts[kBeamMaxLevels + 2], uint(carriedSteps), unused);
            return false;
        }
        if (params.beamQueue != 0)
        {
            beamEnqueue(position);
            return false;
        }
        return true;
    }

    /// The dirty query's ray for a thread index: which listed block, which of its points, and whether it is a corner or a centre.
    /// Corners and centres go in ONE dispatch - they were two, and at a few hundred rays each dispatch costs the latency of its
    /// longest ray, so two back to back cost two of them.
    bool beamDirtyRay(uint ray, out uint2 point, out bool centre)
    {
        const uint rb = max(params.beamDirtyPointStride, 1u);
        const uint per = rb * rb;
        const uint passes = params.beamCentreless != 0 ? 1u : 2u;
        const uint index = ray / (per * passes);
        point = 0;
        centre = false;
        if (index >= min(hstrBeamDirtyCount[0], params.beamDirtyCapacity))
            return false;
        const uint local = ray % per;
        centre = ((ray / per) % passes) == 1u;
        const uint packed = hstrBeamDirty[index];
        point = uint2(packed & 0xFFFFu, packed >> 16) * rb + uint2(local % rb, local / rb);
        return true;
    }

    /// beamShadowCarry: a block the probe carries is certified again from HERE, with its values left as they are. Leaving its guard
    /// camera where the values came from instead kept it listed on every later build - it can never re-certify - so the dirty set
    /// only grew (4K walk, shadow carry: 52k -> 117k blocks a build). Drift still cannot pass unseen: the lattice keeps the anchor's
    /// values, so when this certificate expires the block is compared with those, not with anything written since.
    void beamShadowCertify(uint2 cell)
    {
        hstrBeamGuardCamera[cell] = float4(gScene.camera.data.posW, 1.f);
        beamGuardDropCoarse(cell);
    }

    void marchBeamAt(uint2 pixel)
    {
        const float3 w = cameraRayDirection(float2(pixel) + 0.5f);
        const float3 albedo = gScene.getGridVolume(0).data.albedo;
        float4 value;
        // The per-pixel fallback keeps marching in the reference frame: it is the part of the image the basis cannot carry, its
        // carry resamples the previous SCREEN, and isolating the basis is what makes the frame's effect measurable.
        if (params.beamRefFrame == 0 && params.beamRefresh != 0 &&
            !beamRefreshes(pixel / beamTileEdge(params.beamLevels - 1)) && beamReprojectPixel(pixel, w, value))
        {
            color[pixel] = float4(value.rgb, 1.f);
            hstrBeamPixelOutput[pixel] = value;
            uint unused;
            InterlockedAdd(hstrBeamCounts[kBeamMaxLevels], 1u, unused);
            return;
        }
        const BeamSample exact = marchBeam(w, float2(0.f, 1.f));
        const float3 radiance = beamRadiance(exact.cache, exact.single, exact.depth, w, albedo);
        color[pixel] = float4(radiance, 1.f);
        // With beamScreenResidual the beam image is not the screen, so a screen-indexed write would land on unrelated directions.
        if (params.beamRefresh != 0 && params.beamScreenResidual == 0)
            hstrBeamPixelOutput[pixel] = float4(radiance, exact.distance);
    }

    // ---------------------------------------------------------------------------------------------
    // Beam refresh (beamRefresh): the lattice and the marched pixels are carried from the previous build where they reproject.

    /// Whether grid cell p (a root lattice point, or a finest tile) is marched this build: one cell of an n x n pattern per build.
    bool beamRefreshes(uint2 p, bool centre = false)
    {
        if (params.beamRefresh == 0 || params.beamHistoryValid == 0)
            return true;
        const uint n = centre && params.beamRefreshCentres != 0 ? params.beamRefreshCentres : params.beamRefresh;
        p /= max(params.beamRefreshBlock, 1u);
        // For 2 x 2, cells 0, 3, 1, 2 (packed as nibbles), so that consecutive builds alternate across the diagonal.
        const uint cell = n == 2 ? (0x2130u >> (4u * (params.beamFrame % 4u))) & 0xFu : params.beamFrame % (n * n);
        return (p.x % n) + n * (p.y % n) == cell;
    }

    /// Pixel-corner position of a world point in the previous build's image.
    bool beamPreviousPosition(float3 world, out float2 position)
    {
        position = 0.f;
        const float4 clip = mul(params.beamRefreshDebug == 3 ? gScene.camera.data.viewProjMatNoJitter : params.beamPrevViewProj, float4(world, 1.f));
        if (clip.w <= 1e-6f)
            return false;
        position = (float2(0.5f, -0.5f) * clip.xy / clip.w + 0.5f) * float2(params.frameDim);
        return all(position >= 0.f) && all(position <= float2(params.frameDim));
    }

    /// Distance from the current camera along w of a point the previous build saw at distance d along a ray close to w.
    float beamCurrentDistance(float d, float3 w) { return d + dot(params.beamPrevCamera - gScene.camera.data.posW, w); }

    /// The basis a carry reads. In the reference frame the lattice is PERSISTENT - the index of a basis point is its world
    /// direction, which does not move, so there is nothing for a ping-pong to do and the query pass reads and updates the one
    /// image in place. A point that turns off screen therefore keeps its value and carries again when it turns back, which the
    /// ping-ponged pair could not express. Everywhere else this is the previous build's lattice, as before.
    float4 beamLatticePrevAt(uint2 position, uint slice) { return hstrBeamLatticePrev[beamLatticeTexel(position, slice)]; }

    /// The persistent lattice read through the image the query pass is writing. Kept apart from beamLatticePrevAt because the
    /// screen path calls that one about thirteen times per carried point through beamPreviousLattice, and putting a branch on
    /// beamRefFrame inside it cost the screen carry 1.81 -> 2.81 ms of basis at 4K for a value the compiler cannot hoist.
    float4 beamLatticeHeldAt(uint2 position, uint slice) { return hstrBeamLatticeOutput[beamLatticeTexel(position, slice)]; }

    /// Whether a persistent basis point holds a value marched since the frame was anchored.
    bool beamLatticeValid(uint2 position) { return beamLatticeHeldAt(position, 2).y > 0.5f; }

    /// The guard entry covering a lattice point: blocks are beamRefreshBlock points square, the same blocks the refresh phase uses.
    uint2 beamGuardCell(uint2 position)
    {
        return (position / params.beamLatticeStep) / max(params.beamRefreshBlock, 1u);
    }

    /// Whether the camera is still close enough to where this block was marched from that its values still hold.
    ///
    /// A value in the persistent image is the radiance along a world direction AS SEEN FROM the camera that marched it. From
    /// anywhere else the content behind that direction has shifted, by at most |dP| / distance radians. So a block is valid
    /// while |P - capturedFrom| <= beamGuardParallaxAngle * (the smallest distance any of its samples holds), and past that it
    /// is re-marched, never carried: an in-place carry keeps the radiance of a direction and so is only ever right for a camera
    /// that has not moved.
    ///
    /// This replaced a bound on how far depths within a block could diverge (|dP| * block angle against beamDepthTolerance *
    /// distance), which is a statement about the old per-point depth test and not about the image: at 4K it certified blocks
    /// for tens of thousands of units, nothing was ever invalidated by translation, and walk scored 24.9% and sprint 34.6% of
    /// pixels over 0.02 against the exact march. That was hidden until the benchmark stopped re-anchoring before every scored
    /// frame. Conservative as before: an unverified block is never certified, and a stale minimum is only ever too small.
    bool beamBlockCertified(uint2 position) { return beamCellCertified(beamGuardCell(position)); }

    bool beamCellCertified(uint2 cell)
    {
        if (params.beamGuard == 0 || params.beamRefFrame == 0 || params.beamHistoryValid == 0)
            return false;
        const float radius = beamCellRadius(cell);
        return radius > 0.f && length(gScene.camera.data.posW - hstrBeamGuardCamera[cell].xyz) <= radius;
    }

    /// Level L of the guard's min-depth pyramid: its offset in hstrBeamGuardPyramid and its dimensions.
    void beamGuardPyramidLevel(uint level, out uint offset, out uint2 dims)
    {
        offset = 0;
        dims = params.beamGuardDims;
        for (uint l = 0; l < level; ++l)
        {
            offset += dims.x * dims.y;
            dims = (dims + 1u) / 2u;
        }
    }

    // The pyramid itself is built by buildBeamGuardPyramidTiles / buildBeamGuardPyramidTop: level 0 from the guard depths (never
    // verified: unbounded), each level above the 2 x 2 minimum.

    /// Appends the block to this build's work list if translation could have invalidated anything in it.
    ///
    /// This is what stops the renderer sweeping. Certifying a block per point made each of the million swept threads cheap but
    /// still launched them: measured, a walking camera dispatched 1,111,541 query threads and 8,880,400 residual threads a frame
    /// where a parked one dispatched 32 and 256. Testing the certificate once per block instead, and dispatching only the blocks
    /// that fail, makes translation cost what it invalidates rather than what it might have.
    ///
    /// The list is sized to hold every block, so it cannot overflow and there is no fallback path to get wrong.
    /// The radius a leaf is safe within: the tolerance on its nearest sample, over the block's angular reach. An unverified block
    /// has none, and an empty one - no sample hit anything - has an unbounded one, which is what lets empty sky collapse.
    ///
    /// Everything that can shift into a block has to be counted, not just the block's own content. Bounded by its own depths, a
    /// block of empty sky was certified for ever and cloud slid into it unseen (4K walk: edges cut into a sawtooth). Bounded by
    /// its ring of neighbours it still missed content that crosses more than a block between checks: a walk anchored 128 units
    /// back lost a whole small cloud, 1.05% of pixels over 0.02. So the bound is taken at every scale of a min-depth pyramid:
    /// the 3 x 3 neighbourhood at level L holds everything within 2^L blocks, and content from beyond level L - 1's reach has to
    /// travel at least 2^(L-1) blocks to arrive, which from distance d takes a camera move of that angle times d. The ring is
    /// held to beamGuardParallaxAngle instead, the error a certified block may carry. A move m also brings content nearer, to
    /// no less than d - m, so each level allows m <= a d / (1 + a).
    float beamCellRadius(uint2 cell)
    {
        if (hstrBeamGuardDepth[cell] == 0xFFFFFFFFu || hstrBeamGuardCamera[cell].w < 0.5f)
            return -1.f;
        const float blockAngle = float(params.beamDirtyBlockEdge) * 1.20f * 2.f / float(max(params.beamFrameDim.x, 1u));
        float radius = 3.4e38f;
        uint offset = 0;
        uint2 dims = params.beamGuardDims;
        for (uint level = 0; level < max(params.beamGuardPyramidLevels, 1u); ++level)
        {
            const int2 at = int2(cell >> level);
            float nearest = 3.4e38f;
            for (int y = -1; y <= 1; ++y)
                for (int x = -1; x <= 1; ++x)
                {
                    const int2 n = at + int2(x, y);
                    if (all(n >= 0) && all(n < int2(dims)))
                        nearest = min(nearest, hstrBeamGuardPyramid[offset + uint(n.y) * dims.x + uint(n.x)]);
                }
            if (!(nearest > 0.f))
                return -1.f;
            const float budget = level == 0 ? params.beamGuardParallaxAngle : float(1u << (level - 1u)) * blockAngle;
            // MEASURED and REMOVED (budgetcertify1, 4K, 2x steps + tstep + beamOctScale 0.5, with beamWarp): a differential
            // certificate for the warp - a block held while camera motion times the spread of inverse cloud depth around it
            // (1 / near - 1 / far over a max-depth pyramid beside this one, sky excluded, every level) stayed under the budget, and a
            // listed block restarted its depth range. Walk at 0.25 / 0.5 / 1 / 2 texels: 8,275 - 5,999 dirty blocks (15,313 at
            // parallax 1) but 1.578 / 2.225 / 2.999 / 3.874% of pixels over 0.02, against parallax 8 + warp's 0.960% at 11,462 -
            // it held blocks the warp cannot correct (the depths are one opacity-weighted distance per ray, so a ray through a
            // thick cloud reads as one depth). Sprint held 28% of its 30,027 blocks for the first time, and saved nothing: query
            // 0.82 -> 0.79 ms, units unchanged - the blocks it can hold are the cheap ones next to sky, and near dense cloud expires
            // at 20 units a frame under any certificate.
            if (nearest < 3.4e38f)
                radius = min(radius, budget * nearest / (1.f + budget));
            offset += dims.x * dims.y;
            dims = (dims + 1u) / 2u;
        }
        return radius;
    }

    /// The beam-image box a coarse cell covers, so its on-screen test is the same test its children would have run.
    void beamCoarseBox(uint2 coarse, out float2 low, out float2 high)
    {
        const float edge = float(params.beamDirtyBlockEdge << params.beamCoarseShift);
        low = float2(coarse) * edge - 2.f * float(params.beamTileSize);
        high = float2(coarse) * edge + edge + 2.f * float(params.beamTileSize);
    }

    bool beamCoarseCertified(uint2 coarse)
    {
        // The same preconditions as a leaf's: without history there is nothing a certificate could be vouching for.
        if (params.beamGuard == 0 || params.beamRefFrame == 0 || params.beamHistoryValid == 0)
            return false;
        const float4 held = hstrBeamCoarse[coarse];
        return held.w > 0.f && length(gScene.camera.data.posW - held.xyz) <= held.w;
    }

    /// The children of a coarse cell that lie inside the guard grid, as a mask of the same layout as its state.
    uint beamCoarseInGrid(uint2 coarse)
    {
        const uint side = 1u << params.beamCoarseShift;
        const uint2 reach = min(params.beamGuardDims - min(coarse * side, params.beamGuardDims), uint2(side, side));
        const uint row = (1u << reach.x) - 1u;
        uint mask = 0;
        for (uint y = 0; y < reach.y; ++y)
            mask |= row << (y * side);
        return mask;
    }

    /// Queues a coarse cell for rebuilding once per build, however many of its children ask.
    void beamQueueRebuild(uint2 coarse)
    {
        uint previous;
        InterlockedOr(hstrBeamCoarseState[coarse], kBeamCoarseQueued, previous);
        if ((previous & kBeamCoarseQueued) != 0)
            return;
        uint slot;
        InterlockedAdd(hstrBeamDescendCount[1], 1u, slot);
        if (slot < params.beamCoarseCapacity)
            hstrBeamRebuild[slot] = (coarse.y << 16) | (coarse.x & 0xFFFFu);
    }

    /// One thread per coarse cell. It lists, as individual blocks, only the children its certificate cannot answer for.
    ///
    /// A cell's certificate is not all-or-nothing. It carries a mask of the children it does NOT cover - never verified, off the
    /// screen when it was built, or already expired then - and while the certificate holds only those are classified. Before
    /// this, one such child zeroed the whole cell and its fifteen verified siblings were classified again on every frame; a cell
    /// straddling the screen edge could never be certified at all, because its off-screen children are never verified.
    ///
    /// The uncovered children are NOT assumed safe: they are simply the only ones still asked.
    void classifyBeamCoarseCell(uint2 coarse)
    {
        if (any(coarse >= params.beamCoarseDims))
            return;
        const bool held = beamCoarseCertified(coarse);
        const uint mask = (held ? hstrBeamCoarseState[coarse] & kBeamCoarseChildren : kBeamCoarseChildren) & beamCoarseInGrid(coarse);
        if (mask == 0)
            return;
        float2 low, high;
        beamCoarseBox(coarse, low, high);
        if (!beamBoxOnScreen(low, high))
            return;
        // An expired certificate has to be re-anchored even if none of its children turns out dirty, or the cell would descend
        // on every frame from now on. A held one is rebuilt only if one of its uncovered children changes (classifyBeamLeaf).
        if (!held)
            beamQueueRebuild(coarse);
        uint base;
        InterlockedAdd(hstrBeamDescendCount[0], countbits(mask), base);
        const uint side = 1u << params.beamCoarseShift;
        for (uint bits = mask; bits != 0; bits &= bits - 1u, ++base)
        {
            const uint child = firstbitlow(bits);
            const uint2 cell = coarse * side + uint2(child % side, child / side);
            if (base < params.beamDirtyCapacity)
                hstrBeamDescend[base] = (cell.y << 16) | (cell.x & 0xFFFFu);
        }
    }

    /// One thread per block the coarse level could not answer for. A block that turns out dirty will be re-verified by this
    /// build's query, so its cell is queued to take a new certificate from it: rebuilds follow the blocks that changed.
    void classifyBeamLeaf(uint index)
    {
        if (index >= min(hstrBeamDescendCount[0], params.beamDirtyCapacity))
            return;
        const uint packed = hstrBeamDescend[index];
        const uint2 cell = uint2(packed & 0xFFFFu, packed >> 16);
        if (classifyBeamGuardCell(cell))
            beamQueueRebuild(cell >> params.beamCoarseShift);
    }

    /// Rebuilds a queued cell's certificate from its children, after this build has re-verified the ones that needed it.
    ///
    /// The anchor is the current camera and the radius min over covered children of (child radius - how far the child was
    /// verified from here), which by the triangle inequality puts the cell's sphere inside every covered child's. A child with
    /// no radius, or none left from here, is left uncovered rather than allowed to zero the cell: it goes into the mask and is
    /// classified on its own while the rest stay certified. With nothing covered the radius is unbounded and the mask is full,
    /// which is the same as descending into every child.
    void updateBeamCoarseCell(uint index)
    {
        if (index >= min(hstrBeamDescendCount[1], params.beamCoarseCapacity))
            return;
        const uint packed = hstrBeamRebuild[index];
        const uint2 coarse = uint2(packed & 0xFFFFu, packed >> 16);
        const uint side = 1u << params.beamCoarseShift;
        const float3 anchor = gScene.camera.data.posW;
        float radius = 3.4e38f;
        uint uncovered = 0;
        for (uint bits = beamCoarseInGrid(coarse); bits != 0; bits &= bits - 1u)
        {
            const uint child = firstbitlow(bits);
            const uint2 cell = coarse * side + uint2(child % side, child / side);
            const float leaf = beamCellRadius(cell);
            const float reach = leaf - length(hstrBeamGuardCamera[cell].xyz - anchor);
            if (leaf < 0.f || !(reach > 0.f))
                uncovered |= 1u << child;
            else
                radius = min(radius, reach);
        }
        hstrBeamCoarse[coarse] = float4(anchor, radius);
        hstrBeamCoarseState[coarse] = uncovered; // Also clears the queued bit for the next build.
    }

    /// Lists the block if translation could have invalidated anything in it, and says whether it did.
    bool classifyBeamGuardCell(uint2 thread)
    {
        const uint2 cell = thread + params.beamGuardOrigin;
        if (any(cell >= params.beamGuardDims))
            return false;
        // Exactly the test that decides whether this block's points are built (beamPointOnScreen), so a block is listed if and
        // only if the build would have touched it.
        if (!beamGuardBlockOnScreen(cell, beamGuardReachBlocks()))
            return false;
        if (beamCellCertified(cell))
        {
            // Held: read from an older camera, so the warp has something to do (beamWarpAuto reads the count back).
            const uint held = WaveActiveCountBits(true);
            if (WaveIsFirstLane())
                InterlockedAdd(hstrBeamDirtyCount[2], held);
            return false;
        }
        // Split the reason, because the two have opposite fixes: a block the guard knows nothing about is one whose points never
        // reached the code that vouches for them, and a block whose bound failed is doing its job.
        if (hstrBeamGuardDepth[cell] == 0xFFFFFFFFu || hstrBeamGuardCamera[cell].w < 0.5f)
        {
            uint unverified;
            InterlockedAdd(hstrBeamCounts[kBeamDirtyUnverified], 1u, unverified);
        }
        uint slot;
        InterlockedAdd(hstrBeamDirtyCount[0], 1u, slot);
        if (slot < params.beamDirtyCapacity)
        {
            hstrBeamDirty[slot] = (cell.y << 16) | (cell.x & 0xFFFFu);
            uint first;
            InterlockedMin(hstrBeamDirtyMark[cell], slot, first);
        }
        uint counted;
        InterlockedAdd(hstrBeamCounts[kBeamDirtyBlocks], 1u, counted);
        return true;
    }

    /// Appends this build's refresh-phase blocks to the dirty list, so the refresh is marched by the split dirty query like
    /// everything else the guard lists. One thread per guard block of each phase block (beamGridOrigin + k * beamGridBlockStride,
    /// in refresh blocks), which is a handful.
    ///
    /// The refresh used to go through the grid query at one thread per ray: a few hundred long rays, so it cost the latency of
    /// the longest - 4K flick over a prebuilt sphere, 0.4-0.9 ms of queries and 0.1-0.4 of march whenever the phase landed on
    /// dense cloud, which it does parked as well.
    void listBeamRefreshBlock(uint2 thread)
    {
        const uint edge = max(params.beamDirtyBlockEdge, 1u);
        const uint g = max(max(params.beamRefreshBlock, 1u) * params.beamTileSize / edge, 1u);
        const uint2 cell = (params.beamGridOrigin + (thread / g) * params.beamGridBlockStride) * g + thread % g;
        if (any(cell >= params.beamGuardDims) || !beamGuardBlockOnScreen(cell, beamGuardReachBlocks()))
            return;
        uint slot;
        InterlockedAdd(hstrBeamDirtyCount[0], 1u, slot);
        if (slot < params.beamDirtyCapacity)
        {
            hstrBeamDirty[slot] = (cell.y << 16) | (cell.x & 0xFFFFu);
            uint first;
            InterlockedMin(hstrBeamDirtyMark[cell], slot, first);
        }
        // Its points are about to be verified again, which zeroes the coarse certificate over it (beamGuardVerified).
        if (params.beamCoarseShift != 0)
            beamQueueRebuild(cell >> params.beamCoarseShift);
    }

    /// The dirty unit march: one thread per unit the dirty tile pass listed (listBeamFailedTileUnits), every one of which marches.
    /// Marching in place instead, 4K walk: 30.0M threads for 413k marches (1.4%) scattered through them, so nearly every wave paid
    /// a full march for one or two lanes - 20.8 ms of a 28 ms frame.
    void marchBeamDirtyListedUnit(uint thread)
    {
        if (thread >= hstrBeamDirtyCount[1])
            return;
        const uint packed = hstrBeamDirtyUnits[thread];
        marchBeamUnitExact(uint2(packed & 0xFFFFu, packed >> 16));
    }

    // MEASURED and REMOVED (2026-09-22): letting each sample the guard lists decide for itself whether to march, from its own history.
    // Near cloud shifts by more than a block every frame at walk (90% of the listed blocks with content, all of them at sprint), so
    // no per-block budget can hold it - one learned per block up to the pyramid's next level changed nothing, 0.136% / 15.9 ms either
    // way - yet 91.5% of the rays it re-marched were within 0.02 of their old value. Each lattice point and unit kept the build it
    // was marched at and a camera distance learned from how much it had changed over the distance before (tolerance * d / change,
    // at most twice the distance it had held over), and was kept while the camera stayed within it: 75% of the rays were kept, but
    // 4K walk scored 2.890% of pixels over 0.02 (p99.9 0.474) against 0.136%, sprint 5.486% against 0.025%, for 16.1 -> 14.2 ms and
    // 15.9 -> 15.8. The rays it kept were the cheap ones - the query only went 6.3 -> 4.6 ms - and a carried corner beside a fresh
    // centre trips the surplus test, so the tiles failing it rose by half. A change is not linear in the camera distance.

    /// Records what a point that took the full test this build proves about its block: the camera it was verified from, and a
    /// running minimum of the distances its points hold. The minimum is never raised - a block that has drifted nearer keeps the
    /// smaller value and is simply re-verified sooner, which costs work rather than correctness.
    void beamGuardVerified(uint2 position, float distance)
    {
        if (params.beamGuard == 0)
            return;
        const uint2 cell = beamGuardCell(position);
        // A sample that hit nothing has no parallax to fail, so it stands in as infinitely far rather than recording nothing.
        // Skipping it instead left every block of empty sky - most of a cloud sea - without a depth, and a block with no depth is
        // never certified, so it was re-listed as invalidated every frame for ever. Measured: 12,349 dirty blocks out of 6,195
        // that exist, counted twice by the double-buffered readback. That is the whole of the residual's cost under translation.
        uint previous;
        InterlockedMin(hstrBeamGuardDepth[cell], asuint(distance > 0.f ? distance : 3.4e38f), previous);
        // Every thread of the block writes the same camera, so the race between them has one outcome.
        hstrBeamGuardCamera[cell] = float4(gScene.camera.data.posW, 1.f);
        // The coarse certificate over this block was derived from the sphere just replaced, so it no longer follows. A block the
        // dirty list reached has its cell queued and rebuilt after this pass anyway; this is for the paths that verify blocks
        // behind the hierarchy's back - a refreshing point, a full sweep - whose cells would otherwise keep vouching for them.
        beamGuardDropCoarse(cell);
    }

    /// Unverifies the block if any of its directions, from the camera it was verified from, can pass through a region whose content
    /// changed (hstrBeamInvalidations), so the classifier lists it and its points and units are marched again.
    ///
    /// The certificate only ever bounded CAMERA motion. The sea replaces tiles as the camera crosses them and rebuilds the sun pages
    /// of tiles whose light changed, and a direction marched before that kept the old radiance for as long as the camera did not
    /// move it out of its certificate. Moving, that is a frame or two (a far block's radius is under a unit); parked or turning it
    /// is for ever. MEASURED 4K, parked with the sun moving 0.002 rad a frame: 1.681% of pixels over 0.02 (p99.9 0.086) without
    /// this, 0.021% (p99.9 0.015) with it, at 0.56 -> 0.59 ms. Walk and sprint unchanged (0.136% / 0.025%, within 0.1 ms).
    void invalidateBeamGuardBlock(uint2 cell)
    {
        if (any(cell >= params.beamGuardDims))
            return;
        const float4 verified = hstrBeamGuardCamera[cell];
        if (verified.w < 0.5f)
            return; // Already unverified.
        // The block's cone: its centre direction and the widest angle to any of its corners.
        const float edge = float(max(params.beamDirtyBlockEdge, 1u));
        const float2 lo = float2(cell) * edge;
        const float3 axis = normalize(beamRayDirection(lo + 0.5f * edge));
        float cosCone = 1.f;
        for (uint k = 0; k < 4; ++k)
            cosCone = min(cosCone, dot(axis, normalize(beamRayDirection(lo + edge * float2(k & 1u, k >> 1)))));
        const float cone = acos(clamp(cosCone, -1.f, 1.f));
        // From the camera its values were marched from, and from the current one: the query that re-verifies it marches from here.
        bool hit = false;
        for (uint i = 0; i < params.beamInvalidateCount && !hit; ++i)
        {
            const float4 sphere = hstrBeamInvalidations[i];
            for (uint from = 0; from < 2 && !hit; ++from)
            {
                const float3 v = sphere.xyz - (from == 0 ? verified.xyz : gScene.camera.data.posW);
                const float d = length(v);
                if (d <= sphere.w)
                    hit = true;
                else
                    hit = acos(clamp(dot(axis, v / d), -1.f, 1.f)) <= cone + asin(sphere.w / d);
            }
        }
        if (!hit)
            return;
        hstrBeamGuardCamera[cell] = float4(verified.xyz, 0.f);
        beamGuardDropCoarse(cell);
    }

    /// Zeroes the coarse certificates that may have been derived from this block's depth: its own cell's, and - because a
    /// block's certificate reads its neighbours' depths (beamCellReachDepth) - the cells of the eight around it.
    void beamGuardDropCoarse(uint2 cell)
    {
        if (params.beamCoarseShift == 0)
            return;
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                const int2 n = int2(cell) + int2(x, y);
                if (all(n >= 0) && all(n < int2(params.beamGuardDims)))
                    hstrBeamCoarse[uint2(n) >> params.beamCoarseShift] = float4(0.f);
            }
    }

    /// Updates only the distance a carried point is measured from, so the next build's parallax test still reads it relative to
    /// that build's camera. The radiance and the validity are already in place and are not rewritten.
    void carryBeamQuery(uint2 position, float4 single)
    {
        hstrBeamLatticeOutput[beamLatticeTexel(position, 1)] = single;
    }

    /// Whether the previous camera's ray through the point at distance d along w is within beamParallax pixels of w.
    bool beamParallaxAllows(float d, float3 w)
    {
        if (params.beamParallax <= 0.f || d <= 0.f)
            return true;
        const float3 camera = gScene.camera.data.posW;
        const float3 previous = normalize(camera + w * d - params.beamPrevCamera);
        return length(cross(previous, w)) <= params.beamParallax * params.cloudPixelAngle;
    }

    /// Distance the previous build would have stored, along a ray close to w, for a point at distance d from the current camera.
    float beamPreviousDistance(float d, float3 w) { return d - dot(params.beamPrevCamera - gScene.camera.data.posW, w); }

    /// Whether a previous sample with stored distance sample may stand for a point at previous distance d (0: no opacity) with
    /// interpolation weight weight: it carries no weight, or it saw the same surface - both empty, or distances within
    /// beamDepthTolerance.
    bool beamSampleAgrees(float sample, float d, float weight)
    {
        if (weight <= 1e-3f || params.beamRefreshDebug == 4)
            return true;
        if ((sample > 0.f) != (d > 0.f))
            return false;
        return d <= 0.f || abs(sample - d) <= params.beamDepthTolerance * max(sample, d);
    }

    /// The previous build's root lattice reconstructed at pixel-corner position q, as its resolve would. With check set, valid
    /// where every sample the reconstruction weighs agrees with previous distance d.
    bool beamPreviousLattice(float2 q, float d, bool check, out float4 a, out float4 b)
    {
        const uint s = params.beamTileSize;
        const uint2 tile = min(uint2(q) / s, params.beamTileDims - 1u);
        const uint2 origin = tile * s;
        if (params.beamPageIndirect == 2 && beamPageMissing(origin))
        {
            a = 0.f;
            b = 0.f;
            return false;
        }
        const float2 f = saturate((q - float2(origin)) / float(s));
        const float4 a00 = beamLatticePrevAt(origin, 0);
        const float4 a10 = beamLatticePrevAt(origin + uint2(s, 0), 0);
        const float4 a01 = beamLatticePrevAt(origin + uint2(0, s), 0);
        const float4 a11 = beamLatticePrevAt(origin + uint2(s, s), 0);
        const float4 b00 = beamLatticePrevAt(origin, 1);
        const float4 b10 = beamLatticePrevAt(origin + uint2(s, 0), 1);
        const float4 b01 = beamLatticePrevAt(origin + uint2(0, s), 1);
        const float4 b11 = beamLatticePrevAt(origin + uint2(s, s), 1);
        a = bilinear(a00, a10, a01, a11, f);
        b = bilinear(b00, b10, b01, b11, f);
        const float4 weights = float4((1.f - f.x) * (1.f - f.y), f.x * (1.f - f.y), (1.f - f.x) * f.y, f.x * f.y);
        // The stored distance is interpolated over the samples that saw opacity only: an empty neighbour has no distance.
        const float4 distances = float4(b00.a, b10.a, b01.a, b11.a);
        const float4 seen = weights * float4(distances > 0.f);
        b.a = dot(seen, distances) / max(dot(seen, 1.f), 1e-6f);
        bool valid = !check || (beamSampleAgrees(b00.a, d, weights.x) && beamSampleAgrees(b10.a, d, weights.y) &&
                                beamSampleAgrees(b01.a, d, weights.z) && beamSampleAgrees(b11.a, d, weights.w));
        // A value that blends samples is only as good as the tile's basis: the previous build must have accepted the tile. Without
        // this a turning camera, which reprojects exactly, still tripled the error - resampled through failing tiles, every frame.
        const bool blends = max(max(weights.x, weights.y), max(weights.z, weights.w)) < 1.f - 1e-3f;
        if (check && blends && params.beamRefreshDebug != 5 && hstrBeamLevelPrev[origin / beamTileEdge(params.beamLevels - 1)] >= params.beamLevels)
            valid = false;
        if (check && blends && params.beamCarryTolerance > 0.f && params.beamRefreshDebug != 6)
        {
            // The corners' spread (and the centre's, which the bubble weighs) in log radiance proxy and in transmittance.
            float3 lo = log(1.f + max(a00.rgb + b00.rgb, 0.f));
            float3 hi = lo;
            float tLo = exp(-a00.a);
            float tHi = tLo;
            const float4 ac = beamLatticePrevAt(origin + s / 2, 0);
            const float4 bcSample = beamLatticePrevAt(origin + s / 2, 1);
            const float4 as[4] = { a10, a01, a11, ac };
            const float4 bs[4] = { b10, b01, b11, bcSample };
            for (uint k = 0; k < (params.beamCentreless == 0 ? 4u : 3u); ++k)
            {
                const float3 l = log(1.f + max(as[k].rgb + bs[k].rgb, 0.f));
                lo = min(lo, l);
                hi = max(hi, l);
                tLo = min(tLo, exp(-as[k].a));
                tHi = max(tHi, exp(-as[k].a));
            }
            if (maxComponent(hi - lo) > params.beamCarryTolerance || tHi - tLo > params.beamCarryTolerance)
                valid = false;
        }
        if (params.beamCentreless == 0)
        {
            const float bubble = 16.f * f.x * (1.f - f.x) * f.y * (1.f - f.y);
            const float4 bc = beamLatticePrevAt(origin + s / 2, 1);
            a += (beamLatticePrevAt(origin + s / 2, 0) - 0.25f * (a00 + a10 + a01 + a11)) * bubble;
            b.rgb += (bc.rgb - 0.25f * (b00.rgb + b10.rgb + b01.rgb + b11.rgb)) * bubble;
            valid = valid && (!check || beamSampleAgrees(bc.a, d, bubble));
        }
        return valid;
    }

    /// A lattice query carried from the previous build: found at a first distance (its own previous one, or the middle of the
    /// medium), then at the distance the previous build has there, where every sample it interpolates must agree.
    bool beamReprojectQuery(uint2 position, out float4 depthCache, out float4 single)
    {
        depthCache = 0.f;
        single = 0.f;
        const float3 w = beamRayDirection(float2(position));
        const float3 camera = gScene.camera.data.posW;
        const float stored = params.beamRefFrame != 0 ? beamLatticeHeldAt(position, 1).a : beamLatticePrevAt(position, 1).a;
        // In the reference frame a basis point IS a world direction, and the frame did not turn with the camera, so the previous
        // build marched this very ray and its value sits at this very index. Nothing is reprojected and nothing is resampled: the
        // only thing that can invalidate it is the camera having MOVED, which parallax answers. This is the whole point of the
        // frame - rotation costs nothing, and the resampling that "tripled the error" for a turning camera cannot happen.
        if (params.beamRefFrame != 0)
        {
            // With the guard, a point whose block is certified never gets here (beamQueryNeedsMarch keeps it as it is), so one
            // that does is in a block the camera has moved too far from. Its value is the radiance seen from somewhere else, and
            // keeping it in place with a new distance is exactly what made translation invisible: it is marched.
            if (params.beamGuard != 0)
                return false;
            if (params.beamRefValid == 0 || !beamLatticeValid(position))
                return false;
            depthCache = 0.f; // Already in place: the persistent lattice is not rewritten for a carry.
            single = beamLatticeHeldAt(position, 1);
            if (stored <= 0.f)
                return true; // The ray saw no opacity: no parallax can move what is not there.
            const float moved = beamCurrentDistance(stored, w);
            if (moved <= 0.f || !beamParallaxAllows(moved, w))
                return false;
            single.a = moved;
            return true;
        }
        float d;
        if (stored > 0.f)
            d = beamCurrentDistance(stored, w);
        else
        {
            CutRay ray;
            if (!makeCutRay(camera, w, gScene.getGridVolume(0), ray))
                return true; // Outside the medium: nothing to carry, nothing to march.
            d = ray.entryDistance + 0.5f * ray.worldLength;
        }
        float2 q;
        if (!beamPreviousPosition(camera + w * d, q))
            return false;
        if (params.beamRefreshDebug == 1 && stored > 0.f && length(q - float2(position)) < 0.5f)
        {
            uint unused;
            InterlockedAdd(hstrBeamCounts[2], 1u, unused);
        }
        if (params.beamRefreshDebug == 2 || params.beamRefreshDebug == 3)
        {
            beamPreviousLattice(q, 0.f, false, depthCache, single);
            return true;
        }
        // Onto itself (a camera that did not move): the previous value is this point's own, with nothing resampled to check.
        if (all(abs(q - float2(position)) < 0.01f))
        {
            depthCache = beamLatticePrevAt(position, 0);
            single = beamLatticePrevAt(position, 1);
            single.a = stored > 0.f ? d : 0.f;
            return true;
        }
        // One checked reconstruction where the point's own distance lands (an empty point: where the middle of the medium does, and
        // everything it weighs must be empty too). Every sample it weighs must agree with that distance, so re-centring on the
        // interpolated distance, as a second reconstruction, moved it by less than the depth tolerance - at twice the fetches.
        const float previous = stored > 0.f ? stored : 0.f;
        if (!beamPreviousLattice(q, previous, true, depthCache, single) || !beamParallaxAllows(d, w))
            return false;
        single.a = single.a > 0.f ? beamCurrentDistance(single.a, w) : 0.f;
        return true;
    }

    /// A marched pixel carried from the previous build: every previous pixel its reprojection weighs must have been marched and
    /// agree with the distance.
    bool beamReprojectPixel(uint2 pixel, float3 w, out float4 value)
    {
        value = 0.f;
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const float3 camera = gScene.camera.data.posW;
        // First guess: this pixel's own previous distance where it was marched, else the lattice's here.
        float d;
        if (hstrBeamLevelPrev[pixel / finest] >= params.beamLevels)
            d = hstrBeamPixelPrev[pixel].a;
        else
        {
            const uint s = params.beamTileSize;
            const uint2 origin = pixel / s * s;
            d = max(max(beamLatticeAt(origin, 1).a, beamLatticeAt(origin + uint2(s, 0), 1).a),
                    max(beamLatticeAt(origin + uint2(0, s), 1).a, beamLatticeAt(origin + uint2(s, s), 1).a));
        }
        if (d > 0.f)
            d = beamCurrentDistance(d, w);
        for (uint i = 0; i < 2; ++i)
        {
            float2 q;
            if (d <= 0.f || !beamPreviousPosition(camera + w * d, q))
                return false;
            // Onto itself: this pixel's own previous march, if it had one that saw something.
            if (i == 0 && all(abs(q - float2(pixel) - 0.5f) < 0.01f))
            {
                if (hstrBeamLevelPrev[pixel / finest] < params.beamLevels || hstrBeamPixelPrev[pixel].a <= 0.f)
                    return false;
                value = float4(hstrBeamPixelPrev[pixel].rgb, d);
                return true;
            }
            const float2 texel = q - 0.5f;
            const int2 base = int2(floor(texel));
            const float2 g = texel - float2(base);
            if (any(base < 0) || any(base + 1 >= int2(params.frameDim)) || !beamParallaxAllows(d, w))
                return false;
            const float4 weights = float4((1.f - g.x) * (1.f - g.y), g.x * (1.f - g.y), (1.f - g.x) * g.y, g.x * g.y);
            const float previous = beamPreviousDistance(d, w);
            float4 sum = 0.f;
            float seen = 0.f;
            float3 logLo = 1e30f;
            float3 logHi = -1e30f;
            for (uint k = 0; k < 4; ++k)
            {
                if (weights[k] <= 1e-3f)
                    continue;
                const uint2 p = uint2(base + int2(k & 1u, k >> 1));
                if (hstrBeamLevelPrev[p / finest] < params.beamLevels)
                    return false;
                const float4 v = hstrBeamPixelPrev[p];
                // The first pass only finds the surface; the second requires agreement with it.
                if (i == 1 && !beamSampleAgrees(v.a, previous, weights[k]))
                    return false;
                // Marched pixels are the edges by construction: blending across one is exactly what they were marched to avoid.
                logLo = min(logLo, log(1.f + max(v.rgb, 0.f)));
                logHi = max(logHi, log(1.f + max(v.rgb, 0.f)));
                if (i == 1 && maxComponent(logHi - logLo) > (params.beamCarryTolerance > 0.f ? params.beamCarryTolerance : params.beamTolerance))
                    return false;
                sum.rgb += weights[k] * v.rgb;
                sum.a += v.a > 0.f ? weights[k] * v.a : 0.f;
                seen += v.a > 0.f ? weights[k] : 0.f;
            }
            value = float4(sum.rgb / max(dot(weights, float4(weights > 1e-3f)), 1e-6f), 0.f);
            if (seen <= 0.f)
                return false; // A marched pixel that saw nothing is cheap to march again.
            d = beamCurrentDistance(sum.a / seen, w);
            value.a = d;
        }
        return true;
    }

    /// Full-resolution guide of one beam pixel from an extinction-only march (the stepping of marchSegment, no lighting): x the
    /// optical depth of the whole ray, y the world distance at which it reaches one (0: never), z the sun optical depth there.
    void beamGuidePixel(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        float4 guide = 0.f;
        CutRay ray;
        if (makeCutRay(gScene.camera.data.posW, cameraRayDirection(float2(pixel) + 0.5f), gScene.getGridVolume(0), ray))
        {
            const float cellWidth = float(params.hstrCellWidth);
            const float voxelsPerUnit = length(ray.direction) * cellWidth;
            const float worldPerVoxel = ray.worldLength / max(voxelsPerUnit, 1e-6f);
            const float minDepth = -log(kMinTransmittance);
            float depth = 0.f;
            float t = 0.f;
            float fineVoxels = 1.f;
            uint steps = 0;
            [loop]
            while (voxelsPerUnit > 1e-6f && t < 1.f && depth < minDepth && steps < params.maxMarchSteps)
            {
                const float3 v = (ray.position + t * ray.direction) * cellWidth;
                const float majorant = majorantAt(v);
                const float minimumStep = clamp(fineVoxels, params.cloudFineMinVoxels, 1.f) * params.minStepVoxels;
                const float stepVoxels = clamp(params.stepOpticalDepth / max(1e-6f, majorant * worldPerVoxel), minimumStep, params.maxStepVoxels);
                const float dt = min(stepVoxels / voxelsPerUnit, 1.f - t);
                if (majorant > 0.f)
                {
                    const float3 midpoint = (ray.position + (t + 0.5f * dt) * ray.direction) * cellWidth;
                    const float stepDepth = cameraExtinctionAtVoxel(midpoint, rayFootprint(ray, t + 0.5f * dt), fineVoxels) * dt * ray.worldLength;
                    if (depth < 1.f && depth + stepDepth >= 1.f)
                    {
                        // Where the optical depth crosses one, linearly within the step.
                        const float s = t + dt * (1.f - depth) / max(stepDepth, 1e-6f);
                        guide.y = ray.entryDistance + s * ray.worldLength;
                        const float3 p = (ray.position + s * ray.direction) * cellWidth;
                        guide.z = sunDepthAt(p, rayFootprint(ray, s));
                    }
                    depth += stepDepth;
                }
                t += dt;
                ++steps;
            }
            guide.x = depth;
        }
        hstrBeamGuideOutput[pixel] = guide;
    }

    /// compareReferenceRow on compareBlock x compareBlock pixel means: one row of blocks. Averaging removes most of the
    /// reference's per-pixel noise (and the halves' estimate of it shrinks accordingly), so what remains is systematic error.
    void compareReferenceBlockRow(uint blockRow)
    {
        const uint block = params.compareBlock;
        const uint2 blocks = params.frameDim / block;
        if (blockRow >= blocks.y)
            return;
        float4 error = 0.f;
        uint count = 0;
        for (uint bx = 0; bx < blocks.x; ++bx)
        {
            float3 value = 0.f;
            float3 reference = 0.f;
            float3 first = 0.f;
            float3 second = 0.f;
            for (uint y = 0; y < block; ++y)
                for (uint x = 0; x < block; ++x)
                {
                    const uint2 pixel = uint2(bx, blockRow) * block + uint2(x, y);
                    value += params.compareExact != 0 ? color[pixel].rgb
                                                      : color[pixel].rgb + referenceHalves(pixel, params.compareSubstitute).both;
                    if (params.compareExact != 0)
                    {
                        reference += hstrExactFrame[pixel].rgb;
                        continue;
                    }
                    const ReferenceHalves target = referenceHalves(pixel, params.compareTarget);
                    reference += target.both;
                    first += target.first;
                    second += target.second;
                }
            const float inverse = 1.f / float(block * block);
            value *= inverse;
            reference *= inverse;
            first *= inverse;
            second *= inverse;
            if (any(isnan(value) || isinf(value) || isnan(reference) || isinf(reference)))
                continue;
            error += float4(
                dot(abs(value - reference), 1.f / 3.f),
                dot(abs(log(1.f + max(value, 0.f)) - log(1.f + max(reference, 0.f))), 1.f / 3.f),
                0.5f * dot(abs(first - second), 1.f / 3.f),
                0.5f * dot(abs(log(1.f + max(first, 0.f)) - log(1.f + max(second, 0.f))), 1.f / 3.f)
            );
            ++count;
            if (params.compareMapScale > 0.f)
            {
                // Error map: the signed block log error, red where the frame is brighter than the reference and blue where
                // darker, over the reference's log luminance in grey.
                const float signedError = dot(log(1.f + max(value, 0.f)) - log(1.f + max(reference, 0.f)), 1.f / 3.f);
                const float grey = 0.4f * saturate(log(1.f + dot(max(reference, 0.f), 1.f / 3.f)) / 3.f);
                const float red = saturate(signedError / params.compareMapScale);
                const float blue = saturate(-signedError / params.compareMapScale);
                const float3 map = float3(grey + (1.f - grey) * red, grey * (1.f - max(red, blue)), grey + (1.f - grey) * blue);
                for (uint y = 0; y < block; ++y)
                    for (uint x = 0; x < block; ++x)
                        color[uint2(bx, blockRow) * block + uint2(x, y)] = float4(map, 1.f);
            }
        }
        hstrReferenceRowError[blockRow] = error / float(max(count, 1u));
    }

    // ---------------------------------------------------------------------------------------------
    // Full-resolution resolve: L = (1 - tau) * basis source + tau * background, or the exact per-pixel integral in refined tiles.

    /// The camera pixel of the cloud sea: see renderCloudCamera.
    void executeCamera(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        const float3 direction = cameraRayDirection(float2(pixel) + 0.5f);
        const float3 sky = params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * direction.y));
        const GridVolume volume = gScene.getGridVolume(0);
        CutRay ray;
        if (!makeCutRay(gScene.camera.data.posW, direction, volume, ray))
        {
            color[pixel] = float4((params.hstComponents & kComponentBackground) != 0 ? sky : 0.f, 1.f);
            return;
        }
        ray.sunOctaves.yzw = 0.f;
        float3 cached = 0.f;
        float3 single = 0.f;
        float transmittance = 1.f;
        uint steps = 0;
        const bool withCache = (params.hstComponents & (kComponentSunMultiple | kComponentSkyScattered)) != 0;
        const bool withSingle = (params.hstComponents & kComponentSunSingle) != 0;
        marchSegment(ray, float2(0.f, 1.f), withCache, withSingle, cached, single, transmittance, steps);
        const float3 background = (params.hstComponents & kComponentBackground) != 0 ? transmittance * sky : 0.f;
        color[pixel] = float4((cached + single) * volume.data.albedo + background, 1.f);
    }

    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        const float3 direction = cameraRayDirection(float2(pixel) + 0.5f);
        const float3 sky = params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * direction.y));
        if (gScene.getGridVolumeCount() == 0)
        {
            color[pixel] = float4(sky, 1.f);
            transportError[pixel] = 0.f;
            cutStats[pixel] = 0.f;
            return;
        }

        const GridVolume volume = gScene.getGridVolume(0);
        const AABB bounds = volume.getBounds();
        const float3 origin = gScene.camera.data.posW;
        if (params.cloudDomain == 0 && all(origin > bounds.minPoint) && all(origin < bounds.maxPoint))
        {
            Ray ray;
            ray.origin = origin;
            ray.dir = direction;
            ray.tMin = 0.f;
            ray.tMax = 1e30f;
            float residualError;
            const uint stepCount = min(1024u, (params.baseSteps << params.refinementLevel) << 1);
            color[pixel] = float4(integrateVolume(ray, volume, stepCount, pixel, residualError).rgb, 1.f);
            transportError[pixel] = residualError;
            cutStats[pixel] = 0.f;
            return;
        }

        CutRay ray;
        if (!makeCutRay(origin, direction, volume, ray))
        {
            color[pixel] = float4((params.hstComponents & kComponentBackground) != 0 ? sky : 0.f, 1.f);
            transportError[pixel] = 0.f;
            cutStats[pixel] = 0.f;
            return;
        }

        if (params.debugView == kWorldCacheView)
        {
            // World cache experiment, marched exactly per pixel: the cache supplies sun-multiple plus scattered sky, the fine sun
            // field supplies single scattering (octave 0 only), and the background is transmitted; each gated by hstComponents.
            ray.sunOctaves.yzw = 0.f;
            float3 cached = 0.f;
            float3 single = 0.f;
            float cachedTransmittance = 1.f;
            uint cachedSteps = 0;
            const bool withCache = (params.hstComponents & (kComponentSunMultiple | kComponentSkyScattered)) != 0;
            const bool withSingle = (params.hstComponents & kComponentSunSingle) != 0;
            marchSegment(ray, float2(0.f, 1.f), withCache, withSingle, cached, single, cachedTransmittance, cachedSteps);
            const float3 background = (params.hstComponents & kComponentBackground) != 0 ? cachedTransmittance * sky : 0.f;
            color[pixel] = float4((cached + single) * volume.data.albedo + background, 1.f);
            if (params.marchProbe != 0)
                color[pixel] = float4(probeMarch(ray, params.marchProbe).xxx, 1.f);
            transportError[pixel] = 0.f;
            cutStats[pixel] = 0.f;
            return;
        }

        if (params.debugView == 5)
        {
            // Resident residual page value (cell-averaged sun transmittance) where the ray's opacity reaches one half.
            const float cellWidth = float(params.hstrCellWidth);
            float T = 1.f;
            float t = 0.f;
            float3 debug = 0.f;
            for (uint i = 0; i < 2048 && t < 1.f && T > 0.5f; ++i)
            {
                const float dt = 0.5f / max(1.f, length(ray.direction) * cellWidth);
                const float3 v = (ray.position + (t + 0.5f * dt) * ray.direction) * cellWidth;
                T *= exp(-cameraExtinctionAtVoxel(v, rayFootprint(ray, t + 0.5f * dt)) * dt * ray.worldLength);
                if (T <= 0.5f)
                {
                    debug = exp(-hstrFineSun.SampleLevel(hstrLinearSampler, (seaVoxel(v) + 0.5f) / float3(params.hstrFineDims), 0.f));
                }
                t += dt;
            }
            color[pixel] = float4(debug, 1.f);
            return;
        }

        const uint2 tile = pixel / params.hstrTileSize;
        const uint tileID = tile.x + params.hstrTileDims.x * tile.y;
        const HSTRTileBasis basis = hstrTileBasis[tileID];
        bool refine = basis.info.y != 0.f || params.debugView == 4;
        float transmittance = 1.f;
        uint visited = 0;
        uint steps = 0;
        float3 residual = 0.f;
        float3 radiance = traverseCut(ray, tileID, refine, true, residual, transmittance, visited, steps);
        if (!refine)
        {
            const float2 f = (float2(pixel - tile * params.hstrTileSize) + 0.5f) / float(params.hstrTileSize);
            const float4 fitted = evaluateTileBasis(basis, f);
            const float opacity = 1.f - transmittance;
            if (fitted.a > 1e-3f || opacity <= 1e-3f)
                radiance = max(0.f, fitted.rgb) / max(fitted.a, 1e-4f) * opacity;
            else
            {
                // The tile basis carries no source where this pixel has opacity: exact fallback for this pixel only.
                refine = true;
                transmittance = 1.f;
                residual = 0.f;
                radiance = traverseCut(ray, tileID, true, true, residual, transmittance, visited, steps) * volume.data.albedo;
            }
        }
        else
            radiance *= volume.data.albedo;

        // L = L_base + sum of rasterized residuals + tau * background.
        residual *= volume.data.albedo;
        // Baseline for the HST sky: a constant source at the sphere-averaged sky radiance, weighted by this pixel's opacity.
        if (params.skyAmbient > 0.f)
            radiance = params.skyAmbient * volume.data.albedo * params.skyRadiance * 0.675f * (1.f - transmittance);
        radiance *= params.skyScale;
        if ((params.hstComponents & kComponentSkyScattered) == 0)
            radiance = 0.f;
        const float3 background = (params.hstComponents & kComponentBackground) != 0 ? transmittance * sky : 0.f;
        float3 result = max(0.f, radiance + residual) + background;
        if (params.debugView == 7)
            result = 0.5f + 4.f * residual;
        if (params.debugView == 1 && refine)
            result = lerp(result, float3(1.f, 0.1f, 0.1f), 0.35f);
        else if (params.debugView == 2)
            result = float3(steps / 64.f, visited / 32.f, 0.f);
        else if (params.debugView == 3)
            result = 1.f - transmittance;
        color[pixel] = float4(result, 1.f);
        transportError[pixel] = basis.info.x;
        cutStats[pixel] = float4(refine ? 1.f : 0.f, visited, steps, transmittance);
    }
};

cbuffer CB
{
    HSTRCloudRenderer gHSTRCloud;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.execute(dispatchThreadId.xy);
}

/// The steady-state camera pixel: execute()'s kWorldCacheView branch alone, without the other views, the probes, the reference display
/// or the grid path tracer that share its entry point. Same frames; the point is what the program does NOT contain, since the march's
/// occupancy is set by the worst path through its program (see HSTR_SUN_LIVE).
[numthreads(8, 8, 1)]
void renderCloudCamera(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.executeCamera(dispatchThreadId.xy);
}

groupshared float4 gBeamDepthCache[64];
groupshared float4 gBeamSingle[64];

/// Beam queries of one level: beamSegments consecutive threads march consecutive segments of one query ray from unit
/// transmittance, and the first composes them front to back, (C, T) = (C_a + T_a C_b, T_a T_b).
[numthreads(64, 1, 1)]
void buildBeamQueries(uint3 dispatchThreadId: SV_DispatchThreadID, uint3 groupThreadId: SV_GroupThreadID)
{
    const uint segments = gHSTRCloud.params.beamSegments;
    const uint segment = dispatchThreadId.x % segments;
    uint2 position;
    const bool valid = gHSTRCloud.beamQueryPosition(dispatchThreadId.x / segments, position);
    BeamSample sample = {};
    if (valid)
        sample = gHSTRCloud.marchBeam(
            gHSTRCloud.beamRayDirection(float2(position)), float2(float(segment), float(segment + 1)) / float(segments)
        );
    gBeamDepthCache[groupThreadId.x] = float4(sample.cache, sample.depth);
    gBeamSingle[groupThreadId.x] = float4(sample.single, 0.f);
    GroupMemoryBarrierWithGroupSync();
    if (!valid || segment != 0)
        return;
    float depth = 0.f;
    float3 cache = 0.f;
    float3 single = 0.f;
    for (uint k = 0; k < segments; ++k)
    {
        const float transmittance = exp(-depth);
        cache += transmittance * gBeamDepthCache[groupThreadId.x + k].rgb;
        single += transmittance * gBeamSingle[groupThreadId.x + k].rgb;
        depth += gBeamDepthCache[groupThreadId.x + k].a;
    }
    gHSTRCloud.writeBeamQuery(position, float4(cache, depth), float4(single, 0.f));
}

[numthreads(64, 1, 1)]
void testBeamTiles(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.testBeamTile(dispatchThreadId.xy);
}

[numthreads(8, 8, 1)]
void invalidateBeamGuardBlocks(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.invalidateBeamGuardBlock(dispatchThreadId.xy);
}

groupshared float gPyramid[16 * 16]; ///< buildBeamGuardPyramidTiles: one level of the group's tile at a time.

/// The guard's min-depth pyramid in two dispatches instead of one per level (twelve at 4K, 0.07 ms of launches and barriers:
/// classify/pyramid in policy4). One 16 x 16 group per 32 x 32 tile of level 0 writes levels 0 - 5 of that tile, reducing in
/// groupshared memory. Exact: level L entry j is the minimum over level 0's [j 2^L, (j + 1) 2^L), which the ceil-halving levels
/// cover aligned from the origin, and a cell past the edge is +inf, which the minimum ignores just as the clamped read did.
[numthreads(16, 16, 1)]
void buildBeamGuardPyramidTiles(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    const uint levels = gHSTRCloud.params.beamGuardPyramidLevels;
    const uint2 local = groupThreadId.xy;
    // Level 0: this thread's 2 x 2 cells, then level 1 from them.
    uint offset = 0;
    uint2 dims = gHSTRCloud.params.beamGuardDims;
    float nearest = 3.4e38f;
    for (uint k = 0; k < 4; ++k)
    {
        const uint2 cell = groupId.xy * 32u + local * 2u + uint2(k & 1u, k >> 1);
        if (all(cell < dims))
        {
            const uint stored = gHSTRCloud.hstrBeamGuardDepth[cell];
            const float depth = stored == 0xFFFFFFFFu ? 3.4e38f : asfloat(stored);
            gHSTRCloud.hstrBeamGuardPyramid[cell.y * dims.x + cell.x] = depth;
            nearest = min(nearest, depth);
        }
    }
    // Levels 1 - 5: at level L the tile holds 32 >> L entries a side, the first (32 >> L)^2 threads each one.
    for (uint level = 1; level < min(levels, 6u); ++level)
    {
        offset += dims.x * dims.y;
        dims = (dims + 1u) / 2u;
        const uint side = 32u >> level;
        if (level > 1)
        {
            GroupMemoryBarrierWithGroupSync();
            if (all(local < side))
            {
                const uint2 c = local * 2u;
                const uint below = side * 2u;
                nearest = min(min(gPyramid[c.y * below + c.x], gPyramid[c.y * below + c.x + 1u]),
                              min(gPyramid[(c.y + 1u) * below + c.x], gPyramid[(c.y + 1u) * below + c.x + 1u]));
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (all(local < side))
        {
            gPyramid[local.y * side + local.x] = nearest;
            const uint2 entry = groupId.xy * side + local;
            if (all(entry < dims))
                gHSTRCloud.hstrBeamGuardPyramid[offset + entry.y * dims.x + entry.x] = nearest;
        }
    }
}

/// The levels above 5, which the tiles leave (levels 6 and up of a 560 x 560 guard grid are at most 9 x 9 entries): one group,
/// level after level, each from the one below in memory.
[numthreads(256, 1, 1)]
void buildBeamGuardPyramidTop(uint3 groupThreadId: SV_GroupThreadID)
{
    const uint levels = gHSTRCloud.params.beamGuardPyramidLevels;
    uint below = 0;
    uint2 belowDims = gHSTRCloud.params.beamGuardDims;
    for (uint l = 1; l < 6u && l < levels; ++l)
    {
        below += belowDims.x * belowDims.y;
        belowDims = (belowDims + 1u) / 2u;
    }
    // below / belowDims now describe level 5.
    for (uint level = 6; level < levels; ++level)
    {
        const uint offset = below + belowDims.x * belowDims.y;
        const uint2 dims = (belowDims + 1u) / 2u;
        for (uint i = groupThreadId.x; i < dims.x * dims.y; i += 256u)
        {
            const uint2 thread = uint2(i % dims.x, i / dims.x);
            float nearest = 3.4e38f;
            for (uint y = 0; y < 2; ++y)
                for (uint x = 0; x < 2; ++x)
                {
                    const uint2 c = min(thread * 2u + uint2(x, y), belowDims - 1u);
                    nearest = min(nearest, gHSTRCloud.hstrBeamGuardPyramid[below + c.y * belowDims.x + c.x]);
                }
            gHSTRCloud.hstrBeamGuardPyramid[offset + thread.y * dims.x + thread.x] = nearest;
        }
        DeviceMemoryBarrierWithGroupSync();
        below = offset;
        belowDims = dims;
    }
}

[numthreads(8, 8, 1)]
void listBeamRefreshBlocks(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.listBeamRefreshBlock(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void testBeamDirtyTiles(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    gHSTRCloud.testBeamDirtyTile(foldedDirtyThread(groupId, groupThreadId.x));
}

[numthreads(64, 1, 1)]
void testBeamTilesTemporal(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.testBeamTileTemporal(dispatchThreadId.x);
}

/// The per-pixel fallback in the rotation-invariant frame: one thread per beam unit of a tile the basis could not carry.
[numthreads(8, 8, 1)]
void marchBeamUnits(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.marchBeamUnit(dispatchThreadId.xy);
}

/// One thread per block of the translation guard: does this block need work at all?
[numthreads(8, 8, 1)]
void classifyBeamGuard(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.classifyBeamGuardCell(dispatchThreadId.xy);
}

/// One thread per COARSE cell: one certificate and one on-screen test answering for all of its blocks.
[numthreads(8, 8, 1)]
void classifyBeamCoarse(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.classifyBeamCoarseCell(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void classifyBeamLeaves(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.classifyBeamLeaf(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void updateBeamCoarse(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.updateBeamCoarseCell(dispatchThreadId.x);
}

/// The dirty passes' 64-thread groups in rows of kDirtyRow, because a dispatch dimension stops at 65,535 groups. A walking
/// camera near cloud lists ~100k blocks, and at 8 rays x 64 slices each the query alone wanted ~800k groups in x: the dispatch
/// was out of range, the blocks were never re-marched, and walk scored ~5% of pixels over 0.02 whatever the parallax budget.
static const uint kDirtyRow = 32768;

void writeFoldedDirtyArgs(uint at, uint threads)
{
    const uint groups = (threads + 63u) / 64u;
    gHSTRCloud.hstrBeamDirtyArgs[at] = min(groups, kDirtyRow);
    gHSTRCloud.hstrBeamDirtyArgs[at + 1] = (groups + kDirtyRow - 1u) / kDirtyRow;
    gHSTRCloud.hstrBeamDirtyArgs[at + 2] = 1u;
}

uint foldedDirtyThread(uint3 groupId, uint groupThread) { return (groupId.y * kDirtyRow + groupId.x) * 64u + groupThread; }

// MEASURED and REMOVED (leanorder1, 4K, errors identical): the dirty passes' 64-thread waves dispatched in a scrambled order, each
// wave keeping its own rays - walk 6.03 / 6.04 -> 6.63 ms (query 2.21 -> 2.50, units 2.47 -> 2.77), sprint 7.51 / 7.55 -> 8.22.
// Random order is the worst an ordering can do and costs only 10%, so the listed order (guard blocks appended roughly in raster,
// a tile's units together) already gets nearly all the cross-wave brick reuse there is: reordering (Morton, tile sorts) cannot buy
// back the frame's DRAM reads. The whole resident sea is 140 MB, so those reads are per-sample sector misses, not brick refetch.

/// Dispatch arguments for the blocks the coarse level could not answer for.
[numthreads(1, 1, 1)]
void writeBeamDescendArgs()
{
    const uint blocks = min(gHSTRCloud.hstrBeamDescendCount[0], gHSTRCloud.params.beamDirtyCapacity);
    gHSTRCloud.hstrBeamDirtyArgs[6] = (blocks + 63u) / 64u;
    gHSTRCloud.hstrBeamDirtyArgs[7] = 1u;
    gHSTRCloud.hstrBeamDirtyArgs[8] = 1u;
}

/// Dispatch arguments for the blocks the classification listed, for the query pass and for the residual.
[numthreads(1, 1, 1)]
void writeBeamDirtyArgs()
{
    const uint blocks = min(gHSTRCloud.hstrBeamDirtyCount[0], gHSTRCloud.params.beamDirtyCapacity);
    const uint rb = max(gHSTRCloud.params.beamDirtyPointStride, 1u);
    const uint edge = max(gHSTRCloud.params.beamDirtyBlockEdge, 1u);
    // Corners and centres of every block, times the threads each ray is split across: one dispatch, see buildBeamDirtyQueries.
    const uint passes = gHSTRCloud.params.beamCentreless != 0 ? 1u : 2u;
    const uint segments = clamp(gHSTRCloud.params.beamDirtySegments, 1u, 64u);
    writeFoldedDirtyArgs(0, blocks * rb * rb * passes * segments);
    // The coarse cells to rebuild are only all known once the blocks have been classified, which is now.
    const uint cells = min(gHSTRCloud.hstrBeamDescendCount[1], gHSTRCloud.params.beamCoarseCapacity);
    gHSTRCloud.hstrBeamDirtyArgs[9] = (cells + 63u) / 64u;
    gHSTRCloud.hstrBeamDirtyArgs[10] = 1u;
    gHSTRCloud.hstrBeamDirtyArgs[11] = 1u;
    // The root tiles of the listed blocks and their aprons (testBeamDirtyTile, which also lists the units to march).
    const uint side = max(edge / max(gHSTRCloud.params.beamTileSize, 1u), 1u) + 2u * kBeamDirtyApron;
    writeFoldedDirtyArgs(12, blocks * side * side);
}

#ifndef HSTR_BEAM_DIRTY_SLICES
#define HSTR_BEAM_DIRTY_SLICES 0 // Nonzero: beamDirtySegments > 1 (set by the host), the sliced query below.
#endif

groupshared uint gDirtyRay[64];      ///< Slice 0's answer for the ray: its position packed, or ~0 when nothing is marched.
groupshared float2 gDirtyExtra[64];  ///< Per slice: opacity-weighted distance times opacity, and steps.

/// The dirty query: every corner and centre of every listed block, in one dispatch, with beamDirtySegments threads per ray. At one
/// thread per ray (the default, HSTR_BEAM_DIRTY_SLICES 0) it is a plain march: no groupshared, barriers or composition.
///
/// Slice 0 alone runs the tests and any carry, and shares whether the ray needs marching through groupshared. Letting each slice
/// decide for itself would race with neighbouring points' carry writes, and two slices of one ray disagreeing would compose a
/// ray from pieces of two different answers. Every thread reaches both barriers; the early-outs only choose what is marched.
///
/// Slices compose front to back: radiance as C += T C_k, optical depth as a sum, and the opacity-weighted distance as
/// sum(T_before d_k (1 - T_k)) / (1 - T_total). The COMPOSITION is exact given the slices; the slices themselves are not the
/// unbroken march cut into pieces. Each marchBeam starts fresh - held step and sigma, cached lighting, empty-cell traversal state -
/// a step boundary is forced at every slice edge, and maxMarchSteps and cloudMinTransmittance apply per slice, so a ray that hits
/// the step cap or goes opaque can end differently. Measured, 720p sea: the error statistics matched to every printed digit
/// across 1 to 64 slices; that is not a pixel diff, and a dense or long volume could still separate them.
[numthreads(64, 1, 1)]
void buildBeamDirtyQueries(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    const uint thread = foldedDirtyThread(groupId, groupThreadId.x);
#if HSTR_BEAM_DIRTY_SLICES == 0
    uint2 point;
    bool centre;
    uint2 position = 0;
    const bool march = gHSTRCloud.beamDirtyRay(thread, point, centre) && gHSTRCloud.beamQueryNeedsMarch(point, centre, true, position);
#if HSTR_BEAM_REPAIR_PROBE
    // The probe reads its block's other lanes, so every lane has to reach it.
    BeamSample ray = {};
    if (march)
    {
        ray = gHSTRCloud.marchBeam(gHSTRCloud.beamRayDirection(float2(position)), float2(0.f, 1.f));
        if (!(1.f - exp(-ray.depth) > 1e-4f))
            ray.distance = 0.f;
    }
    const bool shadowCarried = gHSTRCloud.beamProbeQuery(position, ray, march);
    // beamShadowCarry: an accepted block keeps its old values and is certified from here, as a real carry would be.
    if (march && !shadowCarried)
        gHSTRCloud.beamQueryMarched(position, ray);
    else if (shadowCarried)
        gHSTRCloud.beamShadowCertify(gHSTRCloud.beamGuardCell(position));
#else
    if (!march)
        return;
    BeamSample ray = gHSTRCloud.marchBeam(gHSTRCloud.beamRayDirection(float2(position)), float2(0.f, 1.f));
#if HSTR_STRIP != 0
    gHSTRCloud.beamStripSink(ray);
    return;
#endif
    if (!(1.f - exp(-ray.depth) > 1e-4f))
        ray.distance = 0.f; // As the sliced composition leaves it.
    gHSTRCloud.beamQueryMarched(position, ray);
#endif
#else
    const uint segments = clamp(gHSTRCloud.params.beamDirtySegments, 1u, 64u);
    const uint slice = thread % segments;
    const uint lead = groupThreadId.x - slice;
    if (slice == 0)
    {
        uint2 point;
        bool centre;
        uint2 position;
        bool march = gHSTRCloud.beamDirtyRay(thread / segments, point, centre) &&
                     gHSTRCloud.beamQueryNeedsMarch(point, centre, true, position);
        gDirtyRay[groupThreadId.x] = march ? (position.x | (position.y << 16)) : 0xFFFFFFFFu;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint packed = gDirtyRay[lead];
    const bool march = packed != 0xFFFFFFFFu;
    const uint2 position = uint2(packed & 0xFFFFu, packed >> 16);
    BeamSample sample = {};
    if (march)
        sample = gHSTRCloud.marchBeam(
            gHSTRCloud.beamRayDirection(float2(position)), float2(float(slice), float(slice + 1)) / float(segments)
        );
    gBeamDepthCache[groupThreadId.x] = float4(sample.cache, sample.depth);
    gBeamSingle[groupThreadId.x] = float4(sample.single, 0.f);
    gDirtyExtra[groupThreadId.x] = float2(sample.distance * (1.f - exp(-sample.depth)), sample.steps);
    GroupMemoryBarrierWithGroupSync();
    if (!march || slice != 0)
        return;
    BeamSample ray = {};
    float weighted = 0.f;
    for (uint k = 0; k < segments; ++k)
    {
        const float transmittance = exp(-ray.depth);
        ray.cache += transmittance * gBeamDepthCache[lead + k].rgb;
        ray.single += transmittance * gBeamSingle[lead + k].rgb;
        weighted += transmittance * gDirtyExtra[lead + k].x;
        ray.depth += gBeamDepthCache[lead + k].a;
        ray.steps += gDirtyExtra[lead + k].y;
    }
    const float opacity = 1.f - exp(-ray.depth);
    ray.distance = opacity > 1e-4f ? weighted / opacity : 0.f;
    gHSTRCloud.beamQueryMarched(position, ray);
#endif
}

/// Dispatch arguments for the compacted units, hstrBeamDirtyArgs[15..17].
[numthreads(1, 1, 1)]
void writeBeamDirtyUnitArgs()
{
    writeFoldedDirtyArgs(15, gHSTRCloud.hstrBeamDirtyCount[1]);
}

[numthreads(64, 1, 1)]
void marchBeamDirtyUnits(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    gHSTRCloud.marchBeamDirtyListedUnit(foldedDirtyThread(groupId, groupThreadId.x));
}

/// Span probe: the evaluator's dispatch, one thread per recorded ray (in rows, as the dirty passes'), from the recorded count.
[numthreads(1, 1, 1)]
void writeSpanArgs()
{
    uint capacity, stride;
    gHSTRCloud.hstrSpanRays.GetDimensions(capacity, stride);
    const uint rays = min(gHSTRCloud.hstrSpanCounts[kSpanRays], capacity / kSpanRayWords);
    const uint groups = (rays + 63u) / 64u;
    gHSTRCloud.hstrSpanArgs[0] = min(groups, kDirtyRow);
    gHSTRCloud.hstrSpanArgs[1] = (groups + kDirtyRow - 1u) / kDirtyRow;
    gHSTRCloud.hstrSpanArgs[2] = 1u;
}

/// Span probe: the timed evaluator (writes each ray's result only).
[numthreads(64, 1, 1)]
void evaluateSpans(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    gHSTRCloud.evaluateSpanRay(foldedDirtyThread(groupId, groupThreadId.x), false);
}

/// Span probe: the same integration, untimed, compared with the march's and counted.
[numthreads(64, 1, 1)]
void checkSpans(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    gHSTRCloud.evaluateSpanRay(foldedDirtyThread(groupId, groupThreadId.x), true);
}

/// Cell views: the build dispatch's arguments, one group per queued cell.
[numthreads(1, 1, 1)]
void writeCellViewArgs()
{
    const uint queued = gHSTRCloud.hstrCellCounters[0];
    gHSTRCloud.hstrCellArgs[0] = min(queued, gHSTRCloud.params.cellViewBuildCapacity);
    gHSTRCloud.hstrCellArgs[1] = 1u;
    gHSTRCloud.hstrCellArgs[2] = 1u;
    gHSTRCloud.hstrBeamCounts[kCellViewRequests] += queued;
}

groupshared uint gCellViewSlot;
groupshared uint gCellViewContent;

/// Cell views: one group per queued cell builds its view, each thread a share of the texels. An unoccupied cell is flagged empty
/// without marching (lookups skip those before they ask, so this only happens after the content changed).
[numthreads(8, 8, 1)]
void buildCellViews(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID, uint groupIndex: SV_GroupIndex)
{
    const uint index = groupId.x;
    if (index >= min(gHSTRCloud.hstrCellCounters[0], gHSTRCloud.params.cellViewBuildCapacity))
        return;
    const int3 cell = gHSTRCloud.hstrCellBuild[index].xyz;
    if (groupIndex == 0)
    {
        gCellViewSlot = gHSTRCloud.beginCellView(cell);
        gCellViewContent = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint slot = gCellViewSlot;
    if (slot == ~0u)
        return;
    const uint edge = gHSTRCloud.params.cellViewEdge;
    bool content = false;
    if (gHSTRCloud.hstrCellOccupancy.Load(int4(int3(gHSTRCloud.cellViewKey(float3(cell))), 0)) != 0u)
    {
        for (uint y = groupThreadId.y; y < edge; y += 8u)
            for (uint x = groupThreadId.x; x < edge; x += 8u)
                content = gHSTRCloud.buildCellViewTexel(cell, slot, uint2(x, y)) || content;
    }
    if (content)
        InterlockedOr(gCellViewContent, 1u);
    GroupMemoryBarrierWithGroupSync();
    if (groupIndex == 0)
        gHSTRCloud.endCellView(cell, slot, gCellViewContent != 0u);
}

/// Cell views: the occupancy of every domain cell, from the conservative maxima of its 27 voxels (rebuilt when content changes).
[numthreads(4, 4, 4)]
void buildCellOccupancy(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId >= gHSTRCloud.params.hstrExtinctionDims))
        return;
    bool occupied = false;
    for (uint k = 0; k < 27u && !occupied; ++k)
        occupied = gHSTRCloud.cellViewOccupied(int3(dispatchThreadId), k);
    gHSTRCloud.hstrCellOccupancyOutput[dispatchThreadId] = occupied ? 1u : 0u;
}

/// Cell views: one thread per slot, unmapping the views of changed content.
[numthreads(64, 1, 1)]
void invalidateCellViews(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.invalidateCellView(dispatchThreadId.x);
}

/// beamPushProbe: see projectPushCell.
[numthreads(4, 4, 4)]
void listPushCandidates(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.listPushCandidate(dispatchThreadId);
}

[numthreads(1, 1, 1)]
void writePushArgs()
{
    const uint cells = min(gHSTRCloud.hstrPushCandidateCount[0], gHSTRCloud.params.pushCandidateCapacity);
    gHSTRCloud.hstrPushArgs[0] = (cells + 63u) / 64u;
    gHSTRCloud.hstrPushArgs[1] = 1u;
    gHSTRCloud.hstrPushArgs[2] = 1u;
}

[numthreads(64, 1, 1)]
void projectPushCells(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.projectPushCell(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void allocatePushTiles(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.allocatePushTile(dispatchThreadId.x);
}

[numthreads(8, 8, 1)]
void evaluatePushSamples(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.evaluatePushSample(dispatchThreadId.xy);
}

/// pushEval: 27 threads per visible cell build its operator (groups in rows of 32768, see allocatePushTile).
[numthreads(64, 1, 1)]
void buildPushOperators(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    gHSTRCloud.buildPushOperator((groupId.x + 32768u * groupId.y) * 64u + groupThreadId.x);
}

[numthreads(8, 8, 1)]
void comparePushSamples(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.comparePushSample(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void countPushShares(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.countPushShare(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void histogramPushShares(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.histogramPushShare(dispatchThreadId.x);
}

/// The sort's arguments: one group per listed tile, in rows of kPushSortRow groups (a dispatch dimension stops at 65535).
static const uint kPushSortRow = 32768;
[numthreads(1, 1, 1)]
void writePushSortArgs()
{
    const uint tiles = gHSTRCloud.hstrPushCounts[kPushSortTiles];
    gHSTRCloud.hstrPushArgs[6] = min(tiles, kPushSortRow);
    gHSTRCloud.hstrPushArgs[7] = (tiles + kPushSortRow - 1u) / kPushSortRow;
    gHSTRCloud.hstrPushArgs[8] = 1u;
}

[numthreads(64, 1, 1)]
void scatterPushPieces(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.scatterPushPiece(dispatchThreadId.x);
}

groupshared uint gPushKeys[kPushSortMax];
groupshared uint gPushIDs[kPushSortMax];

/// beamPushProbe: one group per tile of the screen's box puts its list in front-to-back order, by the distance from the camera to
/// each cell's centre. The cells are equal cubes of one grid, and for those that order agrees with every ray through the tile.
[numthreads(kSortGroupSize, 1, 1)]
void sortPushTiles(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    const HSTRCloudParams params = gHSTRCloud.params;
    const uint entry = groupId.x + kPushSortRow * groupId.y;
    const uint lane = groupThreadId.x;
    if (entry >= gHSTRCloud.hstrPushCounts[kPushSortTiles])
        return;
    const uint id = gHSTRCloud.hstrPushSortTiles[entry];
    const uint base = gHSTRCloud.hstrPushTileOffsets[id];
    const uint listed = gHSTRCloud.hstrPushTileCounts[id];
    const uint count = base < params.pushListCapacity ? min(listed, params.pushListCapacity - base) : 0u;
    // count is uniform across the group, so returning here cannot split a barrier.
    if (count < 2u)
        return;
    if (count > kPushSortMax)
    {
        if (lane == 0u)
        {
            uint unused;
            InterlockedAdd(gHSTRCloud.hstrPushCounts[kPushSortOverflow], 1u, unused);
        }
        return;
    }
    uint sortSize = 2u;
    while (sortSize < count)
        sortSize <<= 1;
    for (uint i = lane; i < sortSize; i += kSortGroupSize)
    {
        const uint cell = i < count ? gHSTRCloud.hstrPushList[base + i] : ~0u;
        // Positive floats order as their bits.
        gPushKeys[i] = i < count ? asuint(length(gHSTRCloud.pushCellOffset(gHSTRCloud.hstrPushSlotKeys[cell]))) : ~0u;
        gPushIDs[i] = cell;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 2; k <= sortSize; k <<= 1)
    {
        for (uint j = k >> 1; j > 0; j >>= 1)
        {
            for (uint i = lane; i < sortSize; i += kSortGroupSize)
            {
                const uint partner = i ^ j;
                if (partner > i)
                {
                    const bool ascending = (i & k) == 0;
                    const uint a = gPushKeys[i];
                    const uint b = gPushKeys[partner];
                    if ((a > b) == ascending)
                    {
                        gPushKeys[i] = b;
                        gPushKeys[partner] = a;
                        const uint t = gPushIDs[i];
                        gPushIDs[i] = gPushIDs[partner];
                        gPushIDs[partner] = t;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for (uint i = lane; i < count; i += kSortGroupSize)
        gHSTRCloud.hstrPushList[base + i] = gPushIDs[i];
}

groupshared uint gResidualMask[2]; ///< Which of the group's 64 pixels the residual resolve has to write.

/// The resolve, and the queue of 8 x 8 groups holding pixels the residual resolve has to write: one entry per such group, with
/// the mask of which pixels, appended once by the group's first thread.
[numthreads(8, 8, 1)]
void resolveBeam(uint3 dispatchThreadId: SV_DispatchThreadID, uint3 groupId: SV_GroupID, uint groupIndex: SV_GroupIndex)
{
    if (groupIndex < 2)
        gResidualMask[groupIndex] = 0;
    GroupMemoryBarrierWithGroupSync();
    if (gHSTRCloud.resolveBeamPixel(dispatchThreadId.xy))
        InterlockedOr(gResidualMask[groupIndex >> 5], 1u << (groupIndex & 31u));
    GroupMemoryBarrierWithGroupSync();
    if (groupIndex == 0 && (gResidualMask[0] | gResidualMask[1]) != 0)
    {
        uint entry;
        InterlockedAdd(gHSTRCloud.hstrBeamResidualArgs[0], 1u, entry);
        if (entry < gHSTRCloud.beamResidualGroups())
        {
            gHSTRCloud.hstrBeamResidualList[3 * entry] = groupId.x | (groupId.y << 16);
            gHSTRCloud.hstrBeamResidualList[3 * entry + 1] = gResidualMask[0];
            gHSTRCloud.hstrBeamResidualList[3 * entry + 2] = gResidualMask[1];
        }
    }
}

/// One group per queued resolve group, laid out in rows of kResidualRow groups because a dispatch dimension stops at 65,535 and
/// a 4K frame has 129,600 resolve groups.
static const uint kResidualRow = 32768;

/// beamWarp: the warp field over the on-screen lattice box (buildBeamWarpFieldAt), before the resolve reads it.
[numthreads(8, 8, 1)]
void buildBeamWarpField(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.buildBeamWarpFieldAt(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void resolveBeamResidual(uint3 groupId: SV_GroupID, uint groupIndex: SV_GroupIndex)
{
    gHSTRCloud.resolveBeamResidualPixel(groupId.y * kResidualRow + groupId.x, groupIndex);
}

/// Dispatch arguments for the groups the resolve queued, after the count in element 0.
[numthreads(1, 1, 1)]
void writeBeamResidualArgs()
{
    const uint groups = min(gHSTRCloud.hstrBeamResidualArgs[0], gHSTRCloud.beamResidualGroups());
    // Under beamWarpAuto the plain [1..3] and warped [4..6] residual resolves are both dispatched; the decision gives one the groups.
    const bool warped = gHSTRCloud.params.beamWarpAuto > 0.f && gHSTRCloud.hstrBeamWarpArgs[9] != 0;
    const uint used = warped ? 4u : 1u;
    const uint idle = warped ? 1u : 4u;
    gHSTRCloud.hstrBeamResidualArgs[used] = min(groups, kResidualRow);
    gHSTRCloud.hstrBeamResidualArgs[used + 1] = (groups + kResidualRow - 1u) / kResidualRow;
    gHSTRCloud.hstrBeamResidualArgs[used + 2] = 1u;
    gHSTRCloud.hstrBeamResidualArgs[idle] = 0u;
    gHSTRCloud.hstrBeamResidualArgs[idle + 1] = 1u;
    gHSTRCloud.hstrBeamResidualArgs[idle + 2] = 1u;
}

/// Right after the classification, from the held and listed guard blocks it just counted (hstrBeamDirtyCount[2] and [0]): whether
/// the resolve warps (beamWarpAuto) and the tile tolerance (beamPolicy), for this build's tiles and its resolve. Only a build that
/// classified while translating runs it; any other keeps the last decision. On the GPU, so the decision is this frame's and nothing
/// waits on a readback. Sprint expires every block, so it holds none: no warp, and the error its held blocks would have carried
/// goes into a looser tile test instead.
[numthreads(1, 1, 1)]
void decideBeamPolicy()
{
    const HSTRCloudParams p = gHSTRCloud.params;
    const uint listed = gHSTRCloud.hstrBeamDirtyCount[0];
    const uint held = gHSTRCloud.hstrBeamDirtyCount[2];
    const float share = held + listed == 0 ? 1.f : float(held) / float(held + listed);
    if (p.beamWarpAuto > 0.f)
        gHSTRCloud.hstrBeamWarpArgs[9] = share >= p.beamWarpAuto ? 1u : 0u;
    if (p.beamPolicy != 0)
    {
        const float s = saturate((p.beamPolicyHeldHigh - share) / max(p.beamPolicyHeldHigh - p.beamPolicyHeldLow, 1e-6f));
        gHSTRCloud.hstrBeamWarpArgs[11] = asuint(lerp(p.beamTolerance, p.beamPolicyTolerance, s));
    }
}

/// beamWarpAuto, before the resolve: the warp field's and both resolves' arguments from decideBeamPolicy's decision.
[numthreads(1, 1, 1)]
void writeBeamWarpArgs()
{
    const bool warp = gHSTRCloud.hstrBeamWarpArgs[9] != 0;
    uint2 origin, end;
    gHSTRCloud.beamWarpFieldBox(origin, end);
    const uint2 box = max(end, origin) - origin;
    const uint2 fieldGroups = warp ? (box + 7u) / 8u : uint2(0);
    gHSTRCloud.hstrBeamWarpArgs[0] = fieldGroups.x;
    gHSTRCloud.hstrBeamWarpArgs[1] = fieldGroups.y;
    gHSTRCloud.hstrBeamWarpArgs[2] = 1u;
    const uint2 frameGroups = (gHSTRCloud.params.frameDim + 7u) / 8u;
    gHSTRCloud.hstrBeamWarpArgs[3] = warp ? frameGroups.x : 0u;
    gHSTRCloud.hstrBeamWarpArgs[4] = frameGroups.y;
    gHSTRCloud.hstrBeamWarpArgs[5] = 1u;
    gHSTRCloud.hstrBeamWarpArgs[6] = warp ? 0u : frameGroups.x;
    gHSTRCloud.hstrBeamWarpArgs[7] = frameGroups.y;
    gHSTRCloud.hstrBeamWarpArgs[8] = 1u;
}

[numthreads(8, 8, 1)]
void resolveBeamSparse(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.resolveBeamSparsePixel(dispatchThreadId.xy);
}

[numthreads(8, 8, 1)]
void beamGuide(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.beamGuidePixel(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void marchBeamPixels(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.marchBeamPixel(dispatchThreadId.x);
}

[numthreads(8, 8, 1)]
void marchBeamGrid(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.marchBeamGridPixel(dispatchThreadId.xy);
}

[numthreads(8, 8, 1)]
void buildBeamGridQueries(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.buildBeamGridQuery(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void marchBeamQueue(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.marchBeamQueued(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void queueBeamTiles(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.queueBeamTile(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void marchBeamQueuePixels(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.marchBeamQueuedPixel(dispatchThreadId.x);
}

[numthreads(1, 1, 1)]
void writeBeamQueueArgs(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.writeBeamQueueArgs();
}

[numthreads(1, 1, 1)]
void writeBeamArgs(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.writeBeamArgs();
}

[numthreads(64, 1, 1)]
void classifyBeamRoots(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.classifyBeamRoot(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void emitBeamSparseQueries(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.emitBeamSparseQuery(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void verifyBeamSparseTiles(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.verifyBeamSparseTile(dispatchThreadId.x);
}

[numthreads(1, 1, 1)]
void writeBeamSparseArgs(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.writeBeamSparseArgs();
}

[numthreads(8, 8, 1)]
void referencePathTrace(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.referencePixel(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void compareReference(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.compareReferenceRow(dispatchThreadId.x);
}

[numthreads(4, 4, 4)]
void updateWorldCache(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.updateWorldCacheCell(dispatchThreadId);
}

[numthreads(64, 1, 1)]
void traceWorldCachePhotons(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.tracePhoton(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void advanceWorldCachePhotons(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.advancePhoton(dispatchThreadId.x);
}

[numthreads(4, 4, 4)]
void resolveWorldCache(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.resolveWorldCacheCell(dispatchThreadId);
}

[numthreads(4, 4, 4)]
void bakeWorldCache(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.bakeWorldCacheCell(dispatchThreadId);
}

[numthreads(4, 4, 4)]
void updateCameraLighting(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.writeCameraLighting(dispatchThreadId);
}

[numthreads(8, 8, 1)]
void buildCameraQueries(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.writeCameraQuery(dispatchThreadId.xy);
}

[numthreads(8, 8, 1)]
void buildTileBases(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.buildTileBasis(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void projectCutNodes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.projectCutNode(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void binCutNodes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.binCutNode(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void buildDomainCutNodes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.buildDomainCutNode(dispatchThreadId.x);
}

groupshared uint gSortKeys[kMaxTileNodes];
groupshared uint gSortIDs[kMaxTileNodes];

/// One group per tile: a bitonic sort orders the tile's cut by BSP front-to-back key.
[numthreads(kSortGroupSize, 1, 1)]
void sortTileCuts(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    const HSTRCloudParams params = gHSTRCloud.params;
    const uint2 tile = groupId.xy;
    const uint lane = groupThreadId.x;
    const bool valid = all(tile < params.hstrTileDims) && gScene.getGridVolumeCount() > 0;
    const uint tileID = tile.x + params.hstrTileDims.x * tile.y;
    const uint count = valid ? min(gHSTRCloud.hstrTileNodeCounts[tileID], params.hstrMaxTileNodes) : 0u;
    const uint base = tileID * params.hstrMaxTileNodes;
    // count is uniform across the group, so returning here cannot split a barrier.
    if (count < 2u)
        return;
    uint sortSize = 2u;
    while (sortSize < count)
        sortSize <<= 1;

    for (uint i = lane; i < sortSize; i += kSortGroupSize)
    {
        const uint id = i < count ? gHSTRCloud.hstrTileNodes[base + i] : ~0u;
        gSortKeys[i] = i < count ? gHSTRCloud.hstrNodeOrder[id] : ~0u;
        gSortIDs[i] = id;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 2; k <= sortSize; k <<= 1)
    {
        for (uint j = k >> 1; j > 0; j >>= 1)
        {
            for (uint i = lane; i < sortSize; i += kSortGroupSize)
            {
                const uint partner = i ^ j;
                if (partner > i)
                {
                    const bool ascending = (i & k) == 0;
                    const uint a = gSortKeys[i];
                    const uint b = gSortKeys[partner];
                    if ((a > b) == ascending)
                    {
                        gSortKeys[i] = b;
                        gSortKeys[partner] = a;
                        const uint id = gSortIDs[i];
                        gSortIDs[i] = gSortIDs[partner];
                        gSortIDs[partner] = id;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for (uint i = lane; i < count; i += kSortGroupSize)
        gHSTRCloud.hstrTileNodes[base + i] = gSortIDs[i];
}

[numthreads(4, 4, 4)]
void computeFineSun(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const HSTRCloudParams params = gHSTRCloud.params;
    if (any(params.fineSunSize > 0u) && any(dispatchThreadId >= params.fineSunSize))
        return;
    gHSTRCloud.computeFineSun(dispatchThreadId + params.fineSunOffset);
}

[numthreads(4, 4, 4)]
void gatherSunOctaves(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.gatherSunOctaves(dispatchThreadId);
}

[numthreads(4, 4, 4)]
void blurSunOctaves(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.blurSunOctaves(dispatchThreadId);
}

[numthreads(4, 4, 4)]
void maskResidual(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.maskResidual(dispatchThreadId);
}

/// Atlas texel origin of a brick slot.
uint3 cloudSlotOrigin(uint slot, uint shift)
{
    const uint mask = (1u << shift) - 1u;
    return uint3(slot & mask, (slot >> shift) & mask, slot >> (2u * shift)) * kCloudBrickEdge;
}

/// Reconstructs the staged bricks of one level into their atlas slots (reconstructBrick in CloudFormat.h is the same rule):
/// value = max(0, trilinear parent prediction + residual on the core), requantised as sqrt-spaced unorm. Thread (x, y, z) is
/// value (x, y, z % 10) of staged brick cloudCommitOffset + z / 10; the parents were written by earlier dispatches.
[numthreads(4, 4, 4)]
void commitCloudBricks(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const HSTRCloudParams params = gHSTRCloud.params;
    const uint local = dispatchThreadId.z / kCloudBrickEdge;
    if (local >= params.cloudCommitCount || any(dispatchThreadId.xy >= kCloudBrickEdge))
        return;
    const uint staged = params.cloudCommitOffset + local;
    const HSTRCloudStaging info = gHSTRCloud.hstrCloudStagingInfo[staged];
    const uint3 texel = uint3(dispatchThreadId.xy, dispatchThreadId.z % kCloudBrickEdge);

    float prediction = 0.f;
    if (info.parentSlot != kCloudRefNone)
    {
        const float3 parity = float3((info.parity & 1u) != 0 ? 1.f : 0.f, (info.parity & 2u) != 0 ? 1.f : 0.f, (info.parity & 4u) != 0 ? 1.f : 0.f);
        const float3 position = 4.f * parity + (float3(texel) - 1.f + 0.5f) * 0.5f - 0.5f + 1.f;
        const int3 base = int3(floor(position));
        const float3 f = position - float3(base);
        const uint3 parentOrigin = cloudSlotOrigin(info.parentSlot, params.cloudAtlasShift);
        for (uint corner = 0; corner < 8; ++corner)
        {
            const uint3 offset = uint3(corner & 1u, (corner >> 1) & 1u, corner >> 2);
            const float3 w = select(offset != 0u, f, 1.f - f);
            const float t = gHSTRCloud.hstrCloudAtlasOutput[parentOrigin + uint3(base) + offset];
            prediction += w.x * w.y * w.z * (info.parentMin + info.parentRange * t * t);
        }
    }
    float residual = 0.f;
    if (info.coded != 0 && all(texel >= 1u) && all(texel <= 8u))
    {
        const uint3 core = texel - 1u;
        residual = gHSTRCloud.hstrCloudResiduals[staged * 512u + core.x + 8u * (core.y + 8u * core.z)];
    }
    const float density = max(0.f, prediction + residual);
    const float t = info.valueRange > 0.f ? sqrt(saturate((density - info.valueMin) / info.valueRange)) : 0.f;
    gHSTRCloud.hstrCloudAtlasOutput[cloudSlotOrigin(info.slot, params.cloudAtlasShift) + texel] = t;
}

/// Occupancy of the staged bricks after their commit dispatches: cell (x, y, z) of a brick's 4^3 covers level voxels 2c and 2c + 1
/// per axis, whose trilinear samples read atlas texels 2c to 2c + 3 (the apron shifts them by one); its bit is set if any of those
/// decodes to non-zero density (valueMin + valueRange t^2, so t > 0 or valueMin > 0). One thread per brick.
[numthreads(16, 1, 1)]
void occupancyCloudBricks(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const HSTRCloudParams params = gHSTRCloud.params;
    if (dispatchThreadId.x >= params.cloudStagedCount)
        return;
    const HSTRCloudStaging info = gHSTRCloud.hstrCloudStagingInfo[dispatchThreadId.x];
    const uint3 origin = cloudSlotOrigin(info.slot, params.cloudAtlasShift);
    uint words[kCloudCellWords] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
    for (uint cz = 0; cz < 4; ++cz)
        for (uint cy = 0; cy < 4; ++cy)
            for (uint cx = 0; cx < 4; ++cx)
            {
                float t = 0.f;
                for (uint z = 2 * cz; z < 2 * cz + 4; ++z)
                    for (uint y = 2 * cy; y < 2 * cy + 4; ++y)
                        for (uint x = 2 * cx; x < 2 * cx + 4; ++x)
                            t = max(t, gHSTRCloud.hstrCloudAtlas.Load(int4(origin + uint3(x, y, z), 0)));
                const uint cell = cx + 4u * cy + 16u * cz;
                const uint code = cloudCellCode(t > 0.f ? info.valueMin + info.valueRange * t * t : info.valueMin);
                words[cell >> 3u] |= code << ((cell & 7u) * 4u);
                // The octant (4 voxels) this cell belongs to bounds all eight of its cells.
                const uint octant = (cx >> 1u) + 2u * (cy >> 1u) + 4u * (cz >> 1u);
                const uint shifted = octant * 4u;
                words[8] = (words[8] & ~(15u << shifted)) | (max((words[8] >> shifted) & 15u, code) << shifted);
            }
    for (uint w = 0; w < kCloudCellWords; ++w)
        gHSTRCloud.hstrCloudOccupancyOutput[info.brick * kCloudCellWords + w] = words[w];
}

/// Clears only bitmap entries named by the previous compact queue.
[numthreads(256, 1, 1)]
void clearDirtyCloudPages(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= gHSTRCloud.hstrCloudDirtyPageCount[0]) return;
    const uint page = gHSTRCloud.hstrCloudDirtyPages[id.x];
    uint unused;
    InterlockedAnd(gHSTRCloud.hstrCloudDirtyBits[page >> 5u], ~(1u << (page & 31u)), unused);
}

/// Expands invalidated regions and appends only the first marker for each virtual page.
[numthreads(256, 1, 1)]
void markDirtyCloudPages(uint3 id: SV_DispatchThreadID)
{
    const uint count = gHSTRCloud.params.cloudPageRegionCount;
    if (count == 0 || id.x >= gHSTRCloud.hstrCloudDirtyRegions[count - 1u].workEnd) return;
    uint lo = 0, hi = count;
    while (lo < hi)
    {
        const uint mid = (lo + hi) >> 1u;
        if (id.x < gHSTRCloud.hstrCloudDirtyRegions[mid].workEnd) hi = mid; else lo = mid + 1u;
    }
    const HSTRCloudDirtyPageRegion region = gHSTRCloud.hstrCloudDirtyRegions[lo];
    const uint begin = lo == 0 ? 0u : gHSTRCloud.hstrCloudDirtyRegions[lo - 1u].workEnd;
    const uint local = id.x - begin;
    const uint3 p = uint3(local % region.extent.x, (local / region.extent.x) % region.extent.y,
                          local / (region.extent.x * region.extent.y)) + region.origin;
    const HSTRCloudAsset asset = gHSTRCloud.hstrCloudAssets[region.asset];
    const uint3 dims = asset.dims >> 3u;
    const uint page = asset.pageOffset + p.x + dims.x * (p.y + dims.y * p.z);
    const uint bit = 1u << (page & 31u);
    uint previous;
    InterlockedOr(gHSTRCloud.hstrCloudDirtyBits[page >> 5u], bit, previous);
    if ((previous & bit) == 0)
    {
        uint slot;
        InterlockedAdd(gHSTRCloud.hstrCloudDirtyPageCount[0], 1u, slot);
        gHSTRCloud.hstrCloudDirtyPages[slot] = page;
    }
}

[numthreads(1, 1, 1)]
void writeDirtyCloudPageArgs()
{
    gHSTRCloud.hstrCloudDirtyPageArgs[0] = (gHSTRCloud.hstrCloudDirtyPageCount[0] + 255u) / 256u;
    gHSTRCloud.hstrCloudDirtyPageArgs[1] = 1u;
    gHSTRCloud.hstrCloudDirtyPageArgs[2] = 1u;
}

/// Resolves final canonical hierarchy state once per unique page; no operation ordering or physical-ID comparison is involved.
[numthreads(256, 1, 1)]
void resolveDirtyCloudPages(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= gHSTRCloud.hstrCloudDirtyPageCount[0]) return;
    const uint pageIndex = gHSTRCloud.hstrCloudDirtyPages[id.x];
    uint assetCount, stride;
    gHSTRCloud.hstrCloudAssets.GetDimensions(assetCount, stride);
    HSTRCloudAsset asset = {};
    bool found = false;
    for (uint a = 0; a < assetCount; ++a)
    {
        const HSTRCloudAsset candidate = gHSTRCloud.hstrCloudAssets[a];
        const uint3 d = candidate.dims >> 3u;
        const uint n = d.x * d.y * d.z;
        if (pageIndex >= candidate.pageOffset && pageIndex - candidate.pageOffset < n)
        {
            asset = candidate;
            found = true;
            break;
        }
    }
    if (!found) return;
    const uint3 dims = asset.dims >> 3u;
    const uint linear = pageIndex - asset.pageOffset;
    const uint3 brick = uint3(linear % dims.x, (linear / dims.x) % dims.y, linear / (dims.x * dims.y));
    const uint3 chunk = brick >> 4u;
    uint entry = gHSTRCloud.hstrCloudDirectory[asset.directoryOffset + chunk.x + asset.chunkDims.x * (chunk.y + asset.chunkDims.y * chunk.z)];
    if (entry != kCloudRefNone && (entry & kCloudRefNode) != 0)
    {
        const uint3 cell = (brick >> 2u) & 3u;
        entry = gHSTRCloud.hstrCloudNodes[(entry & ~kCloudRefNode) * 64u + cell.x + 4u * (cell.y + 4u * cell.z)];
        if (entry != kCloudRefNone && (entry & kCloudRefNode) != 0)
        {
            const uint3 sub = brick & 3u;
            entry = gHSTRCloud.hstrCloudNodes[(entry & ~kCloudRefNode) * 64u + sub.x + 4u * (sub.y + 4u * sub.z)];
        }
    }
    HSTRCloudVirtualPage page = {entry, entry == kCloudRefNone ? 0u : (gHSTRCloud.hstrCloudBricks[entry].info & 15u)};
    gHSTRCloud.hstrCloudPagesOutput[pageIndex] = page;

    // The resolved level pages: what cloudAssetBrickAt returns for a sample in this page asking for level L, for every L. It is the
    // same for every page of a level-L brick - the hierarchy is closed under parents, so every page there climbs to the same
    // level-L ancestor, and a missing child octant of a coarser brick covers the whole level-L brick - so each dirty page writes its
    // level-L brick's word and pages sharing one write the same value.
    uint base = asset.levelPageBase;
    for (uint level = 0; level <= asset.topLevel; ++level)
    {
        const uint3 levelDims = (dims + (1u << level) - 1u) >> level;
        const uint3 c = brick >> level;
        uint word = kCloudRefNone;
        if (entry != kCloudRefNone)
        {
            uint resolved = entry;
            uint resolvedLevel = page.mappedLevel;
            const uint3 child = (brick >> max(page.mappedLevel, 1u) - 1u) & 1u;
            if (level < page.mappedLevel &&
                ((gHSTRCloud.hstrCloudBricks[entry].info >> 4) & (1u << (child.x + 2u * child.y + 4u * child.z))) == 0)
                word = kCloudLevelEmpty | page.mappedLevel;
            else
            {
                for (uint k = 0; k < 15 && resolvedLevel < level && gHSTRCloud.hstrCloudBricks[resolved].parent != kCloudRefNone; ++k)
                {
                    resolved = gHSTRCloud.hstrCloudBricks[resolved].parent;
                    resolvedLevel = gHSTRCloud.hstrCloudBricks[resolved].info & 15u;
                }
                word = resolved | (resolvedLevel << 24);
            }
        }
        gHSTRCloud.hstrCloudLevelPagesOutput[base + c.x + levelDims.x * (c.y + levelDims.y * c.z)] = word;
        base += levelDims.x * levelDims.y * levelDims.z;
    }
}

/// The sun bake every brick's samples read, per orientation class, resolved once a frame (after the frame's bakes and slot changes)
/// so the lean march's lit samples read one pair instead of searching kCloudSunBakesPerBrick keys on the brick and its ancestors.
[numthreads(64, 1, 1)]
void resolveCloudSunSlots(uint3 id: SV_DispatchThreadID)
{
    uint bricks, stride;
    gHSTRCloud.hstrCloudBricks.GetDimensions(bricks, stride);
    if (id.x >= bricks)
        return;
    for (uint c = 0; c < kCloudSunClasses; ++c)
    {
        uint entry = id.x;
        const uint slot = gHSTRCloud.cloudSunSlotUp(entry, c);
        gHSTRCloud.hstrCloudSunResolvedOutput[id.x * kCloudSunClasses + c] =
            uint2(slot, entry | ((gHSTRCloud.hstrCloudBricks[entry].info & 15u) << 24));
    }
}

/// Sun bakes of this frame (cloudCommitOffset, cloudCommitCount): for every atlas texel of a brick, the optical depth (in source
/// voxel lengths, density scale applied) along the sun over job.reach source voxels, on the asset's own bricks at the brick's level
/// (the density sunDepthAt's live march reads for a camera sample at that level), skipping exactly empty cells and octants.
/// Thread (x, y, z) is texel (x, y, z % 10) of job z / 10.
[numthreads(4, 4, 4)]
void bakeCloudSun(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const HSTRCloudParams params = gHSTRCloud.params;
    const uint local = dispatchThreadId.z / kCloudBrickEdge;
    if (local >= params.cloudCommitCount || any(dispatchThreadId.xy >= kCloudBrickEdge))
        return;
    const HSTRCloudSunBake job = gHSTRCloud.hstrCloudSunBakes[params.cloudCommitOffset + local];
    if (job.slot == kCloudRefNone)
        return; // A GPU-scheduled entry that found no slot (or no candidate).
    const uint3 texel = uint3(dispatchThreadId.xy, dispatchThreadId.z % kCloudBrickEdge);
    const HSTRCloudAsset asset = gHSTRCloud.hstrCloudAssets[job.asset];
    const float scale = gScene.getGridVolume(0).data.densityScale * params.densityScale;
    const float levelScale = float(1u << job.level);
    const float3 start = (job.origin + float3(texel) - 1.f + 0.5f) * levelScale - 0.5f;
    float depth = 0.f;
    float u = 0.f;
    float stepVoxels = 0.5f * levelScale;
    [loop]
    for (uint k = 0; k < 512 && u < job.reach && depth <= job.cap; ++k)
    {
        const float dt = min(stepVoxels, job.reach - u);
        const float3 x = start + (u + 0.5f * dt) * job.direction;
        CloudSample sample;
        float next = u + dt;
        if (gHSTRCloud.cloudAssetBrickAt(asset, x, job.level, false, sample))
        {
            stepVoxels = 0.5f * float(1u << sample.level);
            if (sample.empty)
            {
                float exit = 1e30f;
                for (uint a = 0; a < 3; ++a)
                    if (abs(job.direction[a]) > 1e-8f)
                        exit = min(exit, ((job.direction[a] > 0.f ? sample.hi[a] : sample.lo[a]) - x[a]) / job.direction[a]);
                next = max(next, u + 0.5f * dt + max(exit, 0.f));
            }
            else
                depth += gHSTRCloud.cloudBrickDensity(sample.entry, x, int3(floor(x + 0.5f))) * scale * dt;
        }
        else
            stepVoxels = 4.f * levelScale; // Outside the asset: no density until the march re-enters it.
        u = min(next, job.reach);
    }
    gHSTRCloud.hstrCloudSunAtlasOutput[cloudSlotOrigin(job.slot, params.cloudAtlasShift) + texel] = depth;
}

/// Byte k of the packed coefficients starting at payload word p.
uint cloudPayloadByte(uint p, uint k)
{
    return (gHSTRCloud.hstrCloudPayload[p + (k >> 2)] >> (8u * (k & 3u))) & 255u;
}

/// Inverse orthonormal Haar step of residual values base + offset + k * stride, k < n (haarStepInverse in CloudFormat.h).
void cloudHaarStepInverse(uint base, uint offset, uint stride, uint n)
{
    float temp[8];
    const float s = 0.70710678f;
    for (uint k = 0; k < n / 2; ++k)
    {
        const float a = gHSTRCloud.hstrCloudResiduals[base + offset + k * stride];
        const float d = gHSTRCloud.hstrCloudResiduals[base + offset + (n / 2 + k) * stride];
        temp[2 * k] = (a + d) * s;
        temp[2 * k + 1] = (a - d) * s;
    }
    for (uint k = 0; k < n; ++k)
        gHSTRCloud.hstrCloudResiduals[base + offset + k * stride] = temp[k];
}

/// Decodes the residual of every staged brick into hstrCloudResiduals (512 per staged brick) before the level dispatches of
/// commitCloudBricks: the packed coefficients (packCoefficients in CloudFormat.h: significance bitmap, zigzag varints in scan
/// order) dequantised by the brick's step, then the inverse transform (haarInverse). One thread per brick.
[numthreads(16, 1, 1)]
void decodeCloudResiduals(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint staged = dispatchThreadId.x;
    if (staged >= gHSTRCloud.params.cloudStagedCount)
        return;
    const HSTRCloudStaging info = gHSTRCloud.hstrCloudStagingInfo[staged];
    if (info.coded == 0)
        return;
    const uint base = staged * 512u;
    const uint p = info.payloadWord;
    const uint lo[4] = { 0, 1, 2, 4 };
    const uint hi[4] = { 1, 2, 4, 8 };
    uint s = 0;
    uint k = 64;
    for (uint band = 0; band < 4; ++band)
        for (uint z = 0; z < hi[band]; ++z)
            for (uint y = 0; y < hi[band]; ++y)
                for (uint x = 0; x < hi[band]; ++x)
                {
                    if (max(x, max(y, z)) < lo[band])
                        continue;
                    float value = 0.f;
                    if ((cloudPayloadByte(p, s / 8u) & (1u << (s % 8u))) != 0)
                    {
                        uint u = 0;
                        for (uint shift = 0; shift < 35; shift += 7)
                        {
                            const uint byte = cloudPayloadByte(p, k++);
                            u |= (byte & 0x7Fu) << shift;
                            if ((byte & 0x80u) == 0)
                                break;
                        }
                        const int q = int(u >> 1) ^ -int(u & 1u);
                        value = float(q) * info.step;
                    }
                    gHSTRCloud.hstrCloudResiduals[base + x + 8u * (y + 8u * z)] = value;
                    ++s;
                }
    if (info.transform != kCloudTransformHaar)
        return;
    for (uint n = 2; n <= 8; n *= 2)
    {
        for (uint y = 0; y < n; ++y)
            for (uint x = 0; x < n; ++x)
                cloudHaarStepInverse(base, x + 8u * y, 64u, n);
        for (uint z = 0; z < n; ++z)
            for (uint x = 0; x < n; ++x)
                cloudHaarStepInverse(base, x + 64u * z, 8u, n);
        for (uint z = 0; z < n; ++z)
            for (uint y = 0; y < n; ++y)
                cloudHaarStepInverse(base, 8u * (y + 8u * z), 1u, n);
    }
}

/// Scales every world cache deposit by worldCacheKeep: after a sun change the old estimate fades as new photons arrive.
[numthreads(4, 4, 4)]
void decayWorldCache(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId >= gHSTRCloud.params.worldCacheDims))
        return;
    const uint index = gHSTRCloud.worldCacheIndex(dispatchThreadId);
    for (uint slot = 0; slot < kWorldCacheSlots; ++slot)
    {
        const uint i = kWorldCacheSlots * index + slot;
        gHSTRCloud.hstrWorldCacheDeposit[i] = int(round(float(gHSTRCloud.hstrWorldCacheDeposit[i]) * gHSTRCloud.params.worldCacheKeep));
    }
}

/// The running fades of the residency's bricks at this frame (CloudResidency::setFade computes the same).
[numthreads(256, 1, 1)]
void advanceCloudFades(uint3 id: SV_DispatchThreadID)
{
    const HSTRCloudFadeFrame frame = gHSTRCloud.hstrCloudFadeFrame[0];
    if (id.x >= frame.brickCount)
        return;
    const HSTRCloudBrick brick = gHSTRCloud.hstrCloudBricksOutput[id.x];
    if ((brick.fadeStart & (kCloudFadeIn | kCloudFadeOut)) == 0)
        return;
    const float delta = frame.step * float((frame.frame - brick.fadeStart) & kCloudFadeFrameMask);
    gHSTRCloud.hstrCloudBricksOutput[id.x].fade =
        (brick.fadeStart & kCloudFadeIn) != 0 ? min(1.f, brick.fadeBase + delta) : max(0.f, brick.fadeBase - delta);
}

[numthreads(4, 4, 4)]
void clearWorldCacheTiles(uint3 cell: SV_DispatchThreadID)
{
    // Clears the deposits of world cache cells in sea tiles whose cloud changed. Bound once like the domain passes, so it reads
    // the domain frame rather than the params (worldCacheIndex and worldCacheTile with the frame's fields).
    const HSTRCloudDomainFrame frame = gHSTRCloud.hstrDomainFrame[0];
    if (any(cell >= frame.cacheDims))
        return;
    const uint2 tile = min(cell.xz * frame.cacheCellVoxels / frame.tileVoxels, frame.tiles - 1u);
    if (gHSTRCloud.hstrCloudTileReset[tile.x + frame.tiles * tile.y] == 0)
        return;
    const uint index = cell.x + frame.cacheDims.x * (cell.y + frame.cacheDims.y * cell.z);
    for (uint slot = 0; slot < kWorldCacheSlots; ++slot)
        gHSTRCloud.hstrWorldCacheDeposit[kWorldCacheSlots * index + slot] = 0;
}

[numthreads(64, 1, 1)]
void solveLeaves(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint leaf = dispatchThreadId.x;
    const uint leafCount = gHSTRCloud.params.hstrLeafDims.x * gHSTRCloud.params.hstrLeafDims.y * gHSTRCloud.params.hstrLeafDims.z;
    if (leaf >= leafCount)
        return;
    const HSTRCloudParams params = gHSTRCloud.params;

    float3 incident[kMaxTraceDofs];
    for (uint dof = 0; dof < params.hstrTraceDofs; ++dof)
        incident[dof] = 0.f;
    float omittedError = 0.f;
    const uint matrixSize = params.hstrTraceDofs * params.hstrStorageRank;
    for (uint mode = 0; mode < params.hstrStorageRank; ++mode)
    {
        float3 coefficient = 0.f;
        for (uint rootDof = 0; rootDof < params.hstrTraceDofs; ++rootDof)
            coefficient += gHSTRCloud.hstrLeafBasisRight[matrixSize * leaf + params.hstrStorageRank * rootDof + mode] *
                           gHSTRCloud.hstrRootIncident[rootDof].rgb;

        const float adjointResponse = gHSTRCloud.hstrLeafAdjointResponses[leaf * params.hstrStorageRank + mode];
        const float importance = adjointResponse * length(coefficient);
        const bool active = mode < params.activeRank && importance >= params.activeThreshold;
        if (active)
            for (uint dof = 0; dof < params.hstrTraceDofs; ++dof)
                incident[dof] += gHSTRCloud.hstrLeafBasisLeft[matrixSize * leaf + params.hstrStorageRank * dof + mode] * coefficient;
        else
            omittedError += importance;
    }
    float3 outgoing[kMaxTraceDofs];
    for (uint outputDof = 0; outputDof < params.hstrTraceDofs; ++outputDof)
        outgoing[outputDof] = 0.f;
    for (uint inputDof = 0; inputDof < params.hstrTraceDofs; ++inputDof)
    {
        const uint inputFace = inputDof / params.hstrFaceDofs;
        const uint spatialMode = inputDof % params.hstrFaceDofs;
        const uint ballisticOutput = (inputFace ^ 1u) * params.hstrFaceDofs + spatialMode;
        outgoing[ballisticOutput] += gHSTRCloud.hstrLeafBallistic[leaf * params.hstrTraceDofs + inputDof] * incident[inputDof];
        for (uint outputFace = 0; outputFace < 6; ++outputFace)
            outgoing[outputFace * params.hstrFaceDofs + spatialMode] +=
                gHSTRCloud.hstrLeafNearScatter[(leaf * params.hstrTraceDofs + inputDof) * 6 + outputFace] * incident[inputDof];
    }
    for (uint mode = 0; mode < gHSTRCloud.hstrLeafDiffuseRanks[leaf]; ++mode)
    {
        float3 coefficient = 0.f;
        for (uint inputDof = 0; inputDof < params.hstrTraceDofs; ++inputDof)
            coefficient +=
                gHSTRCloud
                    .hstrLeafDiffuseRight[params.hstrTraceDofs * params.hstrTraceDofs * leaf + params.hstrTraceDofs * inputDof + mode] *
                incident[inputDof];
        for (uint outputDof = 0; outputDof < params.hstrTraceDofs; ++outputDof)
            outgoing[outputDof] +=
                gHSTRCloud
                    .hstrLeafDiffuseLeft[params.hstrTraceDofs * params.hstrTraceDofs * leaf + params.hstrTraceDofs * outputDof + mode] *
                coefficient;
    }
    const float correctionFade = saturate(float(params.frameIndex + 1) / max(1.f, float(params.correctionFadeFrames)));
    const uint2 correctionRange = gHSTRCloud.hstrCorrectionRanges[leaf];
    float admittedError = 0.f;
    for (uint i = 0; i < correctionRange.y; ++i)
    {
        const HSTRCorrectionAtom atom = gHSTRCloud.hstrCorrectionAtoms[correctionRange.x + i];
        outgoing[atom.row] += correctionFade * atom.value * incident[atom.col];
        admittedError += correctionFade * atom.goalError;
    }
    float3 incidentFlux = 0.f;
    float3 outgoingFlux = 0.f;
    for (uint dof = 0; dof < params.hstrTraceDofs; ++dof)
    {
        incidentFlux += max(0.f, incident[dof]);
        outgoing[dof] = max(0.f, outgoing[dof]);
        outgoingFlux += outgoing[dof];
    }
    const float3 passiveScale = min(1.f, incidentFlux / max(float3(1e-8f), outgoingFlux));
    for (uint dof = 0; dof < params.hstrTraceDofs; ++dof)
        outgoing[dof] *= passiveScale;
    const float omittedCorrectionError = max(0.f, gHSTRCloud.hstrLeafResidualBounds[leaf] - admittedError);
    for (uint outputDof = 0; outputDof < params.hstrTraceDofs; ++outputDof)
        gHSTRCloud.hstrRadiance[params.hstrTraceDofs * leaf + outputDof] = float4(outgoing[outputDof], omittedCorrectionError + omittedError);
}

// GPU sun bake scheduling (CloudResidencyDesc::gpuSun), in this order every frame; see HSTRCloudRenderer::scanSunBakes.
[numthreads(64, 1, 1)]
void releaseSunBakes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.releaseSunBakes(dispatchThreadId.x);
}

[numthreads(256, 1, 1)]
void resetSunBlocks(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.resetSunField(dispatchThreadId.xy, false);
}

[numthreads(64, 16, 1)]
void resetSunNodes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.resetSunField(dispatchThreadId.xy, true);
}

[numthreads(64, 4, 1)]
void ageSunField(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.ageSunField(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void stampSunChanges(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.stampSunChange(dispatchThreadId.x);
}

[numthreads(256, 1, 1)]
void scanSunBakes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.scanSunBakes(dispatchThreadId.x);
}

[numthreads(1, 1, 1)]
void selectSunBakes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.selectSunBakes();
}

[numthreads(256, 1, 1)]
void emitSunBakes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.emitSunBakes(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void evictSunBakes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.evictSunBakes(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void assignSunBakes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.assignSunBake(dispatchThreadId.x);
}

// The cloud sea's domain proxy (HSTRCloud::uploadDomainExtinction): one dispatch per stage over every changed slot, each after the
// stage it reads. Thread z is a slot's region depth times its index plus the offset in it; regions wrap around the toroidal domain
// in x and z. What each stage writes is what the CPU built before, from the tiles' float16 volumes.

/// float16_t(value), rounded up: majorants must bound what they cover after the half-float write.
float domainRoundedUp(float value)
{
    const float rounded = f16tof32(f32tof16(value));
    return rounded < value ? value * (1.f + 1.f / 1024.f) : value;
}

/// The block (in a region starting `border` blocks before its tile) a thread writes, wrapped; false past the region.
bool domainRegionBlock(uint3 id, uint width, uint height, uint border, uint unit, out uint3 cell)
{
    cell = 0;
    const HSTRCloudDomainFrame frame = gHSTRCloud.hstrDomainFrame[0];
    const uint region = id.z / width;
    if (region >= frame.regionCount || id.x >= width || id.y >= height)
        return false;
    const uint slot = gHSTRCloud.hstrDomainRegions[region] & kDomainRegionSlot;
    const uint tileCells = frame.tileVoxels / unit;
    const uint cellsX = frame.tiles * tileCells;
    const uint x = (slot % frame.tiles * tileCells + cellsX - border + id.x) % cellsX;
    const uint z = (slot / frame.tiles * tileCells + cellsX - border + id.z % width) % cellsX;
    cell = uint3(x, id.y, z);
    return true;
}

/// A changed tile's voxels: its packed volume (region i of the batch is volume i of hstrDomainStaged) and their extinction.
[numthreads(8, 8, 8)]
void domainExtinction(uint3 id: SV_DispatchThreadID)
{
    const HSTRCloudDomainFrame frame = gHSTRCloud.hstrDomainFrame[0];
    uint3 voxel;
    if (!domainRegionBlock(id, frame.tileVoxels, frame.dims.y, 0u, 1u, voxel))
        return;
    const uint r = frame.tileVoxels;
    const uint region = id.z / r;
    const uint flags = gHSTRCloud.hstrDomainRegions[region];
    float2 value = 0.f;
    if ((flags & kDomainRegionKeep) != 0)
        value = gHSTRCloud.hstrDomainVolumeOutput[voxel];
    else if ((flags & kDomainRegionZero) == 0)
    {
        const uint packed = gHSTRCloud.hstrDomainStaged[(region * r + id.z % r) * frame.dims.y * r + id.x + r * id.y];
        value = float2(f16tof32(packed & 0xffff), f16tof32(packed >> 16));
        gHSTRCloud.hstrDomainVolumeOutput[voxel] = value;
    }
    else
        gHSTRCloud.hstrDomainVolumeOutput[voxel] = value;
    gHSTRCloud.hstrExtinctionOutput[voxel] = value.x * frame.scale;
}

/// Block maxima over 4^3 voxels plus the voxels one past the upper faces (trilinear support), for the blocks that read a changed
/// tile: from one block before it.
[numthreads(4, 4, 4)]
void domainBlocks(uint3 id: SV_DispatchThreadID)
{
    const HSTRCloudDomainFrame frame = gHSTRCloud.hstrDomainFrame[0];
    uint3 block;
    if (!domainRegionBlock(id, frame.tileVoxels / kDomainBlock + 1u, frame.blockDims.y, 1u, kDomainBlock, block))
        return;
    float meanValue = 0.f;
    float maxValue = 0.f;
    const uint yLast = min(block.y * kDomainBlock + kDomainBlock, frame.dims.y - 1u);
    for (uint z = 0; z <= kDomainBlock; ++z)
        for (uint y = block.y * kDomainBlock; y <= yLast; ++y)
            for (uint x = 0; x <= kDomainBlock; ++x)
            {
                const int4 voxel = int4((block.x * kDomainBlock + x) % frame.dims.x, y, (block.z * kDomainBlock + z) % frame.dims.z, 0);
                const float2 value = gHSTRCloud.hstrDomainVolume.Load(voxel);
                meanValue = max(meanValue, value.x);
                maxValue = max(maxValue, max(value.x, value.y));
            }
    gHSTRCloud.hstrDomainBlocksOutput[block] = float2(maxValue, meanValue);
}

/// The camera majorant (block maxima dilated by one block, wrapping) and the tight one (undilated block means) of the blocks whose
/// maxima changed: from two blocks before the tile.
[numthreads(4, 4, 4)]
void domainMajorant(uint3 id: SV_DispatchThreadID)
{
    const HSTRCloudDomainFrame frame = gHSTRCloud.hstrDomainFrame[0];
    uint3 block;
    if (!domainRegionBlock(id, frame.tileVoxels / kDomainBlock + 3u, frame.blockDims.y, 2u, kDomainBlock, block))
        return;
    float value = 0.f;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                const int ny = int(block.y) + y;
                if (ny < 0 || ny >= int(frame.blockDims.y))
                    continue;
                const uint3 n = uint3(
                    (int(block.x) + x + int(frame.blockDims.x)) % int(frame.blockDims.x), ny,
                    (int(block.z) + z + int(frame.blockDims.z)) % int(frame.blockDims.z)
                );
                value = max(value, gHSTRCloud.hstrDomainBlocks.Load(int4(n, 0)).x);
            }
    gHSTRCloud.hstrMajorantOutput[block] = domainRoundedUp(value * frame.scale);
    gHSTRCloud.hstrTightMajorantOutput[block] = domainRoundedUp(gHSTRCloud.hstrDomainBlocks.Load(int4(block, 0)).y * frame.scale);
}

/// Occupancy (any block maximum) and majorant-zero (any dilated majorant) of the 16-voxel cells over the changed majorant blocks:
/// from one cell before the tile.
[numthreads(4, 4, 4)]
void domainOccupancy(uint3 id: SV_DispatchThreadID)
{
    const HSTRCloudDomainFrame frame = gHSTRCloud.hstrDomainFrame[0];
    uint3 cell;
    const uint unit = kDomainBlock * kDomainOccupancy;
    if (!domainRegionBlock(id, frame.tileVoxels / unit + 2u, (frame.blockDims.y + kDomainOccupancy - 1u) / kDomainOccupancy, 1u, unit, cell))
        return;
    uint occupied = 0;
    uint majorant = 0;
    for (uint z = 0; z < kDomainOccupancy; ++z)
        for (uint y = 0; y < kDomainOccupancy; ++y)
            for (uint x = 0; x < kDomainOccupancy; ++x)
            {
                const uint3 block = cell * kDomainOccupancy + uint3(x, y, z);
                if (block.y >= frame.blockDims.y)
                    continue;
                occupied |= gHSTRCloud.hstrDomainBlocks.Load(int4(block, 0)).x > 0.f ? 1u : 0u;
                majorant |= gHSTRCloud.hstrMajorant.Load(int4(block, 0)) > 0.f ? 1u : 0u;
            }
    gHSTRCloud.hstrOccupancyOutput[cell] = occupied;
    gHSTRCloud.hstrMajorantZeroOutput[cell] = majorant;
}
