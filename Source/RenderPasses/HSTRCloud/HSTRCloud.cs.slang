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
#ifndef HSTR_SUN_LIVE
#define HSTR_SUN_LIVE 1
#endif

/// A mask of groups of the march's experimental switches compiled to their shipping values (HSTRCloud::beamShipping says when that
/// is exact): a cbuffer value cannot be folded, so every one of them otherwise stays a live branch in the beam programs. The groups
/// exist because folding them all slowed one view while speeding up another.
#ifndef HSTR_SHIP
#define HSTR_SHIP 0
#endif
#ifndef HSTR_BEAM_SPARSE_MIN_LEVEL
#define HSTR_BEAM_SPARSE_MIN_LEVEL 2
#endif
#ifndef HSTR_BEAM_SPARSE_CUT
#define HSTR_BEAM_SPARSE_CUT 1
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
    RWTexture2D<float4> hstrBeamGuardCamera;
    /// The blocks whose certificate failed this build, compacted, with the count and the indirect arguments that dispatch them.
    RWStructuredBuffer<uint> hstrBeamDirty;
    RWStructuredBuffer<uint> hstrBeamDirtyCount;
    RWStructuredBuffer<uint> hstrBeamDirtyArgs;
    /// The coarse certificate level: xyz is the camera the cell was last rebuilt from, w the radius it is safe within.
    RWTexture2D<float4> hstrBeamCoarse;
    /// Per coarse cell: the children its certificate does NOT cover (low 16 bits) and whether it is queued for a rebuild (bit 16).
    RWTexture2D<uint> hstrBeamCoarseState;
    RWStructuredBuffer<uint> hstrBeamDescend;      ///< Guard blocks this build has to classify one by one.
    RWStructuredBuffer<uint> hstrBeamDescendCount; ///< [0] blocks in hstrBeamDescend, [1] cells in hstrBeamRebuild.
    RWStructuredBuffer<uint> hstrBeamRebuild;      ///< Coarse cells whose certificate this build has to rebuild.
    /// Reference frame: per 8 x 8 resolve group holding failed-tile pixels, its packed group position and a 64-bit pixel mask.
    RWStructuredBuffer<uint> hstrBeamResidualList;
    RWStructuredBuffer<uint> hstrBeamResidualArgs; ///< [0] groups queued, [1..3] the residual resolve's dispatch.
    RWTexture2DArray<float4> hstrBeamLatticeOutput;
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

    float cloudEmptyExit(CutRay ray, float t, CloudSample box)
    {
        const HSTRCloudInstance instance = hstrCloudInstances[box.instance];
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
        if (instance.occupied == 0)
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
        return true;
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
        for (uint k = 0; k < kCloudSunAncestors && sunSlot == kCloudRefNone && hstrCloudBricks[entry].parent != kCloudRefNone; ++k)
        {
            entry = hstrCloudBricks[entry].parent;
            sunSlot = cloudSunSlotOf(entry, sunClass);
        }
        return sunSlot;
    }

    /// Baked sun optical depth of a brick at source position x from its bake in sun atlas slot sunSlot, in source voxel lengths.
    float cloudBrickSun(uint brickIndex, uint sunSlot, float3 x)
    {
        const uint level = hstrCloudBricks[brickIndex].info & 15u;
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
        // Voxel edge (domain voxels) of the density the last sample read: fine bricks shorten the minimum step to their voxels.
        float fineVoxels = 1.f;
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
        ray.sunOctaves = float4(1.f, 0.f, 0.f, 0.f);
        float transmittance = 1.f;
        uint steps = 0;
        float opacityDistance = 0.f;
#if (HSTR_SHIP & 512) != 0
        marchBeamShipping(ray, segment, beamWithCache(), beamWithSingle(), result.cache, result.single, transmittance, steps, opacityDistance);
#else
        marchSegmentMoment(ray, segment, beamWithCache(), beamWithSingle(), result.cache, result.single, transmittance, steps, opacityDistance);
#endif
        result.depth = -log(max(transmittance, 1e-12f));
        result.steps = float(steps);
        result.distance = transmittance < 1.f - 1e-4f ? opacityDistance / (1.f - transmittance) : 0.f;
        return result;
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

    bool beamWithCache() { return (params.hstComponents & (kComponentSunMultiple | kComponentSkyScattered)) != 0; }

    bool beamWithSingle() { return (params.hstComponents & kComponentSunSingle) != 0; }

    uint beamTileEdge(uint level) { return params.beamTileSize >> level; }

    /// Whether a box of the beam image is one the screen can see. With the reference frame off the beam image IS the screen and
    /// everything qualifies; with it on, the image reaches past the screen so that turning does not leave it, and that surplus is
    /// never built - a direction only earns a basis when it turns on screen, and then it has no history to carry anyway.
    bool beamBoxOnScreen(float2 low, float2 high)
    {
        if (params.beamRefFrame == 0)
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

    bool beamTileOnScreen(uint level, uint2 tile)
    {
        const float s = float(beamTileEdge(level));
        return beamBoxOnScreen(float2(tile) * s, (float2(tile) + 1.f) * s);
    }

    /// A basis point is kept while any tile that could read it is on screen. beamTransmittanceRange reads the corners and centres
    /// of a tile's eight neighbours at level 0, so the reach is two tiles, not one - at one tile the tests on the outermost built
    /// tiles read lattice this build never wrote. Widened for that reason alone: it did NOT move the 720p screen-edge error the
    /// reference frame shows (max log 7.88e-2 on about 0.002% of pixels, bit-identical before and after), which is still open.
    bool beamPointOnScreen(uint2 position)
    {
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
            bool refine = beamCornerCurvature(tile, sparse) > params.beamTolerance;
            if (!refine && params.beamEdgeContrast > 0.f)
                refine = beamTransmittanceRange(level, tile, sparse) > params.beamEdgeContrast;
            return refine;
        }
        bool refine = maxComponent(abs(log(1.f + max(predictedCentre, 0.f)) - log(1.f + exactCentre))) > params.beamTolerance;
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

    /// Beam view pixel reconstructed from the tile level that accepted it: bilinear corners plus the centre's bubble. No voxel
    /// is touched. Pixels of tiles that failed the finest level are left to marchBeamPixel. Returns whether the pixel is left to
    /// the reference frame's residual resolve instead.
    bool resolveBeamPixel(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return false;
        const float3 w = cameraRayDirection(float2(pixel) + 0.5f);
        float2 beamPosition;
        if (!beamPixelPosition(float2(pixel) + 0.5f, beamPosition))
            return false;
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
    void marchBeamUnitAt(uint2 unit)
    {
        // MEASURED and REVERTED: bounding a SWEPT residual to the box the screen covers, on the theory that dispatching a thread
        // per texel of the octahedral sphere to have almost all of them fail the test below was the 1.28 ms of a 1.79 ms sprint
        // frame. It was not - sprint went 1.79 to 1.85 - and it cost coverage: the resolve reads a 2x2 neighbourhood, so units
        // just outside the box are read after all, and a parked octahedral frame went from 1.663% of pixels over 0.02 to 7.790%.
        // The residual's cost under translation is units genuinely re-marching, not threads arriving to be rejected.
        if (any(unit >= params.beamFrameDim) || !beamBoxOnScreen(float2(unit), float2(unit) + 1.f))
            return;
        // MEASURED and REMOVED: marching a one-unit apron around every failed tile, so the resolve's bilinear never reaches into
        // an accepted neighbour's unwritten units. Failed tiles are 4 units square and scattered, so a one-unit border is 2.25x
        // their area and the march went 0.59 -> 1.71 ms parked and 0.75 -> 3.39 under yaw - and it did not move the seam it was
        // aimed at (720p max log error 7.88e-2, bit-identical before and after), so the seam is something else.
        const uint finest = beamTileEdge(params.beamLevels - 1);
        if (hstrBeamLevel[unit / finest] < params.beamLevels)
            return; // The basis carries this tile; the resolve reconstructs it.
        const float3 w = beamRayDirection(float2(unit) + 0.5f);
        const float3 albedo = gScene.getGridVolume(0).data.albedo;
        // Carried where THIS unit already holds a value marched for this direction, and the camera has not moved past its
        // parallax. The residual image is persistent, so the question is about the unit itself rather than about what the
        // previous build happened to cover: alpha < 0 means never marched since the frame was anchored, and that is the only
        // thing that makes a unit new. That replaces two conditions that were really proxies for it - the previous level map
        // (did this tile fail last build) and beamPrevScreenBounds (did the previous build reach this far) - and it lets a unit
        // that leaves the screen and comes back carry, which neither proxy could express.
        if (params.beamRefValid != 0 && params.beamRefresh != 0 && !beamRefreshes(unit / finest))
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
                    return;
                }
            }
        }
        const BeamSample exact = marchBeam(w, float2(0.f, 1.f));
        hstrBeamPixelOutput[unit] = float4(beamRadiance(exact.cache, exact.single, exact.depth, w, albedo), exact.distance);
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
        const bool stationary =
            params.beamStationary != 0 || params.beamAssumeCarry != 0 || beamBlockCertified(position);
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

    /// Whether translation since this block was last verified is small enough that none of its points can have failed.
    ///
    /// A sample stored at distance s for direction w_s is asked to stand for a point at predicted distance d along w. Under a
    /// camera displacement dP those two move by dot(dP, w_s) and dot(dP, w), so they can separate by at most |dP| * |w_s - w|,
    /// which over a block of angular reach theta is |dP| * theta. The per-point test accepts while the separation stays within
    /// beamDepthTolerance of the distance, so the block as a whole is safe while that bound stays under the tolerance times the
    /// SMALLEST distance any of its points holds. Everything here is conservative: an unverified block (depth still at the clear
    /// value) is never certified, and a stale minimum is always too small rather than too large.
    bool beamBlockCertified(uint2 position) { return beamCellCertified(beamGuardCell(position)); }

    bool beamCellCertified(uint2 cell)
    {
        if (params.beamGuard == 0 || params.beamRefFrame == 0 || params.beamHistoryValid == 0)
            return false;
        const uint stored = hstrBeamGuardDepth[cell];
        if (stored == 0xFFFFFFFFu)
            return false; // Never verified, so nothing is known about it.
        const float minimum = asfloat(stored);
        if (!(minimum > 0.f))
            return false;
        const float4 verified = hstrBeamGuardCamera[cell];
        if (verified.w < 0.5f)
            return false;
        const float moved = length(gScene.camera.data.posW - verified.xyz);
        return moved * params.beamGuardAngle <= params.beamDepthTolerance * minimum;
    }

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
    float beamCellRadius(uint2 cell)
    {
        const uint stored = hstrBeamGuardDepth[cell];
        if (stored == 0xFFFFFFFFu || hstrBeamGuardCamera[cell].w < 0.5f)
            return -1.f;
        const float minimum = asfloat(stored);
        return minimum > 0.f ? params.beamDepthTolerance * minimum / max(params.beamGuardAngle, 1e-9f) : -1.f;
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
        // The block's pixel extent, with the same two-tile reach beamPointOnScreen keeps, so a block on the edge of the screen is
        // not dropped by a test that its own points would have passed.
        const float edge = float(params.beamDirtyBlockEdge);
        const float2 low = float2(cell) * edge;
        if (!beamBoxOnScreen(low - 2.f * float(params.beamTileSize), low + edge + 2.f * float(params.beamTileSize)))
            return false;
        if (beamCellCertified(cell))
            return false;
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
            hstrBeamDirty[slot] = (cell.y << 16) | (cell.x & 0xFFFFu);
        uint counted;
        InterlockedAdd(hstrBeamCounts[kBeamDirtyBlocks], 1u, counted);
        return true;
    }

    /// One thread per beam unit of a listed block: the residual, over the same blocks.
    void marchBeamDirtyUnit(uint thread)
    {
        const uint edge = max(params.beamDirtyBlockEdge, 1u);
        const uint per = edge * edge;
        const uint index = thread / per;
        if (index >= min(hstrBeamDirtyCount[0], params.beamDirtyCapacity))
            return;
        const uint packed = hstrBeamDirty[index];
        const uint local = thread % per;
        marchBeamUnitAt(uint2(packed & 0xFFFFu, packed >> 16) * edge + uint2(local % edge, local / edge));
    }

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
        if (params.beamCoarseShift != 0)
            hstrBeamCoarse[cell >> params.beamCoarseShift] = float4(0.f);
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
    gHSTRCloud.hstrBeamDirtyArgs[0] = (blocks * rb * rb * passes * segments + 63u) / 64u;
    gHSTRCloud.hstrBeamDirtyArgs[1] = 1u;
    gHSTRCloud.hstrBeamDirtyArgs[2] = 1u;
    gHSTRCloud.hstrBeamDirtyArgs[3] = (blocks * edge * edge + 63u) / 64u;
    gHSTRCloud.hstrBeamDirtyArgs[4] = 1u;
    gHSTRCloud.hstrBeamDirtyArgs[5] = 1u;
    // The coarse cells to rebuild are only all known once the blocks have been classified, which is now.
    const uint cells = min(gHSTRCloud.hstrBeamDescendCount[1], gHSTRCloud.params.beamCoarseCapacity);
    gHSTRCloud.hstrBeamDirtyArgs[9] = (cells + 63u) / 64u;
    gHSTRCloud.hstrBeamDirtyArgs[10] = 1u;
    gHSTRCloud.hstrBeamDirtyArgs[11] = 1u;
}

groupshared uint gDirtyRay[64];      ///< Slice 0's answer for the ray: its position packed, or ~0 when nothing is marched.
groupshared float2 gDirtyExtra[64];  ///< Per slice: opacity-weighted distance times opacity, and steps.

/// The dirty query: every corner and centre of every listed block, in one dispatch, with beamDirtySegments threads per ray.
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
void buildBeamDirtyQueries(uint3 dispatchThreadId: SV_DispatchThreadID, uint3 groupThreadId: SV_GroupThreadID)
{
    const uint segments = clamp(gHSTRCloud.params.beamDirtySegments, 1u, 64u);
    const uint slice = dispatchThreadId.x % segments;
    const uint lead = groupThreadId.x - slice;
    if (slice == 0)
    {
        uint2 point;
        bool centre;
        uint2 position;
        bool march = gHSTRCloud.beamDirtyRay(dispatchThreadId.x / segments, point, centre) &&
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
}

[numthreads(64, 1, 1)]
void marchBeamDirtyUnits(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.marchBeamDirtyUnit(dispatchThreadId.x);
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
    gHSTRCloud.hstrBeamResidualArgs[1] = min(groups, kResidualRow);
    gHSTRCloud.hstrBeamResidualArgs[2] = (groups + kResidualRow - 1u) / kResidualRow;
    gHSTRCloud.hstrBeamResidualArgs[3] = 1u;
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
