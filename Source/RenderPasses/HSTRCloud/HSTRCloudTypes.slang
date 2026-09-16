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

static const uint kWorldCacheView = 8;     ///< debugView of the world-space radiance cache experiment.
static const uint kWorldCacheSlots = 25;   ///< Scalar slots per cache cell: unit sun SH 0-8, unit sky SH 9-17, sun extension 18-24.
static const uint kWorldCacheTextures = 7; ///< RGBA textures holding the cache slots (slot s in texture s / 4, channel s % 4).
static const uint kBeamView = 9;           ///< debugView of the world-cache frame reconstructed from hierarchical beam tiles.
static const uint kBeamMaxLevels = 4;      ///< Beam tile levels, each halving the tile edge.
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
    uint occupied;
    uint sunClass; ///< Orientation class (the signed permutation of the rows): instances of one class share baked sun depth.
    uint padInstance0;
    uint padInstance1;
    uint padInstance2;
};

struct HSTRCloudAsset
{
    uint3 dims; ///< Source voxels, multiples of 128.
    uint directoryOffset;
    uint3 chunkDims;
    uint topLevel;
    float sourceVoxelWorld; ///< World size of one source voxel at instance scale 1.
    uint padAsset0;
    uint padAsset1;
    uint padAsset2;
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
    uint padBrick0;
    uint padBrick1;
};

/// hstrCloudOccupancy words per brick: a 4-bit density bound per 2-voxel cell (cloudCellCode) in words 0-7, then the bound of each of
/// the brick's eight 4-voxel octants in word 8 (the brick's own bound is the maximum of those eight).
static const uint kCloudCellWords = 9;
static const uint kCloudSunAncestors = 2;    ///< Levels a sample of an unbaked brick climbs for a baked ancestor before the live march.
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
    uint beamTileSize = 16;
    uint beamLevels = 3;          ///< Beam tile levels; tiles that fail at the finest level are marched per pixel.
    float beamTolerance = 0.02f;  ///< log(1 + radiance) surplus at a tile centre above which the tile is refined.

    uint worldCacheBands = 2; ///< Highest SH band the light tracer deposits (1: 4 coefficients per field, 2: 9).
    uint worldCacheWindow = 1; ///< 1: Hanning-window the evaluated SH bands against truncation ringing.
    uint worldCacheTextured = 1; ///< 1: the camera reads the cache from baked, hardware-filtered 3D textures.
    uint lightingStride = 1;     ///< The camera march re-evaluates the cache every lightingStride steps and holds it between.

    uint2 beamTileDims = { 0, 0 };    ///< Coarsest beam tiles.
    uint2 beamLatticeDims = { 0, 0 }; ///< Query lattice, one texel per beamLatticeStep pixels; every tile corner and centre is a texel.

    float beamEdgeContrast = 0.05f; ///< Transmittance range of the queries around a tile above which it is refined (silhouettes); 0: off.
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
    uint cloudZeroSkip = 1;
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
    uint padHSTRCloud11 = 0;
    /// Domain voxels a camera march may move before it queries the sun depth again (0: every sample). MEASURED (4K c15, 0.1 voxels,
    /// which reuses about every second query): near 42.8 -> 42.3 ms, sea 30.4 -> 33.0 ms - nothing. Halving the sun fetches does not
    /// pay, the same way the brick context did not: what a sample costs is the sample, not the fetches inside it.
    float cloudSunReuse = 0.f;
    /// Camera transmittance at which a march stops (the rest of the ray is behind opaque cloud). MEASURED (4K c15, 0.001 -> 0.02):
    /// near 42.6 -> 38.8 ms, sea 28.6 -> 30.4 ms, so the long-tailed rays are not what the frame waits for either.
    float cloudMinTransmittance = 1e-3f;
    uint padHSTRCloud9 = 0;
};

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
