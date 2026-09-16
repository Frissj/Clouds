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
    /// MEASURED with 16 observations per step (4K, beam view, against the per-pixel 1-voxel march), and the answer is that the pair
    /// buys one doubling of the step at equal quality, consistently, on both views. Sea: a 4-voxel step is p99.9 0.072 as it stands,
    /// 0.051 with the exact optical depth, and 0.026 with the centroid as well - which is exactly the 2-voxel step's 0.026. Near:
    /// 0.122, 0.072, 0.026 against the 2-voxel step's 0.036, so slightly better than the baseline. Doubling again keeps the pattern:
    /// an 8-voxel step with both lands on 0.072 on each view, which is what a 4-voxel step costs without them. The marched share
    /// never moves (3.9-4.0% sea, 5.2-5.3% near), so none of this is bought by refining more of the image.
    ///
    /// The split matters as much as the total. The optical depth is about a third of the gap and the centroid is the rest, so the
    /// source error that survives an exact tau is mostly a question of where the step's one source evaluation is placed, not of
    /// fine structure the step cannot represent. That is what makes the result worth anything: two scalars per position and
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
