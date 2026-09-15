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
};

struct HSTRCloudAsset
{
    uint3 dims; ///< Source voxels, multiples of 128.
    uint directoryOffset;
    uint3 chunkDims;
    uint topLevel;
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
    uint pad;
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
    float minStepVoxels = 1.f;     ///< Midpoint quadrature of trilinear extinction is accurate at one-voxel steps at any density.
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
    uint beamTileSize = 16;       ///< Pixels per edge of the coarsest beam tiles.
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
    uint beamSegments = 4;     ///< Threads that march consecutive segments of one query ray in parallel (divides 64).
    uint beamPassLevel = 0;    ///< Tile level the current beam query or tile pass processes.

    uint2 beamLevelDims = { 0, 0 }; ///< Level map, one texel per finest tile.
    uint beamLatticeStep = 2;       ///< Pixels between lattice texels: half the finest tile.
    uint beamTemporal = 0;          ///< 1: query every level in one pass over the tiles refined last frame (one frame of lag).

    uint compareBlock = 1; ///< Comparisons average compareBlock x compareBlock pixel blocks first (1: per pixel).
    float worldCacheModulation = 0.f;       ///< b of the cache's exp(-b tau_sun) voxel-resolution modulation (0: off); set per frame.
    float worldCacheModulationDepth = 8.f;  ///< Sun optical depth at which the modulation stops falling.
    uint padHSTR26 = 0;

    uint worldCacheZonalBands = 0;  ///< >0: the sun field's zonal part about the sun direction is windowed on its own and extended
                                    ///< by zonal bands 3 up to this (at most 8); 0: off.
    uint worldCacheZonalWindow = 0; ///< Window order of that zonal part (0: max(worldCacheZonalBands, 2)).
    uint worldCacheSunOrder = 2;    ///< 3: the sun field carries all of band 3 (7 extension slots) instead of zonal bands.
    uint padHSTR27 = 0;

    uint padHSTR28 = 0;
    float sunNearVoxels = 0.f; ///< Camera samples march the sun optical depth exactly over this many voxels towards the sun, then
                               ///< read the voxel-centre field; 0: one-voxel midpoint step onto the field (enough for the
                               ///< adaptive march, which integrates the sun attenuation analytically).
    uint padHSTR29 = 0;
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
    uint padHSTRCloud2 = 0;
    uint3 fineSunSize = { 0, 0, 0 };   ///< Voxels of that region (0: the whole grid).
    uint padHSTRCloud3 = 0;

    float3 cloudAtlasInvSize = float3(1.f); ///< Reciprocal texel dimensions of the brick atlas.
    float worldCacheKeep = 1.f;              ///< Share of the world cache deposits decayWorldCache keeps.
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
