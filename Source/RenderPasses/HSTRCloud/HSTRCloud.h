/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "Rendering/Volumes/HSTRHierarchy.h"
#include "RenderGraph/RenderPass.h"
#include "HSTRCloudTypes.slang"
#include "CloudResidency.h"

#include <array>
#include <memory>

using namespace Falcor;

/** Hierarchical Schur transport renderer for heterogeneous cloud volumes.
 *
 * Lighting is solved through persistent spatial-angular six-face boundary
 * operators. NanoVDB supplies visibility and unresolved residual detail only.
 *
 * The image is produced in three bands so that no transport work runs per pixel:
 * - World space (sun or density changes): the HST leaf solve and fine residual pages
 *   carrying the ballistic sun transmittance for the forward-peaked single scattering
 *   that the isotropic boundary basis cannot represent.
 * - Camera space (camera changes): the camera lighting basis, the projected cut with
 *   split/merge hysteresis, and one tile camera basis per 8x8 tile fitted to tile-corner
 *   and tile-centre queries, with a hierarchical-surplus error flag.
 * - Every pixel: full-resolution transmittance through the cut, multiplied by the
 *   reconstructed tile basis, or the exact per-pixel integral in refined tiles.
 */
class HSTRCloud : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(HSTRCloud, "HSTRCloud", "Hierarchical Schur transport for heterogeneous clouds.");

    static ref<HSTRCloud> create(ref<Device> pDevice, const Properties& props) { return make_ref<HSTRCloud>(pDevice, props); }

    HSTRCloud(ref<Device> pDevice, const Properties& props);

    Properties getProperties() const override;
    void setProperties(const Properties& props) override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;

private:
    struct CorrectionAtom
    {
        uint32_t row = 0;
        uint32_t col = 0;
        float value = 0.f;
        float goalError = 0.f;
    };
    static_assert(sizeof(CorrectionAtom) == 16);
    static_assert(sizeof(HSTRCutNode) == 96);
    static_assert(sizeof(HSTRTileBasis) == 96);

    void parseProperties(const Properties& props);
    void buildHierarchy();
    void buildCloudDomain();
    void uploadDomainExtinction(const std::vector<uint32_t>& slots);
    void updateCloudDomain(RenderContext* pRenderContext);
    void updateSunVoxelDirection();
    void onLightingChanged(const HSTRCloudParams& previous);
    void createSamplers();
    void updateHierarchy();
    void uploadHierarchy();
    void uploadExtinction();
    void uploadCorrectionPool(const hstr::DenseMatrix& rootIncident);
    void solveLighting();
    void dispatchLightingSolve();
    void dispatchResidualPages(RenderContext* pRenderContext);
    bool mSunPagesDirty = false;
    std::vector<uint32_t> mSunPageSlots; ///< Sea tiles whose sun pages are stale (all tiles when mResidualDirty).
    void bindRenderer(RenderContext* pRenderContext, const ref<ComputePass>& pPass);
    void saveReference(RenderContext* pRenderContext, const std::string& path);
    void loadReference(RenderContext* pRenderContext, const std::string& path);
    void ensureCameraResources();
    std::vector<float> sampleLeafDensities() const;
    hstr::DenseMatrix makeNestedLeafTransport(uint32_t leafIndex) const;

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;
    ref<ComputePass> mpSolvePass;
    ref<ComputePass> mpCameraLightingPass;
    ref<ComputePass> mpProjectPass;
    ref<ComputePass> mpCutPass;
    ref<ComputePass> mpSortPass;
    ref<ComputePass> mpQueryPass;
    ref<ComputePass> mpTileBasisPass;
    ref<ComputePass> mpFineSunPass;
    ref<ComputePass> mpResidualMaskPass;
    ref<ComputePass> mpGatherOctavesPass;
    ref<ComputePass> mpBlurOctavesPass;
    ref<ComputePass> mpReferencePass;
    ref<ComputePass> mpCompareReferencePass;
    ref<Texture> mpReferenceSum;     ///< Running sums of path-traced reference frames (rgb) and their count (a), per component and half.
    ref<Buffer> mpReferenceRowError; ///< Per-row mean error of the HST frame against the reference average.
    ref<Buffer> mpReferenceRowHistogram; ///< Exact comparisons: per-row log error histogram and maximum.
    bool mCompareReference = false;
    float mReferenceLogP999 = -1.f; ///< Exact comparisons: 99.9th percentile log error (upper edge of its bin).
    float mReferenceLogMax = -1.f;  ///< Exact comparisons: largest pixel log error.
    float mReferenceError = -1.f;         ///< Mean |HST - reference| in linear radiance.
    float mReferenceLogError = -1.f;      ///< Mean |log(1 + HST) - log(1 + reference)|.
    float mReferenceNoiseError = -1.f;    ///< Expected linear error of the reference average itself, from its two halves.
    float mReferenceNoiseLogError = -1.f; ///< Same in log(1 + radiance).
    ref<ComputePass> mpWorldCachePass;
    ref<ComputePass> mpWorldCachePhotonPass;
    ref<ComputePass> mpWorldCacheResolvePass;
    ref<Buffer> mpWorldCache;        ///< World-space radiance cache experiment: SH running sums per cell.
    ref<Buffer> mpWorldCacheDeposit; ///< Fixed-point light-tracing deposits of the current batch.
    ref<ComputePass> mpWorldCacheBakePass;
    ref<ComputePass> mpWorldCacheAdvancePass;
    ref<Buffer> mpPhotonPool;                                           ///< Persistent light-tracing photons (48 bytes each).
    ref<Buffer> mpPhotonEmitted;                                        ///< Photons emitted by the pool since the last restart.
    std::array<ref<Texture>, kWorldCacheTextures> mpWorldCacheTextures; ///< Cache means packed for hardware-filtered lookups.
    bool mWorldCacheBakeDirty = true;
    uint32_t mWorldCacheBakeInterval = 8; ///< Batches between bakes of the camera textures while updating.
    uint32_t mWorldCacheUpdates = 1;      ///< Cache gather passes per frame in the world cache view.
    float mWorldCacheModulation = -1.f;   ///< Cache modulation b; negative: the medium's diffusion attenuation.
    ref<ComputePass> mpBeamQueryPass;
    ref<ComputePass> mpBeamTilePass;
    ref<ComputePass> mpBeamArgsPass;
    ref<ComputePass> mpBeamResolvePass;
    ref<ComputePass> mpBeamMarchPass;
    ref<Texture> mpBeamLattice; ///< Beam view queries at every tile corner and centre (2 slices).
    ref<Texture> mpBeamLevel;   ///< Level that finalised every finest beam tile.
    ref<ComputePass> mpBeamGuidePass;
    ref<Texture> mpBeamGuide; ///< Full-resolution beam guide: optical depth, distance and sun depth where it reaches one.
    ref<ComputePass> mpBeamTemporalTilePass;
    std::array<ref<Buffer>, 2> mpBeamLists;    ///< Refined tiles per level, this frame's and last frame's by parity.
    std::array<ref<Buffer>, 2> mpBeamCounts;   ///< Refined tile count per level, by parity.
    std::array<ref<Texture>, 2> mpBeamHistory; ///< Levels refined per finest cell, by parity.
    uint32_t mBeamParity = 0;                  ///< Index of this frame's lists, counts and history.
    bool mBeamHistoryValid = false;            ///< Last frame's beam lists belong to the current tile layout.
    bool mBeamReusable = false;                ///< The beam queries and tile levels of the last frame are still exact.
    uint32_t mWorldCacheBakes = 0;             ///< Bakes of the world cache textures, to detect lighting changes.
    uint32_t mBeamBakes = 0;                   ///< mWorldCacheBakes when the beam queries were last built.
    uint4 mBeamLayout = uint4(0);              ///< Frame size, tile size and levels the beam lists were built for.
    float3 mBeamCameraPosition = float3(0.f);
    float3 mBeamCameraTarget = float3(0.f);
    ref<Buffer> mpBeamArgs;            ///< Indirect arguments of the query and tile passes per level.
    ref<Texture> mpExactFrame;         ///< Stored frame that compareExact measures against.
    bool mStoreExact = false;          ///< Copy the next frame into mpExactFrame.
    float mBeamMarchedFraction = -1.f; ///< Share of pixels the beam view marched per pixel, read with the comparison.
    ref<Buffer> mpTransferProbe;       ///< Entry-cell masks per (asset brick, direction class) and two counters (cloudTransferClasses).
    uint32_t mTransferCrossings = 0;   ///< Brick crossings the last probed frame marched.
    uint32_t mTransferEntries = 0;     ///< Distinct entries a transfer cache would have had to produce for them.
    uint32_t mBeamLevelCounts[kBeamMaxLevels + 1] = {}; ///< Tiles entering each level, read with the comparison; the last is the
                                                        ///< per-pixel march list. Eight queries per entry, so these are what the
                                                        ///< query pass costs.
    std::string mSaveReferencePath;    ///< When set, the reference sums are written to <path>_s<slice>.exr at the end of the frame.
    std::string mLoadReferencePath;    ///< When set, the reference sums are read from <path>_s<slice>.exr at the start of the frame.
    float3 mReferencePosition = float3(0.f);
    float3 mReferenceDirection = float3(0.f);
    ref<Buffer> mpLeafRadiance;
    ref<Buffer> mpLeafBasisLeft;
    ref<Buffer> mpLeafBasisRight;
    ref<Buffer> mpLeafBallistic;
    ref<Buffer> mpLeafNearScatter;
    ref<Buffer> mpLeafDiffuseLeft;
    ref<Buffer> mpLeafDiffuseRight;
    ref<Buffer> mpLeafDiffuseRanks;
    ref<Buffer> mpLeafResidualBounds;
    ref<Buffer> mpRootIncident;
    ref<Buffer> mpLeafAdjointResponses;
    ref<Buffer> mpCorrectionRanges;
    ref<Buffer> mpCorrectionAtoms;
    ref<Buffer> mpCutNodes;
    ref<Buffer> mpCutNodeParents;
    ref<Buffer> mpNodeProjection;
    ref<Buffer> mpNodeDepth;
    ref<Buffer> mpNodeOrder; ///< BSP front-to-back key of every node for the current camera.
    ref<Buffer> mpTileNodeCounts;
    ref<Buffer> mpTileNodes;
    std::array<ref<Buffer>, 2> mpCutState; ///< Per-node acceptance of the previous and current cut, for hysteresis.
    ref<Texture> mpExtinction;
    ref<Texture> mpMajorant;
    ref<Texture> mpTightMajorant; ///< Undilated block maxima, for delta tracking.
    ref<Texture> mpOccupancy;     ///< Non-empty 16-voxel blocks, for delta tracking's empty-space skipping.
    ref<Texture> mpMajorantZero;  ///< 16-voxel blocks with any non-zero (dilated) majorant, for the camera march's zero-block skipping.
    ref<Texture> mpCameraLighting;
    ref<Texture> mpCameraQueries;
    ref<Texture> mpTileCenters;
    ref<Buffer> mpTileBasis;
    ref<Buffer> mpTileState;
    ref<Texture> mpFineSun;
    ref<Buffer> mpLeafResidual;
    std::array<ref<Texture>, 2> mpSunOctaves; ///< Ping-pong storage for the octave spread.
    ref<Texture> mpSunOctaveField;            ///< The spread octave field the renderer samples.
    ref<Sampler> mpExtinctionSampler;
    ref<Sampler> mpLinearSampler;
    hstr::Hierarchy mHierarchy;
    HSTRCloudParams mParams;
    uint3 mActualLeafDims = uint3(0);
    int3 mGridMin = int3(0);
    int3 mGridMax = int3(0);
    float3 mVoxelSize = float3(0.f);
    std::vector<float> mLeafDensity;
    // Cloud sea: an endless procedural layer of library clouds (CloudSea) with virtualized fine density (CloudResidency).
    std::string mCloudLibraryPath; ///< A cloud package, or a directory of them.
    std::vector<std::filesystem::path> mCloudLibraryFiles;
    uint32_t mCloudProxyResolution = 64; ///< Domain voxels per sea tile edge.
    uint32_t mCloudBrickPoolMB = 256;
    uint32_t mCloudPayloadPoolMB = 128; ///< GPU memory of the packed coefficients of resident pages.
    bool mCloudDirectStorage = true;    ///< Load page payloads with DirectStorage (GPU GDeflate, RTX IO); false: CPU decompression.
    uint32_t mCloudBrickLoadsPerFrame = 1024;
    uint32_t mCloudSeaTiles = 8;
    uint32_t mCloudSeaSeed = 1;
    float mCloudSeaCoverage = 0.85f;
    float mCloudLodPixels = 1.f;
    uint32_t mCloudFadeFrames = 8;
    bool mCloudVirtual = true;
    std::string mCloudSeaKey;       ///< Settings the resident sea was built for.
    std::string mCloudResidencyKey; ///< Settings the residency was built for.
    std::unique_ptr<hstrcloud::CloudSea> mpCloudSea;
    std::unique_ptr<hstrcloud::CloudResidency> mpCloudResidency;
    bool mCloudInstancesUploaded = false;
    uint32_t mCloudFrames = 0;
    float mSeaViewDistance = 4000.f; ///< Requested view distance; the sea clamps it to its resident window.
    float mCloudCacheKeep = 1.f;         ///< Pending decay of the world cache after sun changes.
    std::vector<uint32_t> mSunPageQueue; ///< Sea tiles whose sun pages refresh over the next frames, nearest first.
    uint32_t mCloudSunTilesPerFrame = 8; ///< Sea tiles whose sun pages are recomputed per frame after a sun change.
    uint32_t mCloudSunBakesPerFrame = 256; ///< Bricks whose sun depth is baked per frame (CloudResidencyDesc::sunBakesPerFrame).
    ref<ComputePass> mpBakeCloudSunPass;
    float4 mCloudSunBakeInputs = float4(0.f); ///< Sun direction and density scale the sun generation was last bumped for.
    float mCloudSunBakeNear = -1.f;           ///< sunNearVoxels the sun generation was last bumped for.
    float mCloudSunBakeAngle = 0.25f;         ///< Degrees the sun moves from the current generation's bake direction before the next.
    bool mCloudSunLiveMarch = true;           ///< Whether the camera program keeps sunDepthAt's live near march (HSTR_SUN_LIVE).
    bool mCloudCameraKernel = false;          ///< Whether the per-pixel cloud view renders from its own entry point (renderCloudCamera).
    ref<ComputePass> mpCameraPass;            ///< That entry point.
    ref<ComputePass> mpDecayWorldCachePass;
    std::vector<float> mCloudMeanBlocks; ///< Domain majorant blocks of the mean density (transport), unscaled.
    std::vector<float> mCloudMaxBlocks;  ///< Domain majorant blocks of the maximum density (camera), unscaled.
    /// World Y of the occupied band, from the majorant blocks, dilated by one. mParams.seaContentY carries this when cloudSlabClamp
    /// is on and an unbounded range when it is off, so the shader clamps without a branch.
    float2 mSeaContentBand = float2(-std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    std::vector<float> mCloudTileBatches;
    std::vector<uint32_t> mCloudTileReset;
    ref<Buffer> mpCloudTileBatches;
    ref<Buffer> mpCloudTileReset;
    ref<ComputePass> mpCommitCloudPass;
    ref<ComputePass> mpDecodeCloudPass;
    ref<ComputePass> mpOccupancyCloudPass;
    ref<ComputePass> mpClearWorldCacheTilesPass;
    ref<Sampler> mpLinearClampSampler;
    uint32_t mSamplerSeaMode = ~0u;
    std::vector<float> mHierarchyResidualBounds;
    std::vector<hstr::DenseMatrix> mLeafTransfers; ///< Root-to-leaf incident maps, reused by every lighting solve.
    std::vector<hstr::DenseMatrix> mLeafCorrections;
    std::vector<hstr::DenseMatrix> mLeafBaseTransport;
    std::vector<hstr::DenseMatrix> mOperatorDictionary;
    std::vector<uint32_t> mLeafOperatorIDs;
    std::vector<hstr::DenseMatrix> mDictionaryCorrections;
    bool mOptionsChanged = false;
    bool mFirstFrame = true;
    bool mCameraLightingDirty = true;
    bool mCutDirty = true;
    bool mResidualDirty = true;
    bool mBasisDirty = true;
    bool mCameraLightingPoseValid = false;
    float3 mCameraLightingPosition = float3(0.f);
    float3 mCameraLightingDirection = float3(0.f);
};
