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

/// The cloud sea's tile uploads: host-visible buffers of one packed tile volume each, kept mapped. A sea worker writes a tile
/// straight into a free one, so taking the tile on costs the main thread one recorded copy. A buffer is free again once the GPU
/// is past its copy (a fence signalled the frame after the copy was recorded).
class DomainStaging : public hstrcloud::TileStaging
{
public:
    DomainStaging(ref<Device> pDevice, uint32_t tileValues, uint32_t count);

    int32_t acquire(uint32_t*& data) override;
    void discard(int32_t handle) override;

    const ref<Buffer>& getBuffer(int32_t handle) const { return mBuffers[handle].pBuffer; }
    /// The buffer's copy is in this frame's command list.
    void recorded(int32_t handle) { mRecorded.push_back(handle); }
    /// Signals the copies recorded last frame and frees the buffers whose copies ran.
    void beginFrame(RenderContext* pRenderContext);

private:
    struct Staged
    {
        ref<Buffer> pBuffer;
        uint32_t* pData = nullptr;
    };
    std::vector<Staged> mBuffers;
    std::mutex mMutex;
    std::vector<int32_t> mFree;                            ///< Guarded by mMutex (workers take buffers).
    std::vector<int32_t> mRecorded;                        ///< Main thread only.
    std::vector<std::pair<uint64_t, int32_t>> mInFlight;   ///< Fence value and buffer, main thread only.
    ref<Fence> mpFence;
};

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
    static_assert(sizeof(BeamResult) == 16);
    static_assert(sizeof(BeamTile) == 32);

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
    ref<ComputePass> mpDomainCutPass; ///< Builds the cloud-sea's regular 16^3-cell cut and trilinear pages on the GPU.
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
    ref<ComputePass> mpBeamSparseResolvePass;
    ref<ComputePass> mpBeamMarchPass;
    ref<ComputePass> mpBeamClassifyPass;
    ref<ComputePass> mpBeamSparseEmitPass;
    ref<ComputePass> mpBeamSparseVerifyPass;
    ref<ComputePass> mpBeamSparseArgsPass;
    ref<ComputePass> mpBeamGridQueryPass;  ///< Root lattice queries as a 2D dispatch over the corner (or centre) grid.
    ref<ComputePass> mpBeamGridMarchPass;  ///< Per-pixel refinement as a full-frame 2D dispatch that skips accepted tiles.
    ref<ComputePass> mpBeamUnitMarchPass;  ///< The same refinement in the rotation-invariant beam image, where it carries (beamRefFrame).
    bool mBeamGridDispatch = false;        ///< Whether the beam view uses the two passes above (beamSegments 1, per-level build).
    bool mBeamSparse = true;               ///< Metadata-led sparse query compiler; false keeps the legacy lattice generator for A/B.
    /// Evaluate sparse bases through the projected transport cut (traverseCut/integratePage) instead of full-volume marchBeam().
    /// This is the first intermediate form of the intended architecture: the basis query still exists, but it composites the tile's
    /// front-to-back cut instead of re-solving transport down the whole ray.
    ///
    /// MEASURED and OFF for the cloud sea (4K near, parked, 2026-09-20, against the same stored exact frame). It is correct and
    /// quality-neutral - log 1.48e-3 against the volume evaluator's 1.39e-3, p99.9 4.31e-2 either way - and it is a large net loss,
    /// for a reason that is about the sea's representation rather than this code: of the 1.28 million cut cells the basis rays hit,
    /// 379 were resolved by their page at the shipping tolerance and 6,744 at five times it, 0.03% and 0.5%. Everything else fell
    /// through to the marcher, so the basis queries rose 10.1 -> 15.3 ms (a cell restart drops the held lighting and sun depth) and
    /// building the cut - projecting 2048 domain cells, binning them into 129,600 8-pixel tiles, sorting each list - cost 23.4 ms.
    ///
    /// The blocker is the resolution of the transport the cut can carry, and it is structural. The sea's only world-space field
    /// outside the virtual brick pyramid is the domain proxy at 5.16 world units per voxel; a cut cell is 16 of those, 83 units. At
    /// the far end of the sea's 960-unit view distance one PROXY VOXEL still subtends about ten 4K pixels, so there is no distance
    /// inside this scene at which a trilinear page over a cut cell resolves what the camera sees. Accumulating BeamTile x node
    /// coefficients from these pages would accumulate the 0.5% and leave the rest to the residual queue. Making the radical path
    /// pay here needs the cut built at fine-brick resolution - transport compressed from the resident pyramid, not downsampled 16x
    /// from the proxy - which is the "solve/compress transport in world space" step the sea currently has no representation for.
    bool mBeamSparseCut = false;
    uint32_t mBeamSparseMinLevel = 2;       ///< Lowest metadata-selected verifier level; benchmark knob outside the shared shader ABI.
    bool mBeamSparseBuilt = false;          ///< The current beam buffers were produced by the sparse compiler.
    /// Rotation-invariant beam basis frame: the whole beam build runs in an image anchored to a world orientation instead of to the
    /// screen, so turning the camera neither invalidates a basis ray nor resamples one. See HSTRCloudParams::beamRefU.
    bool mBeamRefFrame = false;
    /// EXPERIMENT. Build and keep the WHOLE reference image rather than only the part the screen covers, so a turn reveals no
    /// direction that has never been marched. This is the gate on a persistent world-direction cache: if yaw then costs what
    /// parked costs, the yaw penalty really is newly exposed directions and the cache is worth building.
    bool mBeamRefPrebuild = false;
    bool mBeamOct = false;      ///< The beam image is the octahedral sphere: no anchor, so no re-anchoring.
    bool mBeamOctFull = false;  ///< Build the whole sphere rather than the screen's footprint.
    float mBeamOctScale = 1.f;  ///< Texel angular size against a screen pixel at the view centre.
    uint2 mBeamOctDim = { 0, 0 };
    /// How the query dispatch was generated, counted since the view settled: swept over the region, generated from the refresh
    /// phase alone, or that plus the strips a rotation exposed. Reported so an arm that should be generating and is not says so.
    uint32_t mBeamGridSweeps = 0;
    uint32_t mBeamGridGenerated = 0;
    uint32_t mBeamGridStrips = 0;
    uint32_t mBeamGridThreads = 0;
    /// The regions this build's query dispatch covered, replayed by the level-0 tile pass: a tile can only change its
    /// classification where one of its basis points was re-marched. Empty means the tile pass sweeps, as it always did.
    std::vector<uint4> mBeamGridRegions;   ///< origin.xy, size.xy
    std::vector<uint32_t> mBeamGridRegionMode; ///< 1 where that region is block-mapped (the refresh phase), 0 where it is a box.
    float mBeamRefMargin = 0.15f; ///< Fraction of the frame the reference image reaches past each screen edge before re-anchoring.
    bool mBeamRefAnchored = false;
    float3 mBeamRefU = float3(0.f);
    float3 mBeamRefV = float3(0.f);
    float3 mBeamRefW = float3(0.f);
    float3 mBeamRefCameraU = float3(0.f);
    float3 mBeamRefCameraV = float3(0.f);
    float3 mBeamRefCameraW = float3(0.f);
    uint32_t mBeamRefAnchors = 0;  ///< Re-anchors since the view settled, which is what a turn amortises a full rebuild over.
    float4 mBeamBuiltScreenBounds = float4(0.f); ///< beamScreenBounds of the build that most recently wrote the beam image.
    void updateBeamReferenceFrame(const uint2& frameDim);
    void updateBeamOctFrame(const uint2& frameDim, const CameraData& camera);
    // beamRefresh: the previous build's lattice and level map (swapped with the current ones every build), its marched pixels
    // (the two alternate), and its camera.
    ref<Texture> mpBeamLatticePrev;
    ref<Texture> mpBeamLevelPrev;
    std::array<ref<Texture>, 2> mpBeamPixels;
    uint32_t mBeamRefresh = 0; ///< Requested beamRefresh; the shader's copy is 0 where the build cannot refresh.
    bool mBeamQueue = false;   ///< Requested beamQueue; the shader's copy is 0 where the build cannot queue.
    ref<Buffer> mpBeamQueue;
    ref<Buffer> mpBeamQueueCounts;
    ref<Buffer> mpBeamQueueArgs;
    ref<ComputePass> mpBeamQueueMarchPass;
    ref<ComputePass> mpBeamQueueArgsPass;
    ref<ComputePass> mpBeamQueueTilePass;
    ref<ComputePass> mpBeamQueuePixelPass;
    /// Fills the queue's dispatch arguments for entries of threadsPerEntry threads.
    void writeBeamQueueArgs(RenderContext* pRenderContext, uint32_t threadsPerEntry);
    bool mCloudResidencyFrozen = false; ///< Benchmarks: skip the residency update, keeping the resident set as it is.
    float mCloudCutMargin = 8.f;        ///< CloudView::cutMargin (voxels; 0: off).
    float mCloudCutTurn = 3.f;          ///< CloudView::cutTurn (degrees).
    bool mCloudCutAsync = true;         ///< CloudView::cutAsync: the frame only takes on cuts the worker finished.
    bool mBeamShip = true;     ///< Whether the beam marches compile HSTR_SHIP where the settings allow it (beamShipping).
    /// The HSTR_SHIP groups folded when they do. MEASURED (4K, 2026-09-17, same frame everywhere): all of them (511) sped up the sea
    /// 7.25 -> 5.40 ms but slowed near 14.3 -> 19.5 and farside 11.4 -> 14.7 - a DXC codegen cliff no single group causes; every
    /// group but the cost probe (509) is faster everywhere: near 12.5, farside 9.9, sea 5.5, flying 2 units 7.9 -> 5.8, 20 units
    /// 10.0 -> 8.0.
    uint32_t mBeamShipMask = 509;
    /// Whether every switch HSTR_SHIP folds holds its shipping value, so that the folded program renders the same frame.
    bool beamShipping() const;
    bool mBeamRefreshValid = false;
    float4x4 mBeamPrevViewProj;
    float3 mBeamPrevCamera = float3(0.f);
    ref<Texture> mpBeamLattice; ///< Beam view queries at every tile corner and centre (2 slices).
    ref<Buffer> mpBeamPageTable; ///< Identity page table for the beamPageIndirect probe.
    ref<Texture> mpBeamLevel;   ///< Level that finalised every finest beam tile.
    ref<Texture> mpBeamQueryMap; ///< Sparse query ID per possible lattice coordinate.
    ref<Texture> mpBeamTileMap;  ///< Final sparse tile ID per finest cell.
    ref<Buffer> mpBeamSparseCandidates;
    ref<Buffer> mpBeamSparseArgs;
    ref<Buffer> mpBeamResults;
    ref<Buffer> mpBeamFinalTiles;
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
    uint32_t mBeamLevelCounts[kBeamCountSlots] = {};    ///< Tiles entering each level, read with the comparison; the last is the
                                                        ///< per-pixel march list. Eight queries per entry, so these are what the
                                                        ///< query pass costs.
    std::string mSaveReferencePath;    ///< When set, the reference sums are written to <path>_s<slice>.exr at the end of the frame.
    std::string mLoadReferencePath;    ///< When set, the reference sums are read from <path>_s<slice>.exr at the start of the frame.
    float3 mReferencePosition = float3(0.f);
    uint32_t mReferenceBandRows = 0; ///< Rows per separately submitted band of a path-traced sample (0: the whole frame).
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
    std::unique_ptr<DomainStaging> mpDomainStaging; ///< Before the sea: its workers write into it until the sea is gone.
    std::unique_ptr<hstrcloud::CloudSea> mpCloudSea;
    std::unique_ptr<hstrcloud::CloudResidency> mpCloudResidency;
    bool mCloudInstancesUploaded = false;
    uint32_t mCloudFrames = 0;
    /// What invalidated the carried beam basis on the LAST frame, and how often each has since the view settled. A fixed refresh
    /// rate is insurance against exactly these, so their frequency and coverage decide whether it can become event-driven.
    bool mDensityChangedFrame = false;
    uint32_t mDensityChangedFrames = 0;
    uint32_t mSunBakeFrames = 0;
    float mSeaViewDistance = 4000.f; ///< Requested view distance; the sea clamps it to its resident window.
    float mCloudCacheKeep = 1.f;         ///< Pending decay of the world cache after sun changes.
    std::vector<uint32_t> mSunPageQueue; ///< Sea tiles whose sun pages refresh over the next frames, nearest first.
    uint32_t mCloudSunTilesPerFrame = 8; ///< Sea tiles whose sun pages are recomputed per frame after a sun change.
    uint32_t mCloudSunBakesPerFrame = 256; ///< Bricks whose sun depth is baked per frame (CloudResidencyDesc::sunBakesPerFrame).
    ref<ComputePass> mpBakeCloudSunPass;
    // GPU sun bake scheduling (cloudGpuSun, CloudResidencyDesc::gpuSun).
    bool mCloudGpuSun = true;
    ref<ComputePass> mpReleaseSunPass;
    ref<ComputePass> mpScanSunPass;
    ref<ComputePass> mpStampSunPass;
    ref<ComputePass> mpResetSunBlocksPass;
    ref<ComputePass> mpResetSunNodesPass;
    ref<ComputePass> mpAgeSunPass;
    std::vector<const ComputePass*> mResidencyPassesBound; ///< Residency passes holding the current residency's bindings.
    /// Binds a residency pass (bindResidencyPass in the .cpp) and sets its params; true on its first bind.
    bool bindResidencyPass(RenderContext* pRenderContext, const ref<ComputePass>& pPass, bool scene);
    ref<ComputePass> mpSelectSunPass;
    ref<ComputePass> mpEmitSunPass;
    ref<ComputePass> mpEvictSunPass;
    ref<ComputePass> mpAssignSunPass;
    ref<Buffer> mpSunReadback; ///< The bake count of a run, read back without waiting.
    ref<Fence> mpSunFence;
    uint64_t mSunReadbackPending = 0;
    bool mSunReadbackRecorded = false; ///< The copy is in the command list; its fence is signalled next frame.
    void dispatchSunScheduling(RenderContext* pRenderContext);
    float4 mCloudSunBakeInputs = float4(0.f); ///< Sun direction and density scale the sun generation was last bumped for.
    float mCloudSunBakeNear = -1.f;           ///< sunNearVoxels the sun generation was last bumped for.
    float mCloudSunBakeAngle = 0.25f;         ///< Degrees the sun moves from the current generation's bake direction before the next.
    bool mCloudSunLiveMarch = true;           ///< Whether the camera program keeps sunDepthAt's live near march (HSTR_SUN_LIVE).
    bool mCloudCameraKernel = false;          ///< Whether the per-pixel cloud view renders from its own entry point (renderCloudCamera).
    ref<ComputePass> mpCameraPass;            ///< That entry point.
    ref<ComputePass> mpDecayWorldCachePass;
    // The sea's domain proxy on the GPU (uploadDomainExtinction).
    ref<Texture> mpDomainVolume; ///< Per domain voxel: unscaled mean density and conservative maximum (RG16).
    ref<Texture> mpDomainBlocks; ///< Per majorant block: unscaled maximum and mean over its trilinear support (RG16).
    ref<Buffer> mpDomainRegions; ///< Changed slots, flagged (kDomainRegionZero, kDomainRegionKeep).
    ref<Buffer> mpDomainFrame;
    ref<Buffer> mpDomainStaged;  ///< One batch of packed tile volumes, copied from the staging buffers.
    bool mDomainPassesBound = false;         ///< The domain passes and the world cache clear hold the current resources.
    ref<Buffer> mpClearBoundDeposit;         ///< The world cache the clear holds (a reference, so no new buffer reuses its address).
    ref<ComputePass> mpDomainExtinctionPass;
    ref<ComputePass> mpDomainBlocksPass;
    ref<ComputePass> mpDomainMajorantPass;
    ref<ComputePass> mpDomainOccupancyPass;
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
    ref<ComputePass> mpClearDirtyCloudPagesPass;
    ref<ComputePass> mpMarkDirtyCloudPagesPass;
    ref<ComputePass> mpCloudPageArgsPass;
    ref<ComputePass> mpResolveDirtyCloudPagesPass;
    ref<ComputePass> mpClearWorldCacheTilesPass;
    ref<ComputePass> mpAdvanceFadesPass;
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
    bool mDomainCutOriginValid = false; ///< Whether hstrCutOriginLeaf holds a window placed for the current camera.
    bool mResidualDirty = true;
    bool mBasisDirty = true;
    bool mCameraLightingPoseValid = false;
    float3 mCameraLightingPosition = float3(0.f);
    float3 mCameraLightingDirection = float3(0.f);
};
