/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "Core/Pass/RasterPass.h"
#include "Rendering/Volumes/HSTRHierarchy.h"
#include "RenderGraph/RenderPass.h"
#include "HSTRCloudTypes.slang"
#include "CloudResidency.h"
#include "Atmosphere.h"

#include <array>
#include <functional>
#include <memory>

using namespace Falcor;

/// A compute pass created on first use (->, get() or conversion), not when the scene is set. HSTRCloud has 114 passes, most of
/// them probes and measured-and-off variants that a configuration never dispatches; creating one compiles its kernel, which
/// after a shader edit (a cold shader cache) costs 2-10 s, and 190 s for countPushShares alone - 350-430 s of every first load.
class LazyComputePass
{
public:
    LazyComputePass() = default;
    /// Copies share the pass (and its creation).
    LazyComputePass(std::function<ref<ComputePass>()> create) : mpSlot(std::make_shared<Slot>(Slot{nullptr, std::move(create)})) {}
    LazyComputePass& operator=(std::nullptr_t)
    {
        mpSlot = nullptr;
        return *this;
    }
    ComputePass* operator->() const { return get(); }
    ComputePass* get() const { return resolve().get(); }
    operator const ref<ComputePass>&() const { return resolve(); }
    /// Whether there is a pass, created or not (does not create it).
    explicit operator bool() const { return mpSlot != nullptr; }
    bool isCreated() const { return mpSlot && mpSlot->pPass; }

private:
    struct Slot
    {
        ref<ComputePass> pPass;
        std::function<ref<ComputePass>()> create;
    };
    const ref<ComputePass>& resolve() const
    {
        static const ref<ComputePass> kNone;
        if (!mpSlot)
            return kNone;
        if (!mpSlot->pPass)
            mpSlot->pPass = mpSlot->create();
        return mpSlot->pPass;
    }
    std::shared_ptr<Slot> mpSlot;
};

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
    void executeFrame(RenderContext* pRenderContext, const RenderData& renderData);
    /// gpuTiming: the pass's GPU time from a begin / end timestamp pair alone (gpuTimeSum in us, gpuTimeFrames), true with the
    /// overlaps the profiler's inner scopes serialise.
    bool mGpuTiming = true;
    std::array<ref<GpuTimer>, 3> mpGpuTimers;
    uint32_t mGpuTimerFrame = 0;
    uint64_t mGpuTimeSumUs = 0;
    uint64_t mGpuTimeFrames = 0;
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
    std::vector<uint32_t> seaTilesNearestFirst() const; ///< Sea slots by torus distance from the camera's slot.
    bool sunColumnsActive() const;                      ///< The sea's sun field is built by sun ray (sunColumns) this frame.
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
    /// rowOffset / rowStride (seaFarBeside 2): only those rows of the run; the run counts as done after the dispatch with `last`.
    void dispatchFarSea(RenderContext* pRenderContext, uint32_t rowOffset = 0, uint32_t rowStride = 1, bool last = true);
    void saveReference(RenderContext* pRenderContext, const std::string& path);
    void loadReference(RenderContext* pRenderContext, const std::string& path);
    void ensureCameraResources();
    std::vector<float> sampleLeafDensities() const;
    hstr::DenseMatrix makeNestedLeafTransport(uint32_t leafIndex) const;

    ref<Scene> mpScene;
    LazyComputePass mpPass;
    LazyComputePass mpSolvePass;
    LazyComputePass mpCameraLightingPass;
    LazyComputePass mpProjectPass;
    LazyComputePass mpCutPass;
    LazyComputePass mpSortPass;
    LazyComputePass mpDomainCutPass; ///< Builds the cloud-sea's regular 16^3-cell cut and trilinear pages on the GPU.
    LazyComputePass mpQueryPass;
    LazyComputePass mpTileBasisPass;
    LazyComputePass mpFineSunPass;
    LazyComputePass mpSunColumnsPass;
    LazyComputePass mpResidualMaskPass;
    LazyComputePass mpGatherOctavesPass;
    LazyComputePass mpBlurOctavesPass;
    LazyComputePass mpReferencePass;
    LazyComputePass mpCompareReferencePass;
    ref<Texture> mpReferenceSum;     ///< Running sums of path-traced reference frames (rgb) and their count (a), per component and half.
    ref<Buffer> mpReferenceRowError; ///< Per-row mean error of the HST frame against the reference average.
    ref<Buffer> mpReferenceRowHistogram; ///< Exact comparisons: per-row log error histogram and maximum.
    bool mCompareReference = false;
    float mReferenceLogP999 = -1.f; ///< Exact comparisons: 99.9th percentile log error (upper edge of its bin).
    float mReferenceLogP99 = -1.f;  ///< Exact comparisons: 99th percentile log error (upper edge of its bin).
    float mReferenceLogMax = -1.f;  ///< Exact comparisons: largest pixel log error.
    float mReferenceError = -1.f;         ///< Mean |HST - reference| in linear radiance.
    float mReferenceLogError = -1.f;      ///< Mean |log(1 + HST) - log(1 + reference)|.
    float mReferenceNoiseError = -1.f;    ///< Expected linear error of the reference average itself, from its two halves.
    float mReferenceNoiseLogError = -1.f; ///< Same in log(1 + radiance).
    LazyComputePass mpWorldCachePass;
    LazyComputePass mpWorldCachePhotonPass;
    LazyComputePass mpWorldCacheResolvePass;
    ref<Buffer> mpWorldCache;        ///< World-space radiance cache experiment: SH running sums per cell.
    ref<Buffer> mpWorldCacheDeposit; ///< Fixed-point light-tracing deposits of the current batch.
    ref<Buffer> mpWorldCacheRingKernel; ///< Ring edge cosines (worldCacheSunOrder 4).
    ref<Texture> mpWorldCacheRingTable; ///< Ring phase table (worldCacheSunOrder 4), for mWorldCacheRingG.
    float mWorldCacheRingG = -2.f;
    uint32_t mWorldCacheRingShape = 0; ///< worldCacheRingCount * 16 + worldCacheRingLayout the ring buffers were built for.
    float mWorldCacheRingModulation = -1.f; ///< Ring slot modulation b; negative: the medium's transport attenuation.
    LazyComputePass mpWorldCacheBakePass;
    LazyComputePass mpWorldCacheViewPass; ///< worldCacheView: bakeWorldCacheView.
    ref<Texture> mpWorldCacheView;        ///< worldCacheView: the cache's radiance towards the camera, a frame (RGBA16F, cache cells).
    LazyComputePass mpWorldCacheAdvancePass;
    ref<Buffer> mpPhotonPool;                                           ///< Persistent light-tracing photons (48 bytes each).
    ref<Buffer> mpPhotonEmitted;                                        ///< Photons emitted by the pool since the last restart.
    std::array<ref<Texture>, kWorldCacheTextures> mpWorldCacheTextures; ///< Cache means packed for hardware-filtered lookups.
    bool mWorldCacheBakeDirty = true;
    uint32_t mWorldCacheBakeInterval = 8; ///< Batches between bakes of the camera textures while updating.
    uint32_t mWorldCacheUpdates = 1;      ///< Cache gather passes per frame in the world cache view.
    /// Photon updates only while some sea tile holds fewer batches than this (0: every frame): the converged state the benchmarks
    /// score, which the shipped scene (CloudSea.py, worldCacheUpdates 1) otherwise kept paying for every frame.
    uint32_t mWorldCacheTarget = 64;
    /// A sun move with no photon updates to follow bakes the world cache without decaying it (the decay alone leaves the means).
    /// MEASURED (sunoct2, 4K, 0.57 deg/frame sun, cache frozen): worldCache 0.29 -> 0.04 ms, parked 1.84 -> 1.57 and walk 2.45 ->
    /// 2.20 ms, errors identical. Only a frozen cache (the benchmarks) takes this; a live one decays as before.
    bool mWorldCacheFrozenBake = true;
    float mWorldCacheModulation = -1.f;   ///< Cache modulation b; negative: the medium's diffusion attenuation.
    LazyComputePass mpBeamQueryPass;
    LazyComputePass mpBeamTilePass;
    LazyComputePass mpBeamArgsPass;
    LazyComputePass mpBeamResolvePass;
    LazyComputePass mpBeamResidualResolvePass; ///< Reference frame: failed tiles' pixels, from the residual image.
    LazyComputePass mpBeamResidualArgsPass;
    ref<Buffer> mpBeamResidualList;             ///< Pixels the resolve found in failed tiles, one frame's worth.
    ref<Buffer> mpBeamResidualArgs;             ///< [0] their count, [1..3] the residual resolve's indirect dispatch.
    LazyComputePass mpBeamSparseResolvePass;
    LazyComputePass mpBeamMarchPass;
    LazyComputePass mpBeamClassifyPass;
    LazyComputePass mpBeamSparseEmitPass;
    LazyComputePass mpBeamSparseVerifyPass;
    LazyComputePass mpBeamSparseArgsPass;
    LazyComputePass mpBeamGridQueryPass;  ///< Root lattice queries as a 2D dispatch over the corner (or centre) grid.
    LazyComputePass mpBeamGridMarchPass;  ///< Per-pixel refinement as a full-frame 2D dispatch that skips accepted tiles.
    LazyComputePass mpBeamUnitMarchPass;  ///< The same refinement in the rotation-invariant beam image, where it carries (beamRefFrame).
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
    float mBeamOctScale = 0.5f; ///< Texel angular size against a screen pixel at the view centre (CloudSea.py: why 0.5).
    uint2 mBeamOctDim = { 0, 0 };
    uint32_t mBeamOctAxis = 0;  ///< World axis the octahedral map's +z points along, and so where its derivative kinks.
    bool mBeamScreenResidual = false; ///< March failed tiles in screen space instead of resampling the beam-image residual.
    /// How the query dispatch was generated, counted since the view settled: swept over the region, generated from the refresh
    /// phase alone, or that plus the strips a rotation exposed. Reported so an arm that should be generating and is not says so.
    uint32_t mBeamGridSweeps = 0;
    uint32_t mBeamGridGenerated = 0;
    uint32_t mBeamGridStrips = 0;
    uint32_t mBeamGridThreads = 0;
    uint32_t mBeamUnitThreads = 0; ///< Residual-march threads dispatched, cumulative: the query counter says nothing about them.
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
    float beamGuardReach(const uint2& frameDim, const CameraData& camera) const;
    // beamRefresh: the previous build's lattice and level map (swapped with the current ones every build), its marched pixels
    // (the two alternate), and its camera.
    ref<Texture> mpBeamLatticePrev;
    ref<Texture> mpBeamLevelPrev;
    std::array<ref<Texture>, 2> mpBeamPixels;
    ref<Texture> mpBeamUnitTransmittance; ///< Beside mpBeamPixels[0]: each marched unit's transmittance (the far sea is added at resolve).
    uint3 mBeamAllocatedDim = uint3(0); ///< The beam image size and tile size the beam resources were last sized for.
    uint32_t mBeamRefresh = 0; ///< Requested beamRefresh; the shader's copy is 0 where the build cannot refresh.
    bool mBeamQueue = false;   ///< Requested beamQueue; the shader's copy is 0 where the build cannot queue.
    /// beamQueryCompact: the dirty query lists the points it has to march (queue list 0, in listed order) and marchBeamDirtyListed
    /// marches them packed. Why it was tried (querytail1 / livedirty1, 4K walk): ~244k query threads a frame, the order probe's 15.2k
    /// marched rays against 2,367 waves reaching the march read as ~6.4 of 32 lanes a wave.
    /// MEASURED (compact2 / compact3, 4K walk, interleaved, 30 profiled frames an arm; mapped 171-218k of 243k): query 1.353 / 1.288,
    /// 0.974 / 0.715, 1.128 / 1.099, 1.012 / 1.020 ms (base / compact) while units drifted alike (query / units 1.07 / 1.02, 1.09 /
    /// 1.11, 1.02 / 1.02, 1.05 / 1.05). The listed march is 1.057 / 0.978 of it, the listing ~0.04. beamLayerProbe: 2,635 waves
    /// march in place, 2,693 compacted, paid / lane steps 1.45 in both - the in-place waves were already full (the two probes do not
    /// count the same rays), so packing has nothing to pack. Quality (compact1, SCORE 2): 5.73 / 5.56% vs base 5.49 / 5.34 / 6.16%,
    /// knock sanity 53.9%. No gain; off.
    bool mBeamQueryCompact = false;
    LazyComputePass mpBeamDirtyListedPass;
    ref<Buffer> mpBeamQueue;
    ref<Buffer> mpBeamQueueCounts;
    ref<Buffer> mpBeamQueueArgs;
    LazyComputePass mpBeamQueueMarchPass;
    LazyComputePass mpBeamQueueArgsPass;
    LazyComputePass mpBeamQueueTilePass;
    LazyComputePass mpBeamQueuePixelPass;
    /// Fills the queue's dispatch arguments for entries of threadsPerEntry threads.
    void writeBeamQueueArgs(RenderContext* pRenderContext, uint32_t threadsPerEntry);
    bool mCloudResidencyFrozen = false; ///< Benchmarks: skip the residency update, keeping the resident set as it is.
    uint32_t mSeaTilesChanged = 0;      ///< Sea tiles whose cloud was replaced, cumulative (cloudStats).
    std::vector<int2> mSeaSlotWorld;    ///< Per sea slot: the world tile the beam image last saw it hold.
    /// World regions whose density or lighting changed since the last guard build, as spheres (invalidateBeamGuardBlock).
    std::vector<float4> mBeamInvalidations;
    bool mBeamInvalidateAll = false;    ///< Everything changed: every guard block is unverified at the next build.
    bool mBeamInvalidate = true;        ///< beamInvalidate: off only to A/B the stale image it prevents.
    bool mBeamRepairProbe = false;      ///< beamRepairProbe: score in-place and reprojected predictions of re-marched samples.
    ref<Texture> mpBeamPixelsSnapshot;  ///< beamRepairProbe: the residual units before the dirty march.
    ref<Texture> mpBeamLatticeSnapshot; ///< beamRepairProbe: the lattice before the dirty query.
    ref<Texture> mpBeamGuardCameraSnapshot; ///< beamRepairProbe: the guard cameras before the dirty query.
    ref<Texture> mpBeamLevelSnapshot;       ///< beamWarpHistory: the level map before the dirty query.
    ref<Texture> mpBeamProbeAccept;         ///< beamRepairProbe: per block, the witness policies that would have carried it.
    uint32_t mBeamInvalidatedBuilds = 0; ///< Builds that invalidated blocks for changed content, cumulative (cloudStats).
    LazyComputePass mpBeamInvalidatePass;
    ref<Buffer> mpBeamInvalidations;
    /// The GPU change list (markBeamChanges): the residency's settled bricks and the sun bakes, per slot that reads them. The
    /// tile columns above only cover the sea replacing tiles and the sun pages; streaming and baking left the image stale.
    uint32_t mBeamChangeCellVoxels = 4; ///< beamChangeCellVoxels: the change cells' edge in domain voxels; 0: off.
    bool mBeamChangesPending = false;   ///< Marked changes the next build's invalidation has not consumed.
    /// beamChangeInterval: the marked cells are listed every this many frames, and each list is applied to this many interleaved
    /// slices of the guard blocks, one a frame, so the re-marching is spread over the cycle.
    uint32_t mBeamChangeInterval = 8;
    uint32_t mBeamChangeFrames = 0;
    bool mBeamChangeCellsMarked = false; ///< Cells marked since the last listing.
    uint32_t mBeamChangeApplying = 0;    ///< Slices of the current list still to apply.
    LazyComputePass mpBeamMarkChangesPass;
    LazyComputePass mpBeamCompactChangesPass;
    ref<Buffer> mpBeamChangeInput;
    ref<Buffer> mpBeamChangeCells;
    ref<Buffer> mpBeamChangeSpheres;
    ref<Buffer> mpBeamChangeCount;
    /// After the frame's residency update and sun bakes: marks what changed (bakes: this frame's bake jobs) for the next build.
    void markBeamChanges(RenderContext* pRenderContext, uint32_t bakes);
    /// Binds the change list for reading (the invalidation passes).
    void bindBeamChanges(const ref<ComputePass>& pPass);
    /// Queues the column of the cloud layer over a world tile, and what its change reaches, for invalidateBeamGuardBlock.
    void invalidateBeamColumn(int2 worldTile);
    float mCloudCutMargin = 8.f;        ///< CloudView::cutMargin (voxels; 0: off).
    float mCloudCutTurn = 3.f;          ///< CloudView::cutTurn (degrees).
    bool mCloudCutAsync = true;         ///< CloudView::cutAsync: the frame only takes on cuts the worker finished.
    bool mBeamShip = true;     ///< Whether the beam marches compile HSTR_SHIP where the settings allow it (beamShipping).
    /// The HSTR_SHIP groups folded when they do. MEASURED (4K, 2026-09-17, same frame everywhere): all of them (511) sped up the sea
    /// 7.25 -> 5.40 ms but slowed near 14.3 -> 19.5 and farside 11.4 -> 14.7 - a DXC codegen cliff no single group causes; every
    /// group but the cost probe (509) is faster everywhere: near 12.5, farside 9.9, sea 5.5, flying 2 units 7.9 -> 5.8, 20 units
    /// 10.0 -> 8.0.
    /// 509 | 1024 | 2048 | 8192: the lean march (marchBeamLean) on the resolved level pages and sun slots, crossing zero 4-voxel
    /// majorant blocks whole (lean_march.py).
    uint32_t mBeamShipMask = 11773 | 32768; // With the transmittance-scaled steps (CloudSea.py).
    /// Whether every switch HSTR_SHIP folds holds its shipping value, so that the folded program renders the same frame.
    bool beamShipping() const;
    uint32_t beamShipDefine() const;
    bool mBeamRefreshValid = false;
    float4x4 mBeamPrevViewProj;
    float3 mBeamPrevCamera = float3(0.f);
    ref<Texture> mpBeamLattice; ///< Beam view queries at every tile corner and centre (2 slices).
    ref<Texture> mpBeamSpan;    ///< beamUnitSpan: per lattice point, its ray's depth-span record (RGBA32Uint).
    ref<Buffer> mpBeamPageTable; ///< Identity page table for the beamPageIndirect probe.
    ref<Texture> mpBeamGuardDepth;  ///< Per block: the smallest distance any of its lattice points holds.
    ref<Texture> mpBeamGuardFar;    ///< Per block: the largest hit distance (beamGuardWarpParallax).
    ref<Texture> mpBeamGuardCamera; ///< Per block: the camera position its points were last tested against.
    bool mBeamGuard = false;        ///< Certify translation survival per block instead of per point.
    bool mBeamPrebuild = false;     ///< The build that anchors the octahedral image builds all of it (beamBuildAll).
    float mBeamGuardParallax = 8.f; ///< Texels of parallax a guard block may accumulate before it is re-marched (with beamWarp).
    float mBeamGuardIsoParallax = 0.f; ///< beamGuardIsoParallax: the isotropic level-0 budget ORed with beamGuardMotion's (0: off).
    float mBeamGuardWarpParallax = 0.f; ///< beamGuardWarpParallax: texels of near-to-far parallax a held block may carry (0: off).
    uint32_t mBeamGuardMotion = 0;  ///< beamGuardMotion: the motion-aware certificate (HSTRCloudTypes.slang).
    bool mBeamLayerProbe = false;   ///< beamLayerProbe: count the layered lookups by outcome (HSTR_LAYER_PROBE).
    /// worksetProbe: the distinct density bricks and sun bakes the dirty marches read a frame (HSTR_WORKSET_PROBE, hstrWorkset).
    bool mWorksetProbe = false;
    ref<Buffer> mpWorkset;
    std::vector<uint32_t> mWorksetStats; ///< Last frame's brick reads, bricks, bake reads, bake slots.
    bool mBeamUnitRefill = false;   ///< beamUnitRefill: PROBE, the units pass with lane refill (marchBeamDirtyUnitsRefill).
    /// HSTR_SUN_BAKE_PROBE=1 in the environment (fixed from the first bake, so the pass compiles once): count bakeCloudSun's steps
    /// and exits (HSTR_BAKE_PROBE), logged every 240 frames.
    const bool mSunBakeProbe = std::getenv("HSTR_SUN_BAKE_PROBE") != nullptr;
    bool mBeamWarp = true;        ///< The resolve reads a held block where its capture camera saw the content (HSTR_BEAM_WARP).
    /// beamOverlapResolve: the resolve's pixel pass runs without barriers behind the dirty unit march, filling that march's tail
    /// (ngfx8: the dirty passes fill the SMs at launch and then only drain, the last third on a few long rays). The pixel pass reads
    /// none of what the unit march writes (beam pixels, guard depth, coarse certificate); the warp field, which reads the guard
    /// depth, is built before the unit march instead, from the depths the query left.
    /// MEASURED (overlap2, 4K, default / overlap / default again, ms): walk 2.08 / 1.89 / 2.03, jog 2.52 / 2.34 / 2.55, sprint
    /// 2.40 / 2.21 / 2.39, every error statistic identical (walk 0.886%, jog 0.829%, sprint 0.473% over 0.02). overlap1 had measured
    /// nothing: one UAV -> SRV transition was left (hstrBeamPixelPrev, the beam pixels again in the reference frame), found in ngfx9.
    /// With it on, the units scope's time includes the pixel pass running inside it: compare HSTRCloud totals, not these two scopes.
    bool mBeamOverlapResolve = true;
    LazyComputePass mpBeamWarpFieldPass; ///< beamWarp: the warp offset per on-screen lattice point (buildBeamWarpField).
    LazyComputePass mpBeamWarpEdgePass;  ///< beamWarpEdge: held blocks' disoccluded edge tiles re-marched (repairBeamWarpEdge).
    LazyComputePass mpBeamWarpGapPass;   ///< beamWarpGap: held blocks' opened gap strips listed for repair (repairBeamWarpGap).
    LazyComputePass mpBeamWarpCoverPass; ///< beamWarpCover: the warp field corrected where nearer content covers a point.
    ref<Texture> mpBeamWarpFieldCover;   ///< beamWarpCover: the spare field texture it writes (swapped with mpBeamWarpField).
    ref<Texture> mpBeamGapBits;          ///< beamWarpGap: this build's repair texels, a bit each (R32Uint, 32 along x a word).
    ref<Texture> mpBeamRepair;           ///< beamWarpGap: their radiance from this camera (RGBA16Float, beam image size).
    ref<Texture> mpBeamWarpField;         ///< Lattice-sized, RG16Float: offsets are a few texels, so half precision holds them.
    /// beamWarpAuto: the warp runs only while at least this share of the on-screen guard blocks this build classified are held
    /// (certified, so read from an older camera). 0 = always, as beamWarp alone. Sprint expires every block, so the field and the
    /// warped resolve cost it ~0.13 ms for nothing. Decided on the GPU (writeBeamWarpArgs): both resolve variants are dispatched
    /// indirectly and the one not chosen gets no groups.
    float mBeamWarpAuto = 0.25f;
    uint32_t mColorFormat = 1;
    /// PROBE: after the real dirty query and unit passes, a copy of each over the same list (queryStrip, unitsStrip) compiled with
    /// HSTR_STRIP = this: 1 returns after the ray setup, 2 marches one step, 3 the full march (the warm-cache calibration). The
    /// copies compute and never write, so the image and the next frame's work are the real passes'. 0: off.
    uint32_t mBeamStripProbe = 0;
    uint32_t mBeamSunKnock = 0; ///< ORACLE beamSunKnock (HSTR_SUN_KNOCK on the dirty passes), timing only.
    uint32_t mSeaFarOverlapLog = 0; ///< DIAGNOSTIC seaFarOverlapLog: frames whose far-sea + unit-march barriers are logged.
    /// DIAGNOSTIC frameDispatchLog: frames whose every dispatch, clear and barrier is logged (DISPATCH / CLEAR / barrier lines),
    /// to find the small dispatches and the barriers between them.
    uint32_t mFrameDispatchLog = 0;
    bool mSeaFarOverlapScopes = true; ///< seaFarOverlapScopes: false drops the farSea / units profiler scopes while they overlap.
    /// seaFarBeside: the overlapped far-sea run beside the unit march (0) or the dirty query (1). The query leaves the GPU 6-9% busy
    /// (ngfxlive1, sm__throughput over its 0.6-0.9 ms), the march ~40%. Beside the query nothing may transition at the run: the far
    /// layer's pair is put in its states before the query and the beam lattice / repair / snapshot are unbound from it (farquery1-3
    /// logged each). MEASURED (farquery4, 4K walk 2, live, profiler scopes off for the pair, arms interleaved serial / query / units
    /// x2): HSTRCloud 3.275 / 2.847 / 2.695, 2.677 / 2.458 / 2.751 ms - beside the query -0.43 / -0.22 against the serial arm before
    /// it; the query's scope grows 0.13-0.15 while the far sea's 0.36-0.45 goes. Exact: the run reads nothing either pass writes.
    /// Timed without the profiler (gpuTiming, overlap6, live walk, interleaved x3): serial 3.238 / 3.091 / 2.707, beside the query
    /// 2.724 / 2.583 / 2.863, and with beamOverlapResolve 2.447 / 2.708 / 2.829 ms - means 3.01 / 2.72 / 2.66.
    /// 2: the run's even rows beside the query, its odd rows beside the unit march (HSTRCloudParams::seaFarRowOffset / Stride).
    /// MEASURED and left off (ngfxb1noscope / ngfxsplit2, 4K walk, seaFarOverlapScopes false, residency fixed, frames 60-61):
    /// HSTRCloud less cloudSea 2.82 / 2.75 ms (1) against 2.94 / 3.01 (2) - the march range grows 0.88-0.91 -> 1.11-1.18 while the
    /// query side saves 0-0.09: the far rows do not hide behind the unit march. farsplit2's frozen gpuTime "win" (~0.6 ms) was
    /// the laptop's two clock states (chunks at ~2.5-3.5 and ~5-6.4 ms whatever the arm).
    uint32_t mSeaFarBeside = 1;
    bool mFarBesideNow = false;       ///< Set while dispatchFarSea runs beside the unit march.
    bool mFarSplitRest = false;       ///< seaFarBeside 2: the even rows ran beside the query; the odd rows go beside the unit march.
    LazyComputePass mpBeamDirtyQueryStripPass;
    LazyComputePass mpBeamDirtyMarchStripPass; ///< The colour output: 0 RGBA32Float, 1 RGBA16Float, 2 R11G11B10Float (see reflect).
    LazyComputePass mpBeamWarpArgsPass;
    LazyComputePass mpBeamPolicyPass; ///< decideBeamPolicy: the warp decision and beamPolicy's step scale and tolerance.
    /// beamPolicy (see HSTRCloudParams): what a build holding few guard blocks spends on a looser tile test (and could on longer
    /// steps, which fail on harder content: policy3). Shipped tolerance-only, 0.01 -> 0.05 as the held share falls 0.25 -> 0.05.
    bool mBeamPolicy = true;
    float mBeamPolicyTolerance = 0.05f;
    float mBeamPolicyHeldLow = 0.05f;
    float mBeamPolicyHeldHigh = 0.25f;
    float mBeamPolicyToleranceNow = 0.f; ///< The compared frame's tolerance (stats, read with the comparison).
    LazyComputePass mpBeamResolveWarpPass;         ///< resolveBeam with HSTR_BEAM_WARP 1, beside the plain one.
    LazyComputePass mpBeamHalfShadePass;           ///< beamHalfResolve: resolveBeamHalf (the shaded parity).
    LazyComputePass mpBeamHalfShadeWarpPass;       ///< resolveBeamHalf with HSTR_BEAM_WARP 1.
    LazyComputePass mpBeamHalfFillPass;            ///< beamHalfResolve: fillBeamHalf (the filled parity, no resolve inside).
    LazyComputePass mpBeamHalfExactArgsPass;       ///< beamHalfResolve: writeBeamHalfExactArgs.
    LazyComputePass mpBeamHalfExactPass;           ///< beamHalfResolve: exactBeamHalf (the fill's listed pixels).
    LazyComputePass mpBeamHalfExactWarpPass;       ///< exactBeamHalf with HSTR_BEAM_WARP 1.
    ref<Texture> mpBeamHalfFlag;                   ///< beamHalfResolve: hstrBeamHalfFlag, R8Uint, (width + 1) / 2 x height.
    ref<Buffer> mpBeamResidualEntry;               ///< beamHalfResolve: hstrBeamResidualEntry, one uint per 8 x 8 screen group.
    ref<Buffer> mpBeamHalfExactList;               ///< beamHalfResolve: hstrBeamHalfExactList, three uints per fill group.
    ref<Buffer> mpBeamHalfExactArgs;               ///< beamHalfResolve: hstrBeamHalfExactArgs (count, two dispatches).
    LazyComputePass mpBeamContinuePass;            ///< beamQueryContinueSlices: continueBeamQueries.
    LazyComputePass mpBeamContinueArgsPass;        ///< beamQueryContinueSlices: writeBeamContinueArgs.
    ref<Buffer> mpBeamContinueList;                ///< hstrBeamContinueList: three uint4 a cut ray.
    ref<Buffer> mpBeamContinueArgs;                ///< hstrBeamContinueArgs: count, then the dispatch.
    LazyComputePass mpBeamResidualResolveWarpPass; ///< resolveBeamResidual with HSTR_BEAM_WARP 1.
    ref<Buffer> mpBeamWarpArgs;                     ///< See hstrBeamWarpArgs.
    uint32_t mBeamWarpHeld = 0;   ///< The compared frame's guard blocks held (hstrBeamDirtyCount[2]) ...
    uint32_t mBeamWarpListed = 0; ///< ... and listed dirty (hstrBeamDirtyCount[0]) ...
    uint32_t mBeamWarpOn = 0;     ///< ... and whether its resolve was warped (stats, read with the comparison).
    bool mBeamGuardCleared = false; ///< The guard holds no claim about any block until a build clears it.
    ref<Buffer> mpBeamDirty;        ///< Blocks whose certificate failed, compacted; sized to hold every block, so it cannot overflow.
    ref<Buffer> mpBeamOrderProbe;   ///< beamOrderProbe: the dirty query's step counters (hstrBeamOrderProbe), 21 uints.
    /// beamDirtyFused: the query, tile test, rebuild, warp field and the units there is time for in one persistent dispatch
    /// (runBeamDirtyFused). Its claim counters, per-block ray counts and finished-block queue, and per-tile reader counts.
    bool mBeamDirtyFused = false;
    bool mBeamFusedBuild = false; ///< This build runs the fused chain (mBeamDirtyFused and nothing it does not cover).
    /// The fused kernel marches units itself (HSTR_FUSED_UNITS). ngfx12 (4K sprint): with them it takes 168 registers, 12 warps
    /// an SM, where the separate query takes 96; without, every unit is the units pass's.
    bool mBeamFusedUnits = true;
    /// DIAGNOSTIC (HSTR_FUSED_RAYS_ONLY): the fused kernel compiled as its ray loop alone - wrong images, for a trace's registers.
    bool mBeamFusedRaysOnly = false;
    /// With beamDirtyFused: the separate dirty query (its own 96 registers) counts finished blocks, and runBeamDirtyFused, compiled
    /// without its ray loop, tests their tiles in a dispatch right behind it with no barrier between - filling the query's tail.
    bool mBeamFusedOverlap = false;
    /// With beamDirtyFused: the separate dirty query tests the tiles of its own wave's blocks once past its march, through the
    /// fused chain's per-tile reader join - no queue, no consumer, no dirty tile pass (HSTR_DIRTY_OVERLAP 2). MEASURED and off
    /// (inline1/2, 4K, tiles/units/errors identical): walk 1.91 -> 2.12, jog 2.41 -> 2.75, sprint 2.33 -> 2.54 ms. At sprint the
    /// program alone (tiles skipped) cost +0.05 over query + rebuild, and its tile work 0.45 ms against the padded-square pass's
    /// 0.32: the join hid nothing. A native Work Graph keeps the same join, so it cannot beat the dense tile pass's 0.115 either.
    bool mBeamFusedInline = false;
    /// The dirty tile test as a thread per tile of the grid (testBeamDirtyTilesDense) instead of a thread per padded tile of each
    /// listed block. MEASURED (dense2, 4K, tiles/units/errors identical): walk 1.94 -> 1.85, jog 2.49 -> 2.30, sprint 2.37 -> 2.17
    /// ms - the padded squares launched 36 threads a block for ~1 test each (0.31 ms of tile pass at sprint, 0.115 dense), and the
    /// units, now listed in tile order, march 0.02-0.06 ms faster.
    bool mBeamDirtyTilesDense = true;
    uint32_t mBeamFusedTilesTested = 0;  ///< The compared frame's dirty tiles tested (either path) ...
    uint32_t mBeamFusedUnitsMarched = 0; ///< ... units the fused chain marched itself ...
    uint32_t mBeamFusedUnitsListed = 0;  ///< ... and units listed (hstrBeamDirtyCount[1]).
    uint32_t mBeamFusedDiag[kFusedDiagCount] = {}; ///< Diagnostic: waits that gave up, sticky (see kFusedDiagCount).
    ref<Buffer> mpBeamFusedState;
    ref<Buffer> mpBeamFusedBlocks;
    ref<Buffer> mpBeamFusedTiles;
    LazyComputePass mpBeamDirtyFusedSetupPass;
    LazyComputePass mpBeamDirtyFusedPass;
    uint32_t mBeamOrderProbeValues[21] = {}; ///< The scored frame's counters.
    ref<Buffer> mpBeamDirtyCount;
    ref<Buffer> mpBeamDirtyArgs;    ///< Two indirect dispatches: the query over those blocks, then the residual over them.
    LazyComputePass mpBeamClassifyGuardPass;
    ref<Texture> mpBeamCoarse;        ///< Coarse certificate: anchor camera and safe radius per cell of blocks.
    ref<Texture> mpBeamCoarseState;   ///< Per coarse cell: children its certificate does not cover, and a queued-for-rebuild bit.
    ref<Buffer> mpBeamDescend;        ///< Blocks the coarse level could not answer for, to classify one by one.
    ref<Buffer> mpBeamDescendCount;   ///< [0] blocks in mpBeamDescend, [1] cells in mpBeamRebuild.
    ref<Buffer> mpBeamRebuild;        ///< Coarse cells to rebuild after this build's query: expired, or a child re-verified.
    LazyComputePass mpBeamCoarsePass;
    LazyComputePass mpBeamLeafPass;
    LazyComputePass mpBeamCoarseUpdatePass;
    LazyComputePass mpBeamDescendArgsPass;
    LazyComputePass mpBeamDirtyArgsPass;
    LazyComputePass mpBeamDirtyQueryPass;
    LazyComputePass mpBeamDirtyMarchPass;
    LazyComputePass mpBeamDirtyMarchRefillPass; ///< beamUnitRefill's units pass (marchBeamDirtyUnitsRefill).
    ref<Buffer> mpBeamRefill;                   ///< Its claim counter.
    LazyComputePass mpBeamDirtyUnitArgsPass;
    bool mBeamSubTiles = false;          ///< beamSubTiles (HSTRCloudParams::beamSubTiles), where the dirty build can run it.
    bool mBeamUnitStartOracle = false;   ///< ORACLE (beamUnitStartOracle): HSTRCloudParams::beamUnitOracle's record pass and start.
    ref<Buffer> mpUnitStartOracle;       ///< Per listed unit, its own first-density distance (hstrUnitStartOracle).
    LazyComputePass mpBeamSubRefinePass; ///< beamSubTiles: refineBeamSubTiles.
    LazyComputePass mpBeamSubArgsPass;   ///< beamSubTiles: writeBeamSubUnitArgs.
    // Cell views (cellViews): cached per-cell views of the transfer. No reader since composeBeam's removal.
    bool mCellViews = false;
    bool mCellViewsClear = true;           ///< The map and the views must be emptied before the next use.
    uint32_t mCellViewFrame = 0;
    ref<Texture> mpCellMap;                ///< Per domain cell: view slot + 1 (R32Uint, hstrCellMap).
    ref<Texture> mpCellRequest;            ///< Per domain cell: the frame + 1 it was last queued in.
    ref<Texture> mpCellOccupancy;          ///< Per domain cell: whether any sample in it can read density (R8Uint).
    bool mCellOccupancyDirty = true;       ///< The domain changed since the occupancy was built.
    LazyComputePass mpCellOccupancyPass;
    ref<Buffer> mpCellViews;               ///< CellView per slot.
    ref<Texture> mpCellTexels;             ///< Atlas: cache rgb, transmittance.
    ref<Texture> mpCellTexelsSingle;       ///< Atlas: single rgb, opacity centroid.
    ref<Buffer> mpCellBuild;               ///< Cells queued for a build this frame.
    ref<Buffer> mpCellCounters;            ///< [0] cells queued this frame.
    ref<Buffer> mpCellCursor;              ///< [0] the slot ring's cursor.
    ref<Buffer> mpCellArgs;                ///< The build dispatch's arguments.
    LazyComputePass mpCellArgsPass;
    LazyComputePass mpCellBuildPass;
    LazyComputePass mpCellInvalidatePass;
    /// Creates (or re-creates, on a size change) the cell view resources, empties them when asked, and sets their parameters.
    void ensureCellViews(RenderContext* pRenderContext);
    /// Creates the cell occupancy, rebuilds it when the domain changed, and relists the occupied cells for beamPushProbe.
    void ensureCellOccupancy(RenderContext* pRenderContext);
    // beamPushProbe: occupied world cells projected into BeamTiles, counted and binned, no radiance (see projectPushCell).
    bool mPushProbe = false;
    bool mPushEval = false;        ///< Also consume the lists at sample directions (needs cellViews); see evaluatePushSample.
    ref<Buffer> mpPushSamples;
    ref<Buffer> mpPushSlotKeys;    ///< Per visible cell: its packed key (the lists hold slots).
    ref<Buffer> mpPushOps;         ///< Per visible cell: its operator, 27 uint4 (buildPushOperator).
    LazyComputePass mpPushOpsPass;
    LazyComputePass mpPushEvalPass;
    LazyComputePass mpPushComparePass;
    /// pushShare: after the dirty passes, count what the push lists' cell x tile interactions could share (see countPushShare).
    bool mPushShare = false;
    /// rtSpanProbe (DIAGNOSTIC): each frame, boxes of every asset's mapped bricks, a bottom-level acceleration structure per asset, a
    /// top-level one over the sea instances, and every pixel's camera ray gathered through them with RayQuery (HSTRCloudRtProbe).
    void runRtSpanProbe(RenderContext* pRenderContext);
    bool mRtSpanProbe = false;
    uint32_t mRtTraceSpans = 0; ///< rtSpanProbeSpans: 0 gathers every box in one query, n > 0 asks n ordered nearest-box queries.
    ref<ComputePass> mpRtBoxesPass;
    ref<ComputePass> mpRtTracePass;
    ref<ComputePass> mpRtReducePass;
    ref<Buffer> mpRtBoxes;
    ref<Buffer> mpRtStats;
    ref<Buffer> mpRtBlasBuffer;
    ref<Buffer> mpRtBlasScratch;
    ref<Buffer> mpRtTlasBuffer;
    ref<Buffer> mpRtTlasScratch;
    ref<Texture> mpRtCount;
    ref<Texture> mpRtSpan;
    std::vector<ref<RtAccelerationStructure>> mRtBlas;
    std::vector<uint64_t> mRtBlasOffset;  ///< Per asset: its structure's byte offset in mpRtBlasBuffer.
    std::vector<uint64_t> mRtBlasScratchOffset;
    std::vector<uint32_t> mRtBoxOffset;   ///< Per asset: its first box slot (one per virtual page).
    ref<RtAccelerationStructure> mRtTlas;
    uint32_t mRtTlasCapacity = 0;
    std::vector<uint32_t> mRtStats;       ///< Last probe frame's gProbeStats (HSTRCloudRtProbe.cs.slang slots).
    uint32_t mRtInstances = 0;            ///< Instances in the last top-level structure.
    /// rasterProbe (DIAGNOSTIC): each frame, the cloud sea drawn by the rasterizer instead of marched (HSTRCloudRaster.slang): the
    /// boxes of the bricks the march would read, cut into view-aligned slices at its step, depth-sorted and drawn front to back.
    void runRasterProbe(RenderContext* pRenderContext);
    /// The apron-free 8^3 copies of the density atlas (and the sun's) behind spanLoop 4096 / 8192 and rasterProbeFetchMask 4.
    void packCompactAtlases(RenderContext* pRenderContext, bool sun);
    uint32_t mCompactDensity = 0; ///< compactDensity: 1 the march's density reads from the apron-free copy, 2 no read (oracle).
    bool mRasterProbe = false;
    float mRasterProbeScale = 0.5f; ///< rasterProbeScale: the probe's target over the frame (0.5: about a beam texel a pixel).
    bool mRasterProbeCount = true;  ///< rasterProbeCount: a second, untimed draw counts the fragments.
    bool mRasterProbeFetch = true;  ///< rasterProbeFetch: false drops the fragments' atlas and bake reads (cost split only).
    uint32_t mRasterProbeFetchMask = 3u; ///< rasterProbeFetchMask: 1 the density read, 2 the sun bake read (cost split only).
    bool mRasterProbePrebuilt = false; ///< rasterProbePrebuilt: polygons built once a slice in rasterScatter, one indexed list.
    ref<Buffer> mpRasterPolygons;      ///< Six world vertices a slice (rasterProbePrebuilt).
    ref<Vao> mpRasterListVao;          ///< Index buffer 6 s + fan over every slice s (rasterProbePrebuilt).
    ref<ComputePass> mpRasterArgsPass;
    ref<ComputePass> mpRasterLevelPass;
    ref<ComputePass> mpRasterScanPass;
    ref<ComputePass> mpRasterScatterPass;
    ref<RasterPass> mpRasterDrawPass;
    ref<RasterPass> mpRasterCountPass;
    ref<Buffer> mpRasterInstances;
    ref<Buffer> mpRasterQueue[2];
    ref<Buffer> mpRasterCounters;
    ref<Buffer> mpRasterArgs;
    ref<Buffer> mpRasterBoxes;
    ref<Buffer> mpRasterBins;
    ref<Buffer> mpRasterItems;
    ref<Texture> mpRasterTarget;
    ref<Fbo> mpRasterFbo;
    ref<Vao> mpRasterVao;
    std::vector<uint32_t> mRasterStats; ///< Last probe frame's gCounters (HSTRCloudRaster.slang slots).
    uint32_t mRasterInstanceCount = 0;  ///< Sea instances in range in the last probe frame.
    uint32_t mRasterSeeds = 0;          ///< Regions it started from, at mRasterSeedLevel.
    uint32_t mRasterSeedLevel = 0;
    /// formR (LIT_VOLUME.md section 6c, gate R1): the lean marches cross empty space between shared occupancy hulls by inline ray
    /// query (1), or only count what that would do (2: HSTR_FORM_R 2, kFormRProbe). Hulls are triangle meshes per (asset, level)
    /// from form_r_hulls.py (formRHulls: their directory; formRDilate false loads the undilated sanity hulls), one bottom-level
    /// structure each, built once; the top-level structure over the sea instances, each at its tile copy nearest the camera with the
    /// hull of the coarsest level its farthest point reads, is rebuilt every frame formR is on.
    /// MEASURED and REJECTED (formr1-4, sunset launcher 4K walk, live residency): R1's stop rule (units + query down 15%) fails.
    /// formR 2 counts: 6.3M lean steps a build, 51% outside every hull, 1.08M queries (6 a ray), but 38k brick density samples
    /// outside the dilated hulls (undilated 124k; formRLevelBias 1 / 2: 17.7k / 3.2k, outside steps 42% / 26%) - samples read
    /// coarse resident ancestors whose reach passes a finer hull. formR 1 cuts warp-paid steps 42% (beamLayerProbe: query 16.4M ->
    /// 9.3M, units 13.4M -> 7.7M) yet units + query 3.86 / 3.70 -> 3.90 / 3.50 ms (formr1) and +2.2% over 0.02 moving. GPU Trace
    /// (formr4, ngfx/formr4{off,on}_export): units 2.76 -> 2.91 ms, instructions only -8%, warps active and register-allocation
    /// stalls unchanged (not occupancy), SM throughput 35 -> 31%, L1 hit 78 -> 74%, DRAM up; query 2.05 -> 1.82 ms. The jumped
    /// steps were already cheap (majorant-zero and layer-mask skips); the cost is the brick chain at samples inside the hulls,
    /// which the hulls cannot remove, and the queries' traversal and BVH traffic outweigh what they skip. Off by default.
    void updateHullTlas(RenderContext* pRenderContext);
    uint32_t mFormR = 0;
    bool mFormRDilate = true;
    int mFormRLevelBias = 0;              ///< formRLevelBias: hull level = the footprint's + this (coarser, more conservative).
    std::string mFormRHulls;
    std::string mFormRLoaded;             ///< The hull set the bottom-level structures hold (directory + dilation), empty if none.
    std::vector<ref<Buffer>> mHullMeshBuffers;
    std::vector<ref<Buffer>> mHullBlasBuffers;
    std::vector<ref<RtAccelerationStructure>> mHullBlas; ///< Per asset * kFormRLevels + level (null where no file).
    ref<Buffer> mpHullTlasBuffer;
    ref<Buffer> mpHullTlasScratch;
    ref<RtAccelerationStructure> mHullTlas;
    uint32_t mHullTlasCapacity = 0;
    uint64_t mHullTlasFrame = ~0ull;      ///< mExecuteFrames of the last build.
    static constexpr uint32_t kFormRLevels = 8;
    /// formB (LIT_VOLUME.md section 4b, gate S2): over an aligned sea (cloudSeaAlign 2) the lean marches read a world table of
    /// level-2 bricks (buildFormBTable) naming each brick's shared packed (density, sun) texel, instead of the instance chain (1),
    /// and count what it answered (2). Built when first wanted and when the camera has moved formBRebuildDistance from where the
    /// table was centred; it reads the residency as it stood then, so S2 runs with residency frozen.
    void updateFormBTable(RenderContext* pRenderContext);
    uint32_t mFormB = 0;
    float mFormBChildDistance1 = -1.f; ///< formBChild1: level-1 children for bricks nearer than this (world units); < 0 from the LOD rule.
    float mFormBChildDistance0 = -1.f; ///< formBChild0: level-0 children likewise.
    float mFormBRebuildDistance = 300.f;
    bool mFormBValid = false;
    float3 mFormBCentre = float3(0.f);
    uint32_t mFormBBuilds = 0;
    uint32_t mFormBStats[13] = {};      ///< The last build's hstrFormBCountsOutput (counts, then fallback reasons 5-12).
    ref<Buffer> mpFormBTable;
    ref<Buffer> mpFormBRecords;
    ref<Buffer> mpFormBChildren;
    ref<Buffer> mpFormBCounts;
    LazyComputePass mpFormBBuildPass;
    uint32_t mHullInstances = 0;          ///< Instances in the last hull top-level structure.
    uint32_t mHullLevelCounts[kFormRLevels] = {}; ///< Of those, by hull level.
    uint64_t mHullTriangles = 0;          ///< Triangles over the loaded hulls.
    /// spanProbe: the dirty march records its rays as spans (HSTR_SHIP bit 262144) and evaluateSpans integrates them alone.
    bool mSpanProbe = false;
    ref<Buffer> mpSpanRays;
    ref<Buffer> mpSpanRecords;
    ref<Buffer> mpSpanCounts;
    ref<Buffer> mpSpanArgs;
    LazyComputePass mpSpanArgsPass;
    LazyComputePass mpSpanEvalPass;
    LazyComputePass mpSpanCheckPass;
    /// spanEval / spanShare (HSTRCloudParams): the per-span evaluation's list and outputs, and the shared-transfer oracle's table.
    ref<Buffer> mpSpanOut;
    ref<Buffer> mpSpanList;
    ref<Buffer> mpSpanBuckets;
    ref<Buffer> mpSpanListArgs;
    ref<Buffer> mpSpanShareKeys;
    ref<Buffer> mpSpanShareCounts;
    ref<Buffer> mpSpanShareValues;
    LazyComputePass mpSpanCountPass;
    LazyComputePass mpSpanScanPass;
    LazyComputePass mpSpanScatterPass;
    LazyComputePass mpSpanListedPass;
    LazyComputePass mpSpanComposePass;
    LazyComputePass mpSpanShareInsertPass;
    LazyComputePass mpSpanShareCheckPass;
    LazyComputePass mpSpanShareCountPass;
    ref<Texture> mpSpanPacked; ///< spanLoop 256: the sun atlas' layout, each bake's brick density beside it.
    LazyComputePass mpSpanPackPass;
    ref<Texture> mpSpanCompact;    ///< spanLoop 4096: the density atlas' core texels, 8^3 a slot at slot x 8.
    ref<Texture> mpSpanCompactSun; ///< spanLoop 8192: the sun atlas' likewise.
    LazyComputePass mpSpanCompactPass;
    LazyComputePass mpSpanCompactSunPass;
    void runSpanProbe(RenderContext* pRenderContext);
    ref<Buffer> mpPushShareRays;    ///< The dirty query's marched points this frame.
    ref<Buffer> mpPushShareEntries; ///< Per list entry: dirty rays crossing it, their steps inside it.
    ref<Buffer> mpPushShareAges;    ///< Per list entry: the temporal oracle's worst crossing error per age.
    ref<Buffer> mpPushBrickVisits;  ///< pushShareMode 8: per brick visit of a dirty chord, tile and entry | samples << 20.
    float3 mPushShareCamera = float3(0.f);
    float3 mPushShareMotion = float3(0.f);
    LazyComputePass mpPushShareCountPass;
    LazyComputePass mpPushShareHistogramPass;
    void runPushShare(RenderContext* pRenderContext);
    ref<Buffer> mpPushCandidates;  ///< Occupied domain cells, packed keys (hstrPushCandidates).
    ref<Buffer> mpPushCandidateCount; ///< [0] cells listed, [1] of them with density of their own.
    ref<Buffer> mpPushArgs;        ///< [0..2] the projection's dispatch, [3..5] the scatter's, [6..8] the sort's.
    ref<Buffer> mpPushPieces;      ///< This frame's projected octant pieces: cell key and tile rectangle.
    ref<Buffer> mpPushTileCounts;  ///< Per tile: cells binned; then the running cursor of the scatter.
    ref<Buffer> mpPushTileOffsets; ///< Per tile: first entry of its list.
    ref<Buffer> mpPushList;        ///< The tiles' cell lists, compact.
    ref<Buffer> mpPushCounts;      ///< kPush* counters; [kPushCountSlots] pieces appended, [+1] list entries allocated.
    ref<Buffer> mpPushSortTiles;   ///< Tiles holding two or more cells, for the sort's dispatch.
    LazyComputePass mpPushListPass;
    LazyComputePass mpPushArgsPass;
    LazyComputePass mpPushSortArgsPass;
    LazyComputePass mpPushProjectPass;
    LazyComputePass mpPushAllocatePass;
    LazyComputePass mpPushScatterPass;
    LazyComputePass mpPushSortPass;
    bool mPushCandidatesDirty = true;
    uint32_t mPushListings = 0;    ///< Candidate listings, cumulative (one per content change).
    /// Runs the probe for this frame's camera (beam view, octahedral image, cloud sea).
    void runPushProbe(RenderContext* pRenderContext);
    ref<Texture> mpBeamDirtyMark;  ///< Per guard block: its first dirty-list slot this build (hstrBeamDirtyMark).
    ref<Buffer> mpBeamDirtyUnits;  ///< The dirty march's compacted units (hstrBeamDirtyUnits).
    /// The HSTR_SUN_LIVE and HSTR_SHIP a dirty march pass compiles with: those of every other beam march.
    void setBeamDirtyMarchDefines(const ref<ComputePass>& pPass);
    /// HSTR_UNIT_START: 0 off, 1 beamUnitStart, 2 beamUnitStart with beamUnitSpan.
    const char* unitStartDefine() const
    {
        return mParams.beamUnitStart == 0.f ? "0" : (mParams.beamUnitSpan > 0.f ? "2" : "1");
    }
    /// HSTR_SUN_PACKED for the full-build passes (the dirty passes add formB in setBeamDirtyMarchDefines): without it a reset build
    /// never reads the packed atlas, and the path-trace gate cannot see the read (hillpack2's on / off / on were identical).
    const char* sunPackedDefine() const { return mCloudSunPackedRead && mParams.cloudSunPackedAtlas != 0 ? "1" : "0"; }
    /// HSTR_CACHE_VIEW: worldCacheView where its bake can stand in for the cache read (textured light tracing, no ring slots).
    const char* cacheViewDefine() const
    {
        return mParams.worldCacheView != 0 && mpWorldCacheView && mParams.worldCacheTextured != 0 && mParams.worldCacheEstimator != 0 &&
                       mParams.worldCacheSunOrder != 4
                   ? "1"
                   : "0";
    }
    LazyComputePass mpBeamDirtyTilePass;
    LazyComputePass mpBeamDirtyTileDensePass;
    LazyComputePass mpBeamRefreshListPass;
    LazyComputePass mpBeamGuardPyramidPass;    ///< Levels 0 - 5 per 32 x 32 tile (buildBeamGuardPyramidTiles) ...
    LazyComputePass mpBeamGuardPyramidTopPass; ///< ... and the levels above, in one group (buildBeamGuardPyramidTop).
    ref<Buffer> mpBeamGuardPyramid;     ///< Min-depth pyramid over the guard blocks (beamCellRadius).
    uint32_t mBeamGuardPyramidLevels = 0;
    bool mBeamGuardDriven = false; ///< This build's blocks all came from the guard's dirty list: no grid regions, no sweeps.
    bool mBeamDirtyActive = false; ///< This build listed invalidated blocks, so the residual has to cover them too.
    uint32_t mBeamClassifyCells = 0; ///< Guard blocks the classification examined, against the whole grid it used to walk.
    ref<Texture> mpBeamLevel;   ///< Level that finalised every finest beam tile.
    ref<Texture> mpBeamQueryMap; ///< Sparse query ID per possible lattice coordinate.
    ref<Texture> mpBeamTileMap;  ///< Final sparse tile ID per finest cell.
    ref<Buffer> mpBeamSparseCandidates;
    ref<Buffer> mpBeamSparseArgs;
    ref<Buffer> mpBeamResults;
    ref<Buffer> mpBeamFinalTiles;
    LazyComputePass mpBeamGuidePass;
    ref<Texture> mpBeamGuide; ///< Full-resolution beam guide: optical depth, distance and sun depth where it reaches one.
    LazyComputePass mpBeamTemporalTilePass;
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
    ref<Buffer> mpProbe;               ///< DIAGNOSTIC (probePixel): one uint4 record per march step of the probed pixel.
    float mCloudVisibilityFloor = 0.01f; ///< CloudView::visibilityFloor.
    float mCloudOutsideImportance = 0.125f; ///< CloudView::outsideImportance.
    uint32_t mCloudTraceSlot = ~0u;      ///< DIAGNOSTIC: CloudResidency::mTraceSlot.
    uint32_t mTransferCrossings = 0;   ///< Brick crossings the last probed frame marched.
    uint32_t mTransferEntries = 0;     ///< Distinct entries a transfer cache would have had to produce for them.
    uint32_t mBeamLevelCounts[kBeamCountSlots] = {};    ///< Tiles entering each level, read with the comparison; the last is the
                                                        ///< per-pixel march list. Eight queries per entry, so these are what the
                                                        ///< query pass costs.
    std::string mSaveReferencePath;    ///< When set, the reference sums are written to <path>_s<slice>.exr at the end of the frame.
    /// DIAGNOSTIC (saveColorRaw): when set, this frame's color output is written to the path raw - uint32 width, height, bytes a
    /// texel, then the texels - for offline oracles (Mogwai's Python has no numpy, the EXRs are PIZ). Cleared once written.
    std::string mSaveColorRawPath;
    std::string mBeamPathDump;   ///< beamPathDump: when set, this frame's per-pixel resolve path is written to <path> (raw R32Uint).
    ref<Texture> mpBeamPathCode; ///< beamPathDump: the per-pixel code (resolveBeamPixel).
    ref<Texture> mpBeamGuardWhy; ///< beamPathDump: per guard block, why it was held or listed (classifyBeamGuardCell).
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
    ref<Texture> mpSunShear; ///< Sun optical depth per sun ray and layer (sunColumns).
    /// The sea's sun field by sun ray (computeSunColumns + resampleFineSun), the whole domain at once, instead of a march per voxel
    /// of the stale tiles.
    /// MEASURED (sunpages3, 4K, against a fresh per-voxel exact frame): parked under a 0.57 deg/frame sun 3.44 -> 3.09 ms at 0.501 ->
    /// 0.363% of pixels over 0.02 (the whole field is current every frame instead of 8 tiles lagging), walk the same 3.33 -> 2.99,
    /// static-sun sprint 2.06 -> 1.96 at 0.358 -> 0.363%. A refresh is 0.23 ms (a march per voxel of 8 tiles was 0.57).
    bool mSunColumns = true;
    /// The sun octaves' gather and blur textures in RGBA16F instead of RGBA32F. MEASURED (sunpages3): errors identical, parked and
    /// walk under a moving sun -0.04 to -0.05 ms (gather 0.18 -> 0.14; the blur 0.31 -> 0.30 is not bandwidth bound).
    bool mSunOctavesHalf = true;
    /// With sunColumns, a moving sun re-marches the beam columns of a tile only once the sun has turned this far (radians) since
    /// its last refresh, nearest first, at most mCloudSunTilesPerFrame a frame. 0: every frame the sun moves, which re-marched the
    /// same nearest tiles every frame and never reached the far ones. MEASURED (suninval3, 4K, 0.57 deg/frame sun, two steps, % of
    /// pixels over 0.02 mean / worst): parked 0 -> 4 deg 3.10 -> 2.46 ms at 0.367/0.373 -> 0.368/0.373 (query + units 1.40 ->
    /// 0.53); walk 3.00 -> 2.92 at 0.369/0.375 -> 0.311/0.320. 8 deg is past the edge: parked 2.11 ms at 0.434/0.445, walk
    /// 0.436/0.596.
    float mSunInvalidateAngle = 0.07f;
    std::vector<float3> mTileSun; ///< Per sea slot: the sun its beam columns were last marched under.
    float3 mFieldSun = float3(0.f); ///< The sun the sunColumns field was last built for.
    /// With sunColumns, a moving sun rebuilds the field and the octaves only once it has turned this far (radians) from the sun they
    /// were built for, and once more when it stops. 0: every frame it moves. MEASURED and off (suncost1, 4K, parked under a 0.57
    /// deg/frame sun): 1 deg 2.03 -> 1.54 ms but 1.78 -> 4.81% of pixels over 0.02, 2 deg 1.43 ms at 6.86% - single scattering
    /// shows any lag. sunOctaveAngle lags the octaves alone.
    float mSunFieldAngle = 0.f;
    float3 mPreviousSun = float3(0.f); ///< Last frame's sun, to tell a sun that has stopped.
    bool mSunStill = true;             ///< This frame's sun is last frame's.
    /// With sunColumns, the sun octaves (multiple scattering) follow a moving sun only once it has turned this far (radians) from
    /// their build, and once more when it stops. 0: every frame it moves. MEASURED (sunoct2, 4K, 0.57 deg/frame sun, scored against
    /// octaves rebuilt at the exact frame's own sun, % over 0.02 mean / worst): parked 0 -> 4 deg 2.16 -> 1.84 ms at 0.374/0.376 ->
    /// 0.368/0.373, walk 2.69 -> 2.45 at 0.312/0.320 both. The field cannot lag like this (suncost1: 1 deg, parked 1.78 -> 4.81%).
    float mSunOctaveAngle = 0.07f;
    float3 mOctaveSun = float3(0.f); ///< The sun the octaves were last built for.
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
    /// Bricks committed a frame: a runtime cap, up to kCloudLoadCapacity (it used to rebuild the residency, so arms could not A/B it).
    /// MEASURED (4K sunset walk at 2 units a frame, 120-frame chunks, interleaved): at 1024 the cap is hit on 112-115 of 120 frames
    /// (loadcap3) - each async cut enters 50-100k bricks at once into a full atlas; at 4096 on 28-42, the rest draining the cut's
    /// list. Chunk ends mapped / backlog: 226k / 18k vs 164k / 89k (cutcadence1), 213k / 31k vs 207k / 71k (loadcap3 chunk 1); wall
    /// p50 3-5 ms lower (loadcap2). GPU time drifts 2.9 -> 5.3 ms over a run whatever the arm; the commit costs ~0.04 ms a 1024.
    /// cloudCutMargin 8 / 4 / 2 (cutmargin1) leaves 2-4 cuts a 120 frames in every arm: the remaining lag is the walk itself, ~630 ms
    /// of worker time a cut (cutcadence1: 1269 ms over 2 cuts, 1896 over 3, worker busy on 117-118 of 120 frames).
    uint32_t mCloudBrickLoadsPerFrame = 4096;
    static constexpr uint32_t kCloudLoadCapacity = 4096; ///< The residency's staging capacity (8 MB of residuals).
    uint32_t mCloudSeaTiles = 8;
    uint32_t mCloudSeaSeed = 1;
    float mCloudSeaCoverage = 0.85f;
    uint32_t mCloudSeaLayers = 1; ///< Staggered cloud layers (CloudSeaDesc::layers).
    uint32_t mCloudSeaAlign = 0;  ///< cloudSeaAlign: Form B's aligned sea (CloudSeaDesc::alignLevel), 0 off.
    float mCloudLodPixels = 1.f;
    uint32_t mCloudFadeFrames = 8;
    bool mCloudVirtual = true;
    std::string mCloudSeaKey;       ///< Settings the resident sea was built for.
    std::string mCloudResidencyKey; ///< Settings the residency was built for.
    std::unique_ptr<DomainStaging> mpDomainStaging; ///< Before the sea: its workers write into it until the sea is gone.
    std::unique_ptr<hstrcloud::CloudSea> mpCloudSea;
    std::unique_ptr<hstrcloud::CloudResidency> mpCloudResidency;
    /// skyModel 2: pl-sky's atmosphere (Atmosphere.h). Its sun radiance outside the atmosphere is atmosphereSunColor times
    /// atmosphereSunIntensity; the clouds' sunRadiance and skyRadiance are then derived from it (updateAtmosphere), not set.
    std::unique_ptr<hstrcloud::Atmosphere> mpAtmosphere;
    AtmosphereParams mAtmosphere;
    float3 mAtmosphereSunColor = float3(1.f, 0.95f, 0.85f);
    float mAtmosphereSunIntensity = 3.f;
    void updateAtmosphere(RenderContext* pRenderContext);
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
    uint64_t mSunPageTilesTotal = 0;     ///< Stats, cumulative: sea tiles whose sun pages were recomputed ...
    uint64_t mSunPageChangedTotal = 0;   ///< ... tiles whose cloud changed (each re-suns the tiles it shadows too) ...
    uint64_t mSunPageQueuedTotal = 0;    ///< ... and tiles the sun-move queue released.
    uint32_t mCloudSunBakesPerFrame = 256; ///< Bricks whose sun depth is baked per frame (CloudResidencyDesc::sunBakesPerFrame).
    uint32_t mCloudSunPoolScale = 1;       ///< Sun atlas slots per density slot (CloudResidencyDesc::sunPoolScale).
    uint32_t mCloudSunPacked = 0;          ///< cloudSunPacked: CloudResidencyDesc::sunPacked (0 none, 1 RG8, 2 RG16F).
    bool mCloudSunPackedRead = false;      ///< cloudSunPackedRead: the beam marches read density + bake from it (HSTR_SUN_PACKED).
    bool mBeamListsFull = false;           ///< beamListsFull (DIAGNOSTIC): the beam lists at four levels' 85 per tile, the old size.
    LazyComputePass mpBakeCloudSunPass;
    // GPU sun bake scheduling (cloudGpuSun, CloudResidencyDesc::gpuSun).
    bool mCloudGpuSun = true;
    LazyComputePass mpReleaseSunPass;
    LazyComputePass mpScanSunPass;
    LazyComputePass mpStampSunPass;
    LazyComputePass mpResetSunBlocksPass;
    LazyComputePass mpResetSunNodesPass;
    LazyComputePass mpAgeSunPass;
    std::vector<const ComputePass*> mResidencyPassesBound; ///< Residency passes holding the current residency's bindings.
    double mBindCpuMs = 0.0;  ///< CPU time in bindRenderer since load, its calls, and the frames executed (cloudStats).
    uint64_t mBindCalls = 0;
    uint64_t mExecuteFrames = 0;
    uint64_t mSceneBoundFrame = ~0ull; ///< The frame bindRenderer last wrote the scene's TLAS and camera.
    /// Per pass: its gHSTRCloud fields as bindRenderer resolved them, by the (literal) name's address; valid while its vars live.
    struct BindCache
    {
        ref<ProgramVars> pVars;
        ShaderVar root;
        std::unordered_map<const char*, ShaderVar> fields;
    };
    std::unordered_map<const ComputePass*, BindCache> mBindCache;
    /// Binds a residency pass (bindResidencyPass in the .cpp) and sets its params; true on its first bind.
    bool bindResidencyPass(RenderContext* pRenderContext, const ref<ComputePass>& pPass, bool scene);
    LazyComputePass mpSelectSunPass;
    LazyComputePass mpEmitSunPass;
    LazyComputePass mpEvictSunPass;
    LazyComputePass mpAssignSunPass;
    ref<Buffer> mpSunReadback; ///< The bake count of a run, read back without waiting.
    ref<Fence> mpSunFence;
    uint64_t mSunReadbackPending = 0;
    bool mSunReadbackRecorded = false; ///< The copy is in the command list; its fence is signalled next frame.
    /// Returns whether a scheduling run was dispatched (it can evict and assign sun slots).
    bool dispatchSunScheduling(RenderContext* pRenderContext);
    bool mSunResolveAlways = false; ///< cloudSunResolveAlways: resolve every frame (the A/B anchor for the dirty test).
    bool mSunResolveValid = false; ///< hstrCloudSunResolved holds the resolve of mSunResolveInputs and the current slot and brick tables.
    uint4 mSunResolveInputs = uint4(0); ///< Sun generation, oldest generation, ancestor reach, resolved buffer it was written to.
    float4 mCloudSunBakeInputs = float4(0.f); ///< Sun direction and density scale the sun generation was last bumped for.
    float mCloudSunBakeNear = -1.f;           ///< sunNearVoxels the sun generation was last bumped for.
    float mCloudSunBakeStep = -1.f;           ///< cloudSunBakeStep the sun generation was last bumped for.
    bool mCloudSunBakeStepResets = true;      ///< cloudSunBakeStepResets: a bake step change rebakes everything.
    /// cloudSunScanWave: scanSunBakes' atomics summed per wave and bin first (HSTRCloudSunFrame::scanWave). Exact. MEASURED
    /// (scanwave1, 4K sunset walk, two pairs): scan 0.334 / 0.292 -> 0.144 / 0.127 ms, chunk-end counts and scores alike.
    bool mCloudSunScanWave = true;
    /// cloudSunStampSplit: stampSunChanges per change and class (HSTRCloudSunFrame::stampSplit). Exact (the same epoch stores).
    /// MEASURED (stampsplit2, 4K sunset walk, two pairs, 60 profiled frames each): stamp 0.188 / 0.069 -> 0.000 / 0.031 ms a frame
    /// (it varies with the frame's changes), chunk-end sunStale 101k / 99k -> 104k / 96k.
    bool mCloudSunStampSplit = true;
    /// cloudSunStampRows: threads per change and class in stampSunChanges, each stamping every nth row of its ranges (exact: the
    /// same epoch stores). stamp1 (4K sunset walk, residency fixed): ~1.9k changes a run (14k before the dedupe), each class a
    /// thread looping over long low-sun sweeps, 0.9 ms a scheduling run.
    /// MEASURED, left at 1 (stamprows1, same walk, arms alternating in one launch, profiled): stamp 0.231 / 0.172 / 0.243 / 0.547 ms
    /// a frame at 1 / 4 / 16 / 64 - not short of threads: each split repeats the ranges' setup, and the stores are the cost.
    uint32_t mCloudSunStampRows = 1;
    float mCloudSunBakeAngle = 0.25f;         ///< Degrees the sun moves from the current generation's bake direction before the next.
    bool mCloudSunLiveMarch = true;           ///< Whether the camera program keeps sunDepthAt's live near march (HSTR_SUN_LIVE).
    /// cloudSunLiveDirty: whether the dirty query and unit march keep it too. Off: compiled out of those two passes only.
    /// MEASURED (4K sunset walk, residency fixed): sunprobe1 (beamLayerProbe, 8 counted frames) 508k lit samples a frame in the
    /// dirty passes, every one answered by a resolved bake (0 unbaked, 0 coarse, 0 brickless) - the live march never runs there;
    /// nolive2 (interleaved, profiled 30 frames each) query 1.418 / 1.290 -> 1.154 / 1.130 ms, units 1.287 / 1.197 -> 1.074 /
    /// 1.121 ms with it compiled out: its untaken code costs the passes ~0.35 ms. livedirty1 (SCORE=2, 4 counted frames):
    /// walk 427k / sprint 658k lit samples a frame, all resolved, 0 fallbacks; walk moving quality >0.02 base 5.87%, off
    /// 4.95%, probe (on) 5.01% - p99.9 0.487 / 0.344 / 0.344: within the arm-to-arm spread, no loss.
    /// Why (ngfx Shader Pipelines, ngfxlive2 on vs ngfxnolive1 off): buildBeamDirtyQueries 168 registers / 12 warps (160 live)
    /// -> 128 / 16 (124 live); marchBeamDirtyUnits 128 / 16 in both (118 / 121 live). The query's L1 hit 77.6 -> 67.5%.
    /// Next step (ngfxregs1, source lines; --debug-shaders puts the query back at 168, so its lines are not the shipped ones): the
    /// units peak 124 is cloudOctantSkirt, the page / cell lookup around it 104-114, 74 lines above 96 holding 31% of samples - 20
    /// warps (96) needs ~28 registers off the whole density lookup, bounded earlier at ~0.1-0.15 ms (occupancy slope). renderFarSea
    /// 168 / 12 peaks in the same skirt (165, 155 without it), so the skirt alone cannot take it to 128.
    bool mCloudSunLiveDirty = false;
    bool mCloudSunKeepStale = true;           ///< Outdated sun bakes answer until they rebake (CloudResidency::setSunKeepStale).
    uint32_t mCloudSunBakesCap = 0;           ///< Runtime cap on sun bakes a frame under cloudSunBakesPerFrame; 0: none.
    uint32_t mCloudSunBakesMoving = 256;      ///< Sun bakes a frame at most while the camera moves; 0: no cap.
    float3 mSunCapCamera[2] = {};             ///< Last frame's camera position and target (the moving cap).
    bool mCloudSunLevelStamps = true;         ///< A mapping change outdates only bakes of its level or finer (stampSunChange).
    bool mCloudCameraKernel = false;          ///< Whether the per-pixel cloud view renders from its own entry point (renderCloudCamera).
    LazyComputePass mpCameraPass;            ///< That entry point.
    LazyComputePass mpFarSeaPass;            ///< renderFarSea: the far sea behind the near march.
    ref<Texture> mpFarField[2];              ///< Its layer (radiance and transmittance), ping-ponged: [mFarCurrent] is this frame's.
    ref<Texture> mpFarDistance[2];           ///< And its opacity-weighted distance, for the reprojection.
    uint32_t mFarCurrent = 0;
    bool mFarLayerValid = false;             ///< mpFarField[mFarCurrent] holds a finished layer.
    float mSeaFarFadeWidth = 0.05f;          ///< seaFarFadeWidth: the near / far cross-fade, a fraction of seaViewDistance below it.
    bool mFarSeaMarchDefines = false;      ///< The far-sea pass holds the march program's defines (seaFarBricks).
    bool mSeaFarOverlap = true;             ///< seaFarOverlap: the run is dispatched beside the dirty unit march (see dispatchFarSea).
    bool mFarRunDeferred = false;            ///< This frame's run waits for the unit march (or the end of the frame).
    uint2 mFarRunDims = uint2(0);            ///< Its far-layer size.
    ref<Buffer> mpFarCounts;                 ///< DIAGNOSTIC: seaFarProbe bit 6 counters.
    bool mFarPending = false;                ///< This frame's run wrote the other one; it becomes current next frame.
    CameraData mFarLayerCamera = {};         ///< The camera of mpFarField[mFarCurrent].
    CameraData mFarPendingCamera = {};
    bool mSeaFarField = true;                ///< seaFarField: composite the far sea (sea views).
    bool mFarSeaActive = false;              ///< This frame's view composites it (sea beam or exact view, seaFarField).
    bool mFarSeaDirty = true;                ///< A property or the frame size changed: the next run recomputes every texel.
    bool mFarSeaTilesChanged = false;        ///< The sea's tiles changed: a refresh cycle.
    float4x4 mFarSeaViewProj;                ///< The view the far sea was last run for.
    uint32_t mFarSeaBakes = ~0u;             ///< mWorldCacheBakes it was last run with.
    uint32_t mFarCountdown = 0;              ///< Runs left in the refresh cycle since the last change.
    uint32_t mFarPhase = 0;
    uint32_t mFarSeaRuns = 0;                ///< Frames that ran it (cloudStats farSeaRuns).
    LazyComputePass mpDecayWorldCachePass;
    // The sea's domain proxy on the GPU (uploadDomainExtinction).
    ref<Texture> mpDomainVolume; ///< Per domain voxel: unscaled mean density and conservative maximum (RG16).
    ref<Texture> mpDomainBlocks; ///< Per majorant block: unscaled maximum and mean over its trilinear support (RG16).
    ref<Texture> mpDomainLayers; ///< Per majorant block: bit L where layer L has density over that support (R8Uint).
    /// Per majorant block: Chebyshev distance in blocks to the nearest one holding a layer (R8Uint, cloudLayerRunDistance), and
    /// the passes' intermediate.
    ref<Texture> mpDomainLayerDistance;
    ref<Texture> mpDomainLayerDistanceTemp;
    bool mLayerDistanceRebuild = false; ///< DIAGNOSTIC (cloudLayerDistanceRebuild): rebuild the field every frame.
    float3 mTravelPosition = float3(0.f); ///< Last frame's camera position (for beamTravel).
    bool mTravelValid = false;
    ref<Buffer> mpDomainRegions; ///< Changed slots, flagged (kDomainRegionZero, kDomainRegionKeep).
    ref<Buffer> mpDomainFrame;
    ref<Buffer> mpDomainStaged;  ///< One batch of packed tile volumes, copied from the staging buffers.
    bool mDomainPassesBound = false;         ///< The domain passes and the world cache clear hold the current resources.
    ref<Buffer> mpClearBoundDeposit;         ///< The world cache the clear holds (a reference, so no new buffer reuses its address).
    LazyComputePass mpDomainExtinctionPass;
    LazyComputePass mpDomainBlocksPass;
    LazyComputePass mpDomainMajorantPass;
    LazyComputePass mpDomainLayerDistancePass[3]; ///< cloudLayerRunDistance's field, one axis each.
    LazyComputePass mpDomainOccupancyPass;
    /// World Y of the occupied band, from the majorant blocks, dilated by one. mParams.seaContentY carries this when cloudSlabClamp
    /// is on and an unbounded range when it is off, so the shader clamps without a branch.
    float2 mSeaContentBand = float2(-std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    std::vector<float> mCloudTileBatches;
    std::vector<uint32_t> mCloudTileReset;
    ref<Buffer> mpCloudTileBatches;
    ref<Buffer> mpCloudTileReset;
    LazyComputePass mpCommitCloudPass;
    LazyComputePass mpDecodeCloudPass;
    LazyComputePass mpOccupancyCloudPass;
    LazyComputePass mpDecodeCloudSerialPass;    ///< The thread-a-brick reference (cloudCommitParallel off, cloudCommitCheck).
    LazyComputePass mpOccupancyCloudSerialPass; ///< Likewise.
    /// cloudCommitParallel: decode and occupancy of loaded bricks with a group of 64 a brick, not a thread.
    bool mCloudCommitParallel = true;
    uint32_t mCloudCommitCheck = 0;      ///< DIAGNOSTIC cloudCommitCheck: frames whose parallel outputs are compared with the serial.
    uint32_t mCommitChecks = 0;          ///< DIAGNOSTIC: commits compared (cumulative).
    uint32_t mCommitMismatches = 0;      ///< DIAGNOSTIC: residuals / occupancy words that differed (cumulative).
    LazyComputePass mpClearDirtyCloudPagesPass;
    LazyComputePass mpMarkDirtyCloudPagesPass;
    LazyComputePass mpCloudPageArgsPass;
    LazyComputePass mpResolveDirtyCloudPagesPass;
    LazyComputePass mpResolveSkirtMasksPass;
    LazyComputePass mpResolveSkirtRegionsPass; ///< Only the masks the frame's page changes can reach (cloudSkirtRegions).
    bool mCloudSkirtRegions = true;
    uint64_t mSkirtFrames = 0;       ///< Frames that rewrote skirt masks.
    uint64_t mSkirtRegionFrames = 0; ///< ... through the regions.
    uint64_t mSkirtRegionPages = 0;  ///< Pages the region passes rewrote (overlaps counted).
    uint64_t mSkirtFullPages = 0;    ///< Pages the full passes rewrote.
    uint64_t mSkirtDirtyRegions = 0; ///< Changed bricks on those frames (dirtyPages regions).
    uint64_t mSkirtDirtyWork = 0;    ///< Their pages, before the skirt widening (overlaps counted).
    LazyComputePass mpCheckSkirtMasksPass;    ///< DIAGNOSTIC (cloudSkirtCheck).
    bool mCloudSkirtCheck = false;            ///< DIAGNOSTIC: count stale skirt masks every frame (blocking readback).
    ref<Buffer> mpSkirtCheckCount;
    ref<Buffer> mpSunBakeProbe; ///< mSunBakeProbe's counts of the frame's bakes (gSunBakeProbe).
    std::array<uint64_t, 11> mSunBakeProbeTotals = {}; ///< Their sums since the last log.
    uint32_t mSkirtMaskChecks = 0;            ///< Frames checked (cloudSkirtCheck).
    uint32_t mSkirtMaskMismatches = 0;        ///< Pages whose stored mask differed from this frame's pages, summed over them.
    bool mBeamDirtyStats = false;             ///< DIAGNOSTIC: sum the dirty blocks and units listed per frame (blocking readback).
    uint32_t mDirtyStatFrames = 0;            ///< Frames summed (beamDirtyStats).
    uint32_t mDirtyStatBlocks = 0;            ///< Dirty blocks listed, summed over them (each block's 2 x stride^2 query rays).
    uint32_t mDirtyStatUnits = 0;             ///< Units listed for the unit march, summed over them.
    LazyComputePass mpResolveCloudSunSlotsPass; ///< Per frame, for the lean march's flat lookups (HSTR_SHIP bit 2048).
    LazyComputePass mpResolveCloudSunCheckPass; ///< DIAGNOSTIC (cloudSunTouchCheck): the full resolve into mpSunResolveCheck.
    LazyComputePass mpCompareCloudSunPass;      ///< DIAGNOSTIC (cloudSunTouchCheck).
    LazyComputePass mpStampCloudSunTouchedPass; ///< The CPU's loads into hstrCloudSunTouched (cloudSunTouch).
    ref<Buffer> mpSunTouchListBuffer;
    /// cloudSunTouch: resolve only the bricks whose chain the scheduler's slot writes or the CPU's loads stamped. MEASURED (suntouch1 /
    /// suntouch2, 4K walk 2, arms alternated in one launch): resolveCloudSun 0.114 / 0.118 ms full against 0.116 / 0.123 incremental
    /// with every arm frame incremental - visiting the capacity's 268k bricks (brick record, parent links) is the cost, not their
    /// slot pairs; on the live walk no frame qualified (0 of 112: an input changed every frame, full 0.203 ms). Off; its exactness
    /// check (cloudSunTouchCheck) never ran on an incremental frame.
    bool mCloudSunTouch = false;
    bool mCloudSunTouchCheck = false;           ///< DIAGNOSTIC: count incremental resolve entries differing from a full one.
    /// cloudSunResolveLoads: between scheduling runs resolveCloudSunSlots resolves only the bricks loaded since the last resolve.
    bool mCloudSunResolveLoads = true;
    uint32_t mCloudSunResolveCheck = 0;  ///< DIAGNOSTIC cloudSunResolveCheck: frames compared with a full resolve over loaded bricks.
    uint32_t mSunResolveListFrames = 0;  ///< DIAGNOSTIC: frames that resolved a list (cumulative).
    uint32_t mSunResolveListBricks = 0;  ///< DIAGNOSTIC: bricks they resolved (cumulative).
    uint32_t mSunResolveListChecks = 0;  ///< DIAGNOSTIC (cumulative).
    uint32_t mSunResolveListMismatches = 0; ///< DIAGNOSTIC: loaded bricks' entries differing from the full resolve (cumulative).
    uint32_t mCloudSunEvery = 1;                ///< cloudSunEvery: the GPU sun scheduling runs every nth frame with n times the cap.
    bool mCloudLoadsBesideCut = true;           ///< cloudLoadsBesideCut: CloudResidency loads while an async cut walk runs.
    float mCloudCutStick = 0.f;                 ///< cloudCutStick: CloudView::cutStick (levels of priority).
    bool mCloudCutThreshold = true;             ///< cloudCutThreshold: CloudView::cutThreshold (MEASURED there).
    bool mCloudCutThresholdCheck = false;       ///< cloudCutThresholdCheck: CloudView::cutThresholdCheck (DIAGNOSTIC).
    bool mCloudScatterUploads = true;           ///< cloudScatterUploads: CloudResidency uploads changed table elements by scatter.
    /// cloudScatterDense: CloudResidency::setScatterDense. MEASURED (scatterdense2, 4K sunset walk 2, warm, arms interleaved,
    /// 120 frames): upload MB per 1k committed bricks sparse 0.54 / 0.57 -> dense 0.28 / 0.33; residency/upload 0.122 / 0.100 ->
    /// 0.116 / 0.061 ms; scatterdense1 check arm 20 compared uploads, 0 mismatches.
    bool mCloudScatterDense = true;
    ref<Buffer> mpSunResolveCheck;
    ref<Buffer> mpSunCheckCount;
    uint64_t mSunResolveFrames = 0;      ///< Frames that resolved the sun slots.
    uint64_t mSunResolveTouchFrames = 0; ///< ... incrementally.
    uint64_t mSunTouchChecks = 0;
    uint64_t mSunTouchMismatches = 0;
    LazyComputePass mpClearWorldCacheTilesPass;
    LazyComputePass mpAdvanceFadesPass;
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
