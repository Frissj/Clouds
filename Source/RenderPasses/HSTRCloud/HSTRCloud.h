/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "Rendering/Volumes/HSTRHierarchy.h"
#include "RenderGraph/RenderPass.h"
#include "HSTRCloudTypes.slang"

#include <array>

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
    void updateHierarchy();
    void uploadHierarchy();
    void uploadExtinction();
    void uploadCorrectionPool(const hstr::DenseMatrix& rootIncident);
    void solveLighting();
    void dispatchLightingSolve();
    void dispatchResidualPages(RenderContext* pRenderContext);
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
    bool mCompareReference = false;
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
