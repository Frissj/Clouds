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
    ref<Buffer> mpWorldCache;        ///< World-space radiance cache experiment: SH running sums per cell.
    uint32_t mWorldCacheUpdates = 1; ///< Cache gather passes per frame in the world cache view.
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
