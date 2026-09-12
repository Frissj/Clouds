/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "Rendering/Volumes/HSTRHierarchy.h"
#include "RenderGraph/RenderPass.h"
#include "HSTRCloudTypes.slang"

using namespace Falcor;

/** Hierarchical Schur transport renderer for heterogeneous cloud volumes.
 *
 * Lighting is solved through persistent six-face boundary operators. NanoVDB
 * supplies primary visibility and the deterministic unresolved residual only.
 */
class HSTRCloud : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(HSTRCloud, "HSTRCloud", "Hierarchical Schur transport for heterogeneous clouds.");

    static ref<HSTRCloud> create(ref<Device> pDevice, const Properties& props) { return make_ref<HSTRCloud>(pDevice, props); }

    HSTRCloud(ref<Device> pDevice, const Properties& props);

    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;

private:
    void buildHierarchy();
    void updateHierarchy();
    void uploadHierarchy();
    void solveLighting();
    std::vector<float> sampleLeafDensities() const;
    hstr::DenseMatrix makeNestedLeafTransport(uint32_t leafIndex) const;

    ref<Scene> mpScene;
    ref<ComputePass> mpPass;
    ref<ComputePass> mpSolvePass;
    ref<Buffer> mpLeafRadiance;
    ref<Buffer> mpLeafBasisLeft;
    ref<Buffer> mpLeafBasisRight;
    ref<Buffer> mpLeafTransport;
    ref<Buffer> mpLeafResidualBounds;
    ref<Buffer> mpRootIncident;
    hstr::Hierarchy mHierarchy;
    HSTRCloudParams mParams;
    uint3 mActualLeafDims = uint3(0);
    int3 mGridMin = int3(0);
    int3 mGridMax = int3(0);
    float3 mVoxelSize = float3(0.f);
    std::vector<float> mLeafDensity;
    bool mOptionsChanged = false;
    bool mFirstFrame = true;
};
