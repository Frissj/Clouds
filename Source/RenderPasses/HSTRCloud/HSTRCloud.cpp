/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "HSTRCloud.h"
#include "RenderGraph/RenderPassStandardFlags.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <unordered_map>

namespace
{
const char kShaderFile[] = "RenderPasses/HSTRCloud/HSTRCloud.cs.slang";
const char kColor[] = "color";
const char kTransportError[] = "transportError";
const char kCutStats[] = "cutStats";

const char kBaseSteps[] = "baseSteps";
const char kRefinementLevel[] = "refinementLevel";
const char kResidualBlend[] = "residualBlend";
const char kDensityScale[] = "densityScale";
const char kSunDirection[] = "sunDirection";
const char kSunRadiance[] = "sunRadiance";
const char kSkyRadiance[] = "skyRadiance";
const char kAnisotropy[] = "anisotropy";
const char kActiveRank[] = "activeRank";
const char kActiveThreshold[] = "activeThreshold";
const char kTraceSpatialOrder[] = "traceSpatialOrder";
const char kCorrectionBudget[] = "correctionBudget";
const char kCorrectionFadeFrames[] = "correctionFadeFrames";
const char kBallisticBudget[] = "ballisticBudget";
const char kNearScatterBudget[] = "nearScatterBudget";
const char kDiffuseRankBudget[] = "diffuseRankBudget";
const char kSchurWindowRadius[] = "schurWindowRadius";
const char kLocalSourceRadiance[] = "localSourceRadiance";
const char kMixedFidelityThreshold[] = "mixedFidelityThreshold";
const char kOperatorDictionaryTolerance[] = "operatorDictionaryTolerance";
const char kCutTransmittanceTolerance[] = "cutTransmittanceTolerance";

uint32_t nextPowerOfTwo(uint32_t value)
{
    uint32_t result = 1;
    while (result < value)
        result <<= 1;
    return result;
}

bool insideWindow(uint3 leaf, uint3 center, uint32_t radius)
{
    return all(abs(int3(leaf) - int3(center)) <= int(radius));
}

/// Runs f(i) for i in [0, count) on all hardware threads. Iterations must write disjoint data.
template<typename F>
void parallelFor(size_t count, F&& f)
{
    const size_t threadCount = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), count);
    std::atomic<size_t> next{0};
    auto worker = [&]()
    {
        for (size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1))
            f(i);
    };
    std::vector<std::thread> threads;
    for (size_t t = 1; t < threadCount; ++t)
        threads.emplace_back(worker);
    worker();
    for (auto& thread : threads)
        thread.join();
}

/// NanoVDB accessors cache tree nodes, so every worker needs its own.
nanovdb::FloatGrid::AccessorType makeAccessor(const ref<Grid>& grid)
{
    return grid->getGridHandle().grid<float>()->getAccessor();
}

float gridValue(const nanovdb::FloatGrid::AccessorType& accessor, int3 p)
{
    return accessor.getValue(nanovdb::Coord(p.x, p.y, p.z));
}

struct LogTimer
{
    const char* label;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~LogTimer()
    {
        logInfo(
            "HSTRCloud: {} took {:.1f} ms.",
            label,
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count()
        );
    }
};

template<typename F>
auto timed(const char* label, F&& f)
{
    LogTimer timer{label};
    return f();
}

} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, HSTRCloud>();
}

HSTRCloud::HSTRCloud(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kBaseSteps)
            mParams.baseSteps = value;
        else if (key == kRefinementLevel)
            mParams.refinementLevel = value;
        else if (key == kResidualBlend)
            mParams.residualBlend = value;
        else if (key == kDensityScale)
            mParams.densityScale = value;
        else if (key == kSunDirection)
            mParams.sunDirection = value;
        else if (key == kSunRadiance)
            mParams.sunRadiance = value;
        else if (key == kSkyRadiance)
            mParams.skyRadiance = value;
        else if (key == kAnisotropy)
            mParams.anisotropy = value;
        else if (key == kActiveRank)
            mParams.activeRank = value;
        else if (key == kActiveThreshold)
            mParams.activeThreshold = value;
        else if (key == kTraceSpatialOrder)
            mParams.traceSpatialOrder = value;
        else if (key == kCorrectionBudget)
            mParams.correctionBudget = value;
        else if (key == kCorrectionFadeFrames)
            mParams.correctionFadeFrames = value;
        else if (key == kBallisticBudget)
            mParams.ballisticBudget = value;
        else if (key == kNearScatterBudget)
            mParams.nearScatterBudget = value;
        else if (key == kDiffuseRankBudget)
            mParams.diffuseRankBudget = value;
        else if (key == kSchurWindowRadius)
            mParams.schurWindowRadius = value;
        else if (key == kLocalSourceRadiance)
            mParams.localSourceRadiance = value;
        else if (key == kMixedFidelityThreshold)
            mParams.mixedFidelityThreshold = value;
        else if (key == kOperatorDictionaryTolerance)
            mParams.operatorDictionaryTolerance = value;
        else if (key == kCutTransmittanceTolerance)
            mParams.cutTransmittanceTolerance = value;
        else
            logWarning("Unknown property '{}' in HSTRCloud.", key);
    }
}

Properties HSTRCloud::getProperties() const
{
    Properties props;
    props[kBaseSteps] = mParams.baseSteps;
    props[kRefinementLevel] = mParams.refinementLevel;
    props[kResidualBlend] = mParams.residualBlend;
    props[kDensityScale] = mParams.densityScale;
    props[kSunDirection] = mParams.sunDirection;
    props[kSunRadiance] = mParams.sunRadiance;
    props[kSkyRadiance] = mParams.skyRadiance;
    props[kAnisotropy] = mParams.anisotropy;
    props[kActiveRank] = mParams.activeRank;
    props[kActiveThreshold] = mParams.activeThreshold;
    props[kTraceSpatialOrder] = mParams.traceSpatialOrder;
    props[kCorrectionBudget] = mParams.correctionBudget;
    props[kCorrectionFadeFrames] = mParams.correctionFadeFrames;
    props[kBallisticBudget] = mParams.ballisticBudget;
    props[kNearScatterBudget] = mParams.nearScatterBudget;
    props[kDiffuseRankBudget] = mParams.diffuseRankBudget;
    props[kSchurWindowRadius] = mParams.schurWindowRadius;
    props[kLocalSourceRadiance] = mParams.localSourceRadiance;
    props[kMixedFidelityThreshold] = mParams.mixedFidelityThreshold;
    props[kOperatorDictionaryTolerance] = mParams.operatorDictionaryTolerance;
    props[kCutTransmittanceTolerance] = mParams.cutTransmittanceTolerance;
    return props;
}

RenderPassReflection HSTRCloud::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput(kColor, "Hierarchical Schur transport cloud radiance")
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .format(ResourceFormat::RGBA32Float);
    reflector.addOutput(kTransportError, "Goal-oriented omitted transport bound")
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .format(ResourceFormat::R32Float);
    reflector.addOutput(kCutStats, "HST cut traversal statistics")
        .bindFlags(ResourceBindFlags::UnorderedAccess)
        .format(ResourceFormat::RGBA32Float);
    return reflector;
}

void HSTRCloud::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mParams.frameDim = compileData.defaultTexDims;
    mCutDirty = true;
}

void HSTRCloud::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass = nullptr;
    mpSolvePass = nullptr;
    mpCameraLightingPass = nullptr;
    mpProjectPass = nullptr;
    mpCutPass = nullptr;
    mpSortPass = nullptr;
    mpQueryPass = nullptr;
    mFirstFrame = true;
    mCameraLightingDirty = true;
    mCutDirty = true;
    mCameraLightingPoseValid = false;
    if (!mpScene)
        return;

    if (!mpScene->getGridVolumes().empty())
    {
        const auto bounds = mpScene->getGridVolume(0)->getBounds();
        logInfo("HSTRCloud: first grid-volume bounds are {} to {}.", bounds.minPoint, bounds.maxPoint);
    }

    ProgramDesc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary(kShaderFile).csEntry("main");
    desc.addTypeConformances(mpScene->getTypeConformances());
    mpPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    ProgramDesc solveDesc;
    solveDesc.addShaderModules(mpScene->getShaderModules());
    solveDesc.addShaderLibrary(kShaderFile).csEntry("solveLeaves");
    solveDesc.addTypeConformances(mpScene->getTypeConformances());
    mpSolvePass = ComputePass::create(mpDevice, solveDesc, mpScene->getSceneDefines());
    ProgramDesc cameraLightingDesc;
    cameraLightingDesc.addShaderModules(mpScene->getShaderModules());
    cameraLightingDesc.addShaderLibrary(kShaderFile).csEntry("updateCameraLighting");
    cameraLightingDesc.addTypeConformances(mpScene->getTypeConformances());
    mpCameraLightingPass = ComputePass::create(mpDevice, cameraLightingDesc, mpScene->getSceneDefines());
    ProgramDesc projectDesc;
    projectDesc.addShaderModules(mpScene->getShaderModules());
    projectDesc.addShaderLibrary(kShaderFile).csEntry("projectCutNodes");
    projectDesc.addTypeConformances(mpScene->getTypeConformances());
    mpProjectPass = ComputePass::create(mpDevice, projectDesc, mpScene->getSceneDefines());
    ProgramDesc cutDesc;
    cutDesc.addShaderModules(mpScene->getShaderModules());
    cutDesc.addShaderLibrary(kShaderFile).csEntry("binCutNodes");
    cutDesc.addTypeConformances(mpScene->getTypeConformances());
    mpCutPass = ComputePass::create(mpDevice, cutDesc, mpScene->getSceneDefines());
    ProgramDesc sortDesc;
    sortDesc.addShaderModules(mpScene->getShaderModules());
    sortDesc.addShaderLibrary(kShaderFile).csEntry("sortTileCuts");
    sortDesc.addTypeConformances(mpScene->getTypeConformances());
    mpSortPass = ComputePass::create(mpDevice, sortDesc, mpScene->getSceneDefines());
    ProgramDesc queryDesc;
    queryDesc.addShaderModules(mpScene->getShaderModules());
    queryDesc.addShaderLibrary(kShaderFile).csEntry("buildCameraQueries");
    queryDesc.addTypeConformances(mpScene->getTypeConformances());
    mpQueryPass = ComputePass::create(mpDevice, queryDesc, mpScene->getSceneDefines());
    buildHierarchy();
}

void HSTRCloud::buildHierarchy()
{
    mpLeafRadiance = nullptr;
    if (!mpScene || mpScene->getGridVolumes().empty())
        return;

    const auto startTime = std::chrono::steady_clock::now();
    const auto& volume = mpScene->getGridVolume(0);
    const auto& grid = volume->getDensityGrid();
    if (!grid)
        return;

    constexpr uint32_t kCellWidth = 16;
    mGridMin = grid->getMinIndex();
    mGridMax = grid->getMaxIndex();
    const uint3 gridExtent = uint3(mGridMax - mGridMin + 1);
    mActualLeafDims = (gridExtent + kCellWidth - 1u) / kCellWidth;
    const uint3 leafDims(nextPowerOfTwo(mActualLeafDims.x), nextPowerOfTwo(mActualLeafDims.y), nextPowerOfTwo(mActualLeafDims.z));
    const size_t leafCount = size_t(leafDims.x) * leafDims.y * leafDims.z;

    mVoxelSize = volume->getBounds().extent() / float3(gridExtent);
    mParams.hstrLeafDims = leafDims;
    mParams.hstrGridMin = mGridMin;
    mParams.hstrCellWidth = kCellWidth;
    mParams.traceSpatialOrder = std::clamp(mParams.traceSpatialOrder, 1u, 2u);
    mParams.hstrFaceDofs = mParams.traceSpatialOrder * mParams.traceSpatialOrder;
    mParams.hstrTraceDofs = 6 * mParams.hstrFaceDofs;
    mParams.hstrStorageRank = mParams.hstrTraceDofs;
    mParams.activeRank = std::clamp(mParams.activeRank, 1u, mParams.hstrStorageRank);
    const float3 windowPosition = clamp(
        (mpScene->getCamera()->getPosition() - volume->getBounds().minPoint) / volume->getBounds().extent(), float3(0.f), float3(0.999999f)
    );
    mParams.schurWindowCenter = min(uint3(windowPosition * float3(mActualLeafDims)), mActualLeafDims - 1u);
    logInfo(
        "HSTRCloud: density grid {} voxels, voxel size {}, {} leaves of {}^3 voxels.", gridExtent, mVoxelSize, mActualLeafDims, kCellWidth
    );
    mLeafDensity = sampleLeafDensities();
    std::vector<hstr::DenseMatrix> leafTransport;
    leafTransport.reserve(leafCount);
    mOperatorDictionary.clear();
    mLeafOperatorIDs.resize(leafCount);
    mDictionaryCorrections.assign(leafCount, hstr::DenseMatrix(mParams.hstrTraceDofs, mParams.hstrTraceDofs));
    std::unordered_map<size_t, std::vector<uint32_t>> dictionaryBuckets;
    const float dictionaryTolerance = std::max(1e-7f, mParams.operatorDictionaryTolerance);
    std::vector<hstr::DenseMatrix> candidates(leafCount);
    {
        LogTimer timer{"nested leaf transport"};
        parallelFor(leafCount, [&](size_t leaf) { candidates[leaf] = makeNestedLeafTransport(uint32_t(leaf)); });
    }
    for (uint32_t leaf = 0; leaf < leafCount; ++leaf)
    {
        hstr::DenseMatrix candidate = std::move(candidates[leaf]);
        size_t hash = 1469598103934665603ull;
        for (float value : candidate.data())
        {
            hash ^= size_t(std::llround(value / dictionaryTolerance));
            hash *= 1099511628211ull;
        }
        uint32_t dictionaryID = ~0u;
        for (uint32_t id : dictionaryBuckets[hash])
        {
            bool matches = true;
            for (size_t i = 0; i < candidate.data().size(); ++i)
                matches &= std::abs(candidate.data()[i] - mOperatorDictionary[id].data()[i]) <= dictionaryTolerance;
            if (matches)
            {
                dictionaryID = id;
                mDictionaryCorrections[leaf] = hstr::subtract(candidate, mOperatorDictionary[id]);
                break;
            }
        }
        if (dictionaryID == ~0u)
        {
            dictionaryID = uint32_t(mOperatorDictionary.size());
            dictionaryBuckets[hash].push_back(dictionaryID);
            mOperatorDictionary.push_back(std::move(candidate));
        }
        mLeafOperatorIDs[leaf] = dictionaryID;
        leafTransport.push_back(mOperatorDictionary[dictionaryID]);
    }
    mParams.operatorDictionarySize = uint32_t(mOperatorDictionary.size());

    {
        LogTimer timer{"Schur hierarchy compile"};
        mHierarchy = hstr::Hierarchy::compile(leafDims, leafTransport, mParams.traceSpatialOrder);
    }
    {
        LogTimer timer{"extinction upload"};
        uploadExtinction();
    }
    const float milliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - startTime).count();
    logInfo("HSTRCloud: compiled {} leaves and {} Schur nodes in {:.1f} ms.", leafCount, mHierarchy.getNodes().size(), milliseconds);
    logInfo("HSTRCloud: operator dictionary contains {} prototypes for {} leaves.", mOperatorDictionary.size(), leafCount);

    {
        LogTimer timer{"hierarchy upload"};
        uploadHierarchy();
    }
    {
        LogTimer timer{"lighting solve"};
        solveLighting();
    }
}

void HSTRCloud::uploadExtinction()
{
    const auto& volume = mpScene->getGridVolume(0);
    const auto& grid = volume->getDensityGrid();
    const uint3 dims = uint3(mGridMax - mGridMin + 1);
    std::vector<float16_t> extinction(size_t(dims.x) * dims.y * dims.z);
    const auto& nodes = mHierarchy.getNodes();
    const uint3 leafDims = mHierarchy.getLeafDims();
    const size_t leafCount = size_t(leafDims.x) * leafDims.y * leafDims.z;
    std::vector<float> minimum(nodes.size(), std::numeric_limits<float>::max());
    std::vector<float> maximum(nodes.size(), 0.f);
    std::vector<uint32_t> sampleCount(nodes.size(), 0u);
    // Leaves partition the voxels, so each worker owns one leaf's voxels and statistics.
    parallelFor(
        leafCount,
        [&](size_t id)
        {
            const uint3 leaf(uint32_t(id % leafDims.x), uint32_t((id / leafDims.x) % leafDims.y), uint32_t(id / (leafDims.x * leafDims.y)));
            const uint3 begin = leaf * mParams.hstrCellWidth;
            if (any(begin >= dims))
                return;
            const uint3 end = min(begin + mParams.hstrCellWidth, dims);
            const auto accessor = makeAccessor(grid);
            for (uint32_t z = begin.z; z < end.z; ++z)
                for (uint32_t y = begin.y; y < end.y; ++y)
                    for (uint32_t x = begin.x; x < end.x; ++x)
                    {
                        const float value =
                            std::max(0.f, gridValue(accessor, mGridMin + int3(x, y, z))) * volume->getDensityScale() * mParams.densityScale;
                        extinction[size_t(x) + size_t(dims.x) * (size_t(y) + size_t(dims.y) * z)] = float16_t(value);
                        minimum[id] = std::min(minimum[id], value);
                        maximum[id] = std::max(maximum[id], value);
                        ++sampleCount[id];
                    }
        }
    );
    std::vector<HSTRCutNode> packed(nodes.size());
    std::vector<uint32_t> parents(nodes.size(), ~0u);
    for (uint32_t z = 0; z < leafDims.z; ++z)
        for (uint32_t y = 0; y < leafDims.y; ++y)
            for (uint32_t x = 0; x < leafDims.x; ++x)
            {
                const size_t id = size_t(x) + size_t(leafDims.x) * (size_t(y) + size_t(leafDims.y) * z);
                if (minimum[id] == std::numeric_limits<float>::max())
                    minimum[id] = 0.f;
                packed[id].minLeft = uint4(x, y, z, hstr::HierarchyNode::kInvalid);
                packed[id].maxRight = uint4(x + 1, y + 1, z + 1, hstr::HierarchyNode::kInvalid);
            }
    for (size_t id = leafCount; id < nodes.size(); ++id)
    {
        const auto& node = nodes[id];
        const HSTRCutNode& left = packed[node.left];
        const HSTRCutNode& right = packed[node.right];
        packed[id].minLeft = uint4(
            std::min(left.minLeft.x, right.minLeft.x),
            std::min(left.minLeft.y, right.minLeft.y),
            std::min(left.minLeft.z, right.minLeft.z),
            node.left
        );
        packed[id].maxRight = uint4(
            std::max(left.maxRight.x, right.maxRight.x),
            std::max(left.maxRight.y, right.maxRight.y),
            std::max(left.maxRight.z, right.maxRight.z),
            node.right
        );
        parents[node.left] = uint32_t(id);
        parents[node.right] = uint32_t(id);
        minimum[id] = std::min(minimum[node.left], minimum[node.right]);
        maximum[id] = std::max(maximum[node.left], maximum[node.right]);
        sampleCount[id] = sampleCount[node.left] + sampleCount[node.right];
    }
    auto extinctionAtVoxel = [&](uint3 p)
    {
        p = min(p, dims - 1u);
        return float(extinction[size_t(p.x) + size_t(dims.x) * (size_t(p.y) + size_t(dims.y) * p.z)]);
    };
    auto pageValue = [](const std::array<float, 8>& corners, float3 p)
    {
        auto mix = [](float a, float b, float t) { return a + t * (b - a); };
        const float x00 = mix(corners[0], corners[1], p.x);
        const float x10 = mix(corners[2], corners[3], p.x);
        const float x01 = mix(corners[4], corners[5], p.x);
        const float x11 = mix(corners[6], corners[7], p.x);
        return mix(mix(x00, x10, p.y), mix(x01, x11, p.y), p.z);
    };
    std::vector<std::array<float, 8>> pageCorners(nodes.size());
    std::vector<float> pageResidual(nodes.size(), 0.f);
    parallelFor(
        nodes.size(),
        [&](size_t id)
        {
            const uint3 minimumVoxel = packed[id].minLeft.xyz() * mParams.hstrCellWidth;
            const uint3 maximumVoxel = packed[id].maxRight.xyz() * mParams.hstrCellWidth;
            for (uint32_t corner = 0; corner < 8; ++corner)
            {
                const uint3 p(
                    (corner & 1u) ? maximumVoxel.x : minimumVoxel.x,
                    (corner & 2u) ? maximumVoxel.y : minimumVoxel.y,
                    (corner & 4u) ? maximumVoxel.z : minimumVoxel.z
                );
                pageCorners[id][corner] = sampleCount[id] > 0 ? extinctionAtVoxel(p) : 0.f;
            }
        }
    );
    // Leaf residuals are independent; internal nodes follow their children, which always have lower IDs.
    parallelFor(
        leafCount,
        [&](size_t id)
        {
            if (!nodes[id].isLeaf() || sampleCount[id] == 0)
                return;
            const uint3 minimumVoxel = packed[id].minLeft.xyz() * mParams.hstrCellWidth;
            const uint3 maximumVoxel = packed[id].maxRight.xyz() * mParams.hstrCellWidth;
            const uint3 end = min(maximumVoxel, dims - 1u);
            const float3 extent = max(float3(maximumVoxel - minimumVoxel), float3(1.f));
            for (uint32_t z = minimumVoxel.z; z <= end.z; ++z)
                for (uint32_t y = minimumVoxel.y; y <= end.y; ++y)
                    for (uint32_t x = minimumVoxel.x; x <= end.x; ++x)
                    {
                        const uint3 p(x, y, z);
                        const float3 local = float3(p - minimumVoxel) / extent;
                        pageResidual[id] = std::max(pageResidual[id], std::abs(extinctionAtVoxel(p) - pageValue(pageCorners[id], local)));
                    }
        }
    );
    for (size_t id = leafCount; id < nodes.size(); ++id)
    {
        const uint3 minimumVoxel = packed[id].minLeft.xyz() * mParams.hstrCellWidth;
        const uint3 maximumVoxel = packed[id].maxRight.xyz() * mParams.hstrCellWidth;
        {
            for (uint32_t childID : {nodes[id].left, nodes[id].right})
            {
                const uint3 childMinimum = packed[childID].minLeft.xyz() * mParams.hstrCellWidth;
                const uint3 childMaximum = packed[childID].maxRight.xyz() * mParams.hstrCellWidth;
                const float3 extent = max(float3(maximumVoxel - minimumVoxel), float3(1.f));
                float pageDifference = 0.f;
                for (uint32_t corner = 0; corner < 8; ++corner)
                {
                    const uint3 p(
                        (corner & 1u) ? childMaximum.x : childMinimum.x,
                        (corner & 2u) ? childMaximum.y : childMinimum.y,
                        (corner & 4u) ? childMaximum.z : childMinimum.z
                    );
                    const float3 local = float3(p - minimumVoxel) / extent;
                    pageDifference = std::max(pageDifference, std::abs(pageCorners[childID][corner] - pageValue(pageCorners[id], local)));
                }
                pageResidual[id] = std::max(pageResidual[id], pageResidual[childID] + pageDifference);
            }
        }
    }
    for (size_t id = 0; id < nodes.size(); ++id)
    {
        packed[id].statistics = float4(minimum[id], maximum[id], nodes[id].residualNorm, 0.f);
        packed[id].transmittancePage = float4(pageResidual[id], nodes[id].isLeaf() && sampleCount[id] > 0 ? 1.f : 0.f, 0.f, 0.f);
        packed[id].transmittanceCorners0 = float4(pageCorners[id][0], pageCorners[id][1], pageCorners[id][2], pageCorners[id][3]);
        packed[id].transmittanceCorners1 = float4(pageCorners[id][4], pageCorners[id][5], pageCorners[id][6], pageCorners[id][7]);
    }
    mParams.hstrExtinctionDims = dims;
    mParams.hstrRootNode = mHierarchy.getRoot();
    mParams.hstrNodeCount = uint32_t(packed.size());
    mCutDirty = true;
    const HSTRCutNode& root = packed[mParams.hstrRootNode];
    logInfo(
        "HSTRCloud: cut root [{}, {}, {}]-[{}, {}, {}], extinction [{}, {}].",
        root.minLeft.x,
        root.minLeft.y,
        root.minLeft.z,
        root.maxRight.x,
        root.maxRight.y,
        root.maxRight.z,
        root.statistics.x,
        root.statistics.y
    );
    if (!mpExtinction || mpExtinction->getWidth() != dims.x || mpExtinction->getHeight() != dims.y || mpExtinction->getDepth() != dims.z)
        mpExtinction = mpDevice->createTexture3D(dims.x, dims.y, dims.z, ResourceFormat::R16Float, 1, extinction.data());
    else
        mpDevice->getRenderContext()->updateTextureData(mpExtinction.get(), extinction.data());
    if (!mpCutNodes || mpCutNodes->getElementCount() != packed.size())
        mpCutNodes = mpDevice->createStructuredBuffer(
            sizeof(HSTRCutNode), packed.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packed.data(), false
        );
    else
        mpCutNodes->setBlob(packed.data(), 0, packed.size() * sizeof(HSTRCutNode));
    if (!mpCutNodeParents || mpCutNodeParents->getElementCount() != parents.size())
        mpCutNodeParents = mpDevice->createStructuredBuffer(
            sizeof(uint32_t), parents.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, parents.data(), false
        );
    else
        mpCutNodeParents->setBlob(parents.data(), 0, parents.size() * sizeof(uint32_t));
    if (!mpExtinctionSampler)
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        samplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpExtinctionSampler = mpDevice->createSampler(samplerDesc);
    }
}

void HSTRCloud::uploadHierarchy()
{
    const uint3 leafDims = mHierarchy.getLeafDims();
    const size_t leafCount = size_t(leafDims.x) * leafDims.y * leafDims.z;
    mLeafTransfers = timed("leaf transfers", [&] { return mHierarchy.getLeafTransferMatrices(); });
    const auto& transfers = mLeafTransfers;
    const auto transports = mHierarchy.getLeafTransportMatrices();
    mHierarchyResidualBounds = mHierarchy.getLeafResidualBounds();
    const size_t traceDofs = mParams.hstrTraceDofs;
    const size_t matrixSize = traceDofs * traceDofs;
    std::vector<float> packedLeft(matrixSize * leafCount, 0.f);
    std::vector<float> packedRight(matrixSize * leafCount, 0.f);
    std::vector<float> packedBallistic(traceDofs * leafCount, 0.f);
    std::vector<float> packedNearScatter(6 * traceDofs * leafCount, 0.f);
    std::vector<float> packedDiffuseLeft(matrixSize * leafCount, 0.f);
    std::vector<float> packedDiffuseRight(matrixSize * leafCount, 0.f);
    std::vector<uint32_t> packedDiffuseRanks(leafCount, 0u);
    mLeafCorrections.assign(leafCount, hstr::DenseMatrix(traceDofs, traceDofs));
    mLeafBaseTransport.assign(leafCount, hstr::DenseMatrix(traceDofs, traceDofs));
    const hstr::TraceTransfer spatialTransfer = hstr::makeConservativeTraceTransfer(6, mParams.hstrFaceDofs);
    mParams.ballisticBudget = std::min<uint32_t>(mParams.ballisticBudget, uint32_t(traceDofs));
    mParams.nearScatterBudget = std::min<uint32_t>(mParams.nearScatterBudget, uint32_t(5 * traceDofs));
    mParams.diffuseRankBudget = std::clamp<uint32_t>(mParams.diffuseRankBudget, 1u, uint32_t(traceDofs));
    mParams.maximumDiffuseRank = 0;
    auto retainLargest = [](const hstr::DenseMatrix& source, size_t budget)
    {
        struct Entry
        {
            size_t row;
            size_t col;
            float value;
        };
        std::vector<Entry> entries;
        for (size_t row = 0; row < source.rows(); ++row)
            for (size_t col = 0; col < source.cols(); ++col)
                if (source(row, col) > 0.f)
                    entries.push_back({row, col, source(row, col)});
        if (entries.size() > budget)
        {
            std::nth_element(
                entries.begin(), entries.begin() + budget, entries.end(), [](const Entry& a, const Entry& b) { return a.value > b.value; }
            );
            entries.resize(budget);
        }
        hstr::DenseMatrix result(source.rows(), source.cols());
        for (const Entry& entry : entries)
            result(entry.row, entry.col) = entry.value;
        return result;
    };
    parallelFor(
        leafCount,
        [&](size_t leaf)
        {
            hstr::DenseMatrix targetBase = transports[leaf];
            if (mParams.traceSpatialOrder > 1)
            {
                const hstr::DenseMatrix coarse =
                    hstr::multiply(spatialTransfer.restriction, hstr::multiply(transports[leaf], spatialTransfer.prolongation));
                targetBase = hstr::multiply(spatialTransfer.prolongation, hstr::multiply(coarse, spatialTransfer.restriction));
            }
            const hstr::TransportCharacterSplit split = hstr::splitTransportCharacters(targetBase, mParams.hstrFaceDofs);
            const hstr::DenseMatrix ballistic = retainLargest(split.ballistic, mParams.ballisticBudget);
            const hstr::DenseMatrix nearScatter = retainLargest(split.nearScatter, mParams.nearScatterBudget);
            struct DiffuseColumn
            {
                size_t col;
                float flux;
            };
            std::vector<DiffuseColumn> diffuseColumns;
            for (size_t col = 0; col < traceDofs; ++col)
            {
                float flux = 0.f;
                for (size_t row = 0; row < traceDofs; ++row)
                    flux += split.diffuse(row, col);
                if (flux > 0.f)
                    diffuseColumns.push_back({col, flux});
            }
            if (diffuseColumns.size() > mParams.diffuseRankBudget)
            {
                std::nth_element(
                    diffuseColumns.begin(),
                    diffuseColumns.begin() + mParams.diffuseRankBudget,
                    diffuseColumns.end(),
                    [](const DiffuseColumn& a, const DiffuseColumn& b) { return a.flux > b.flux; }
                );
                diffuseColumns.resize(mParams.diffuseRankBudget);
            }
            hstr::LowRankOperator diffuse;
            diffuse.left = hstr::DenseMatrix(traceDofs, diffuseColumns.size());
            diffuse.right = hstr::DenseMatrix(traceDofs, diffuseColumns.size());
            for (size_t mode = 0; mode < diffuseColumns.size(); ++mode)
            {
                diffuse.right(diffuseColumns[mode].col, mode) = 1.f;
                for (size_t row = 0; row < traceDofs; ++row)
                    diffuse.left(row, mode) = split.diffuse(row, diffuseColumns[mode].col);
            }
            hstr::DenseMatrix diffuseMatrix = diffuse.reconstruct();
            hstr::DenseMatrix base(traceDofs, traceDofs);
            for (size_t row = 0; row < traceDofs; ++row)
                for (size_t col = 0; col < traceDofs; ++col)
                    base(row, col) = ballistic(row, col) + nearScatter(row, col) + diffuseMatrix(row, col);
            FALCOR_ASSERT(hstr::certifyPassivity(base).valid);
            mLeafBaseTransport[leaf] = base;
            mLeafCorrections[leaf] = hstr::subtract(transports[leaf], base);
            if (leaf < mDictionaryCorrections.size())
                for (size_t row = 0; row < traceDofs; ++row)
                    for (size_t col = 0; col < traceDofs; ++col)
                        mLeafCorrections[leaf](row, col) += mDictionaryCorrections[leaf](row, col);
            for (size_t col = 0; col < traceDofs; ++col)
            {
                const size_t inputFace = col / mParams.hstrFaceDofs;
                const size_t mode = col % mParams.hstrFaceDofs;
                packedBallistic[traceDofs * leaf + col] = ballistic((inputFace ^ 1u) * mParams.hstrFaceDofs + mode, col);
                for (size_t outputFace = 0; outputFace < 6; ++outputFace)
                    packedNearScatter[(traceDofs * leaf + col) * 6 + outputFace] =
                        nearScatter(outputFace * mParams.hstrFaceDofs + mode, col);
            }
            packedDiffuseRanks[leaf] = uint32_t(diffuse.rank());
            for (size_t row = 0; row < traceDofs; ++row)
                for (size_t mode = 0; mode < diffuse.rank(); ++mode)
                {
                    packedDiffuseLeft[matrixSize * leaf + traceDofs * row + mode] = diffuse.left(row, mode);
                    packedDiffuseRight[matrixSize * leaf + traceDofs * row + mode] = diffuse.right(row, mode);
                }
            if (mParams.traceSpatialOrder > 1)
            {
                for (size_t row = 0; row < traceDofs; ++row)
                    for (size_t mode = 0; mode < traceDofs; ++mode)
                        packedLeft[matrixSize * leaf + traceDofs * row + mode] = transfers[leaf](row, mode);
                for (size_t mode = 0; mode < traceDofs; ++mode)
                    packedRight[matrixSize * leaf + traceDofs * mode + mode] = 1.f;
            }
            else
            {
                const hstr::LowRankOperator compressed = hstr::compress(transfers[leaf], 1e-6f, traceDofs);
                for (size_t row = 0; row < traceDofs; ++row)
                    for (size_t mode = 0; mode < compressed.rank(); ++mode)
                    {
                        packedLeft[matrixSize * leaf + traceDofs * row + mode] = compressed.left(row, mode);
                        packedRight[matrixSize * leaf + traceDofs * row + mode] = compressed.right(row, mode);
                    }
            }
        }
    );
    mParams.maximumDiffuseRank = *std::max_element(packedDiffuseRanks.begin(), packedDiffuseRanks.end());
    mpLeafBasisLeft = mpDevice->createStructuredBuffer(
        sizeof(float), packedLeft.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedLeft.data(), false
    );
    mpLeafBasisRight = mpDevice->createStructuredBuffer(
        sizeof(float), packedRight.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedRight.data(), false
    );
    mpLeafBallistic = mpDevice->createStructuredBuffer(
        sizeof(float), packedBallistic.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedBallistic.data(), false
    );
    mpLeafNearScatter = mpDevice->createStructuredBuffer(
        sizeof(float), packedNearScatter.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedNearScatter.data(), false
    );
    mpLeafDiffuseLeft = mpDevice->createStructuredBuffer(
        sizeof(float), packedDiffuseLeft.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedDiffuseLeft.data(), false
    );
    mpLeafDiffuseRight = mpDevice->createStructuredBuffer(
        sizeof(float),
        packedDiffuseRight.size(),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        packedDiffuseRight.data(),
        false
    );
    mpLeafDiffuseRanks = mpDevice->createStructuredBuffer(
        sizeof(uint32_t),
        packedDiffuseRanks.size(),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        packedDiffuseRanks.data(),
        false
    );
    mpLeafRadiance = mpDevice->createStructuredBuffer(
        sizeof(float4),
        traceDofs * leafCount,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );
    const uint3 cameraLightingDims = leafDims;
    if (!mpCameraLighting || mpCameraLighting->getWidth() != cameraLightingDims.x ||
        mpCameraLighting->getHeight() != cameraLightingDims.y || mpCameraLighting->getDepth() != cameraLightingDims.z)
        mpCameraLighting = mpDevice->createTexture3D(
            cameraLightingDims.x,
            cameraLightingDims.y,
            cameraLightingDims.z,
            ResourceFormat::RGBA16Float,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
    mCameraLightingDirty = true;
    logInfo(
        "HSTRCloud: split transport budgets ballistic={} near={} diffuse={} (certified max rank {}).",
        mParams.ballisticBudget,
        mParams.nearScatterBudget,
        mParams.diffuseRankBudget,
        mParams.maximumDiffuseRank
    );
}

void HSTRCloud::uploadCorrectionPool(const hstr::DenseMatrix& rootIncident)
{
    struct CandidateColumn
    {
        uint32_t leaf;
        uint32_t col;
        float goalError = 0.f;
        std::vector<CorrectionAtom> atoms;
    };
    const auto& transfers = mLeafTransfers;
    const float3 sun = normalize(mParams.sunDirection);
    uint32_t sunAxis = 0;
    if (std::abs(sun.y) > std::abs(sun.x))
        sunAxis = 1;
    if (std::abs(sun.z) > std::abs(sun[sunAxis]))
        sunAxis = 2;
    const uint32_t sunFace = 2 * sunAxis + (sun[sunAxis] >= 0.f ? 1u : 0u);
    // Rank residual corrections against every outgoing trace degree of freedom.
    // This keeps the correction pool view-independent, so camera motion never
    // requires a CPU adjoint rebuild or a transport re-solve.
    mParams.adjointGoalCount = mParams.hstrTraceDofs + 2;
    hstr::DenseMatrix rootGoals(mParams.hstrTraceDofs, mParams.adjointGoalCount);
    for (uint32_t dof = 0; dof < mParams.hstrTraceDofs; ++dof)
        rootGoals(dof, dof) = 1.f;
    for (uint32_t mode = 0; mode < mParams.hstrFaceDofs; ++mode)
        rootGoals(sunFace * mParams.hstrFaceDofs + mode, mParams.hstrTraceDofs) = 1.f / float(mParams.hstrFaceDofs);
    for (uint32_t dof = 0; dof < mParams.hstrTraceDofs; ++dof)
        rootGoals(dof, mParams.hstrTraceDofs + 1) = 1.f / float(mParams.hstrTraceDofs);
    const auto goals = timed("adjoint goals", [&] { return mHierarchy.getLeafAdjointGoalMatrices(rootGoals); });
    const size_t leafCount = transfers.size();
    std::vector<float> packedResponses(leafCount * mParams.hstrStorageRank, 0.f);
    parallelFor(
        leafCount,
        [&](size_t leaf)
        {
            hstr::DenseMatrix basis = transfers[leaf];
            if (mParams.traceSpatialOrder == 1)
            {
                const hstr::DenseMatrix compressed = hstr::compress(transfers[leaf], 1e-6f, mParams.hstrTraceDofs).left;
                basis = hstr::DenseMatrix(mParams.hstrTraceDofs, mParams.hstrStorageRank);
                for (size_t row = 0; row < compressed.rows(); ++row)
                    for (size_t mode = 0; mode < compressed.cols(); ++mode)
                        basis(row, mode) = compressed(row, mode);
            }
            const hstr::DenseMatrix responses =
                hstr::multiply(hstr::transpose(goals[leaf]), hstr::multiply(mLeafBaseTransport[leaf], basis));
            for (uint32_t mode = 0; mode < mParams.hstrStorageRank; ++mode)
                for (uint32_t goal = 0; goal < mParams.adjointGoalCount; ++goal)
                    packedResponses[leaf * mParams.hstrStorageRank + mode] =
                        std::max(packedResponses[leaf * mParams.hstrStorageRank + mode], std::abs(responses(goal, mode)));
        }
    );
    mpLeafAdjointResponses = mpDevice->createStructuredBuffer(
        sizeof(float), packedResponses.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedResponses.data(), false
    );
    std::vector<float> omitted = mHierarchyResidualBounds;
    std::vector<std::vector<CandidateColumn>> leafCandidates(leafCount);
    parallelFor(
        leafCount,
        [&](size_t leafIndex)
        {
            const uint32_t leaf = uint32_t(leafIndex);
            std::vector<CandidateColumn>& candidates = leafCandidates[leaf];
            const hstr::DenseMatrix incident = hstr::multiply(transfers[leaf], rootIncident);
            const hstr::DenseMatrix& detail = mLeafCorrections[leaf];
            for (uint32_t col = 0; col < mParams.hstrTraceDofs; ++col)
            {
                CandidateColumn candidate{leaf, col};
                float incidentMagnitude = 0.f;
                for (uint32_t channel = 0; channel < 3; ++channel)
                    incidentMagnitude += std::abs(incident(col, channel));
                for (uint32_t row = 0; row < mParams.hstrTraceDofs; ++row)
                {
                    const float value = detail(row, col);
                    if (std::abs(value) <= 1e-12f)
                        continue;
                    float goalError = 0.f;
                    for (uint32_t goal = 0; goal < mParams.adjointGoalCount; ++goal)
                        goalError += std::abs(goals[leaf](row, goal) * value) * incidentMagnitude;
                    candidate.goalError += goalError;
                    candidate.atoms.push_back({row, col, value, goalError});
                }
                if (candidate.goalError <= 1e-12f)
                    continue;
                omitted[leaf] += candidate.goalError;
                candidates.push_back(std::move(candidate));
            }
        }
    );
    std::vector<CandidateColumn> candidates;
    for (auto& perLeaf : leafCandidates)
        std::move(perLeaf.begin(), perLeaf.end(), std::back_inserter(candidates));
    leafCandidates.clear();

    const size_t candidateColumnCount = candidates.size();
    size_t candidateAtomCount = 0;
    for (const CandidateColumn& candidate : candidates)
        candidateAtomCount += candidate.atoms.size();
    std::stable_sort(
        candidates.begin(),
        candidates.end(),
        [](const CandidateColumn& a, const CandidateColumn& b)
        { return a.goalError / float(a.atoms.size()) > b.goalError / float(b.atoms.size()); }
    );
    size_t admittedAtomCount = 0;
    std::vector<CandidateColumn> admitted;
    admitted.reserve(candidates.size());
    for (CandidateColumn& candidate : candidates)
    {
        if (candidate.atoms.size() > size_t(mParams.correctionBudget) - admittedAtomCount)
            continue;
        admittedAtomCount += candidate.atoms.size();
        admitted.push_back(std::move(candidate));
    }
    std::sort(
        admitted.begin(),
        admitted.end(),
        [](const CandidateColumn& a, const CandidateColumn& b) { return a.leaf != b.leaf ? a.leaf < b.leaf : a.col < b.col; }
    );

    std::vector<uint2> ranges(leafCount, uint2(0));
    std::vector<CorrectionAtom> atoms;
    atoms.reserve(admittedAtomCount);
    for (const CandidateColumn& candidate : admitted)
    {
        uint2& range = ranges[candidate.leaf];
        if (range.y == 0)
            range.x = uint32_t(atoms.size());
        range.y += uint32_t(candidate.atoms.size());
        atoms.insert(atoms.end(), candidate.atoms.begin(), candidate.atoms.end());
    }
    mParams.correctionAtomCount = uint32_t(atoms.size());
    const CorrectionAtom dummy;
    mpCorrectionAtoms = mpDevice->createStructuredBuffer(
        sizeof(CorrectionAtom),
        std::max<size_t>(1, atoms.size()),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        atoms.empty() ? &dummy : atoms.data(),
        false
    );
    mpCorrectionRanges = mpDevice->createStructuredBuffer(
        sizeof(uint2), ranges.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, ranges.data(), false
    );
    mpLeafResidualBounds = mpDevice->createStructuredBuffer(
        sizeof(float), omitted.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, omitted.data(), false
    );
    logInfo(
        "HSTRCloud: admitted {} of {} residual atoms in {} of {} complete columns against {} persistent adjoint goals.",
        atoms.size(),
        candidateAtomCount,
        admitted.size(),
        candidateColumnCount,
        mParams.adjointGoalCount
    );
}

std::vector<float> HSTRCloud::sampleLeafDensities() const
{
    constexpr uint32_t kSamplesPerAxis = 2;
    const uint32_t cellWidth = mParams.hstrCellWidth;
    const auto& grid = mpScene->getGridVolume(0)->getDensityGrid();
    const uint3 leafDims = mParams.hstrLeafDims;
    std::vector<float> density(size_t(leafDims.x) * leafDims.y * leafDims.z, 0.f);
    for (uint32_t z = 0; z < leafDims.z; ++z)
        for (uint32_t y = 0; y < leafDims.y; ++y)
            for (uint32_t x = 0; x < leafDims.x; ++x)
            {
                const uint3 cell(x, y, z);
                if (!all(cell < mActualLeafDims))
                    continue;
                const int3 begin = mGridMin + int3(cell * cellWidth);
                const int3 end = min(mGridMax + 1, begin + int(cellWidth));
                float mean = 0.f;
                for (uint32_t sz = 0; sz < kSamplesPerAxis; ++sz)
                    for (uint32_t sy = 0; sy < kSamplesPerAxis; ++sy)
                        for (uint32_t sx = 0; sx < kSamplesPerAxis; ++sx)
                        {
                            const float3 u = (float3(sx, sy, sz) + 0.5f) / float(kSamplesPerAxis);
                            const int3 p = min(end - 1, begin + int3(u * float3(end - begin)));
                            mean += std::max(0.f, grid->getValue(p));
                        }
                const size_t index = size_t(x) + size_t(leafDims.x) * (size_t(y) + size_t(leafDims.y) * z);
                density[index] = mean / float(kSamplesPerAxis * kSamplesPerAxis * kSamplesPerAxis);
            }
    return density;
}

hstr::DenseMatrix HSTRCloud::makeNestedLeafTransport(uint32_t leafIndex) const
{
    constexpr uint32_t kSubcellsPerAxis = 2;
    const uint3 leafDims = mParams.hstrLeafDims;
    const uint3 cell(leafIndex % leafDims.x, (leafIndex / leafDims.x) % leafDims.y, leafIndex / (leafDims.x * leafDims.y));
    if (!all(cell < mActualLeafDims))
    {
        const hstr::DenseMatrix clear = hstr::makeLeafTransport(float3(0.f), 0.f, mParams.anisotropy, 0.65f);
        return mParams.traceSpatialOrder == 1 ? clear
                                              : hstr::makeGridBoundaryTransport(kSubcellsPerAxis, std::vector<hstr::DenseMatrix>(8, clear));
    }

    const auto& volume = mpScene->getGridVolume(0);
    const auto accessor = makeAccessor(volume->getDensityGrid());
    const int3 begin = mGridMin + int3(cell * mParams.hstrCellWidth);
    const int3 end = min(mGridMax + 1, begin + int(mParams.hstrCellWidth));
    const float3 subcellSize = mVoxelSize * float3(end - begin) / float(kSubcellsPerAxis);
    const float3 albedo = volume->getAlbedo();
    const float meanAlbedo = (albedo.x + albedo.y + albedo.z) / 3.f;
    std::vector<hstr::DenseMatrix> subcellTransport;
    std::vector<float> subcellDensity;
    subcellTransport.reserve(kSubcellsPerAxis * kSubcellsPerAxis * kSubcellsPerAxis);
    subcellDensity.reserve(kSubcellsPerAxis * kSubcellsPerAxis * kSubcellsPerAxis);
    for (uint32_t z = 0; z < kSubcellsPerAxis; ++z)
        for (uint32_t y = 0; y < kSubcellsPerAxis; ++y)
            for (uint32_t x = 0; x < kSubcellsPerAxis; ++x)
            {
                const float3 u = (float3(x, y, z) + 0.5f) / float(kSubcellsPerAxis);
                const int3 p = min(end - 1, begin + int3(u * float3(end - begin)));
                const float density = std::max(0.f, gridValue(accessor, p)) * volume->getDensityScale() * mParams.densityScale;
                subcellDensity.push_back(density);
                subcellTransport.push_back(hstr::makeLeafTransport(density * subcellSize, meanAlbedo, mParams.anisotropy, 0.65f));
            }
    if (mParams.traceSpatialOrder > 1)
    {
        if (insideWindow(cell, mParams.schurWindowCenter, mParams.schurWindowRadius))
            return hstr::makeGridBoundaryTransport(kSubcellsPerAxis, subcellTransport);
        const auto [minimumDensity, maximumDensity] = std::minmax_element(subcellDensity.begin(), subcellDensity.end());
        const float relativeVariation = (*maximumDensity - *minimumDensity) / std::max(1e-4f, *maximumDensity);
        if (relativeVariation <= mParams.mixedFidelityThreshold)
        {
            const hstr::Hierarchy local = hstr::Hierarchy::compile(uint3(kSubcellsPerAxis), subcellTransport);
            const hstr::DenseMatrix p0 = local.getNodes()[local.getRoot()].transport;
            const hstr::TraceTransfer lift = hstr::makeConservativeTraceTransfer(6, mParams.hstrFaceDofs);
            return hstr::multiply(lift.prolongation, hstr::multiply(p0, lift.restriction));
        }
        return hstr::makeGridBoundaryTransport(kSubcellsPerAxis, subcellTransport);
    }
    const hstr::Hierarchy local = hstr::Hierarchy::compile(uint3(kSubcellsPerAxis), subcellTransport);
    return local.getNodes()[local.getRoot()].transport;
}

void HSTRCloud::updateHierarchy()
{
    const std::vector<float> updatedDensity = sampleLeafDensities();
    std::vector<uint32_t> changed;
    for (uint32_t leaf = 0; leaf < updatedDensity.size(); ++leaf)
        if (std::abs(updatedDensity[leaf] - mLeafDensity[leaf]) > 1e-4f * std::max(1.f, mLeafDensity[leaf]))
            changed.push_back(leaf);
    if (changed.empty())
        return;
    const auto oldTransport = mHierarchy.getLeafTransportMatrices();
    std::vector<hstr::DenseMatrix> newTransport;
    newTransport.reserve(changed.size());
    size_t maximumUpdateRank = 0;
    for (uint32_t leaf : changed)
    {
        newTransport.push_back(makeNestedLeafTransport(leaf));
        const hstr::DenseMatrix delta = hstr::subtract(newTransport.back(), oldTransport[leaf]);
        maximumUpdateRank = std::max(maximumUpdateRank, hstr::compress(delta, 1e-4f, mParams.hstrTraceDofs).rank());
    }

    hstr::UpdateStrategy strategy = hstr::UpdateStrategy::SubtreeRepair;
    if (changed.size() * 4 >= updatedDensity.size())
        strategy = hstr::UpdateStrategy::FullRefactorization;
    else if (maximumUpdateRank <= 2 && changed.size() <= 32)
        strategy = hstr::UpdateStrategy::Woodbury;
    else if (maximumUpdateRank <= 8 && changed.size() <= 128)
        strategy = hstr::UpdateStrategy::WoodburyKrylov;

    if (strategy == hstr::UpdateStrategy::FullRefactorization)
    {
        buildHierarchy();
        logInfo("HSTRCloud: full refactorization for {} changed leaves (measured update rank {}).", changed.size(), maximumUpdateRank);
        return;
    }

    uint32_t repairedNodes = 0;
    for (size_t i = 0; i < changed.size(); ++i)
    {
        repairedNodes += mHierarchy.updateLeaf(changed[i], newTransport[i], strategy);
        mDictionaryCorrections[changed[i]] = hstr::DenseMatrix(mParams.hstrTraceDofs, mParams.hstrTraceDofs);
        mLeafDensity[changed[i]] = updatedDensity[changed[i]];
    }
    uploadHierarchy();
    uploadExtinction();
    solveLighting();
    mOptionsChanged = true;
    const char* strategyName = strategy == hstr::UpdateStrategy::Woodbury         ? "Woodbury"
                               : strategy == hstr::UpdateStrategy::WoodburyKrylov ? "Woodbury-plus-Krylov"
                                                                                  : "subtree";
    logInfo(
        "HSTRCloud: {} repair of {} changed leaves at measured rank {} across {} ancestor updates.",
        strategyName,
        changed.size(),
        maximumUpdateRank,
        repairedNodes
    );
}

void HSTRCloud::solveLighting()
{
    if (mHierarchy.getRoot() == hstr::HierarchyNode::kInvalid)
        return;

    hstr::DenseMatrix incident(mParams.hstrTraceDofs, 3);
    for (uint32_t face = 0; face < 6; ++face)
        for (uint32_t mode = 0; mode < mParams.hstrFaceDofs; ++mode)
            for (uint32_t channel = 0; channel < 3; ++channel)
                incident(face * mParams.hstrFaceDofs + mode, channel) = mParams.skyRadiance[channel];
    const float3 sun = normalize(mParams.sunDirection);
    uint32_t sunAxis = 0;
    if (std::abs(sun.y) > std::abs(sun.x))
        sunAxis = 1;
    if (std::abs(sun.z) > std::abs(sun[sunAxis]))
        sunAxis = 2;
    const uint32_t sunFace = 2 * sunAxis + (sun[sunAxis] >= 0.f ? 1u : 0u);
    for (uint32_t mode = 0; mode < mParams.hstrFaceDofs; ++mode)
        for (uint32_t channel = 0; channel < 3; ++channel)
            incident(sunFace * mParams.hstrFaceDofs + mode, channel) += 0.2f * std::abs(sun[sunAxis]) * mParams.sunRadiance[channel];

    if (any(mParams.localSourceRadiance > 0.f))
    {
        const auto sourceToRoot = mHierarchy.getLeafSourceToRootMatrices();
        const uint3 leafDims = mHierarchy.getLeafDims();
        const size_t sourceLeaf =
            size_t(mParams.schurWindowCenter.x) +
            size_t(leafDims.x) * (size_t(mParams.schurWindowCenter.y) + size_t(leafDims.y) * mParams.schurWindowCenter.z);
        hstr::DenseMatrix localSource(mParams.hstrTraceDofs, 3);
        for (uint32_t dof = 0; dof < mParams.hstrTraceDofs; ++dof)
            for (uint32_t channel = 0; channel < 3; ++channel)
                localSource(dof, channel) = mParams.localSourceRadiance[channel] / float(mParams.hstrTraceDofs);
        const hstr::DenseMatrix rootSource = hstr::multiply(sourceToRoot[sourceLeaf], localSource);
        for (uint32_t dof = 0; dof < mParams.hstrTraceDofs; ++dof)
            for (uint32_t channel = 0; channel < 3; ++channel)
                incident(dof, channel) += rootSource(dof, channel);
    }

    std::vector<float4> rootIncident(mParams.hstrTraceDofs);
    for (uint32_t dof = 0; dof < mParams.hstrTraceDofs; ++dof)
        rootIncident[dof] = float4(incident(dof, 0), incident(dof, 1), incident(dof, 2), 0.f);
    mpRootIncident = mpDevice->createStructuredBuffer(
        sizeof(float4), rootIncident.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, rootIncident.data(), false
    );
    uploadCorrectionPool(incident);
    dispatchLightingSolve();
}

void HSTRCloud::dispatchLightingSolve()
{
    if (!mpSolvePass || !mpLeafRadiance)
        return;
    ShaderVar var = mpSolvePass->getRootVar()["CB"]["gHSTRCloud"];
    var["params"].setBlob(mParams);
    var["hstrRadiance"] = mpLeafRadiance;
    var["hstrLeafBasisLeft"] = mpLeafBasisLeft;
    var["hstrLeafBasisRight"] = mpLeafBasisRight;
    var["hstrLeafBallistic"] = mpLeafBallistic;
    var["hstrLeafNearScatter"] = mpLeafNearScatter;
    var["hstrLeafDiffuseLeft"] = mpLeafDiffuseLeft;
    var["hstrLeafDiffuseRight"] = mpLeafDiffuseRight;
    var["hstrLeafDiffuseRanks"] = mpLeafDiffuseRanks;
    var["hstrLeafResidualBounds"] = mpLeafResidualBounds;
    var["hstrRootIncident"] = mpRootIncident;
    var["hstrLeafAdjointResponses"] = mpLeafAdjointResponses;
    var["hstrCorrectionRanges"] = mpCorrectionRanges;
    var["hstrCorrectionAtoms"] = mpCorrectionAtoms;
    const uint3 leafDims = mHierarchy.getLeafDims();
    mpSolvePass->execute(mpDevice->getRenderContext(), uint3(leafDims.x * leafDims.y * leafDims.z, 1, 1));
    mCameraLightingDirty = true;
}

void HSTRCloud::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mpScene)
    {
        const auto updates = mpScene->getUpdates();
        const float3 cameraPosition = mpScene->getCamera()->getPosition();
        const float3 cameraDirection = mpScene->getCamera()->getTarget() - cameraPosition;
        const bool cameraChanged =
            !mCameraLightingPoseValid || any(cameraPosition != mCameraLightingPosition) || any(cameraDirection != mCameraLightingDirection);
        mCameraLightingDirty |= cameraChanged || is_set(updates, IScene::UpdateFlags::GridVolumesMoved);
        mCutDirty |= cameraChanged || is_set(updates, IScene::UpdateFlags::GridVolumesMoved);
    }
    if (!mFirstFrame && mpScene)
    {
        const auto updates = mpScene->getUpdates();
        if (is_set(updates, IScene::UpdateFlags::GridVolumeBoundsChanged))
            buildHierarchy();
        else if (is_set(updates, IScene::UpdateFlags::GridVolumeGridsChanged))
        {
            const auto& grid = mpScene->getGridVolume(0)->getDensityGrid();
            if (!grid || any(grid->getMinIndex() != mGridMin) || any(grid->getMaxIndex() != mGridMax))
                buildHierarchy();
            else
                updateHierarchy();
        }
        else if (is_set(updates, IScene::UpdateFlags::GridVolumePropertiesChanged))
            buildHierarchy();
    }
    mFirstFrame = false;

    if (mOptionsChanged)
    {
        auto& dict = renderData.getDictionary();
        const auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[kRenderPassRefreshFlags] = flags | RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
        mParams.frameIndex = 0;
    }

    const auto& color = renderData.getTexture(kColor);
    const auto& error = renderData.getTexture(kTransportError);
    const auto& cutStats = renderData.getTexture(kCutStats);
    const uint2 frameDim(color->getWidth(), color->getHeight());
    mCutDirty |= any(frameDim != mParams.frameDim);
    mParams.frameDim = frameDim;
    mParams.hstrTileDims = (frameDim + 7u) / 8u;
    mParams.hstrQueryDims = (frameDim + mParams.hstrQueryStride - 1u) / mParams.hstrQueryStride + 1u;
    if (!mpScene || !mpPass || !mpLeafRadiance)
    {
        pRenderContext->clearUAV(color->getUAV().get(), float4(0.f));
        pRenderContext->clearUAV(error->getUAV().get(), float4(0.f));
        pRenderContext->clearUAV(cutStats->getUAV().get(), float4(0.f));
        return;
    }

    if (mCameraLightingDirty)
    {
        FALCOR_PROFILE(pRenderContext, "cameraLighting");
        mpScene->bindShaderDataForRaytracing(pRenderContext, mpCameraLightingPass->getRootVar()["gScene"]);
        ShaderVar cameraVar = mpCameraLightingPass->getRootVar()["CB"]["gHSTRCloud"];
        cameraVar["params"].setBlob(mParams);
        cameraVar["hstrRadiance"] = mpLeafRadiance;
        cameraVar["hstrCameraLightingOutput"] = mpCameraLighting;
        mpCameraLightingPass->execute(pRenderContext, uint3(mParams.hstrLeafDims));
        mCameraLightingDirty = false;
        mCameraLightingPoseValid = true;
        mCameraLightingPosition = mpScene->getCamera()->getPosition();
        mCameraLightingDirection = mpScene->getCamera()->getTarget() - mCameraLightingPosition;
    }

    const uint32_t nodeCount = mParams.hstrNodeCount;
    const uint32_t tileCount = mParams.hstrTileDims.x * mParams.hstrTileDims.y;
    const uint2 queryDims = mParams.hstrQueryDims;
    const auto cutFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    if (!mpCameraQueries || mpCameraQueries->getWidth() != queryDims.x || mpCameraQueries->getHeight() != queryDims.y)
        mpCameraQueries = mpDevice->createTexture2D(queryDims.x, queryDims.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, cutFlags);
    if (!mpNodeProjection || mpNodeProjection->getElementCount() != nodeCount)
    {
        mpNodeProjection = mpDevice->createStructuredBuffer(sizeof(float4), nodeCount, cutFlags, MemoryType::DeviceLocal, nullptr, false);
        mpNodeDepth = mpDevice->createStructuredBuffer(sizeof(float2), nodeCount, cutFlags, MemoryType::DeviceLocal, nullptr, false);
        mCutDirty = true;
    }
    if (!mpTileNodeCounts || mpTileNodeCounts->getElementCount() != tileCount)
    {
        mpTileNodeCounts = mpDevice->createStructuredBuffer(sizeof(uint32_t), tileCount, cutFlags, MemoryType::DeviceLocal, nullptr, false);
        mpTileNodes = mpDevice->createStructuredBuffer(
            sizeof(uint32_t), size_t(tileCount) * mParams.hstrMaxTileNodes, cutFlags, MemoryType::DeviceLocal, nullptr, false
        );
        mCutDirty = true;
    }
    if (mCutDirty)
    {
        FALCOR_PROFILE(pRenderContext, "cut");
        mpScene->bindShaderDataForRaytracing(pRenderContext, mpProjectPass->getRootVar()["gScene"]);
        ShaderVar projectVar = mpProjectPass->getRootVar()["CB"]["gHSTRCloud"];
        projectVar["params"].setBlob(mParams);
        projectVar["hstrCutNodes"] = mpCutNodes;
        projectVar["hstrCutNodeParents"] = mpCutNodeParents;
        projectVar["hstrNodeProjection"] = mpNodeProjection;
        projectVar["hstrNodeDepth"] = mpNodeDepth;
        mpProjectPass->execute(pRenderContext, uint3(nodeCount, 1, 1));

        pRenderContext->clearUAV(mpTileNodeCounts->getUAV().get(), uint4(0));
        mpScene->bindShaderDataForRaytracing(pRenderContext, mpCutPass->getRootVar()["gScene"]);
        ShaderVar cutVar = mpCutPass->getRootVar()["CB"]["gHSTRCloud"];
        cutVar["params"].setBlob(mParams);
        cutVar["hstrCutNodes"] = mpCutNodes;
        cutVar["hstrCutNodeParents"] = mpCutNodeParents;
        cutVar["hstrNodeProjection"] = mpNodeProjection;
        cutVar["hstrNodeDepth"] = mpNodeDepth;
        cutVar["hstrTileNodeCounts"] = mpTileNodeCounts;
        cutVar["hstrTileNodes"] = mpTileNodes;
        mpCutPass->execute(pRenderContext, uint3(nodeCount, 1, 1));

        mpScene->bindShaderDataForRaytracing(pRenderContext, mpSortPass->getRootVar()["gScene"]);
        ShaderVar sortVar = mpSortPass->getRootVar()["CB"]["gHSTRCloud"];
        sortVar["params"].setBlob(mParams);
        sortVar["hstrCutNodes"] = mpCutNodes;
        sortVar["hstrCutNodeParents"] = mpCutNodeParents;
        sortVar["hstrNodeProjection"] = mpNodeProjection;
        sortVar["hstrNodeDepth"] = mpNodeDepth;
        sortVar["hstrTileNodeCounts"] = mpTileNodeCounts;
        sortVar["hstrTileNodes"] = mpTileNodes;
        mpSortPass->execute(pRenderContext, uint3(mParams.hstrTileDims, 1));
        mCutDirty = false;
    }

    FALCOR_PROFILE(pRenderContext, "queries+resolve");
    mpScene->bindShaderDataForRaytracing(pRenderContext, mpQueryPass->getRootVar()["gScene"]);
    ShaderVar queryVar = mpQueryPass->getRootVar()["CB"]["gHSTRCloud"];
    queryVar["params"].setBlob(mParams);
    queryVar["hstrRadiance"] = mpLeafRadiance;
    queryVar["hstrLeafResidualBounds"] = mpLeafResidualBounds;
    queryVar["hstrCameraLighting"] = mpCameraLighting;
    queryVar["hstrExtinctionSampler"] = mpExtinctionSampler;
    queryVar["hstrCameraQueryOutput"] = mpCameraQueries;
    mpQueryPass->execute(pRenderContext, uint3(queryDims, 1));

    mpScene->bindShaderDataForRaytracing(pRenderContext, mpPass->getRootVar()["gScene"]);
    ShaderVar var = mpPass->getRootVar()["CB"]["gHSTRCloud"];
    var["params"].setBlob(mParams);
    var["hstrRadiance"] = mpLeafRadiance;
    var["hstrLeafResidualBounds"] = mpLeafResidualBounds;
    var["hstrCutNodes"] = mpCutNodes;
    var["hstrCutNodeParents"] = mpCutNodeParents;
    var["hstrNodeProjection"] = mpNodeProjection;
    var["hstrNodeDepth"] = mpNodeDepth;
    var["hstrTileNodeCounts"] = mpTileNodeCounts;
    var["hstrTileNodes"] = mpTileNodes;
    var["hstrExtinction"] = mpExtinction;
    var["hstrCameraLighting"] = mpCameraLighting;
    var["hstrCameraQueries"] = mpCameraQueries;
    var["hstrExtinctionSampler"] = mpExtinctionSampler;
    var["color"] = color;
    var["transportError"] = error;
    var["cutStats"] = cutStats;
    mpPass->execute(pRenderContext, uint3(mParams.frameDim, 1));
    ++mParams.frameIndex;
}

void HSTRCloud::renderUI(Gui::Widgets& widget)
{
    bool renderChanged = false;
    bool lightingChanged = false;
    bool operatorChanged = false;
    renderChanged |= widget.var("Inside-cloud base steps", mParams.baseSteps, 8u, 128u, 8u);
    renderChanged |= widget.var("Inside-cloud refinement", mParams.refinementLevel, 0u, 3u, 1u);
    renderChanged |= widget.var("Inside-cloud residual blend", mParams.residualBlend, 0.f, 1.f, 0.01f);
    renderChanged |= widget.var("Cut transmittance error", mParams.cutTransmittanceTolerance, 0.0001f, 0.05f, 0.0001f);
    lightingChanged |= widget.var("Active transport rank", mParams.activeRank, 1u, mParams.hstrStorageRank, 1u);
    lightingChanged |= widget.var("Active-mode threshold", mParams.activeThreshold, 0.f, 1.f, 0.0001f);
    lightingChanged |= widget.var("Correction atom budget", mParams.correctionBudget, 0u, 1048576u, 4096u);
    renderChanged |= widget.var("Correction fade frames", mParams.correctionFadeFrames, 1u, 64u, 1u);
    operatorChanged |= widget.var("Ballistic entries per leaf", mParams.ballisticBudget, 0u, mParams.hstrTraceDofs, 1u);
    operatorChanged |= widget.var("Near-scatter entries per leaf", mParams.nearScatterBudget, 0u, 5u * mParams.hstrTraceDofs, 4u);
    operatorChanged |= widget.var("Diffuse rank budget", mParams.diffuseRankBudget, 1u, mParams.hstrTraceDofs, 1u);
    operatorChanged |= widget.var("Mixed-fidelity variation", mParams.mixedFidelityThreshold, 0.f, 1.f, 0.01f);
    operatorChanged |= widget.var("Operator dictionary tolerance", mParams.operatorDictionaryTolerance, 1e-7f, 1e-2f, 1e-5f);
    lightingChanged |= widget.var("Moving Schur window radius", mParams.schurWindowRadius, 0u, 64u, 1u);
    lightingChanged |= widget.rgbColor("Local source radiance", mParams.localSourceRadiance);
    const uint32_t previousTraceOrder = mParams.traceSpatialOrder;
    operatorChanged |= widget.var("Face spatial order", mParams.traceSpatialOrder, 1u, 2u, 1u);
    if (mParams.traceSpatialOrder != previousTraceOrder)
        mParams.activeRank = 6 * mParams.traceSpatialOrder * mParams.traceSpatialOrder;
    operatorChanged |= widget.var("Density scale", mParams.densityScale, 0.f, 100.f, 0.01f);
    lightingChanged |= widget.direction("Sun direction", mParams.sunDirection);
    lightingChanged |= widget.rgbColor("Sun radiance", mParams.sunRadiance);
    lightingChanged |= widget.rgbColor("Sky radiance", mParams.skyRadiance);
    operatorChanged |= widget.var("Anisotropy", mParams.anisotropy, -0.99f, 0.99f, 0.01f);
    if (operatorChanged)
        buildHierarchy();
    else if (lightingChanged)
        solveLighting();
    mOptionsChanged |= renderChanged || lightingChanged || operatorChanged;
    mCutDirty |= renderChanged || operatorChanged;
    widget.textWrapped(
        "Outside-cloud views share one projected HST antichain per 8x8 tile. Smooth nodes are accepted, silhouette and high-variation "
        "nodes descend into the half-float transmittance residual, and NanoVDB is not marched per pixel."
    );
    widget.text(fmt::format("Operator dictionary: {} prototypes", mParams.operatorDictionarySize));
}
