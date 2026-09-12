/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "HSTRCloud.h"
#include "RenderGraph/RenderPassStandardFlags.h"

#include <algorithm>
#include <chrono>

namespace
{
const char kShaderFile[] = "RenderPasses/HSTRCloud/HSTRCloud.cs.slang";
const char kColor[] = "color";
const char kTransportError[] = "transportError";

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
const char kGoalFace[] = "goalFace";
const char kTraceSpatialOrder[] = "traceSpatialOrder";
const char kCorrectionBudget[] = "correctionBudget";
const char kCorrectionFadeFrames[] = "correctionFadeFrames";

uint32_t nextPowerOfTwo(uint32_t value)
{
    uint32_t result = 1;
    while (result < value)
        result <<= 1;
    return result;
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
        else if (key == kGoalFace)
            mParams.goalFace = value;
        else if (key == kTraceSpatialOrder)
            mParams.traceSpatialOrder = value;
        else if (key == kCorrectionBudget)
            mParams.correctionBudget = value;
        else if (key == kCorrectionFadeFrames)
            mParams.correctionFadeFrames = value;
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
    props[kGoalFace] = mParams.goalFace;
    props[kTraceSpatialOrder] = mParams.traceSpatialOrder;
    props[kCorrectionBudget] = mParams.correctionBudget;
    props[kCorrectionFadeFrames] = mParams.correctionFadeFrames;
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
    return reflector;
}

void HSTRCloud::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mParams.frameDim = compileData.defaultTexDims;
}

void HSTRCloud::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpPass = nullptr;
    mpSolvePass = nullptr;
    mFirstFrame = true;
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
    mLeafDensity = sampleLeafDensities();
    std::vector<hstr::DenseMatrix> leafTransport;
    leafTransport.reserve(leafCount);
    for (uint32_t leaf = 0; leaf < leafCount; ++leaf)
        leafTransport.push_back(makeNestedLeafTransport(leaf));

    mHierarchy = hstr::Hierarchy::compile(leafDims, leafTransport, mParams.traceSpatialOrder);
    const float milliseconds = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - startTime).count();
    logInfo("HSTRCloud: compiled {} leaves and {} Schur nodes in {:.1f} ms.", leafCount, mHierarchy.getNodes().size(), milliseconds);

    uploadHierarchy();
    solveLighting();
}

void HSTRCloud::uploadHierarchy()
{
    const uint3 leafDims = mHierarchy.getLeafDims();
    const size_t leafCount = size_t(leafDims.x) * leafDims.y * leafDims.z;
    const auto transfers = mHierarchy.getLeafTransferMatrices();
    const auto transports = mHierarchy.getLeafTransportMatrices();
    mHierarchyResidualBounds = mHierarchy.getLeafResidualBounds();
    const size_t traceDofs = mParams.hstrTraceDofs;
    const size_t matrixSize = traceDofs * traceDofs;
    std::vector<float> packedLeft(matrixSize * leafCount, 0.f);
    std::vector<float> packedRight(matrixSize * leafCount, 0.f);
    std::vector<float> packedTransport(matrixSize * leafCount);
    mLeafCorrections.assign(leafCount, hstr::DenseMatrix(traceDofs, traceDofs));
    const hstr::TraceTransfer spatialTransfer = hstr::makeConservativeTraceTransfer(6, mParams.hstrFaceDofs);
    for (size_t leaf = 0; leaf < leafCount; ++leaf)
    {
        hstr::DenseMatrix storedTransport = transports[leaf];
        if (mParams.traceSpatialOrder > 1)
        {
            const hstr::DenseMatrix coarse = hstr::multiply(
                spatialTransfer.restriction,
                hstr::multiply(transports[leaf], spatialTransfer.prolongation)
            );
            storedTransport = hstr::multiply(
                spatialTransfer.prolongation,
                hstr::multiply(coarse, spatialTransfer.restriction)
            );
            mLeafCorrections[leaf] = hstr::subtract(transports[leaf], storedTransport);
        }
        for (size_t i = 0; i < matrixSize; ++i)
            packedTransport[matrixSize * leaf + i] = storedTransport.data()[i];
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
    mpLeafBasisLeft = mpDevice->createStructuredBuffer(
        sizeof(float), packedLeft.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedLeft.data(), false
    );
    mpLeafBasisRight = mpDevice->createStructuredBuffer(
        sizeof(float), packedRight.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedRight.data(), false
    );
    mpLeafTransport = mpDevice->createStructuredBuffer(
        sizeof(float), packedTransport.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedTransport.data(), false
    );
    mpLeafRadiance = mpDevice->createStructuredBuffer(
        sizeof(float4),
        traceDofs * leafCount,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr,
        false
    );
}

void HSTRCloud::uploadCorrectionPool(const hstr::DenseMatrix& rootIncident)
{
    struct Candidate
    {
        uint32_t leaf;
        CorrectionAtom atom;
    };
    const auto transfers = mHierarchy.getLeafTransferMatrices();
    const float3 sun = normalize(mParams.sunDirection);
    uint32_t sunAxis = 0;
    if (std::abs(sun.y) > std::abs(sun.x))
        sunAxis = 1;
    if (std::abs(sun.z) > std::abs(sun[sunAxis]))
        sunAxis = 2;
    const uint32_t sunFace = 2 * sunAxis + (sun[sunAxis] >= 0.f ? 1u : 0u);
    mParams.adjointGoalCount = mParams.hstrFaceDofs + 2;
    hstr::DenseMatrix rootGoals(mParams.hstrTraceDofs, mParams.adjointGoalCount);
    for (uint32_t mode = 0; mode < mParams.hstrFaceDofs; ++mode)
    {
        rootGoals(mParams.goalFace * mParams.hstrFaceDofs + mode, mode) = 1.f;
        rootGoals(sunFace * mParams.hstrFaceDofs + mode, mParams.hstrFaceDofs) = 1.f / float(mParams.hstrFaceDofs);
    }
    for (uint32_t dof = 0; dof < mParams.hstrTraceDofs; ++dof)
        rootGoals(dof, mParams.hstrFaceDofs + 1) = 1.f / float(mParams.hstrTraceDofs);
    const auto goals = mHierarchy.getLeafAdjointGoalMatrices(rootGoals);
    const size_t leafCount = transfers.size();
    const auto transports = mHierarchy.getLeafTransportMatrices();
    std::vector<float> packedResponses(leafCount * mParams.hstrStorageRank, 0.f);
    for (size_t leaf = 0; leaf < leafCount; ++leaf)
    {
        const hstr::DenseMatrix storedTransport = hstr::subtract(transports[leaf], mLeafCorrections[leaf]);
        hstr::DenseMatrix basis = transfers[leaf];
        if (mParams.traceSpatialOrder == 1)
        {
            const hstr::DenseMatrix compressed = hstr::compress(transfers[leaf], 1e-6f, mParams.hstrTraceDofs).left;
            basis = hstr::DenseMatrix(mParams.hstrTraceDofs, mParams.hstrStorageRank);
            for (size_t row = 0; row < compressed.rows(); ++row)
                for (size_t mode = 0; mode < compressed.cols(); ++mode)
                    basis(row, mode) = compressed(row, mode);
        }
        const hstr::DenseMatrix responses = hstr::multiply(hstr::transpose(goals[leaf]), hstr::multiply(storedTransport, basis));
        for (uint32_t mode = 0; mode < mParams.hstrStorageRank; ++mode)
            for (uint32_t goal = 0; goal < mParams.adjointGoalCount; ++goal)
                packedResponses[leaf * mParams.hstrStorageRank + mode] =
                    std::max(packedResponses[leaf * mParams.hstrStorageRank + mode], std::abs(responses(goal, mode)));
    }
    mpLeafAdjointResponses = mpDevice->createStructuredBuffer(
        sizeof(float), packedResponses.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, packedResponses.data(), false
    );
    std::vector<float> omitted = mHierarchyResidualBounds;
    std::vector<Candidate> candidates;
    if (mParams.traceSpatialOrder > 1)
        candidates.reserve(leafCount * mParams.hstrTraceDofs * mParams.hstrTraceDofs / 2);
    for (uint32_t leaf = 0; leaf < leafCount; ++leaf)
    {
        const hstr::DenseMatrix incident = hstr::multiply(transfers[leaf], rootIncident);
        const hstr::DenseMatrix& detail = mLeafCorrections[leaf];
        for (uint32_t row = 0; row < mParams.hstrTraceDofs; ++row)
            for (uint32_t col = 0; col < mParams.hstrTraceDofs; ++col)
            {
                const float value = detail(row, col);
                float incidentMagnitude = 0.f;
                for (uint32_t channel = 0; channel < 3; ++channel)
                    incidentMagnitude += std::abs(incident(col, channel));
                float goalError = 0.f;
                for (uint32_t goal = 0; goal < mParams.adjointGoalCount; ++goal)
                    goalError += std::abs(goals[leaf](row, goal) * value) * incidentMagnitude;
                if (goalError <= 1e-12f)
                    continue;
                omitted[leaf] += goalError;
                candidates.push_back({leaf, {row, col, value, goalError}});
            }
    }

    const size_t candidateCount = candidates.size();
    const size_t admittedCount = std::min<size_t>(mParams.correctionBudget, candidateCount);
    if (admittedCount < candidates.size())
    {
        std::nth_element(
            candidates.begin(),
            candidates.begin() + admittedCount,
            candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.atom.goalError > b.atom.goalError; }
        );
        candidates.resize(admittedCount);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.leaf < b.leaf; });

    std::vector<uint2> ranges(leafCount, uint2(0));
    std::vector<CorrectionAtom> atoms;
    atoms.reserve(candidates.size());
    for (const Candidate& candidate : candidates)
    {
        uint2& range = ranges[candidate.leaf];
        if (range.y == 0)
            range.x = uint32_t(atoms.size());
        ++range.y;
        atoms.push_back(candidate.atom);
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
        "HSTRCloud: admitted {} of {} residual atoms against {} persistent adjoint goals.",
        atoms.size(),
        candidateCount,
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
        return mParams.traceSpatialOrder == 1
            ? clear
            : hstr::makeGridBoundaryTransport(kSubcellsPerAxis, std::vector<hstr::DenseMatrix>(8, clear));
    }

    const auto& volume = mpScene->getGridVolume(0);
    const auto& grid = volume->getDensityGrid();
    const int3 begin = mGridMin + int3(cell * mParams.hstrCellWidth);
    const int3 end = min(mGridMax + 1, begin + int(mParams.hstrCellWidth));
    const float3 subcellSize = mVoxelSize * float3(end - begin) / float(kSubcellsPerAxis);
    const float3 albedo = volume->getAlbedo();
    const float meanAlbedo = (albedo.x + albedo.y + albedo.z) / 3.f;
    std::vector<hstr::DenseMatrix> subcellTransport;
    subcellTransport.reserve(kSubcellsPerAxis * kSubcellsPerAxis * kSubcellsPerAxis);
    for (uint32_t z = 0; z < kSubcellsPerAxis; ++z)
        for (uint32_t y = 0; y < kSubcellsPerAxis; ++y)
            for (uint32_t x = 0; x < kSubcellsPerAxis; ++x)
            {
                const float3 u = (float3(x, y, z) + 0.5f) / float(kSubcellsPerAxis);
                const int3 p = min(end - 1, begin + int3(u * float3(end - begin)));
                const float density = std::max(0.f, grid->getValue(p)) * volume->getDensityScale() * mParams.densityScale;
                subcellTransport.push_back(hstr::makeLeafTransport(density * subcellSize, meanAlbedo, mParams.anisotropy, 0.65f));
            }
    if (mParams.traceSpatialOrder > 1)
        return hstr::makeGridBoundaryTransport(kSubcellsPerAxis, subcellTransport);
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

    // Broad changes deliberately rebuild. Sparse changes use exact local
    // Schur repair and touch only each leaf-to-root ancestor chain.
    if (changed.size() > updatedDensity.size() / 10)
    {
        buildHierarchy();
        return;
    }

    uint32_t repairedNodes = 0;
    for (uint32_t leaf : changed)
    {
        repairedNodes += mHierarchy.updateLeaf(leaf, makeNestedLeafTransport(leaf));
        mLeafDensity[leaf] = updatedDensity[leaf];
    }
    uploadHierarchy();
    solveLighting();
    mOptionsChanged = true;
    logInfo("HSTRCloud: repaired {} changed leaves across {} ancestor updates.", changed.size(), repairedNodes);
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
            incident(sunFace * mParams.hstrFaceDofs + mode, channel) +=
                0.2f * std::abs(sun[sunAxis]) * mParams.sunRadiance[channel];

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
    var["hstrLeafTransport"] = mpLeafTransport;
    var["hstrLeafResidualBounds"] = mpLeafResidualBounds;
    var["hstrRootIncident"] = mpRootIncident;
    var["hstrLeafAdjointResponses"] = mpLeafAdjointResponses;
    var["hstrCorrectionRanges"] = mpCorrectionRanges;
    var["hstrCorrectionAtoms"] = mpCorrectionAtoms;
    const uint3 leafDims = mHierarchy.getLeafDims();
    mpSolvePass->execute(mpDevice->getRenderContext(), uint3(leafDims.x * leafDims.y * leafDims.z, 1, 1));
}

void HSTRCloud::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
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
    mParams.frameDim = uint2(color->getWidth(), color->getHeight());
    if (!mpScene || !mpPass || !mpLeafRadiance)
    {
        pRenderContext->clearUAV(color->getUAV().get(), float4(0.f));
        pRenderContext->clearUAV(error->getUAV().get(), float4(0.f));
        return;
    }

    if (mParams.frameIndex < mParams.correctionFadeFrames)
        dispatchLightingSolve();

    mpScene->bindShaderDataForRaytracing(pRenderContext, mpPass->getRootVar()["gScene"]);
    ShaderVar var = mpPass->getRootVar()["CB"]["gHSTRCloud"];
    var["params"].setBlob(mParams);
    var["hstrRadiance"] = mpLeafRadiance;
    var["color"] = color;
    var["transportError"] = error;
    mpPass->execute(pRenderContext, uint3(mParams.frameDim, 1));
    ++mParams.frameIndex;
}

void HSTRCloud::renderUI(Gui::Widgets& widget)
{
    bool renderChanged = false;
    bool lightingChanged = false;
    bool operatorChanged = false;
    renderChanged |= widget.var("Base view steps", mParams.baseSteps, 8u, 128u, 8u);
    renderChanged |= widget.var("Refinement level", mParams.refinementLevel, 0u, 3u, 1u);
    renderChanged |= widget.var("Residual blend", mParams.residualBlend, 0.f, 1.f, 0.01f);
    lightingChanged |= widget.var("Active transport rank", mParams.activeRank, 1u, mParams.hstrStorageRank, 1u);
    lightingChanged |= widget.var("Active-mode threshold", mParams.activeThreshold, 0.f, 1.f, 0.0001f);
    lightingChanged |= widget.var("Adjoint goal face", mParams.goalFace, 0u, 5u, 1u);
    lightingChanged |= widget.var("Correction atom budget", mParams.correctionBudget, 0u, 1048576u, 4096u);
    renderChanged |= widget.var("Correction fade frames", mParams.correctionFadeFrames, 1u, 64u, 1u);
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
    widget.textWrapped(
        "Lighting is reconstructed from a persistent six-face spatial-angular Schur hierarchy. NanoVDB is queried only for primary "
        "visibility and the deterministic near-field residual."
    );
}
