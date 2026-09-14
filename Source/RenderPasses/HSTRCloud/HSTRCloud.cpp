/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "HSTRCloud.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Image/Bitmap.h"

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
const char kCutHysteresis[] = "cutHysteresis";
const char kBasisTolerance[] = "basisTolerance";
const char kBasisHysteresis[] = "basisHysteresis";
const char kResidualTolerance[] = "residualTolerance";
const char kResidualStrength[] = "residualStrength";
const char kStepOpticalDepth[] = "stepOpticalDepth";
const char kMinStepVoxels[] = "minStepVoxels";
const char kMaxStepVoxels[] = "maxStepVoxels";
const char kDebugView[] = "debugView";
const char kOctaveEnergy[] = "octaveEnergy";
const char kOctaveExtinction[] = "octaveExtinction";
const char kOctavePhase[] = "octavePhase";
const char kHstSunFraction[] = "hstSunFraction";
const char kOctaveBlurSigma[] = "octaveBlurSigma";
const char kCompareReference[] = "compareReference";
const char kHstComponents[] = "hstComponents";
const char kSkyScale[] = "skyScale";
const char kSkyAmbient[] = "skyAmbient";
const char kWorldCacheCellVoxels[] = "worldCacheCellVoxels";
const char kWorldCacheOrder[] = "worldCacheOrder";
const char kWorldCacheUpdates[] = "worldCacheUpdates";
const char kWorldCacheSampleCount[] = "worldCacheSampleCount";
const char kWorldCacheEstimator[] = "worldCacheEstimator";
const char kWorldCachePhotons[] = "worldCachePhotons";
const char kWorldCacheBands[] = "worldCacheBands";
const char kWorldCacheWindow[] = "worldCacheWindow";
const char kWorldCacheTextured[] = "worldCacheTextured";
const char kLightingStride[] = "lightingStride";
const char kWorldCacheBakeInterval[] = "worldCacheBakeInterval";
const char kWorldCacheSegments[] = "worldCacheSegments";
const char kBeamTileSize[] = "beamTileSize";
const char kBeamLevels[] = "beamLevels";
const char kBeamSegments[] = "beamSegments";
const char kBeamTemporal[] = "beamTemporal";
const char kCompareBlock[] = "compareBlock";
const char kWorldCacheModulation[] = "worldCacheModulation";
const char kWorldCacheModulationDepth[] = "worldCacheModulationDepth";
const char kSaveReference[] = "saveReference";
const char kLoadReference[] = "loadReference";
const char kBeamTolerance[] = "beamTolerance";
const char kBeamEdgeDepth[] = "beamEdgeDepth";
const char kStoreExact[] = "storeExact";
const char kCompareExact[] = "compareExact";
const char kBeamMarchedFraction[] = "beamMarchedFraction";
const char kReferenceShow[] = "referenceShow";
const char kCompareSubstitute[] = "compareSubstitute";
const char kCompareTarget[] = "compareTarget";
const char kReferenceError[] = "referenceError";
const char kReferenceLogError[] = "referenceLogError";
const char kReferenceNoiseError[] = "referenceNoiseError";
const char kReferenceNoiseLogError[] = "referenceNoiseLogError";
const char kReferenceSampleCount[] = "referenceSampleCount";
constexpr uint32_t kReferenceView = 6;

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
    parseProperties(props);
}

void HSTRCloud::parseProperties(const Properties& props)
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
        else if (key == kCutHysteresis)
            mParams.cutHysteresis = value;
        else if (key == kBasisTolerance)
            mParams.basisTolerance = value;
        else if (key == kBasisHysteresis)
            mParams.basisHysteresis = value;
        else if (key == kResidualTolerance)
            mParams.residualTolerance = value;
        else if (key == kResidualStrength)
            mParams.residualStrength = value;
        else if (key == kStepOpticalDepth)
            mParams.stepOpticalDepth = value;
        else if (key == kMinStepVoxels)
            mParams.minStepVoxels = value;
        else if (key == kMaxStepVoxels)
            mParams.maxStepVoxels = value;
        else if (key == kDebugView)
            mParams.debugView = value;
        else if (key == kOctaveEnergy)
            mParams.octaveEnergy = value;
        else if (key == kOctaveExtinction)
            mParams.octaveExtinction = value;
        else if (key == kOctavePhase)
            mParams.octavePhase = value;
        else if (key == kHstSunFraction)
            mParams.hstSunFraction = value;
        else if (key == kOctaveBlurSigma)
            mParams.octaveBlurSigma = value;
        else if (key == kCompareReference)
            mCompareReference = value;
        else if (key == kHstComponents)
            mParams.hstComponents = value;
        else if (key == kSkyScale)
            mParams.skyScale = value;
        else if (key == kSkyAmbient)
            mParams.skyAmbient = value;
        else if (key == kWorldCacheCellVoxels)
            mParams.worldCacheCellVoxels = std::max(1u, uint32_t(value));
        else if (key == kWorldCacheOrder)
            mParams.worldCacheOrder = std::min(2u, uint32_t(value));
        else if (key == kWorldCacheUpdates)
            mWorldCacheUpdates = value;
        else if (key == kWorldCacheEstimator)
            mParams.worldCacheEstimator = value;
        else if (key == kWorldCachePhotons)
            mParams.worldCachePhotons = std::max(64u, uint32_t(value));
        else if (key == kWorldCacheBands)
            mParams.worldCacheBands = std::clamp(uint32_t(value), 1u, 2u);
        else if (key == kWorldCacheWindow)
            mParams.worldCacheWindow = value;
        else if (key == kWorldCacheTextured)
            mParams.worldCacheTextured = value;
        else if (key == kLightingStride)
            mParams.lightingStride = std::max(1u, uint32_t(value));
        else if (key == kWorldCacheBakeInterval)
            mWorldCacheBakeInterval = std::max(1u, uint32_t(value));
        else if (key == kWorldCacheSegments)
            mParams.worldCacheSegments = value;
        else if (key == kBeamTileSize)
            mParams.beamTileSize = nextPowerOfTwo(std::clamp(uint32_t(value), 2u, 64u));
        else if (key == kBeamLevels)
            mParams.beamLevels = std::clamp(uint32_t(value), 1u, kBeamMaxLevels);
        else if (key == kBeamSegments)
            mParams.beamSegments = nextPowerOfTwo(std::clamp(uint32_t(value), 1u, 64u));
        else if (key == kBeamTemporal)
            mParams.beamTemporal = bool(value) ? 1u : 0u;
        else if (key == kCompareBlock)
            mParams.compareBlock = std::max(1u, uint32_t(value));
        else if (key == kWorldCacheModulation)
            mWorldCacheModulation = value;
        else if (key == kWorldCacheModulationDepth)
            mParams.worldCacheModulationDepth = std::max(0.f, float(value));
        else if (key == kSaveReference)
            mSaveReferencePath = value.operator std::string();
        else if (key == kLoadReference)
            mLoadReferencePath = value.operator std::string();
        else if (key == kBeamTolerance)
            mParams.beamTolerance = value;
        else if (key == kBeamEdgeDepth)
            mParams.beamEdgeDepth = value;
        else if (key == kStoreExact)
            mStoreExact = value;
        else if (key == kCompareExact)
            mParams.compareExact = bool(value) ? 1u : 0u;
        else if (key == kReferenceShow)
            mParams.referenceShow = value;
        else if (key == kCompareSubstitute)
            mParams.compareSubstitute = value;
        else if (key == kCompareTarget)
            mParams.compareTarget = value;
        else if (key == kReferenceError || key == kReferenceLogError || key == kReferenceNoiseError || key == kReferenceNoiseLogError || key == kReferenceSampleCount || key == kWorldCacheSampleCount || key == kBeamMarchedFraction)
            continue; // Read-only measurements.
        else
            logWarning("Unknown property '{}' in HSTRCloud.", key);
    }
}

void HSTRCloud::setProperties(const Properties& props)
{
    const HSTRCloudParams previous = mParams;
    const float previousModulation = mWorldCacheModulation;
    parseProperties(props);
    if (!mpScene || !mpLeafRadiance)
        return;
    const auto& p = mParams;
    const auto& q = previous;
    const bool operatorChanged = p.densityScale != q.densityScale || p.anisotropy != q.anisotropy ||
                                 p.traceSpatialOrder != q.traceSpatialOrder || p.ballisticBudget != q.ballisticBudget ||
                                 p.nearScatterBudget != q.nearScatterBudget || p.diffuseRankBudget != q.diffuseRankBudget ||
                                 p.mixedFidelityThreshold != q.mixedFidelityThreshold ||
                                 p.operatorDictionaryTolerance != q.operatorDictionaryTolerance;
    const bool lightingChanged =
        any(p.sunDirection != q.sunDirection) || any(p.sunRadiance != q.sunRadiance) || any(p.skyRadiance != q.skyRadiance) ||
        any(p.localSourceRadiance != q.localSourceRadiance) || p.activeRank != q.activeRank || p.activeThreshold != q.activeThreshold ||
        p.correctionBudget != q.correctionBudget || p.schurWindowRadius != q.schurWindowRadius || p.hstSunFraction != q.hstSunFraction;
    if (operatorChanged)
        buildHierarchy();
    else if (lightingChanged)
        solveLighting();
    // The reference and the world cache solve the same lights and medium, so they restart with them.
    if (operatorChanged || lightingChanged)
    {
        mParams.referenceSamples = 0;
        mParams.worldCacheSamples = 0;
    }
    if (p.worldCacheCellVoxels != q.worldCacheCellVoxels || p.worldCacheEstimator != q.worldCacheEstimator ||
        p.worldCachePhotons != q.worldCachePhotons || p.worldCacheBands != q.worldCacheBands ||
        p.worldCacheTextured != q.worldCacheTextured || p.worldCacheSegments != q.worldCacheSegments ||
        mWorldCacheModulation != previousModulation || p.worldCacheModulationDepth != q.worldCacheModulationDepth)
        mParams.worldCacheSamples = 0;
    mResidualDirty |= p.residualTolerance != q.residualTolerance || p.octaveExtinction != q.octaveExtinction ||
                      p.octaveBlurSigma != q.octaveBlurSigma || p.stepOpticalDepth != q.stepOpticalDepth ||
                      p.maxStepVoxels != q.maxStepVoxels;
    mCutDirty = true;
    mBasisDirty = true;
    mBeamReusable = false;
    mOptionsChanged = true;
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
    props[kCutHysteresis] = mParams.cutHysteresis;
    props[kBasisTolerance] = mParams.basisTolerance;
    props[kBasisHysteresis] = mParams.basisHysteresis;
    props[kResidualTolerance] = mParams.residualTolerance;
    props[kResidualStrength] = mParams.residualStrength;
    props[kStepOpticalDepth] = mParams.stepOpticalDepth;
    props[kMinStepVoxels] = mParams.minStepVoxels;
    props[kMaxStepVoxels] = mParams.maxStepVoxels;
    props[kDebugView] = mParams.debugView;
    props[kOctaveEnergy] = mParams.octaveEnergy;
    props[kOctaveExtinction] = mParams.octaveExtinction;
    props[kOctavePhase] = mParams.octavePhase;
    props[kHstSunFraction] = mParams.hstSunFraction;
    props[kOctaveBlurSigma] = mParams.octaveBlurSigma;
    props[kCompareReference] = mCompareReference;
    props[kHstComponents] = mParams.hstComponents;
    props[kSkyScale] = mParams.skyScale;
    props[kSkyAmbient] = mParams.skyAmbient;
    props[kWorldCacheCellVoxels] = mParams.worldCacheCellVoxels;
    props[kWorldCacheOrder] = mParams.worldCacheOrder;
    props[kWorldCacheUpdates] = mWorldCacheUpdates;
    props[kWorldCacheSampleCount] = mParams.worldCacheSamples;
    props[kWorldCacheEstimator] = mParams.worldCacheEstimator;
    props[kWorldCachePhotons] = mParams.worldCachePhotons;
    props[kWorldCacheBands] = mParams.worldCacheBands;
    props[kWorldCacheWindow] = mParams.worldCacheWindow;
    props[kWorldCacheTextured] = mParams.worldCacheTextured;
    props[kLightingStride] = mParams.lightingStride;
    props[kWorldCacheBakeInterval] = mWorldCacheBakeInterval;
    props[kWorldCacheSegments] = mParams.worldCacheSegments;
    props[kBeamTileSize] = mParams.beamTileSize;
    props[kBeamLevels] = mParams.beamLevels;
    props[kBeamSegments] = mParams.beamSegments;
    props[kBeamTemporal] = mParams.beamTemporal != 0;
    props[kCompareBlock] = mParams.compareBlock;
    props[kWorldCacheModulation] = mWorldCacheModulation;
    props[kWorldCacheModulationDepth] = mParams.worldCacheModulationDepth;
    props[kBeamTolerance] = mParams.beamTolerance;
    props[kBeamEdgeDepth] = mParams.beamEdgeDepth;
    props[kStoreExact] = mStoreExact;
    props[kCompareExact] = mParams.compareExact != 0;
    props[kBeamMarchedFraction] = mBeamMarchedFraction;
    props[kReferenceShow] = mParams.referenceShow;
    props[kCompareSubstitute] = mParams.compareSubstitute;
    props[kCompareTarget] = mParams.compareTarget;
    props[kReferenceError] = mReferenceError;
    props[kReferenceLogError] = mReferenceLogError;
    props[kReferenceNoiseError] = mReferenceNoiseError;
    props[kReferenceNoiseLogError] = mReferenceNoiseLogError;
    props[kReferenceSampleCount] = mParams.referenceSamples;
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
    mpTileBasisPass = nullptr;
    mpFineSunPass = nullptr;
    mpResidualMaskPass = nullptr;
    mpGatherOctavesPass = nullptr;
    mpCompareReferencePass = nullptr;
    mpWorldCachePass = nullptr;
    mpWorldCachePhotonPass = nullptr;
    mpWorldCacheResolvePass = nullptr;
    mpWorldCacheBakePass = nullptr;
    mpWorldCacheAdvancePass = nullptr;
    mpBlurOctavesPass = nullptr;
    mpReferencePass = nullptr;
    mpBeamQueryPass = nullptr;
    mpBeamTilePass = nullptr;
    mpBeamArgsPass = nullptr;
    mpBeamResolvePass = nullptr;
    mpBeamMarchPass = nullptr;
    mFirstFrame = true;
    mCameraLightingDirty = true;
    mCutDirty = true;
    mResidualDirty = true;
    mBasisDirty = true;
    mCameraLightingPoseValid = false;
    if (!mpScene)
        return;

    if (!mpScene->getGridVolumes().empty())
    {
        const auto bounds = mpScene->getGridVolume(0)->getBounds();
        logInfo("HSTRCloud: first grid-volume bounds are {} to {}.", bounds.minPoint, bounds.maxPoint);
    }

    auto createPass = [&](const char* entry)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).csEntry(entry);
        desc.addTypeConformances(mpScene->getTypeConformances());
        return ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    };
    mpPass = createPass("main");
    mpSolvePass = createPass("solveLeaves");
    mpCameraLightingPass = createPass("updateCameraLighting");
    mpProjectPass = createPass("projectCutNodes");
    mpCutPass = createPass("binCutNodes");
    mpSortPass = createPass("sortTileCuts");
    mpQueryPass = createPass("buildCameraQueries");
    mpTileBasisPass = createPass("buildTileBases");
    mpFineSunPass = createPass("computeFineSun");
    mpResidualMaskPass = createPass("maskResidual");
    mpGatherOctavesPass = createPass("gatherSunOctaves");
    mpCompareReferencePass = createPass("compareReference");
    mpWorldCachePass = createPass("updateWorldCache");
    mpWorldCachePhotonPass = createPass("traceWorldCachePhotons");
    mpWorldCacheResolvePass = createPass("resolveWorldCache");
    mpWorldCacheBakePass = createPass("bakeWorldCache");
    mpWorldCacheAdvancePass = createPass("advanceWorldCachePhotons");
    mpBlurOctavesPass = createPass("blurSunOctaves");
    mpReferencePass = createPass("referencePathTrace");
    mpBeamQueryPass = createPass("buildBeamQueries");
    mpBeamTilePass = createPass("testBeamTiles");
    mpBeamArgsPass = createPass("writeBeamArgs");
    mpBeamTemporalTilePass = createPass("testBeamTilesTemporal");
    mpBeamResolvePass = createPass("resolveBeam");
    mpBeamMarchPass = createPass("marchBeamPixels");
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

    // Extinction majorants over 4^3-voxel blocks, dilated by one block: a march step of at most one block from any point
    // inside a block never meets extinction above that block's value, which bounds the optical depth of every step.
    constexpr uint32_t kMajorantBlock = 4;
    const uint3 majorantDims = (dims + kMajorantBlock - 1u) / kMajorantBlock;
    const size_t majorantCount = size_t(majorantDims.x) * majorantDims.y * majorantDims.z;
    auto majorantIndex = [&](uint3 b) { return size_t(b.x) + size_t(majorantDims.x) * (size_t(b.y) + size_t(majorantDims.y) * b.z); };
    std::vector<float> blockMaximum(majorantCount, 0.f);
    parallelFor(
        majorantCount,
        [&](size_t i)
        {
            const uint3 b(
                uint32_t(i % majorantDims.x),
                uint32_t((i / majorantDims.x) % majorantDims.y),
                uint32_t(i / (majorantDims.x * majorantDims.y))
            );
            // Trilinear lookups inside the block also read the voxels one past its upper faces.
            const uint3 begin = b * kMajorantBlock;
            const uint3 end = min(begin + kMajorantBlock, dims - 1u);
            float value = 0.f;
            for (uint32_t z = begin.z; z <= end.z; ++z)
                for (uint32_t y = begin.y; y <= end.y; ++y)
                    for (uint32_t x = begin.x; x <= end.x; ++x)
                        value = std::max(value, extinctionAtVoxel(uint3(x, y, z)));
            blockMaximum[i] = value;
        }
    );
    std::vector<float16_t> majorant(majorantCount);
    parallelFor(
        majorantCount,
        [&](size_t i)
        {
            const int3 b(
                int32_t(i % majorantDims.x), int32_t((i / majorantDims.x) % majorantDims.y), int32_t(i / (majorantDims.x * majorantDims.y))
            );
            float value = 0.f;
            for (int32_t z = -1; z <= 1; ++z)
                for (int32_t y = -1; y <= 1; ++y)
                    for (int32_t x = -1; x <= 1; ++x)
                    {
                        const int3 n = b + int3(x, y, z);
                        if (all(n >= 0) && all(n < int3(majorantDims)))
                            value = std::max(value, blockMaximum[majorantIndex(uint3(n))]);
                    }
            // Round up so the half-float majorant stays an upper bound.
            float16_t rounded(value);
            if (float(rounded) < value)
                rounded = float16_t(value * (1.f + 1.f / 1024.f));
            majorant[i] = rounded;
        }
    );
    mParams.hstrMajorantDims = majorantDims;
    mpMajorant = mpDevice->createTexture3D(majorantDims.x, majorantDims.y, majorantDims.z, ResourceFormat::R16Float, 1, majorant.data());

    // Sun residual pages live at voxel resolution.
    mParams.hstrFineDims = dims;
    mResidualDirty = true;
    mBasisDirty = true;

    if (!mpExtinctionSampler)
    {
        // Density is zero outside the active-voxel bounds, so extinction lookups use a zero border instead of clamping.
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        samplerDesc.setAddressingMode(TextureAddressingMode::Border, TextureAddressingMode::Border, TextureAddressingMode::Border);
        samplerDesc.setBorderColor(float4(0.f));
        mpExtinctionSampler = mpDevice->createSampler(samplerDesc);
    }
    if (!mpLinearSampler)
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        samplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpLinearSampler = mpDevice->createSampler(samplerDesc);
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
    // Two lattice points per leaf axis resolve the 2x2 spatial modes of each face trace.
    const uint3 cameraLightingDims = leafDims * 2u;
    mParams.hstrCameraLightingDims = cameraLightingDims;
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

    // Sky radiance entering each face: the cosine-weighted mean of the sky gradient over the hemisphere the face looks at,
    // the same sky the background and the path-traced reference use (dimmer below the horizon).
    hstr::DenseMatrix incident(mParams.hstrTraceDofs, 3);
    for (uint32_t face = 0; face < 6; ++face)
    {
        float3 normal(0.f);
        normal[face / 2] = (face & 1u) ? 1.f : -1.f;
        const float3 tangent = std::abs(normal.y) < 0.5f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        const float3 u = normalize(cross(tangent, normal));
        const float3 v = cross(normal, u);
        constexpr uint32_t kSamples = 32;
        float weightSum = 0.f;
        float skyScale = 0.f;
        for (uint32_t i = 0; i < kSamples; ++i)
            for (uint32_t j = 0; j < kSamples; ++j)
            {
                const float cosTheta = (float(i) + 0.5f) / float(kSamples);
                const float phi = 2.f * 3.14159265f * (float(j) + 0.5f) / float(kSamples);
                const float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);
                const float3 direction = cosTheta * normal + sinTheta * (std::cos(phi) * u + std::sin(phi) * v);
                skyScale += cosTheta * (0.35f + 0.65f * std::clamp(0.5f + 0.5f * direction.y, 0.f, 1.f));
                weightSum += cosTheta;
            }
        skyScale /= weightSum;
        for (uint32_t mode = 0; mode < mParams.hstrFaceDofs; ++mode)
            for (uint32_t channel = 0; channel < 3; ++channel)
                incident(face * mParams.hstrFaceDofs + mode, channel) = skyScale * mParams.skyRadiance[channel];
    }
    const float3 sun = normalize(mParams.sunDirection);
    // One voxel step towards the sun for the sub-voxel sun reconstruction (the grid axes are world aligned).
    const float3 sunVoxels = sun / mVoxelSize;
    mParams.sunVoxelDirection = sunVoxels / length(sunVoxels);
    mParams.sunVoxelWorldLength = 1.f / length(sunVoxels);
    uint32_t sunAxis = 0;
    if (std::abs(sun.y) > std::abs(sun.x))
        sunAxis = 1;
    if (std::abs(sun.z) > std::abs(sun[sunAxis]))
        sunAxis = 2;
    const uint32_t sunFace = 2 * sunAxis + (sun[sunAxis] >= 0.f ? 1u : 0u);
    for (uint32_t mode = 0; mode < mParams.hstrFaceDofs; ++mode)
        for (uint32_t channel = 0; channel < 3; ++channel)
            incident(sunFace * mParams.hstrFaceDofs + mode, channel) +=
                mParams.hstSunFraction * 0.2f * std::abs(sun[sunAxis]) * mParams.sunRadiance[channel];

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
    mResidualDirty = true;
}

/// Binds the scene and every read-only renderer resource. Passes attach their own outputs with bindOutput, which also unbinds
/// the read view of the same resource so no texture is ever bound as SRV and UAV in one dispatch.
void HSTRCloud::bindRenderer(RenderContext* pRenderContext, const ref<ComputePass>& pPass)
{
    mpScene->bindShaderDataForRaytracing(pRenderContext, pPass->getRootVar()["gScene"]);
    ShaderVar var = pPass->getRootVar()["CB"]["gHSTRCloud"];
    var["params"].setBlob(mParams);
    var["hstrRadiance"] = mpLeafRadiance;
    var["hstrLeafResidualBounds"] = mpLeafResidualBounds;
    var["hstrCutNodes"] = mpCutNodes;
    var["hstrCutNodeParents"] = mpCutNodeParents;
    var["hstrNodeProjection"] = mpNodeProjection;
    var["hstrNodeDepth"] = mpNodeDepth;
    var["hstrNodeOrder"] = mpNodeOrder;
    var["hstrCutStatePrevious"] = mpCutState[mParams.cutParity ^ 1u];
    var["hstrCutState"] = mpCutState[mParams.cutParity];
    var["hstrTileNodeCounts"] = mpTileNodeCounts;
    var["hstrTileNodes"] = mpTileNodes;
    var["hstrExtinction"] = mpExtinction;
    var["hstrMajorant"] = mpMajorant;
    var["hstrCameraLighting"] = mpCameraLighting;
    var["hstrFineSun"] = mpFineSun;
    var["hstrLeafResidual"] = mpLeafResidual;
    var["hstrSunOctaves"] = mpSunOctaveField;
    var["hstrCameraQueries"] = mpCameraQueries;
    var["hstrTileCenters"] = mpTileCenters;
    var["hstrTileBasis"] = mpTileBasis;
    var["hstrTileState"] = mpTileState;
    if (mpWorldCache)
        var["hstrWorldCache"] = mpWorldCache;
    if (mpWorldCacheDeposit)
        var["hstrWorldCacheDeposit"] = mpWorldCacheDeposit;
    if (mpPhotonPool)
        var["hstrPhotonPool"] = mpPhotonPool;
    if (mpPhotonEmitted)
        var["hstrPhotonEmitted"] = mpPhotonEmitted;
    for (uint32_t i = 0; i < kWorldCacheTextures; ++i)
        if (mpWorldCacheTextures[i])
            var["hstrWorldCacheTexture"][i] = mpWorldCacheTextures[i];
    // The indirect argument buffer is bound only by the pass that writes it: a dispatch cannot read it as arguments and
    // hold it as a UAV.
    var["hstrBeamLattice"] = mpBeamLattice;
    var["hstrBeamLevel"] = mpBeamLevel;
    var["hstrBeamLists"] = mpBeamLists[mBeamParity];
    var["hstrBeamCounts"] = mpBeamCounts[mBeamParity];
    var["hstrBeamPreviousLists"] = mpBeamLists[mBeamParity ^ 1u];
    var["hstrBeamPreviousCounts"] = mpBeamCounts[mBeamParity ^ 1u];
    var["hstrBeamHistory"] = mpBeamHistory[mBeamParity ^ 1u];
    var["hstrExactFrame"] = mpExactFrame;
    var["hstrExtinctionSampler"] = mpExtinctionSampler;
    var["hstrLinearSampler"] = mpLinearSampler;
}

namespace
{
void bindOutput(const ref<ComputePass>& pPass, const char* output, const ref<Texture>& pTexture, const char* input)
{
    ShaderVar var = pPass->getRootVar()["CB"]["gHSTRCloud"];
    var[input] = ref<Texture>();
    var[output] = pTexture;
}
} // namespace

void HSTRCloud::dispatchResidualPages(RenderContext* pRenderContext)
{
    const uint3 fineDims = mParams.hstrFineDims;
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    if (!mpFineSun || mpFineSun->getWidth() != fineDims.x || mpFineSun->getHeight() != fineDims.y || mpFineSun->getDepth() != fineDims.z)
        mpFineSun = mpDevice->createTexture3D(fineDims.x, fineDims.y, fineDims.z, ResourceFormat::R16Float, 1, nullptr, flags);
    const uint3 leafDims = mParams.hstrLeafDims;
    const size_t leafCount = size_t(leafDims.x) * leafDims.y * leafDims.z;
    if (!mpLeafResidual || mpLeafResidual->getElementCount() != leafCount)
        mpLeafResidual = mpDevice->createStructuredBuffer(sizeof(uint32_t), leafCount, flags, MemoryType::DeviceLocal, nullptr, false);

    // Ballistic sun transmittance per residual cell, with per-leaf activation, then non-resident pages are cleared.
    pRenderContext->clearUAV(mpLeafResidual->getUAV().get(), uint4(0));
    bindRenderer(pRenderContext, mpFineSunPass);
    bindOutput(mpFineSunPass, "hstrFineSunOutput", mpFineSun, "hstrFineSun");
    mpFineSunPass->execute(pRenderContext, fineDims);

    bindRenderer(pRenderContext, mpResidualMaskPass);
    bindOutput(mpResidualMaskPass, "hstrFineSunOutput", mpFineSun, "hstrFineSun");
    mpResidualMaskPass->execute(pRenderContext, fineDims);

    // Octaves 1-3: scatterer-weighted sun arrival on a 2-voxel grid, spread by a separable Gaussian.
    const uint3 octaveDims = (fineDims + 1u) / 2u;
    mParams.hstrOctaveDims = octaveDims;
    for (auto& texture : mpSunOctaves)
        if (!texture || texture->getWidth() != octaveDims.x || texture->getHeight() != octaveDims.y || texture->getDepth() != octaveDims.z)
            texture = mpDevice->createTexture3D(octaveDims.x, octaveDims.y, octaveDims.z, ResourceFormat::RGBA32Float, 1, nullptr, flags);
    bindRenderer(pRenderContext, mpGatherOctavesPass);
    bindOutput(mpGatherOctavesPass, "hstrSunOctavesOutput", mpSunOctaves[0], "hstrSunOctaves");
    mpGatherOctavesPass->execute(pRenderContext, octaveDims);
    uint32_t current = 0;
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        mParams.octaveBlurAxis = axis;
        bindRenderer(pRenderContext, mpBlurOctavesPass);
        ShaderVar var = mpBlurOctavesPass->getRootVar()["CB"]["gHSTRCloud"];
        var["hstrSunOctaves"] = mpSunOctaves[current];
        var["hstrSunOctavesOutput"] = mpSunOctaves[current ^ 1u];
        mpBlurOctavesPass->execute(pRenderContext, octaveDims);
        current ^= 1u;
    }
    mpSunOctaveField = mpSunOctaves[current];
}

void HSTRCloud::ensureCameraResources()
{
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    const uint32_t nodeCount = mParams.hstrNodeCount;
    const uint2 tileDims = mParams.hstrTileDims;
    const uint32_t tileCount = tileDims.x * tileDims.y;
    const uint2 latticeDims = mParams.hstrLatticeDims;
    if (!mpCameraQueries || mpCameraQueries->getWidth() != latticeDims.x || mpCameraQueries->getHeight() != latticeDims.y)
    {
        mpCameraQueries = mpDevice->createTexture2D(latticeDims.x, latticeDims.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
        mpTileCenters = mpDevice->createTexture2D(tileDims.x, tileDims.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
        mBasisDirty = true;
    }
    if (!mpNodeProjection || mpNodeProjection->getElementCount() != nodeCount)
    {
        mpNodeProjection = mpDevice->createStructuredBuffer(sizeof(float4), nodeCount, flags, MemoryType::DeviceLocal, nullptr, false);
        mpNodeDepth = mpDevice->createStructuredBuffer(sizeof(float2), nodeCount, flags, MemoryType::DeviceLocal, nullptr, false);
        mpNodeOrder = mpDevice->createStructuredBuffer(sizeof(uint32_t), nodeCount, flags, MemoryType::DeviceLocal, nullptr, false);
        for (auto& state : mpCutState)
        {
            state = mpDevice->createStructuredBuffer(sizeof(uint32_t), nodeCount, flags, MemoryType::DeviceLocal, nullptr, false);
            mpDevice->getRenderContext()->clearUAV(state->getUAV().get(), uint4(0));
        }
        mCutDirty = true;
    }
    if (!mpTileNodeCounts || mpTileNodeCounts->getElementCount() != tileCount)
    {
        mpTileNodeCounts = mpDevice->createStructuredBuffer(sizeof(uint32_t), tileCount, flags, MemoryType::DeviceLocal, nullptr, false);
        mpTileNodes = mpDevice->createStructuredBuffer(
            sizeof(uint32_t), size_t(tileCount) * mParams.hstrMaxTileNodes, flags, MemoryType::DeviceLocal, nullptr, false
        );
        mpTileBasis = mpDevice->createStructuredBuffer(sizeof(HSTRTileBasis), tileCount, flags, MemoryType::DeviceLocal, nullptr, false);
        mpTileState = mpDevice->createStructuredBuffer(sizeof(uint32_t), tileCount, flags, MemoryType::DeviceLocal, nullptr, false);
        mpDevice->getRenderContext()->clearUAV(mpTileState->getUAV().get(), uint4(0));
        mCutDirty = true;
    }
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
    mParams.hstrTileDims = (frameDim + mParams.hstrTileSize - 1u) / mParams.hstrTileSize;
    mParams.hstrLatticeDims = mParams.hstrTileDims + 1u;
    if (!mpScene || !mpPass || !mpLeafRadiance)
    {
        pRenderContext->clearUAV(color->getUAV().get(), float4(0.f));
        pRenderContext->clearUAV(error->getUAV().get(), float4(0.f));
        pRenderContext->clearUAV(cutStats->getUAV().get(), float4(0.f));
        return;
    }

    // Unbiased reference on the same medium and lights; accumulated over frames by the graph.
    if (!mpReferenceSum || mpReferenceSum->getWidth() != frameDim.x || mpReferenceSum->getHeight() != frameDim.y)
    {
        mpReferenceSum = mpDevice->createTexture2D(
            frameDim.x,
            frameDim.y,
            ResourceFormat::RGBA32Float,
            2 * kComponentCount,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
        mpReferenceRowError = mpDevice->createStructuredBuffer(
            sizeof(float4),
            frameDim.y,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mParams.referenceSamples = 0;
    }
    if (!mLoadReferencePath.empty())
    {
        loadReference(pRenderContext, mLoadReferencePath);
        mLoadReferencePath.clear();
    }
    if (mParams.debugView == kReferenceView)
    {
        // The reference average restarts whenever the camera or the medium changes.
        const float3 position = mpScene->getCamera()->getPosition();
        const float3 direction = mpScene->getCamera()->getTarget() - position;
        if (any(position != mReferencePosition) || any(direction != mReferenceDirection))
            mParams.referenceSamples = 0;
        mReferencePosition = position;
        mReferenceDirection = direction;
        FALCOR_PROFILE(pRenderContext, "reference");
        bindRenderer(pRenderContext, mpReferencePass);
        ShaderVar var = mpReferencePass->getRootVar()["CB"]["gHSTRCloud"];
        var["hstrReferenceSum"] = mpReferenceSum;
        var["color"] = color;
        var["transportError"] = error;
        var["cutStats"] = cutStats;
        mpReferencePass->execute(pRenderContext, uint3(mParams.frameDim, 1));
        ++mParams.referenceSamples;
        ++mParams.frameIndex;
        if (!mSaveReferencePath.empty())
        {
            saveReference(pRenderContext, mSaveReferencePath);
            mSaveReferencePath.clear();
        }
        return;
    }

    // World space: residual pages depend on the sun and density only. The world cache's modulation reads them, so they come first.
    if (mResidualDirty)
    {
        FALCOR_PROFILE(pRenderContext, "residualPages");
        dispatchResidualPages(pRenderContext);
        mResidualDirty = false;
        mBasisDirty = true;
    }

    // The cache modulation's default is the diffusion attenuation per unit optical depth, sqrt(3 (1 - albedo) (1 - albedo g)):
    // multiply scattered light fades into the cloud at that rate, so the stored ratio is nearly flat at cell scale.
    {
        const float albedo = dot(mpScene->getGridVolume(0)->getAlbedo(), float3(1.f / 3.f));
        const float diffusion = std::sqrt(std::max(0.f, 3.f * (1.f - albedo) * (1.f - albedo * mParams.anisotropy)));
        mParams.worldCacheModulation = mWorldCacheModulation < 0.f ? diffusion : mWorldCacheModulation;
    }

    // World cache experiment: camera-independent gather passes accumulated while its view is shown.
    const bool worldCacheFrame = mParams.debugView == kWorldCacheView || mParams.debugView == kBeamView;
    if (worldCacheFrame)
    {
        const uint32_t cellVoxels = mParams.worldCacheCellVoxels;
        const uint3 dims = (mParams.hstrExtinctionDims + cellVoxels - 1u) / cellVoxels;
        if (any(dims != mParams.worldCacheDims))
        {
            mParams.worldCacheDims = dims;
            mParams.worldCacheSamples = 0;
            logInfo("HSTRCloud: world cache {} cells of {} voxels.", dims, cellVoxels);
        }
        // Float running sums are only used by the gather and by untextured light tracing; textured light tracing bakes from its
        // fixed-point deposits.
        const bool floatSums = mParams.worldCacheEstimator == 0 || mParams.worldCacheTextured == 0;
        const size_t floatCount = size_t(dims.x) * dims.y * dims.z * 27;
        if (!floatSums)
            mpWorldCache = nullptr;
        else if (!mpWorldCache || mpWorldCache->getElementCount() != floatCount)
        {
            mpWorldCache = mpDevice->createStructuredBuffer(
                sizeof(float),
                uint32_t(floatCount),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr,
                false
            );
            mParams.worldCacheSamples = 0;
        }
        const size_t depositCount = size_t(dims.x) * dims.y * dims.z * 18;
        if (mParams.worldCacheEstimator != 0 && (!mpWorldCacheDeposit || mpWorldCacheDeposit->getElementCount() != depositCount))
            mpWorldCacheDeposit = mpDevice->createStructuredBuffer(
                sizeof(int32_t),
                uint32_t(depositCount),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr,
                false
            );
        mParams.worldVoxelSize = mVoxelSize;
        FALCOR_PROFILE(pRenderContext, "worldCache");
        for (uint32_t i = 0; i < mWorldCacheUpdates; ++i)
        {
            if (mParams.worldCacheEstimator == 0)
            {
                bindRenderer(pRenderContext, mpWorldCachePass);
                mpWorldCachePass->execute(pRenderContext, dims);
            }
            else if (mParams.worldCacheSegments > 0)
            {
                // Persistent pool: a fixed amount of flight segments per update; deposits and the emitted count accumulate.
                const uint64_t poolBytes = uint64_t(mParams.worldCachePhotons) * 48;
                if (!mpPhotonPool || mpPhotonPool->getSize() != poolBytes)
                {
                    mpPhotonPool = mpDevice->createStructuredBuffer(
                        48,
                        mParams.worldCachePhotons,
                        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                        MemoryType::DeviceLocal,
                        nullptr,
                        false
                    );
                    mParams.worldCacheSamples = 0;
                }
                if (!mpPhotonEmitted)
                    mpPhotonEmitted = mpDevice->createStructuredBuffer(
                        sizeof(uint32_t),
                        1,
                        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                        MemoryType::DeviceLocal,
                        nullptr,
                        false
                    );
                if (mParams.worldCacheSamples == 0)
                {
                    pRenderContext->clearUAV(mpWorldCacheDeposit->getUAV().get(), uint4(0));
                    pRenderContext->clearUAV(mpPhotonPool->getUAV().get(), uint4(0));
                    pRenderContext->clearUAV(mpPhotonEmitted->getUAV().get(), uint4(0));
                }
                bindRenderer(pRenderContext, mpWorldCacheAdvancePass);
                mpWorldCacheAdvancePass->execute(pRenderContext, uint3(mParams.worldCachePhotons, 1, 1));
            }
            else
            {
                // With textures, deposits accumulate across batches and the bake reads them directly: no per-batch clear or
                // resolve. The buffer lookup still folds every batch into the float sums.
                const bool accumulate = mParams.worldCacheTextured != 0;
                if (!accumulate || mParams.worldCacheSamples == 0)
                    pRenderContext->clearUAV(mpWorldCacheDeposit->getUAV().get(), uint4(0));
                bindRenderer(pRenderContext, mpWorldCachePhotonPass);
                mpWorldCachePhotonPass->execute(pRenderContext, uint3(mParams.worldCachePhotons, 1, 1));
                if (!accumulate)
                {
                    bindRenderer(pRenderContext, mpWorldCacheResolvePass);
                    mpWorldCacheResolvePass->execute(pRenderContext, dims);
                }
            }
            ++mParams.worldCacheSamples;
            ++mWorldCacheBakes; // Buffer lookups see every update.
            mWorldCacheBakeDirty = true;
        }

        // Bake the means into hardware-filtered textures for the camera whenever the sums changed.
        if (!mpWorldCacheTextures[0] || mpWorldCacheTextures[0]->getWidth() != dims.x || mpWorldCacheTextures[0]->getHeight() != dims.y ||
            mpWorldCacheTextures[0]->getDepth() != dims.z)
        {
            for (auto& texture : mpWorldCacheTextures)
                texture = mpDevice->createTexture3D(
                    dims.x,
                    dims.y,
                    dims.z,
                    ResourceFormat::RGBA16Float,
                    1,
                    nullptr,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
                );
            mWorldCacheBakeDirty = true;
        }
        const bool bakeDue = mWorldCacheUpdates == 0 || mParams.worldCacheSamples % mWorldCacheBakeInterval == 0;
        if (mWorldCacheBakeDirty && mParams.worldCacheTextured != 0 && bakeDue)
        {
            FALCOR_PROFILE(pRenderContext, "worldCacheBake");
            bindRenderer(pRenderContext, mpWorldCacheBakePass);
            ShaderVar var = mpWorldCacheBakePass->getRootVar()["CB"]["gHSTRCloud"];
            for (uint32_t i = 0; i < kWorldCacheTextures; ++i)
            {
                var["hstrWorldCacheTexture"][i] = ref<Texture>();
                var["hstrWorldCacheTextureOutput"][i] = mpWorldCacheTextures[i];
            }
            mpWorldCacheBakePass->execute(pRenderContext, dims);
            mWorldCacheBakeDirty = false;
            ++mWorldCacheBakes;
        }
    }

    // Camera space: nothing below re-solves transport; it reprojects the solved field. The world cache views integrate
    // every pixel themselves: the HST camera lighting, cut and camera basis stay dirty until they are used again.
    const bool hstCamera = !worldCacheFrame;
    if (mCameraLightingDirty && hstCamera)
    {
        FALCOR_PROFILE(pRenderContext, "cameraLighting");
        bindRenderer(pRenderContext, mpCameraLightingPass);
        bindOutput(mpCameraLightingPass, "hstrCameraLightingOutput", mpCameraLighting, "hstrCameraLighting");
        mpCameraLightingPass->execute(pRenderContext, mParams.hstrCameraLightingDims);
        mCameraLightingDirty = false;
        mCameraLightingPoseValid = true;
        mCameraLightingPosition = mpScene->getCamera()->getPosition();
        mCameraLightingDirection = mpScene->getCamera()->getTarget() - mCameraLightingPosition;
        mBasisDirty = true;
    }

    ensureCameraResources();
    if (mCutDirty && hstCamera)
    {
        FALCOR_PROFILE(pRenderContext, "cut");
        const uint32_t nodeCount = mParams.hstrNodeCount;
        bindRenderer(pRenderContext, mpProjectPass);
        mpProjectPass->execute(pRenderContext, uint3(nodeCount, 1, 1));

        // The previous cut's acceptance feeds the split/merge hysteresis of this one.
        mParams.cutParity ^= 1u;
        pRenderContext->clearUAV(mpTileNodeCounts->getUAV().get(), uint4(0));
        bindRenderer(pRenderContext, mpCutPass);
        mpCutPass->execute(pRenderContext, uint3(nodeCount, 1, 1));

        bindRenderer(pRenderContext, mpSortPass);
        mpSortPass->execute(pRenderContext, uint3(mParams.hstrTileDims.x * 64u, mParams.hstrTileDims.y, 1));
        mCutDirty = false;
        mBasisDirty = true;
    }

    // Tile camera bases are deterministic, so a static camera reuses them.
    if (mBasisDirty && hstCamera)
    {
        FALCOR_PROFILE(pRenderContext, "cameraBasis");
        bindRenderer(pRenderContext, mpQueryPass);
        bindOutput(mpQueryPass, "hstrCameraQueryOutput", mpCameraQueries, "hstrCameraQueries");
        bindOutput(mpQueryPass, "hstrTileCenterOutput", mpTileCenters, "hstrTileCenters");
        mpQueryPass->execute(pRenderContext, uint3(mParams.hstrLatticeDims, 1));

        bindRenderer(pRenderContext, mpTileBasisPass);
        mpTileBasisPass->execute(pRenderContext, uint3(mParams.hstrTileDims, 1));
        mBasisDirty = false;
    }

    // Beam view: hierarchical tiles. Level 0 queries every coarsest tile corner and centre and tests every tile; each finer
    // level queries and tests only the children of the tiles refined above it, through compacted lists and indirect
    // dispatches. Tiles failing the finest level are marched per pixel.
    if (mParams.debugView == kBeamView)
    {
        uint32_t maximumLevels = 0;
        while ((mParams.beamTileSize >> (maximumLevels + 1)) >= 2u && maximumLevels + 1 < kBeamMaxLevels)
            ++maximumLevels;
        mParams.beamLevels = std::clamp(mParams.beamLevels, 1u, maximumLevels + 1);
        const uint32_t finest = mParams.beamTileSize >> (mParams.beamLevels - 1);
        mParams.beamLatticeStep = std::max(1u, finest / 2);
        mParams.beamTileDims = (frameDim + mParams.beamTileSize - 1u) / mParams.beamTileSize;
        mParams.beamLatticeDims = mParams.beamTileDims * mParams.beamTileSize / mParams.beamLatticeStep + 1u;
        mParams.beamLevelDims = (frameDim + finest - 1u) / finest;
        const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        const uint2 latticeDims = mParams.beamLatticeDims;
        const uint2 levelDims = mParams.beamLevelDims;
        const uint32_t tileCount = mParams.beamTileDims.x * mParams.beamTileDims.y;
        if (!mpBeamLattice || mpBeamLattice->getWidth() != latticeDims.x || mpBeamLattice->getHeight() != latticeDims.y)
            mpBeamLattice = mpDevice->createTexture2D(latticeDims.x, latticeDims.y, ResourceFormat::RGBA16Float, 2, 1, nullptr, flags);
        if (!mpBeamLevel || mpBeamLevel->getWidth() != levelDims.x || mpBeamLevel->getHeight() != levelDims.y)
            mpBeamLevel = mpDevice->createTexture2D(levelDims.x, levelDims.y, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
        // List l (1-4) holds up to 4^(l-1) entries per coarsest tile; the list after the finest level is the march list.
        bool layoutChanged = false;
        for (uint32_t i = 0; i < 2; ++i)
        {
            if (!mpBeamLists[i] || mpBeamLists[i]->getElementCount() != tileCount * 85)
            {
                mpBeamLists[i] =
                    mpDevice->createStructuredBuffer(sizeof(uint32_t), tileCount * 85, flags, MemoryType::DeviceLocal, nullptr, false);
                layoutChanged = true;
            }
            if (!mpBeamCounts[i])
                mpBeamCounts[i] =
                    mpDevice->createStructuredBuffer(sizeof(uint32_t), kBeamMaxLevels + 1, flags, MemoryType::DeviceLocal, nullptr, false);
            if (!mpBeamHistory[i] || mpBeamHistory[i]->getWidth() != levelDims.x || mpBeamHistory[i]->getHeight() != levelDims.y)
            {
                mpBeamHistory[i] = mpDevice->createTexture2D(levelDims.x, levelDims.y, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
                layoutChanged = true;
            }
        }
        const uint4 layout(frameDim, mParams.beamTileSize, mParams.beamLevels);
        if (layoutChanged || any(layout != mBeamLayout))
        {
            mBeamLayout = layout;
            mBeamHistoryValid = false;
            mBeamReusable = false;
        }
        if (!mpBeamArgs)
        {
            mpBeamArgs = mpDevice->createStructuredBuffer(
                sizeof(uint32_t),
                6 * (kBeamMaxLevels + 1),
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::IndirectArg,
                MemoryType::DeviceLocal,
                nullptr,
                false
            );
        }
        auto writeArgs = [&](uint32_t level)
        {
            mParams.beamPassLevel = level;
            bindRenderer(pRenderContext, mpBeamArgsPass);
            mpBeamArgsPass->getRootVar()["CB"]["gHSTRCloud"]["hstrBeamArgs"] = mpBeamArgs;
            mpBeamArgsPass->execute(pRenderContext, uint3(1));
        };
        // Temporal: with the camera and the cache unchanged since the last build, its queries and tile levels are still exact.
        const bool temporal = mParams.beamTemporal != 0;
        const float3 cameraPosition = mpScene->getCamera()->getPosition();
        const float3 cameraTarget = mpScene->getCamera()->getTarget();
        const bool reuse = temporal && mBeamReusable && all(cameraPosition == mBeamCameraPosition) &&
                           all(cameraTarget == mBeamCameraTarget) && mBeamBakes == mWorldCacheBakes;
        if (!reuse)
        {
            FALCOR_PROFILE(pRenderContext, "beamQueries");
            if (temporal)
            {
                // One query pass over every level (children of the tiles refined last frame) and one tile pass walking each
                // coarsest tile down those levels; this frame's refinements drive the next frame's queries.
                mBeamParity ^= 1u;
                if (!mBeamHistoryValid)
                {
                    pRenderContext->clearUAV(mpBeamCounts[mBeamParity ^ 1u]->getUAV().get(), uint4(0));
                    pRenderContext->clearUAV(mpBeamHistory[mBeamParity ^ 1u]->getUAV().get(), uint4(0));
                }
                pRenderContext->clearUAV(mpBeamCounts[mBeamParity]->getUAV().get(), uint4(0));
                pRenderContext->clearUAV(mpBeamHistory[mBeamParity]->getUAV().get(), uint4(0));
                writeArgs(0);
                {
                    FALCOR_PROFILE(pRenderContext, "queries0");
                    bindRenderer(pRenderContext, mpBeamQueryPass);
                    bindOutput(mpBeamQueryPass, "hstrBeamLatticeOutput", mpBeamLattice, "hstrBeamLattice");
                    mpBeamQueryPass->executeIndirect(pRenderContext, mpBeamArgs.get(), 0);
                }
                bindRenderer(pRenderContext, mpBeamTemporalTilePass);
                bindOutput(mpBeamTemporalTilePass, "hstrBeamLevelOutput", mpBeamLevel, "hstrBeamLevel");
                mpBeamTemporalTilePass->getRootVar()["CB"]["gHSTRCloud"]["hstrBeamHistoryOutput"] = mpBeamHistory[mBeamParity];
                mpBeamTemporalTilePass->execute(pRenderContext, uint3(tileCount, 1, 1));
                mBeamHistoryValid = true;
            }
            else
            {
                pRenderContext->clearUAV(mpBeamCounts[mBeamParity]->getUAV().get(), uint4(0));
                for (uint32_t level = 0; level < mParams.beamLevels; ++level)
                {
                    FALCOR_PROFILE(pRenderContext, "beamLevel" + std::to_string(level));
                    if (level > 0)
                        writeArgs(level);
                    mParams.beamPassLevel = level;
                    {
                        FALCOR_PROFILE(pRenderContext, "queries" + std::to_string(level));
                        bindRenderer(pRenderContext, mpBeamQueryPass);
                        bindOutput(mpBeamQueryPass, "hstrBeamLatticeOutput", mpBeamLattice, "hstrBeamLattice");
                        if (level == 0)
                        {
                            // Corners and centres in blocks of 8 x 4 lattice points (see beamQueryPosition).
                            auto blockQueries = [](uint2 grid) { return ((grid.x + 7u) / 8u) * ((grid.y + 3u) / 4u) * 32u; };
                            const uint32_t queries = blockQueries(mParams.beamTileDims + 1u) + blockQueries(mParams.beamTileDims);
                            mpBeamQueryPass->execute(pRenderContext, uint3(queries * mParams.beamSegments, 1, 1));
                        }
                        else
                            mpBeamQueryPass->executeIndirect(pRenderContext, mpBeamArgs.get(), 24 * level);
                    }
                    bindRenderer(pRenderContext, mpBeamTilePass);
                    bindOutput(mpBeamTilePass, "hstrBeamLevelOutput", mpBeamLevel, "hstrBeamLevel");
                    if (level == 0)
                        mpBeamTilePass->execute(pRenderContext, uint3(tileCount, 1, 1));
                    else
                        mpBeamTilePass->executeIndirect(pRenderContext, mpBeamArgs.get(), 24 * level + 12);
                }
                mBeamHistoryValid = false;
            }
            writeArgs(mParams.beamLevels);
            mBeamReusable = true;
            mBeamCameraPosition = cameraPosition;
            mBeamCameraTarget = cameraTarget;
            mBeamBakes = mWorldCacheBakes;
        }
    }

    if (mParams.debugView == kBeamView)
    {
        // Reconstruction of every accepted pixel in a light kernel, then the compacted per-pixel march of failed finest tiles.
        FALCOR_PROFILE(pRenderContext, "resolve");
        bindRenderer(pRenderContext, mpBeamResolvePass);
        mpBeamResolvePass->getRootVar()["CB"]["gHSTRCloud"]["color"] = color;
        mpBeamResolvePass->execute(pRenderContext, uint3(mParams.frameDim, 1));
        FALCOR_PROFILE(pRenderContext, "march");
        bindRenderer(pRenderContext, mpBeamMarchPass);
        mpBeamMarchPass->getRootVar()["CB"]["gHSTRCloud"]["color"] = color;
        mpBeamMarchPass->executeIndirect(pRenderContext, mpBeamArgs.get(), 24 * mParams.beamLevels);
    }
    else
    {
        FALCOR_PROFILE(pRenderContext, "resolve");
        bindRenderer(pRenderContext, mpPass);
        ShaderVar var = mpPass->getRootVar()["CB"]["gHSTRCloud"];
        var["color"] = color;
        var["transportError"] = error;
        var["cutStats"] = cutStats;
        mpPass->execute(pRenderContext, uint3(mParams.frameDim, 1));
    }

    if (mStoreExact)
    {
        if (!mpExactFrame || mpExactFrame->getWidth() != frameDim.x || mpExactFrame->getHeight() != frameDim.y)
            mpExactFrame = mpDevice->createTexture2D(
                frameDim.x, frameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::ShaderResource
            );
        pRenderContext->copyResource(mpExactFrame.get(), color.get());
        mStoreExact = false;
    }

    // Error against the accumulated path-traced reference of this view (or the stored exact frame), read by scripts through
    // getProperties().
    if (mCompareReference && (mParams.referenceSamples > 0 || (mParams.compareExact != 0 && mpExactFrame)))
    {
        if (mParams.debugView == kBeamView && mpBeamCounts[mBeamParity])
        {
            const uint32_t finest = mParams.beamTileSize >> (mParams.beamLevels - 1);
            const uint32_t marchedTiles = mpBeamCounts[mBeamParity]->getElement<uint32_t>(mParams.beamLevels);
            mBeamMarchedFraction = float(marchedTiles * finest * finest) / float(frameDim.x * frameDim.y);
        }
        bindRenderer(pRenderContext, mpCompareReferencePass);
        ShaderVar var = mpCompareReferencePass->getRootVar()["CB"]["gHSTRCloud"];
        var["hstrReferenceSum"] = mpReferenceSum;
        var["hstrReferenceRowError"] = mpReferenceRowError;
        var["color"] = color;
        // Block comparisons write one entry per row of blocks.
        const uint32_t rowCount = mParams.compareBlock > 1 ? frameDim.y / mParams.compareBlock : frameDim.y;
        mpCompareReferencePass->execute(pRenderContext, uint3(rowCount, 1, 1));
        const std::vector<float4> rows = mpReferenceRowError->getElements<float4>(0, rowCount);
        float4 total(0.f);
        for (const float4& row : rows)
            total += row;
        mReferenceError = total.x / float(rows.size());
        mReferenceLogError = total.y / float(rows.size());
        mReferenceNoiseError = total.z / float(rows.size());
        mReferenceNoiseLogError = total.w / float(rows.size());
    }
    if (!mSaveReferencePath.empty())
    {
        saveReference(pRenderContext, mSaveReferencePath);
        mSaveReferencePath.clear();
    }
    ++mParams.frameIndex;
}

/// Writes the reference (per component and half: rgb mean, a sample count) as one EXR per slice, so an expensive path-traced
/// reference survives the process and can be reloaded for comparisons. Compressed EXRs hold half floats, so means are stored
/// rather than sums, which would overflow at high sample counts.
void HSTRCloud::saveReference(RenderContext* pRenderContext, const std::string& path)
{
    if (!mpReferenceSum || mParams.referenceSamples == 0)
    {
        logWarning("HSTRCloud: no reference accumulated, nothing saved to '{}'.", path);
        return;
    }
    for (uint32_t slice = 0; slice < mpReferenceSum->getArraySize(); ++slice)
    {
        std::vector<uint8_t> data =
            pRenderContext->readTextureSubresource(mpReferenceSum.get(), mpReferenceSum->getSubresourceIndex(slice, 0));
        float4* texels = reinterpret_cast<float4*>(data.data());
        for (size_t i = 0; i < data.size() / sizeof(float4); ++i)
            if (texels[i].w > 0.f)
                texels[i] = float4(texels[i].xyz() / texels[i].w, texels[i].w);
        Bitmap::saveImage(
            path + "_s" + std::to_string(slice) + ".exr",
            mpReferenceSum->getWidth(),
            mpReferenceSum->getHeight(),
            Bitmap::FileFormat::ExrFile,
            Bitmap::ExportFlags::ExportAlpha,
            ResourceFormat::RGBA32Float,
            true,
            data.data()
        );
    }
    logInfo("HSTRCloud: saved the {}-sample reference to '{}_s*.exr'.", mParams.referenceSamples, path);
}

/// Reads reference running sums written by saveReference into the reference texture; the sample count comes from the
/// halves' counts. The frame size must match. Set the camera and lights first: changing them afterwards restarts the reference.
void HSTRCloud::loadReference(RenderContext* pRenderContext, const std::string& path)
{
    if (!mpReferenceSum)
        return;
    const uint32_t width = mpReferenceSum->getWidth();
    const uint32_t height = mpReferenceSum->getHeight();
    float samples = 0.f;
    for (uint32_t slice = 0; slice < mpReferenceSum->getArraySize(); ++slice)
    {
        const std::filesystem::path file = path + "_s" + std::to_string(slice) + ".exr";
        auto bitmap = Bitmap::createFromFile(file, true);
        const ResourceFormat format = bitmap ? bitmap->getFormat() : ResourceFormat::Unknown;
        if (!bitmap || bitmap->getWidth() != width || bitmap->getHeight() != height ||
            (format != ResourceFormat::RGBA32Float && format != ResourceFormat::RGBA16Float))
        {
            logWarning(
                "HSTRCloud: cannot load reference slice '{}': {}x{} {}, {}x{} RGBA expected.",
                file,
                bitmap ? bitmap->getWidth() : 0,
                bitmap ? bitmap->getHeight() : 0,
                bitmap ? to_string(format) : std::string("unreadable"),
                width,
                height
            );
            return;
        }
        // Means back to running sums.
        std::vector<float4> sums(size_t(width) * height);
        for (size_t i = 0; i < sums.size(); ++i)
        {
            float4 texel;
            if (format == ResourceFormat::RGBA32Float)
                texel = reinterpret_cast<const float4*>(bitmap->getData())[i];
            else
            {
                const uint16_t* bits = reinterpret_cast<const uint16_t*>(bitmap->getData()) + 4 * i;
                texel = float4(
                    math::float16ToFloat32(bits[0]),
                    math::float16ToFloat32(bits[1]),
                    math::float16ToFloat32(bits[2]),
                    math::float16ToFloat32(bits[3])
                );
            }
            sums[i] = float4(texel.xyz() * texel.w, texel.w);
        }
        pRenderContext->updateSubresourceData(mpReferenceSum.get(), mpReferenceSum->getSubresourceIndex(slice, 0), sums.data());
        if (slice < 2)
            samples += sums[0].w;
    }
    mParams.referenceSamples = uint32_t(std::lround(samples));
    mReferencePosition = mpScene->getCamera()->getPosition();
    mReferenceDirection = mpScene->getCamera()->getTarget() - mReferencePosition;
    logInfo("HSTRCloud: loaded a {}-sample reference from '{}_s*.exr'.", mParams.referenceSamples, path);
}

void HSTRCloud::renderUI(Gui::Widgets& widget)
{
    bool renderChanged = false;
    bool lightingChanged = false;
    bool operatorChanged = false;
    bool residualChanged = false;
    renderChanged |= widget.var("Inside-cloud base steps", mParams.baseSteps, 8u, 128u, 8u);
    renderChanged |= widget.var("Inside-cloud refinement", mParams.refinementLevel, 0u, 3u, 1u);
    renderChanged |= widget.var("Inside-cloud residual blend", mParams.residualBlend, 0.f, 1.f, 0.01f);
    renderChanged |= widget.var("Cut transmittance error", mParams.cutTransmittanceTolerance, 0.0001f, 0.05f, 0.0001f);
    renderChanged |= widget.var("Cut split/merge hysteresis", mParams.cutHysteresis, 0.f, 0.9f, 0.01f);
    renderChanged |= widget.var("Tile basis surplus tolerance", mParams.basisTolerance, 0.f, 1.f, 0.001f);
    renderChanged |= widget.var("Tile basis merge fraction", mParams.basisHysteresis, 0.f, 1.f, 0.01f);
    renderChanged |= widget.var("March step optical depth", mParams.stepOpticalDepth, 0.02f, 2.f, 0.01f);
    renderChanged |= widget.var("Minimum march step (voxels)", mParams.minStepVoxels, 0.05f, 2.f, 0.05f);
    renderChanged |= widget.var("Maximum march step (voxels)", mParams.maxStepVoxels, 0.5f, 4.f, 0.25f);
    residualChanged |= widget.var("Residual page tolerance", mParams.residualTolerance, 0.f, 1.f, 0.001f);
    renderChanged |= widget.var("Residual strength", mParams.residualStrength, 0.f, 4.f, 0.01f);
    renderChanged |= widget.var("Sun octave energy", mParams.octaveEnergy, 0.f, 1.f, 0.01f);
    residualChanged |= widget.var("Sun octave extinction", mParams.octaveExtinction, 0.01f, 1.f, 0.01f);
    renderChanged |= widget.var("Sun octave phase", mParams.octavePhase, 0.f, 1.f, 0.01f);
    lightingChanged |= widget.var("HST sun share", mParams.hstSunFraction, 0.f, 1.f, 0.01f);
    {
        Gui::DropdownList debugViews = {
            {0, "Final"},
            {1, "Refined tiles"},
            {2, "March cost"},
            {3, "Opacity"},
            {4, "Exact everywhere"},
            {5, "Residual fields"},
            {kReferenceView, "Path-traced reference"},
            {7, "Rasterized sun residual"},
            {kWorldCacheView, "World cache (multiple scattering + sky)"}};
        renderChanged |= widget.dropdown("Debug view", debugViews, mParams.debugView);
    }
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
    if (operatorChanged || lightingChanged)
    {
        mParams.referenceSamples = 0;
        mParams.worldCacheSamples = 0;
    }
    mResidualDirty |= residualChanged;
    mOptionsChanged |= renderChanged || lightingChanged || operatorChanged || residualChanged;
    mCutDirty |= renderChanged || operatorChanged || residualChanged;
    widget.textWrapped(
        "Outside-cloud views share one projected HST antichain per 8x8 tile. Every pixel integrates its own transmittance through the "
        "cut; its source radiance comes from the tile camera basis, or from an exact per-pixel integral where the basis surplus or a "
        "resident residual page's footprint demands it. Camera motion only reprojects; transport is re-solved for sun or density changes."
    );
    widget.text(fmt::format("Operator dictionary: {} prototypes", mParams.operatorDictionarySize));
}
