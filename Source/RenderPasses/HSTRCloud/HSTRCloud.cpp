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
#include <numeric>
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
const char kBeamAdaptiveRoot[] = "beamAdaptiveRoot";
const char kCompareBlock[] = "compareBlock";
const char kCompareColumnLow[] = "compareColumnLow";
const char kCompareColumnHigh[] = "compareColumnHigh";
const char kCompareMapScale[] = "compareMapScale";
const char kWorldCacheModulation[] = "worldCacheModulation";
const char kWorldCacheModulationDepth[] = "worldCacheModulationDepth";
const char kWorldCacheZonalBands[] = "worldCacheZonalBands";
const char kWorldCacheZonalWindow[] = "worldCacheZonalWindow";
const char kWorldCacheSunOrder[] = "worldCacheSunOrder";
const char kSunNearVoxels[] = "sunNearVoxels";
const char kMaxMarchSteps[] = "maxMarchSteps";
const char kWorldCacheSimilarity[] = "worldCacheSimilarity";
const char kAdaptiveMarch[] = "adaptiveMarch";
const char kMarchTolerance[] = "marchTolerance";
const char kMarchMinVoxels[] = "marchMinVoxels";
const char kMarchCoarseVoxels[] = "marchCoarseVoxels";
const char kSeaMode[] = "seaMode";
const char kSeaViewDistance[] = "seaViewDistance";
const char kCloudLibrary[] = "cloudLibrary";
const char kCloudProxyResolution[] = "cloudProxyResolution";
const char kCloudBrickPoolMB[] = "cloudBrickPoolMB";
const char kCloudPayloadPoolMB[] = "cloudPayloadPoolMB";
const char kCloudDirectStorage[] = "cloudDirectStorage";
const char kCloudBrickLoadsPerFrame[] = "cloudBrickLoadsPerFrame";
const char kCloudSeaTiles[] = "cloudSeaTiles";
const char kCloudSeaSeed[] = "cloudSeaSeed";
const char kCloudSeaCoverage[] = "cloudSeaCoverage";
const char kCloudLodPixels[] = "cloudLodPixels";
const char kCloudLodBias[] = "cloudLodBias";
const char kCloudFadeFrames[] = "cloudFadeFrames";
const char kCloudVirtual[] = "cloudVirtual";
const char kCloudFineMinVoxels[] = "cloudFineMinVoxels";
const char kCloudEmptySkip[] = "cloudEmptySkip";
const char kCloudStats[] = "cloudStats";
const char kCloudSunTilesPerFrame[] = "cloudSunTilesPerFrame";
const char kCloudSunBakesPerFrame[] = "cloudSunBakesPerFrame";
const char kCloudSunBakeAngle[] = "cloudSunBakeAngle";
const char kCloudSunCache[] = "cloudSunCache";
const char kMarchProbe[] = "marchProbe";
const char kCloudThinDepth[] = "cloudThinDepth";
const char kCloudZeroSkip[] = "cloudZeroSkip";
const char kCloudSunReuse[] = "cloudSunReuse";
const char kCloudTightReject[] = "cloudTightReject";
const char kCloudCostProbe[] = "cloudCostProbe";
const char kCloudSlabClamp[] = "cloudSlabClamp";
const char kCloudTrapezoid[] = "cloudTrapezoid";
const char kCloudLocalStep[] = "cloudLocalStep";
const char kCloudStepFootprint[] = "cloudStepFootprint";
const char kCloudQuadrature[] = "cloudQuadrature";
const char kCloudSourceLinear[] = "cloudSourceLinear";
const char kCloudTransferClasses[] = "cloudTransferClasses";
const char kCloudLongitudinalOracle[] = "cloudLongitudinalOracle";
const char kCloudOracleCentroid[] = "cloudOracleCentroid";
const char kCloudSunLiveMarch[] = "cloudSunLiveMarch";
const char kCloudCameraKernel[] = "cloudCameraKernel";
const char kCloudMinTransmittance[] = "cloudMinTransmittance";
const char kSaveReference[] = "saveReference";
const char kLoadReference[] = "loadReference";
const char kBeamTolerance[] = "beamTolerance";
const char kBeamGuide[] = "beamGuide";
const char kBeamGuideDepth[] = "beamGuideDepth";
const char kBeamGuideSun[] = "beamGuideSun";
const char kBeamEdgeContrast[] = "beamEdgeContrast";
const char kBeamOracle[] = "beamOracle";
const char kBeamOracleBar[] = "beamOracleBar";
const char kBeamCentreless[] = "beamCentreless";
const char kBeamGridDispatch[] = "beamGridDispatch";
const char kBeamSparse[] = "beamSparse";
const char kBeamSparseCut[] = "beamSparseCut";
const char kBeamRefFrame[] = "beamRefFrame";
const char kBeamRefMargin[] = "beamRefMargin";
const char kBeamSparseMinLevel[] = "beamSparseMinLevel";
const char kBeamRefresh[] = "beamRefresh";
const char kBeamShip[] = "beamShip";
const char kBeamShipMask[] = "beamShipMask";
const char kBeamQueue[] = "beamQueue";
const char kBeamQueueSteps[] = "beamQueueSteps";
const char kCloudResidencyFrozen[] = "cloudResidencyFrozen";
const char kCloudCutMargin[] = "cloudCutMargin";
const char kCloudCutTurn[] = "cloudCutTurn";
const char kCloudCutAsync[] = "cloudCutAsync";
const char kCloudGpuSun[] = "cloudGpuSun";
const char kBeamRefreshDebug[] = "beamRefreshDebug";
const char kBeamParallax[] = "beamParallax";
const char kBeamCarryTolerance[] = "beamCarryTolerance";
const char kBeamDepthTolerance[] = "beamDepthTolerance";
const char kBeamRefreshBlock[] = "beamRefreshBlock";
const char kBeamRefreshCentres[] = "beamRefreshCentres";
const char kStoreExact[] = "storeExact";
const char kCompareExact[] = "compareExact";
const char kBeamMarchedFraction[] = "beamMarchedFraction";
const char kReferenceShow[] = "referenceShow";
const char kReferenceBandRows[] = "referenceBandRows";
const char kCompareSubstitute[] = "compareSubstitute";
const char kCompareTarget[] = "compareTarget";
const char kReferenceError[] = "referenceError";
const char kReferenceLogError[] = "referenceLogError";
const char kReferenceNoiseError[] = "referenceNoiseError";
const char kReferenceNoiseLogError[] = "referenceNoiseLogError";
const char kReferenceLogP999[] = "referenceLogP999";
const char kReferenceLogMax[] = "referenceLogMax";
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

void bindOutput(const ref<ComputePass>& pPass, const char* output, const ref<Texture>& pTexture, const char* input);

/// 1 for the 16-voxel blocks (4^3 majorant blocks) where any majorant block is non-zero.
template<typename T>
std::vector<uint8_t> majorantZeroBlocks(const std::vector<T>& majorant, uint3 blockDims)
{
    const uint3 dims = (blockDims + 3u) / 4u;
    std::vector<uint8_t> blocks(size_t(dims.x) * dims.y * dims.z, 0);
    for (size_t i = 0; i < majorant.size(); ++i)
        if (float(majorant[i]) > 0.f)
        {
            const uint3 o = uint3(uint32_t(i % blockDims.x), uint32_t((i / blockDims.x) % blockDims.y), uint32_t(i / (size_t(blockDims.x) * blockDims.y))) / 4u;
            blocks[size_t(o.x) + size_t(dims.x) * (size_t(o.y) + size_t(dims.y) * o.z)] = 1;
        }
    return blocks;
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
        // Split from the chain below, which is at MSVC's nesting limit.
        if (key == "beamRefPrebuild")
        {
            mBeamRefPrebuild = bool(value);
            continue;
        }
        if (key == "beamOct")
        {
            mBeamOct = bool(value);
            continue;
        }
        if (key == "beamGuard")
        {
            mBeamGuard = bool(value);
            continue;
        }
        if (key == "beamAssumeCarry")
        {
            mParams.beamAssumeCarry = uint32_t(bool(value));
            continue;
        }
        if (key == "beamReset")
        {
            // Drops the persistent beam image so the next build starts from nothing. The octahedral image is fixed to the WORLD
            // and survives until its dimensions change, so without this a benchmark arm inherits every direction the arms before
            // it marched - including ones marched from other camera positions, which then fail parallax in bulk. That made one
            // configuration measure 0.36 ms and then 2.41 ms in consecutive runs. An arm that resets first measures itself.
            if (bool(value))
            {
                mBeamRefAnchored = false;
                mBeamHistoryValid = false;
                mBeamReusable = false;
                mBeamRefreshValid = false;
                mParams.beamHistoryValid = 0;
                // Entering-strip work is the new screen box minus the box the last build wrote, and the refresh phase is a
                // function of the frame counter. An arm that inherits either of those does not start where its neighbours did.
                mParams.beamFrame = 0;
                mBeamBuiltScreenBounds = float4(0.f);
                mParams.beamPrevScreenBounds = float4(0.f);
                mBeamGuardCleared = false;
            }
            continue;
        }
        if (key == "beamScreenResidual")
        {
            mBeamScreenResidual = bool(value);
            continue;
        }
        if (key == "beamOctAxis")
        {
            mBeamOctAxis = uint32_t(value);
            continue;
        }
        if (key == "beamOctFull")
        {
            mBeamOctFull = bool(value);
            continue;
        }
        if (key == "beamOctScale")
        {
            mBeamOctScale = float(value);
            continue;
        }
        if (key == "beamPageIndirect")
        {
            mParams.beamPageIndirect = uint32_t(value);
            continue;
        }
        if (key == "beamPageShift")
        {
            mParams.beamPageShift = std::max(uint32_t(value), 1u);
            continue;
        }
        if (key == kCompareColumnLow)
        {
            mParams.compareColumnLow = uint32_t(value);
            continue;
        }
        if (key == kCompareColumnHigh)
        {
            mParams.compareColumnHigh = uint32_t(value);
            continue;
        }
        if (key == kBeamOracle)
        {
            mParams.beamOracle = uint32_t(value);
            continue;
        }
        if (key == kBeamOracleBar)
        {
            mParams.beamOracleBar = value;
            continue;
        }
        if (key == kBeamCentreless)
        {
            mParams.beamCentreless = bool(value) ? 1u : 0u;
            continue;
        }
        if (key == kBeamGridDispatch)
        {
            mBeamGridDispatch = value;
            continue;
        }
        if (key == kBeamSparse)
        {
            mBeamSparse = value;
            continue;
        }
        if (key == kBeamSparseCut)
        {
            mBeamSparseCut = value;
            continue;
        }
        if (key == kBeamRefFrame)
        {
            mBeamRefFrame = value;
            mBeamRefAnchored = false;
            continue;
        }
        if (key == kBeamRefMargin)
        {
            mBeamRefMargin = value;
            mBeamRefAnchored = false;
            continue;
        }
        if (key == kBeamSparseMinLevel)
        {
            mBeamSparseMinLevel = uint32_t(value);
            continue;
        }
        if (key == kBeamRefresh)
        {
            mBeamRefresh = uint32_t(value);
            continue;
        }
        if (key == kBeamDepthTolerance)
        {
            mParams.beamDepthTolerance = value;
            continue;
        }
        if (key == kBeamQueue)
        {
            mBeamQueue = value;
            continue;
        }
        if (key == kBeamQueueSteps)
        {
            mParams.beamQueueSteps = value;
            continue;
        }
        if (key == kBeamShip)
        {
            mBeamShip = value;
            continue;
        }
        if (key == kBeamShipMask)
        {
            mBeamShipMask = uint32_t(value);
            continue;
        }
        if (key == kBeamParallax)
        {
            mParams.beamParallax = value;
            continue;
        }
        if (key == kBeamCarryTolerance)
        {
            mParams.beamCarryTolerance = value;
            continue;
        }
        if (key == kReferenceBandRows)
        {
            mReferenceBandRows = uint32_t(value);
            continue;
        }
        if (key == kBeamRefreshBlock)
        {
            mParams.beamRefreshBlock = std::max(uint32_t(value), 1u);
            continue;
        }
        if (key == kBeamRefreshCentres)
        {
            mParams.beamRefreshCentres = uint32_t(value);
            continue;
        }
        if (key == kBeamRefreshDebug)
        {
            mParams.beamRefreshDebug = uint32_t(value);
            continue;
        }
        if (key == kCloudCutMargin)
        {
            mCloudCutMargin = value;
            continue;
        }
        if (key == kCloudCutTurn)
        {
            mCloudCutTurn = value;
            continue;
        }
        if (key == kCloudGpuSun)
        {
            mCloudGpuSun = value; // Read when the residency is created.
            continue;
        }
        if (key == kCloudCutAsync)
        {
            mCloudCutAsync = value;
            continue;
        }
        if (key == kCloudResidencyFrozen)
        {
            mCloudResidencyFrozen = value;
            continue;
        }
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
        else if (key == kBeamAdaptiveRoot)
            mParams.beamAdaptiveRoot = bool(value) ? 1u : 0u;
        else if (key == kCompareBlock)
            mParams.compareBlock = std::max(1u, uint32_t(value));
        else if (key == kCompareMapScale)
            mParams.compareMapScale = std::max(0.f, float(value));
        else if (key == kWorldCacheModulation)
            mWorldCacheModulation = value;
        else if (key == kWorldCacheModulationDepth)
            mParams.worldCacheModulationDepth = std::max(0.f, float(value));
        else if (key == kWorldCacheZonalBands)
            mParams.worldCacheZonalBands = std::min(8u, uint32_t(value));
        else if (key == kWorldCacheZonalWindow)
            mParams.worldCacheZonalWindow = value;
        else if (key == kWorldCacheSunOrder)
            mParams.worldCacheSunOrder = std::clamp(uint32_t(value), 2u, 3u);
        else if (key == kSunNearVoxels)
            mParams.sunNearVoxels = std::clamp(float(value), 0.f, 16.f);
        else if (key == kWorldCacheSimilarity)
            mParams.worldCacheSimilarity = value;
        else if (key == kMaxMarchSteps)
            mParams.maxMarchSteps = std::max(1u, uint32_t(value));
        else if (key == kAdaptiveMarch)
            mParams.adaptiveMarch = bool(value) ? 1u : 0u;
        else if (key == kMarchTolerance)
            mParams.marchTolerance = std::max(0.f, float(value)); // 0: no refinement.
        else if (key == kMarchMinVoxels)
            mParams.marchMinVoxels = std::max(1e-3f, float(value));
        else if (key == kMarchCoarseVoxels)
            mParams.marchCoarseVoxels = std::max(0.1f, float(value));
        else if (key == kSeaMode)
            mParams.seaMode = bool(value) ? 1u : 0u;
        else if (key == kSeaViewDistance)
            mSeaViewDistance = mParams.seaViewDistance = std::max(1.f, float(value));
        else if (key == kCloudLibrary)
            mCloudLibraryPath = value.operator std::string();
        else if (key == kCloudProxyResolution)
            mCloudProxyResolution = (std::clamp(uint32_t(value), 16u, 256u) + 15u) / 16u * 16u;
        else if (key == kCloudBrickPoolMB)
            mCloudBrickPoolMB = std::clamp(uint32_t(value), 16u, 8192u);
        else if (key == kCloudPayloadPoolMB)
            mCloudPayloadPoolMB = std::clamp(uint32_t(value), 8u, 4096u);
        else if (key == kCloudDirectStorage)
            mCloudDirectStorage = value;
        else if (key == kCloudBrickLoadsPerFrame)
            mCloudBrickLoadsPerFrame = std::clamp(uint32_t(value), 1u, 4096u);
        else if (key == kCloudSeaTiles)
            mCloudSeaTiles = std::clamp(uint32_t(value), 2u, 64u);
        else if (key == kCloudSeaSeed)
            mCloudSeaSeed = value;
        else if (key == kCloudSeaCoverage)
            mCloudSeaCoverage = std::clamp(float(value), 0.f, 1.f);
        else if (key == kCloudLodPixels)
            mCloudLodPixels = std::max(0.05f, float(value));
        else if (key == kCloudLodBias)
            mParams.cloudLodBias = value;
        else if (key == kCloudFadeFrames)
            mCloudFadeFrames = std::max(1u, uint32_t(value));
        else if (key == kCloudVirtual)
            mCloudVirtual = value;
        else if (key == kCloudSunBakesPerFrame)
            mCloudSunBakesPerFrame = uint32_t(value);
        else if (key == kCloudSunBakeAngle)
            mCloudSunBakeAngle = std::max(0.f, float(value));
        else if (key == kCloudSunCache)
            mParams.cloudSunCache = bool(value) ? 1u : 0u;
        else if (key == kMarchProbe)
            mParams.marchProbe = uint32_t(value);
        else if (key == kCloudThinDepth)
            mParams.cloudThinDepth = std::max(0.f, float(value));
        else if (key == kCloudZeroSkip)
            // A run length for the shader's bounded walk, not a flag. The clamp was 2 from when this selected a second, coarser zero
            // level (the 64-voxel test that lost); the run-walk measurement recorded beside the parameter goes to eight blocks, and
            // could not be reproduced from the property while the clamp stood.
            mParams.cloudZeroSkip = std::clamp(uint32_t(value), 0u, 16u);
        else if (key == kCloudSunReuse)
            mParams.cloudSunReuse = std::max(0.f, float(value));
        else if (key == kCloudTightReject)
            mParams.cloudTightReject = std::min(uint32_t(value), 2u);
        else if (key == kCloudCostProbe)
            mParams.cloudCostProbe = std::min(uint32_t(value), 6u);
        else if (key == kCloudSlabClamp)
        {
            // The shader has no flag for this: the clamp is two branch-free instructions on seaContentY, and writing the whole grid's
            // height there is what "off" means. A ternary on a parameter here instead cost 19% of the frame.
            mParams.cloudSlabClamp = bool(value) ? 1u : 0u;
            mParams.seaContentY = mParams.cloudSlabClamp != 0 ? mSeaContentBand
                                                              : float2(-std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        }
        else if (key == kCloudTrapezoid)
            mParams.cloudTrapezoid = bool(value) ? 1u : 0u;
        else if (key == kCloudLocalStep)
            mParams.cloudLocalStep = bool(value) ? 1u : 0u;
        else if (key == kCloudStepFootprint)
            mParams.cloudStepFootprint = std::max(0.f, float(value));
        else if (key == kCloudQuadrature)
            mParams.cloudQuadrature = std::clamp(uint32_t(value), 1u, 3u);
        else if (key == kCloudSourceLinear)
            mParams.cloudSourceLinear = bool(value) ? 1u : 0u;
        else if (key == kCloudLongitudinalOracle)
            mParams.cloudLongitudinalOracle = std::min(uint32_t(value), 32u);
        else if (key == kCloudOracleCentroid)
            mParams.cloudOracleCentroid = bool(value) ? 1u : 0u;
        else if (key == kCloudTransferClasses)
            mParams.cloudTransferClasses = std::min(uint32_t(value), 64u);
        else if (key == kCloudSunLiveMarch)
            mCloudSunLiveMarch = value;
        else if (key == kCloudCameraKernel)
            mCloudCameraKernel = value;
        else if (key == kCloudMinTransmittance)
            mParams.cloudMinTransmittance = std::clamp(float(value), 1e-4f, 0.5f);
        else if (key == kCloudSunTilesPerFrame)
            mCloudSunTilesPerFrame = std::max(1u, uint32_t(value));
        else if (key == kCloudEmptySkip)
            mParams.cloudEmptySkip = bool(value) ? 1u : 0u;
        else if (key == kCloudFineMinVoxels)
            mParams.cloudFineMinVoxels = std::clamp(float(value), 1e-3f, 1.f);
        else if (key == kSaveReference)
            mSaveReferencePath = value.operator std::string();
        else if (key == kLoadReference)
            mLoadReferencePath = value.operator std::string();
        else if (key == kBeamTolerance)
            mParams.beamTolerance = value;
        else if (key == kBeamGuide)
            mParams.beamGuide = std::min(uint32_t(value), 2u);
        else if (key == kBeamGuideDepth)
            mParams.beamGuideDepth = value;
        else if (key == kBeamGuideSun)
            mParams.beamGuideSun = value;
        else if (key == kBeamEdgeContrast)
            mParams.beamEdgeContrast = value;
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
        else if (key == kReferenceError || key == kReferenceLogError || key == kReferenceNoiseError || key == kReferenceNoiseLogError || key == kReferenceLogP999 || key == kReferenceLogMax || key == kReferenceSampleCount || key == kWorldCacheSampleCount || key == kBeamMarchedFraction || key == kCloudStats)
            continue; // Read-only measurements.
        else
            logWarning("Unknown property '{}' in HSTRCloud.", key);
    }
}

void HSTRCloud::writeBeamQueueArgs(RenderContext* pRenderContext, uint32_t threadsPerEntry)
{
    mParams.beamQueueThreads = threadsPerEntry;
    bindRenderer(pRenderContext, mpBeamQueueArgsPass);
    mpBeamQueueArgsPass->getRootVar()["CB"]["gHSTRCloud"]["hstrBeamQueueArgs"] = mpBeamQueueArgs;
    mpBeamQueueArgsPass->execute(pRenderContext, uint3(1));
    mParams.beamQueueThreads = 1;
}

bool HSTRCloud::beamShipping() const
{
    const HSTRCloudParams& p = mParams;
    return mBeamShip && p.cloudQuadrature <= 1 && p.cloudSourceLinear == 0 && p.cloudCostProbe == 0 && p.cloudLongitudinalOracle <= 1 &&
           p.cloudSunReuse <= 0.f && p.cloudTransferClasses == 0 && p.cloudTightReject == 0 && p.cloudStepFootprint <= 0.f &&
           p.cloudLocalStep == 0 && p.cloudTrapezoid == 0 && p.adaptiveMarch == 0 && p.cloudEmptySkip == 1 && p.cloudSunCache == 1;
}

void HSTRCloud::setProperties(const Properties& props)
{
    const HSTRCloudParams previous = mParams;
    const float previousModulation = mWorldCacheModulation;
    auto cloudSettings = [&]()
    {
        return fmt::format(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            mCloudLibraryPath,
            mCloudProxyResolution,
            mCloudBrickPoolMB,
            mCloudBrickLoadsPerFrame,
            mCloudSeaTiles,
            mCloudSeaSeed,
            mCloudSeaCoverage,
            mCloudLodPixels,
            mCloudFadeFrames,
            mCloudVirtual
        );
    };
    const std::string previousCloud = cloudSettings();
    parseProperties(props);
    if (!mpScene || (!mpLeafRadiance && !mpCloudSea))
        return;
    if (cloudSettings() != previousCloud)
    {
        buildHierarchy();
        mCutDirty = true;
        mBeamReusable = false;
        mOptionsChanged = true;
        return;
    }
    const auto& p = mParams;
    const auto& q = previous;
    // A beamRefresh history without centres cannot be carried. Other views leave it alone (they do not build), and it still holds the
    // last beam build and its camera, which is what the benchmarks rely on when they interleave exact frames.
    if (p.beamCentreless != q.beamCentreless)
        mBeamRefreshValid = false;
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
        onLightingChanged(previous);
    // The reference and the world cache solve the same lights and medium, so they restart with them (the sea's cache decays instead).
    if (operatorChanged || lightingChanged)
        mParams.referenceSamples = 0;
    if (operatorChanged || (lightingChanged && mParams.cloudDomain == 0))
        mParams.worldCacheSamples = 0;
    if (p.worldCacheCellVoxels != q.worldCacheCellVoxels || p.worldCacheEstimator != q.worldCacheEstimator ||
        p.worldCachePhotons != q.worldCachePhotons || p.worldCacheBands != q.worldCacheBands ||
        p.worldCacheTextured != q.worldCacheTextured || p.worldCacheSegments != q.worldCacheSegments ||
        mWorldCacheModulation != previousModulation || p.worldCacheModulationDepth != q.worldCacheModulationDepth ||
        p.worldCacheZonalBands != q.worldCacheZonalBands || p.worldCacheSunOrder != q.worldCacheSunOrder ||
        p.worldCacheSimilarity != q.worldCacheSimilarity)
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
    props[kBeamAdaptiveRoot] = mParams.beamAdaptiveRoot != 0;
    props[kCompareBlock] = mParams.compareBlock;
    props["beamRefPrebuild"] = mBeamRefPrebuild;
    props["beamOct"] = mBeamOct;
    props["beamOctFull"] = mBeamOctFull;
    props["beamOctAxis"] = mBeamOctAxis;
    props["beamScreenResidual"] = mBeamScreenResidual;
    props["beamAssumeCarry"] = mParams.beamAssumeCarry != 0;
    props["beamGuard"] = mBeamGuard;
    props["beamOctScale"] = mBeamOctScale;
    props["beamOctDim"] = mBeamOct ? mParams.beamFrameDim.x : 0u;
    props["beamPageIndirect"] = mParams.beamPageIndirect;
    props["beamPageShift"] = mParams.beamPageShift;
    props[kCompareColumnLow] = mParams.compareColumnLow;
    props[kCompareColumnHigh] = mParams.compareColumnHigh;
    props[kCompareMapScale] = mParams.compareMapScale;
    props[kWorldCacheModulation] = mWorldCacheModulation;
    props[kWorldCacheModulationDepth] = mParams.worldCacheModulationDepth;
    props[kWorldCacheZonalBands] = mParams.worldCacheZonalBands;
    props[kWorldCacheZonalWindow] = mParams.worldCacheZonalWindow;
    props[kWorldCacheSunOrder] = mParams.worldCacheSunOrder;
    props[kSunNearVoxels] = mParams.sunNearVoxels;
    props[kMaxMarchSteps] = mParams.maxMarchSteps;
    props[kWorldCacheSimilarity] = mParams.worldCacheSimilarity;
    props[kAdaptiveMarch] = mParams.adaptiveMarch != 0;
    props[kMarchTolerance] = mParams.marchTolerance;
    props[kMarchMinVoxels] = mParams.marchMinVoxels;
    props[kMarchCoarseVoxels] = mParams.marchCoarseVoxels;
    props[kSeaMode] = mParams.seaMode != 0;
    props[kSeaViewDistance] = mSeaViewDistance;
    props[kCloudLibrary] = mCloudLibraryPath;
    props[kCloudProxyResolution] = mCloudProxyResolution;
    props[kCloudBrickPoolMB] = mCloudBrickPoolMB;
    props[kCloudPayloadPoolMB] = mCloudPayloadPoolMB;
    props[kCloudDirectStorage] = mCloudDirectStorage;
    props[kCloudBrickLoadsPerFrame] = mCloudBrickLoadsPerFrame;
    props[kCloudSeaTiles] = mCloudSeaTiles;
    props[kCloudSeaSeed] = mCloudSeaSeed;
    props[kCloudSeaCoverage] = mCloudSeaCoverage;
    props[kCloudLodPixels] = mCloudLodPixels;
    props[kCloudLodBias] = mParams.cloudLodBias;
    props[kCloudFadeFrames] = mCloudFadeFrames;
    props[kCloudVirtual] = mCloudVirtual;
    props[kCloudFineMinVoxels] = mParams.cloudFineMinVoxels;
    props[kCloudEmptySkip] = mParams.cloudEmptySkip != 0;
    props[kCloudSunTilesPerFrame] = mCloudSunTilesPerFrame;
    props[kCloudSunBakesPerFrame] = mCloudSunBakesPerFrame;
    props[kCloudSunBakeAngle] = mCloudSunBakeAngle;
    props[kCloudSunCache] = mParams.cloudSunCache != 0;
    props[kMarchProbe] = mParams.marchProbe;
    props[kCloudThinDepth] = mParams.cloudThinDepth;
    props[kCloudZeroSkip] = mParams.cloudZeroSkip;
    props[kCloudSunReuse] = mParams.cloudSunReuse;
    props[kCloudTightReject] = mParams.cloudTightReject;
    props[kCloudCostProbe] = mParams.cloudCostProbe;
    props[kCloudSlabClamp] = mParams.cloudSlabClamp != 0;
    props[kCloudTrapezoid] = mParams.cloudTrapezoid != 0;
    props[kCloudLocalStep] = mParams.cloudLocalStep != 0;
    props[kCloudStepFootprint] = mParams.cloudStepFootprint;
    props[kCloudQuadrature] = mParams.cloudQuadrature;
    props[kCloudSourceLinear] = mParams.cloudSourceLinear != 0;
    props[kCloudTransferClasses] = mParams.cloudTransferClasses;
    props[kCloudLongitudinalOracle] = mParams.cloudLongitudinalOracle;
    props[kCloudOracleCentroid] = mParams.cloudOracleCentroid != 0;
    props[kCloudSunLiveMarch] = mCloudSunLiveMarch;
    props[kCloudCameraKernel] = mCloudCameraKernel;
    props[kCloudMinTransmittance] = mParams.cloudMinTransmittance;
    props[kCloudGpuSun] = mCloudGpuSun;
    if (mpCloudResidency)
    {
        mpCloudResidency->readGpuSunStats();
        mpCloudResidency->readPageStats();
        const auto& stats = mpCloudResidency->getStats();
        Properties cloud;
        cloud["loaded"] = stats.loaded;
        cloud["mapped"] = stats.mapped;
        cloud["desired"] = stats.desired;
        cloud["pending"] = stats.pending;
        cloud["committed"] = stats.committed;
        cloud["slotsUsed"] = stats.slotsUsed;
        cloud["nodesUsed"] = stats.nodesUsed;
        cloud["pagesLoaded"] = stats.pagesLoaded;
        cloud["pageRegions"] = stats.pageRegions;
        cloud["pageExpanded"] = stats.pageExpanded;
        cloud["pageUnique"] = stats.pageUnique;
        cloud["residentMB"] = stats.residentMB;
        cloud["payloadMB"] = stats.payloadMB;
        cloud["cutMs"] = stats.cutMilliseconds;
        cloud["sunBaked"] = stats.sunBaked;
        cloud["sunWaiting"] = stats.sunWaiting;
        cloud["sunSlotsFree"] = stats.sunSlotsFree;
        cloud["sunStale"] = stats.sunStale;
        cloud["sunBakesFrame"] = stats.sunBakesFrame;
        cloud["maps"] = stats.maps;
        cloud["unmaps"] = stats.unmaps;
        cloud["mapBacklog"] = stats.mapBacklog;
        cloud["activeFades"] = stats.activeFades;
        cloud["fadeStarts"] = stats.fadeStarts;
        cloud["fadeEnds"] = stats.fadeEnds;
        cloud["fadeVoid"] = stats.fadeVoid;
        const auto audit = mpCloudResidency->auditMapped();
        cloud["undesiredFadingOut"] = audit.fadingOut;
        cloud["undesiredHeld"] = audit.held;
        cloud["undesiredIdle"] = audit.idle;
        cloud["staleFades"] = audit.stale;
        cloud["cutPops"] = stats.cutPops;
        cloud["cutOrdered"] = stats.cutOrdered;
        cloud["cutTotalMs"] = stats.cutTotalMs;
        cloud["cutMargin"] = stats.cutMargin;
        cloud["cuts"] = stats.cuts;
        cloud["pendingTiles"] = mpCloudSea ? mpCloudSea->pendingTiles() : 0u;
        // What a transfer cache would have had to produce against what it could have served (cloudTransferClasses).
        cloud["transferCrossings"] = mTransferCrossings;
        cloud["transferEntries"] = mTransferEntries;
        // Beam tiles refined into each level, and so the query rays the level costs; the last entry is the per-pixel march list.
        for (uint32_t level = 1; level <= mParams.beamLevels; ++level)
            cloud[level < mParams.beamLevels ? ("beamLevel" + std::to_string(level)) : std::string("beamMarchTiles")] =
                mBeamLevelCounts[level];
        // beamRefresh (one level): root lattice points and marched pixels carried from the previous build.
        if (mBeamRefresh != 0)
        {
            cloud["beamCarriedPoints"] = mBeamLevelCounts[kBeamMaxLevels - 1];
            cloud["beamCarriedPixels"] = mBeamLevelCounts[kBeamMaxLevels];
            cloud["beamRefreshDebugCount"] = mBeamLevelCounts[2];
            if (mParams.beamRefreshDebug == 7)
            {
                cloud["beamMarchedSteps"] = mBeamLevelCounts[kBeamMaxLevels + 1];
                cloud["beamCarriedSteps"] = mBeamLevelCounts[kBeamMaxLevels + 2];
            }
        }
        cloud["beamRootTiles"] = mBeamLevelCounts[kBeamSparseRoots];
        cloud["beamEmptyTiles"] = mBeamLevelCounts[kBeamSparseEmpty];
        cloud["beamCandidates"] = mBeamLevelCounts[kBeamSparseCandidates];
        cloud["beamUniqueQueries"] = mBeamLevelCounts[kBeamSparseUnique];
        cloud["beamFinalTiles"] = mBeamLevelCounts[kBeamSparseFinal];
        cloud["beamCutResidualQueries"] = mBeamLevelCounts[kBeamSparseCutResidualQueries];
        cloud["beamCutResidualSteps"] = mBeamLevelCounts[kBeamSparseCutResidualSteps];
        cloud["beamCutPaged"] = mBeamLevelCounts[kBeamSparseCutPaged];
        cloud["beamRefAnchors"] = mBeamRefAnchors;
        cloud["beamGridSweeps"] = mBeamGridSweeps;
        cloud["beamGridGenerated"] = mBeamGridGenerated;
        cloud["beamGridStrips"] = mBeamGridStrips;
        cloud["beamGridThreads"] = mBeamGridThreads;
        cloud["densityChanged"] = mDensityChangedFrame ? 1u : 0u;
        cloud["densityChangedFrames"] = mDensityChangedFrames;
        cloud["sunBakeFrames"] = mSunBakeFrames;
        cloud["beamCutMarched"] = mBeamLevelCounts[kBeamSparseCutMarched];
        for (uint32_t level = 0; level < kBeamMaxLevels; ++level)
        {
            cloud["beamStart" + std::to_string(level)] = mBeamLevelCounts[kBeamSparseStarts + level];
            cloud["beamCandidates" + std::to_string(level)] = mBeamLevelCounts[level];
            cloud["beamAccepted" + std::to_string(level)] = mBeamLevelCounts[kBeamSparseAccepted + level];
            cloud["beamUniqueQueries" + std::to_string(level)] = mBeamLevelCounts[kBeamSparseLevelUnique + level];
        }
        props[kCloudStats] = cloud;
    }
    props[kBeamTolerance] = mParams.beamTolerance;
    props[kBeamGuide] = mParams.beamGuide;
    props[kBeamGuideDepth] = mParams.beamGuideDepth;
    props[kBeamGuideSun] = mParams.beamGuideSun;
    props[kBeamEdgeContrast] = mParams.beamEdgeContrast;
    props[kBeamOracle] = mParams.beamOracle;
    props[kBeamOracleBar] = mParams.beamOracleBar;
    props[kBeamCentreless] = mParams.beamCentreless != 0;
    props[kBeamGridDispatch] = mBeamGridDispatch;
    props[kBeamSparse] = mBeamSparse;
    props[kBeamSparseCut] = mBeamSparseCut;
    props[kBeamRefFrame] = mBeamRefFrame;
    props[kBeamRefMargin] = mBeamRefMargin;
    props[kBeamSparseMinLevel] = mBeamSparseMinLevel;
    props[kBeamRefresh] = mBeamRefresh;
    props[kBeamDepthTolerance] = mParams.beamDepthTolerance;
    props[kBeamShip] = mBeamShip;
    props[kBeamShipMask] = mBeamShipMask;
    props[kBeamQueue] = mBeamQueue;
    props[kBeamQueueSteps] = mParams.beamQueueSteps;
    props[kCloudResidencyFrozen] = mCloudResidencyFrozen;
    props[kCloudCutMargin] = mCloudCutMargin;
    props[kCloudCutTurn] = mCloudCutTurn;
    props[kCloudCutAsync] = mCloudCutAsync;
    props[kBeamRefreshDebug] = mParams.beamRefreshDebug;
    props[kBeamParallax] = mParams.beamParallax;
    props[kBeamCarryTolerance] = mParams.beamCarryTolerance;
    props[kBeamRefreshBlock] = mParams.beamRefreshBlock;
    props[kBeamRefreshCentres] = mParams.beamRefreshCentres;
    props[kStoreExact] = mStoreExact;
    props[kCompareExact] = mParams.compareExact != 0;
    props[kBeamMarchedFraction] = mBeamMarchedFraction;
    props[kReferenceShow] = mParams.referenceShow;
    props[kReferenceBandRows] = mReferenceBandRows;
    props[kCompareSubstitute] = mParams.compareSubstitute;
    props[kCompareTarget] = mParams.compareTarget;
    props[kReferenceError] = mReferenceError;
    props[kReferenceLogError] = mReferenceLogError;
    props[kReferenceNoiseError] = mReferenceNoiseError;
    props[kReferenceNoiseLogError] = mReferenceNoiseLogError;
    props[kReferenceLogP999] = mReferenceLogP999;
    props[kReferenceLogMax] = mReferenceLogMax;
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
    mpDomainCutPass = nullptr;
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
    mpBeamSparseResolvePass = nullptr;
    mpBeamMarchPass = nullptr;
    mpBeamClassifyPass = nullptr;
    mpBeamSparseEmitPass = nullptr;
    mpBeamSparseVerifyPass = nullptr;
    mpBeamSparseArgsPass = nullptr;
    mpBeamGridQueryPass = nullptr;
    mpBeamGridMarchPass = nullptr;
    mpBeamUnitMarchPass = nullptr;
    mpBeamQueueMarchPass = nullptr;
    mpBeamQueueArgsPass = nullptr;
    mpBeamQueueTilePass = nullptr;
    mpBeamQueuePixelPass = nullptr;
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
    mpCameraPass = createPass("renderCloudCamera");
    mpSolvePass = createPass("solveLeaves");
    mpCameraLightingPass = createPass("updateCameraLighting");
    mpProjectPass = createPass("projectCutNodes");
    mpCutPass = createPass("binCutNodes");
    mpSortPass = createPass("sortTileCuts");
    mpDomainCutPass = createPass("buildDomainCutNodes");
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
    mpBeamSparseResolvePass = createPass("resolveBeamSparse");
    mpBeamMarchPass = createPass("marchBeamPixels");
    mpBeamClassifyPass = createPass("classifyBeamRoots");
    mpBeamSparseEmitPass = createPass("emitBeamSparseQueries");
    mpBeamSparseVerifyPass = createPass("verifyBeamSparseTiles");
    mpBeamSparseArgsPass = createPass("writeBeamSparseArgs");
    mpBeamGridQueryPass = createPass("buildBeamGridQueries");
    mpBeamGridMarchPass = createPass("marchBeamGrid");
    mpBeamUnitMarchPass = createPass("marchBeamUnits");
    mpBeamQueueMarchPass = createPass("marchBeamQueue");
    mpBeamQueueArgsPass = createPass("writeBeamQueueArgs");
    mpBeamQueueTilePass = createPass("queueBeamTiles");
    mpBeamQueuePixelPass = createPass("marchBeamQueuePixels");
    mpBeamGuidePass = createPass("beamGuide");
    mpCommitCloudPass = createPass("commitCloudBricks");
    mpDecodeCloudPass = createPass("decodeCloudResiduals");
    mpOccupancyCloudPass = createPass("occupancyCloudBricks");
    mpClearDirtyCloudPagesPass = createPass("clearDirtyCloudPages");
    mpMarkDirtyCloudPagesPass = createPass("markDirtyCloudPages");
    mpCloudPageArgsPass = createPass("writeDirtyCloudPageArgs");
    mpResolveDirtyCloudPagesPass = createPass("resolveDirtyCloudPages");
    mpBakeCloudSunPass = createPass("bakeCloudSun");
    mpReleaseSunPass = createPass("releaseSunBakes");
    mpScanSunPass = createPass("scanSunBakes");
    mResidencyPassesBound.clear();
    mpDomainExtinctionPass = createPass("domainExtinction");
    mpDomainBlocksPass = createPass("domainBlocks");
    mpDomainMajorantPass = createPass("domainMajorant");
    mpDomainOccupancyPass = createPass("domainOccupancy");
    mpStampSunPass = createPass("stampSunChanges");
    mpResetSunBlocksPass = createPass("resetSunBlocks");
    mpResetSunNodesPass = createPass("resetSunNodes");
    mpAgeSunPass = createPass("ageSunField");
    mpSelectSunPass = createPass("selectSunBakes");
    mpEmitSunPass = createPass("emitSunBakes");
    mpEvictSunPass = createPass("evictSunBakes");
    mpAssignSunPass = createPass("assignSunBakes");
    mpClearWorldCacheTilesPass = createPass("clearWorldCacheTiles");
    mpAdvanceFadesPass = createPass("advanceCloudFades");
    mpDecayWorldCachePass = createPass("decayWorldCache");
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

    if (!mCloudLibraryPath.empty())
    {
        buildCloudDomain();
        return;
    }
    mpCloudResidency.reset();
    mpCloudSea.reset();
    mParams.cloudDomain = 0;
    mParams.cloudVirtual = 0;
    mParams.seaMode = 0;

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
    mParams.hstrCutFlat = 0; // A real hierarchy: ancestors and the Schur tree's BSP path decide the cut and its order.
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
    // Delta tracking re-reads the majorant at every block exit, so it only needs each block's own maximum: far tighter
    // than the dilated one next to dense cores, which spent most tentative collisions on null ones.
    std::vector<float16_t> tightMajorant(majorantCount);
    parallelFor(
        majorantCount,
        [&](size_t i)
        {
            float16_t rounded(blockMaximum[i]);
            if (float(rounded) < blockMaximum[i])
                rounded = float16_t(blockMaximum[i] * (1.f + 1.f / 1024.f));
            tightMajorant[i] = rounded;
        }
    );
    mpTightMajorant =
        mpDevice->createTexture3D(majorantDims.x, majorantDims.y, majorantDims.z, ResourceFormat::R16Float, 1, tightMajorant.data());
    // Occupancy of 4^3-block groups (16-voxel blocks), so delta tracking crosses empty space in 16-voxel jumps.
    const uint3 occupancyDims = (majorantDims + 3u) / 4u;
    std::vector<uint8_t> occupancy(size_t(occupancyDims.x) * occupancyDims.y * occupancyDims.z, 0);
    for (size_t i = 0; i < majorantCount; ++i)
    {
        if (blockMaximum[i] <= 0.f)
            continue;
        const uint3 b(
            uint32_t(i % majorantDims.x), uint32_t((i / majorantDims.x) % majorantDims.y), uint32_t(i / (majorantDims.x * majorantDims.y))
        );
        const uint3 o = b / 4u;
        occupancy[size_t(o.x) + size_t(occupancyDims.x) * (size_t(o.y) + size_t(occupancyDims.y) * o.z)] = 1;
    }
    mpOccupancy = mpDevice->createTexture3D(occupancyDims.x, occupancyDims.y, occupancyDims.z, ResourceFormat::R8Uint, 1, occupancy.data());
    const std::vector<uint8_t> majorantZero = majorantZeroBlocks(majorant, majorantDims);
    mpMajorantZero = mpDevice->createTexture3D(occupancyDims.x, occupancyDims.y, occupancyDims.z, ResourceFormat::R8Uint, 1, majorantZero.data());

    // Sun residual pages live at voxel resolution.
    mParams.hstrFineDims = dims;
    mResidualDirty = true;
    mBasisDirty = true;

    createSamplers();
}

void HSTRCloud::createSamplers()
{
    if (mpExtinctionSampler && mSamplerSeaMode == mParams.seaMode)
        return;
    mSamplerSeaMode = mParams.seaMode;
    // The sea wraps in X/Z. Density is zero outside the active-voxel bounds, so extinction lookups use a zero border elsewhere.
    const auto side = mParams.seaMode != 0 ? TextureAddressingMode::Wrap : TextureAddressingMode::Border;
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    samplerDesc.setAddressingMode(side, TextureAddressingMode::Border, side);
    samplerDesc.setBorderColor(float4(0.f));
    mpExtinctionSampler = mpDevice->createSampler(samplerDesc);
    const auto clampSide = mParams.seaMode != 0 ? TextureAddressingMode::Wrap : TextureAddressingMode::Clamp;
    samplerDesc.setAddressingMode(clampSide, TextureAddressingMode::Clamp, clampSide);
    mpLinearSampler = mpDevice->createSampler(samplerDesc);
    samplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpLinearClampSampler = mpDevice->createSampler(samplerDesc);
}

DomainStaging::DomainStaging(ref<Device> pDevice, uint32_t tileValues, uint32_t count) : mpFence(pDevice->createFence())
{
    mBuffers.resize(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        // Mapped once for its lifetime: the workers write through the pointer, never through the device.
        mBuffers[i].pBuffer = pDevice->createBuffer(size_t(tileValues) * sizeof(uint32_t), ResourceBindFlags::None, MemoryType::Upload);
        mBuffers[i].pData = static_cast<uint32_t*>(mBuffers[i].pBuffer->map());
        mFree.push_back(int32_t(i));
    }
}

int32_t DomainStaging::acquire(uint32_t*& data)
{
    std::lock_guard lock(mMutex);
    if (mFree.empty())
        return -1;
    const int32_t handle = mFree.back();
    mFree.pop_back();
    data = mBuffers[handle].pData;
    return handle;
}

void DomainStaging::discard(int32_t handle)
{
    std::lock_guard lock(mMutex);
    mFree.push_back(handle);
}

void DomainStaging::beginFrame(RenderContext* pRenderContext)
{
    // Last frame's command list is submitted, so a fence signalled now passes once the GPU ran its copies.
    if (!mRecorded.empty())
    {
        const uint64_t value = pRenderContext->signal(mpFence.get());
        for (int32_t handle : mRecorded)
            mInFlight.emplace_back(value, handle);
        mRecorded.clear();
    }
    if (mInFlight.empty())
        return;
    const uint64_t reached = mpFence->getCurrentValue();
    std::lock_guard lock(mMutex);
    auto keep = mInFlight.begin();
    for (auto it = mInFlight.begin(); it != mInFlight.end(); ++it)
        if (it->first <= reached)
            mFree.push_back(it->second);
        else
            *keep++ = *it;
    mInFlight.erase(keep, mInFlight.end());
}

void HSTRCloud::buildCloudDomain()
{
    const auto& volume = mpScene->getGridVolume(0);
    const AABB bounds = volume->getBounds();
    const float3 cameraPosition = mpScene->getCamera()->getPosition();
    const std::string seaKey = fmt::format(
        "{}|{}|{}|{}|{}|{}|{}",
        mCloudLibraryPath,
        mCloudSeaTiles,
        mCloudSeaSeed,
        mCloudSeaCoverage,
        mCloudProxyResolution,
        bounds.minPoint,
        bounds.extent()
    );
    if (!mpCloudSea || seaKey != mCloudSeaKey)
    {
        mpCloudResidency.reset();
        mpCloudSea.reset();
        hstrcloud::CloudLibrary library = timed("cloud library", [&] { return hstrcloud::loadCloudLibrary(mCloudLibraryPath); });
        mCloudLibraryFiles = library.files;
        hstrcloud::CloudSeaDesc seaDesc;
        // The scene's grid volume is only the carrier: its bounds give one tile's footprint and the cloud layer.
        seaDesc.origin = bounds.minPoint;
        seaDesc.tileWorld = bounds.extent().x;
        seaDesc.layerHeight = bounds.extent().y;
        seaDesc.tileVoxels = mCloudProxyResolution;
        seaDesc.tiles = mCloudSeaTiles;
        seaDesc.seed = mCloudSeaSeed;
        seaDesc.coverage = mCloudSeaCoverage;
        mpCloudSea = std::make_unique<hstrcloud::CloudSea>(std::move(library.assets), seaDesc);
        // Staging for three rows of tiles: a flight takes on at most a row and a column at a time. The load's full window
        // overflows it into host memory, which uploads from the main thread.
        const uint3 seaDims = mpCloudSea->getDims();
        mpDomainStaging = std::make_unique<DomainStaging>(mpDevice, seaDesc.tileVoxels * seaDims.y * seaDesc.tileVoxels, 3 * seaDesc.tiles);
        mpCloudSea->setStaging(mpDomainStaging.get());
        timed("cloud sea window", [&] { mpCloudSea->fill(cameraPosition); });
        mCloudSeaKey = seaKey;
        mCloudInstancesUploaded = false;
    }
    const std::string residencyKey = fmt::format(
        "{}|{}|{}|{}|{}|{}|{}",
        mCloudBrickPoolMB,
        mCloudBrickLoadsPerFrame,
        mCloudLodPixels,
        mCloudFadeFrames,
        mCloudPayloadPoolMB,
        mCloudDirectStorage,
        mCloudSunBakesPerFrame
    );
    if (!mCloudVirtual)
        mpCloudResidency.reset();
    else if (!mpCloudResidency || residencyKey != mCloudResidencyKey)
    {
        mpCloudResidency.reset();
        hstrcloud::CloudResidencyDesc residencyDesc;
        residencyDesc.files = mCloudLibraryFiles;
        residencyDesc.poolMB = mCloudBrickPoolMB;
        residencyDesc.loadsPerFrame = mCloudBrickLoadsPerFrame;
        residencyDesc.lodPixels = mCloudLodPixels;
        residencyDesc.fadeFrames = mCloudFadeFrames;
        residencyDesc.payloadPoolMB = mCloudPayloadPoolMB;
        residencyDesc.directStorage = mCloudDirectStorage;
        residencyDesc.sunBakesPerFrame = mCloudSunBakesPerFrame;
        residencyDesc.gpuSun = mCloudGpuSun;
        mpCloudResidency = std::make_unique<hstrcloud::CloudResidency>(mpDevice, *mpCloudSea, residencyDesc);
        mResidencyPassesBound.clear();
        mCloudResidencyKey = residencyKey;
        mCloudInstancesUploaded = false;
    }

    const auto& desc = mpCloudSea->getDesc();
    const uint3 dims = mpCloudSea->getDims();
    const float voxel = mpCloudSea->getVoxelWorld();
    mParams.cloudDomain = 1;
    mParams.seaMode = 1;
    mParams.cloudVirtual = mpCloudResidency ? 1u : 0u;
    mParams.seaOrigin = desc.origin;
    mParams.seaVoxelSize = float3(voxel);
    mParams.cloudTiles = uint2(desc.tiles);
    // The shader wraps a possibly negative tile coordinate into the instance grid. Where an axis' count is a power of two that wrap
    // is exactly an AND, which replaces two emulated integer modulos on the hottest path there is; ~0u keeps the general path.
    const uint32_t n = mParams.cloudTiles.x;
    mParams.cloudTileMask = n == mParams.cloudTiles.y && n != 0 && (n & (n - 1)) == 0 ? n - 1 : ~0u;
    mParams.cloudTileVoxels = desc.tileVoxels;
    mParams.cloudAtlasShift = mpCloudResidency ? mpCloudResidency->getAtlasShift() : 6u;
    if (mpCloudResidency)
    {
        const auto& atlas = mpCloudResidency->getAtlas();
        mParams.cloudAtlasInvSize = 1.f / float3(float(atlas->getWidth()), float(atlas->getHeight()), float(atlas->getDepth()));
        const auto& sunAtlas = mpCloudResidency->getSunAtlas();
        mParams.cloudSunAtlasInvSize = 1.f / float3(float(sunAtlas->getWidth()), float(sunAtlas->getHeight()), float(sunAtlas->getDepth()));
    }
    mVoxelSize = float3(voxel);
    constexpr uint32_t kCellWidth = 16;
    mGridMin = int3(0);
    mGridMax = int3(dims) - 1;
    mActualLeafDims = dims / kCellWidth;
    mParams.hstrLeafDims = mActualLeafDims;
    mParams.hstrGridMin = int3(0);
    mParams.hstrCellWidth = kCellWidth;
    mParams.hstrExtinctionDims = dims;
    mParams.hstrFineDims = dims;
    // The dynamic sea has no CPU Schur tree - its content is rasterized on worker threads into a toroidal proxy - but that proxy
    // already is a compact regular transport cut: one 16^3-voxel node per HSTR leaf, built and paged on the GPU
    // (buildDomainCutNodes) and then projected, binned and sorted by the existing camera pipeline. seaViewDistance is under half
    // the domain edge, so the camera-centred window of these cells tiles everything a ray can reach exactly once.
    const uint32_t domainNodeCount = mActualLeafDims.x * mActualLeafDims.y * mActualLeafDims.z;
    mParams.hstrNodeCount = domainNodeCount;
    mParams.hstrRootNode = ~0u;
    mParams.hstrCutFlat = 1;
    mDomainCutOriginValid = false;
    const auto cutFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    mpCutNodes = mpDevice->createStructuredBuffer(
        sizeof(HSTRCutNode), domainNodeCount, cutFlags, MemoryType::DeviceLocal, nullptr, false
    );
    std::vector<uint32_t> domainParents(domainNodeCount, ~0u);
    mpCutNodeParents = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), domainNodeCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, domainParents.data(), false
    );
    // The sea's frame is the world cache march (per pixel or through beam tiles), and each tile counts its own light-tracing
    // batches, so the photon pool (which counts emitted photons globally) is not used.
    if (mParams.debugView != kReferenceView && mParams.debugView != kWorldCacheView && mParams.debugView != kBeamView)
    {
        logWarning("HSTRCloud: the cloud sea renders through the world cache views; debugView {} becomes {}.", mParams.debugView, kBeamView);
        mParams.debugView = kBeamView;
    }
    mParams.worldCacheSegments = 0;
    // The window reaches at least tiles / 2 - 1 tiles from the camera in every direction; beyond, tiles are stale or wrapped.
    mParams.seaViewDistance = std::min(mSeaViewDistance, std::max(1.f, float(desc.tiles / 2 - 1)) * desc.tileWorld);
    const uint32_t tileCount = desc.tiles * desc.tiles;
    mCloudTileBatches.assign(tileCount, 0.f);
    mCloudTileReset.assign(tileCount, 0u);
    mpCloudTileBatches = mpDevice->createStructuredBuffer(
        sizeof(float), tileCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, mCloudTileBatches.data(), false
    );
    mpCloudTileReset = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), tileCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, mCloudTileReset.data(), false
    );
    mDomainPassesBound = false; // The world cache clear reads the tile resets.
    std::vector<uint32_t> slots(tileCount);
    std::iota(slots.begin(), slots.end(), 0u);
    uploadDomainExtinction(slots);
    // Tile changes dispatch this pass mid-flight: its kernels compile now, not in the first frame that takes on a tile (150 ms).
    mpClearWorldCacheTilesPass->warm();
    mpAdvanceFadesPass->warm();
    updateSunVoxelDirection();
    createSamplers();
    mpLeafRadiance = nullptr;
    mParams.worldCacheSamples = 0;
    mParams.referenceSamples = 0;
    mResidualDirty = true;
    mBasisDirty = true;
    mCutDirty = true;
    mBeamReusable = false;
    logInfo("HSTRCloud: cloud sea domain {} voxels of {:.2f} world units ({} fine density).", dims, voxel, mpCloudResidency ? "virtual" : "proxy");
}

void HSTRCloud::uploadDomainExtinction(const std::vector<uint32_t>& slots)
{
    // The domain proxy of the changed slots: their float16 volumes (converted on the sea's workers, straight into staging buffers)
    // copy to the GPU, which scatters them and derives the extinction, block maxima, majorants and occupancy around them
    // (domainExtinction ... domainOccupancy), a batch of slots at a time.
    // MEASURED before (4K sea flight at 20 units a frame): building all of it on the CPU cost 2.1 ms a frame on average and up to
    // 51 ms in a frame that took on a row of tiles - float16 conversion 17, block maxima 10, the whole majorant 15.5. Uploading
    // the volumes from the main thread still cost 5.4 ms for a row, and binding the rarely run passes 0.7 ms each.
    auto& sea = *mpCloudSea;
    const uint3 dims = sea.getDims();
    const uint32_t r = sea.getDesc().tileVoxels;
    const uint32_t tiles = sea.getDesc().tiles;
    const uint3 blockDims = dims / kDomainBlock;
    const uint3 occupancyDims = (blockDims + kDomainOccupancy - 1u) / kDomainOccupancy;
    const size_t tileBytes = size_t(r) * dims.y * r * sizeof(uint32_t);
    auto* pRenderContext = mpDevice->getRenderContext();
    const auto readWrite = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto ensure = [&](ref<Texture>& texture, uint3 size, ResourceFormat format)
    {
        if (!texture || texture->getWidth() != size.x || texture->getHeight() != size.y || texture->getDepth() != size.z ||
            !is_set(texture->getBindFlags(), ResourceBindFlags::UnorderedAccess))
        {
            texture = mpDevice->createTexture3D(size.x, size.y, size.z, format, 1, nullptr, readWrite);
            mDomainPassesBound = false;
        }
    };
    ensure(mpDomainVolume, dims, ResourceFormat::RG16Float);
    ensure(mpExtinction, dims, ResourceFormat::R16Float);
    ensure(mpDomainBlocks, blockDims, ResourceFormat::RG16Float);
    ensure(mpMajorant, blockDims, ResourceFormat::R16Float);
    ensure(mpTightMajorant, blockDims, ResourceFormat::R16Float);
    ensure(mpOccupancy, occupancyDims, ResourceFormat::R8Uint);
    ensure(mpMajorantZero, occupancyDims, ResourceFormat::R8Uint);
    mParams.hstrMajorantDims = blockDims;
    if (!mpDomainRegions)
    {
        mpDomainRegions = mpDevice->createStructuredBuffer(sizeof(uint32_t), kDomainBatch, ResourceBindFlags::ShaderResource);
        mpDomainFrame = mpDevice->createStructuredBuffer(sizeof(HSTRCloudDomainFrame), 1, ResourceBindFlags::ShaderResource);
        mDomainPassesBound = false;
    }
    if (!mpDomainStaged || mpDomainStaged->getSize() != kDomainBatch * tileBytes)
    {
        mpDomainStaged = mpDevice->createBuffer(kDomainBatch * tileBytes, ResourceBindFlags::ShaderResource);
        mDomainPassesBound = false;
    }
    if (!mDomainPassesBound)
    {
        // Each stage binds only what it reads and writes (an output must not also be bound as the stage's input), once: the
        // resources persist and only their contents change.
        auto bind = [&](const ref<ComputePass>& pPass, std::initializer_list<std::pair<const char*, ref<Resource>>> resources)
        {
            ShaderVar var = pPass->getRootVar()["CB"]["gHSTRCloud"];
            var["hstrDomainRegions"] = mpDomainRegions;
            var["hstrDomainFrame"] = mpDomainFrame;
            for (const auto& [name, pResource] : resources)
                if (auto pTexture = pResource->asTexture())
                    var[name] = pTexture;
                else
                    var[name] = pResource->asBuffer();
        };
        bind(mpDomainExtinctionPass, {{"hstrDomainStaged", mpDomainStaged}, {"hstrDomainVolumeOutput", mpDomainVolume}, {"hstrExtinctionOutput", mpExtinction}});
        bind(mpDomainBlocksPass, {{"hstrDomainVolume", mpDomainVolume}, {"hstrDomainBlocksOutput", mpDomainBlocks}});
        bind(mpDomainMajorantPass, {{"hstrDomainBlocks", mpDomainBlocks}, {"hstrMajorantOutput", mpMajorant}, {"hstrTightMajorantOutput", mpTightMajorant}});
        bind(mpDomainOccupancyPass,
             {{"hstrDomainBlocks", mpDomainBlocks}, {"hstrMajorant", mpMajorant}, {"hstrOccupancyOutput", mpOccupancy}, {"hstrMajorantZeroOutput", mpMajorantZero}});
        bind(mpClearWorldCacheTilesPass, {{"hstrCloudTileReset", mpCloudTileReset}});
        mpClearBoundDeposit = nullptr;
        mDomainPassesBound = true;
    }
    HSTRCloudDomainFrame frame = {};
    frame.scale = mpScene->getGridVolume(0)->getDensityScale() * mParams.densityScale;
    frame.tileVoxels = r;
    frame.tiles = tiles;
    frame.dims = dims;
    frame.blockDims = blockDims;
    frame.cacheDims = mParams.worldCacheDims;
    frame.cacheCellVoxels = mParams.worldCacheCellVoxels;
    if (slots.empty())
    {
        mpDomainFrame->setBlob(&frame, 0, sizeof(frame));
        return;
    }
    const uint32_t blocksWide = r / kDomainBlock + 1;
    const uint32_t majorantWide = r / kDomainBlock + 3;
    const uint32_t cellsWide = r / (kDomainBlock * kDomainOccupancy) + 2;
    for (size_t first = 0; first < slots.size(); first += kDomainBatch)
    {
        const uint32_t count = uint32_t(std::min<size_t>(slots.size() - first, kDomainBatch));
        std::array<uint32_t, kDomainBatch> regions;
        {
            // Region i's volume is at i tile volumes into the batch buffer. A staged tile costs a recorded copy; a tile in host
            // memory (the staging was full) uploads here.
            FALCOR_PROFILE(pRenderContext, "volumes");
            for (uint32_t i = 0; i < count; ++i)
            {
                const uint32_t slot = slots[first + i];
                auto& volume = sea.getUploadVolume(slot);
                using Upload = hstrcloud::CloudSea::TileVolume::Upload;
                regions[i] = slot;
                if (volume.upload == Upload::Staged)
                {
                    pRenderContext->copyBufferRegion(mpDomainStaged.get(), i * tileBytes, mpDomainStaging->getBuffer(volume.staged).get(), 0, tileBytes);
                    mpDomainStaging->recorded(volume.staged);
                    volume.staged = -1;
                }
                else if (volume.upload == Upload::Packed)
                {
                    mpDomainStaged->setBlob(volume.packed.data(), i * tileBytes, tileBytes);
                    volume.packed = {};
                }
                else
                    regions[i] |= volume.upload == Upload::Zero ? kDomainRegionZero : kDomainRegionKeep;
                volume.upload = Upload::Done;
            }
        }
        frame.regionCount = count;
        mpDomainRegions->setBlob(regions.data(), 0, count * sizeof(uint32_t));
        mpDomainFrame->setBlob(&frame, 0, sizeof(frame));
        auto run = [&](const char* name, const ref<ComputePass>& pPass, uint3 threads)
        {
            FALCOR_PROFILE(pRenderContext, name);
            pPass->execute(pRenderContext, threads);
        };
        run("extinction", mpDomainExtinctionPass, uint3(r, dims.y, r * count));
        run("blocks", mpDomainBlocksPass, uint3(blocksWide, blockDims.y, blocksWide * count));
        run("majorant", mpDomainMajorantPass, uint3(majorantWide, blockDims.y, majorantWide * count));
        run("occupancy", mpDomainOccupancyPass, uint3(cellsWide, occupancyDims.y, cellsWide * count));
    }

    // The occupied vertical band, in world units, dilated by one block to cover the camera majorant's own dilation. A sea ray was
    // marched between the floor and ceiling of the whole extinction grid, so every ray crossed the empty sky above and below the
    // clouds to reach the domain boundary. majorantZeroExit skips that at 16-voxel granularity, which costs one texture load per
    // block rather than a step per voxel - 17.81 of the sea's 30 steps a pixel - but clamping the slab removes the blocks from the
    // ray instead of skipping them, and it cannot change a pixel: there is no density outside the band by construction.
    // A voxel layer y is in the blocks whose trilinear support [4 b, 4 b + 4] holds it, so the tiles' content layers give the
    // non-zero blocks' range.
    int32_t lowVoxel = std::numeric_limits<int32_t>::max();
    int32_t highVoxel = -1;
    for (const auto& tile : sea.getTiles())
        if (tile.contentLow >= 0)
        {
            lowVoxel = std::min(lowVoxel, tile.contentLow);
            highVoxel = std::max(highVoxel, tile.contentHigh);
        }
    const float gridTopY = mParams.seaOrigin.y + float(mParams.hstrExtinctionDims.y) * mParams.seaVoxelSize.y;
    float2 band;
    if (highVoxel < 0) // No density anywhere: leave the slab alone rather than inverting it.
        band = float2(mParams.seaOrigin.y, gridTopY);
    else
    {
        const uint32_t lowBlock = uint32_t(std::max(lowVoxel - 1, 0)) / kDomainBlock; // ceil((y - 4) / 4), at least 0.
        const uint32_t highBlock = std::min(uint32_t(highVoxel) / kDomainBlock, blockDims.y - 1);
        const float blockVoxels = float(kDomainBlock); // A majorant block is four DOMAIN voxels: majorantAt indexes with floor(v * 0.25).
        const float low = float(lowBlock > 0 ? lowBlock - 1 : 0) * blockVoxels;
        const float high = float(std::min(highBlock + 2u, blockDims.y)) * blockVoxels;
        band = mParams.seaOrigin.y + float2(low, high) * mParams.seaVoxelSize.y;
    }
    if (any(band != mSeaContentBand))
    {
        mSeaContentBand = band;
        mParams.seaContentY = mSeaContentBand;
        logInfo(
            "HSTRCloud: sea content band y {} to {} of {} to {} ({} of the grid's height).", mSeaContentBand.x, mSeaContentBand.y,
            mParams.seaOrigin.y, gridTopY, (mSeaContentBand.y - mSeaContentBand.x) / std::max(1e-6f, gridTopY - mParams.seaOrigin.y)
        );
    }
}

void HSTRCloud::updateCloudDomain(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "cloudSea");
    mpDomainStaging->beginFrame(pRenderContext);
    const auto& camera = mpScene->getCamera();
    std::vector<uint32_t> changed;
    {
        FALCOR_PROFILE(pRenderContext, "seaTiles");
        // A residency cut running on its worker reads the tiles: finished tiles wait until it is back.
        changed = mpCloudSea->update(camera->getPosition(), !(mpCloudResidency && mpCloudResidency->cutInFlight()));
    }
    if (!mCloudInstancesUploaded)
    {
        changed.resize(mCloudTileBatches.size());
        std::iota(changed.begin(), changed.end(), 0u);
        mCloudInstancesUploaded = true;
    }
    if (!changed.empty())
    {
        FALCOR_PROFILE(pRenderContext, "tileUpload");
        uploadDomainExtinction(changed);
        // The world cache of tiles whose cloud changed restarts; the rest keeps converging.
        std::fill(mCloudTileReset.begin(), mCloudTileReset.end(), 0u);
        for (uint32_t slot : changed)
        {
            mCloudTileReset[slot] = 1;
            mCloudTileBatches[slot] = 0.f;
        }
        mpCloudTileReset->setBlob(mCloudTileReset.data(), 0, mCloudTileReset.size() * sizeof(uint32_t));
        mpCloudTileBatches->setBlob(mCloudTileBatches.data(), 0, mCloudTileBatches.size() * sizeof(float));
        if (mpWorldCacheDeposit && all(mParams.worldCacheDims > uint3(0)))
        {
            // The clear reads the domain frame and the tile resets, bound with the domain passes (uploadDomainExtinction).
            FALCOR_PROFILE(pRenderContext, "clearCache");
            if (mpClearBoundDeposit != mpWorldCacheDeposit)
            {
                mpClearWorldCacheTilesPass->getRootVar()["CB"]["gHSTRCloud"]["hstrWorldCacheDeposit"] = mpWorldCacheDeposit;
                mpClearBoundDeposit = mpWorldCacheDeposit;
            }
            mpClearWorldCacheTilesPass->execute(pRenderContext, mParams.worldCacheDims);
        }
        mWorldCacheBakeDirty = true;
        mCutDirty = true; // Domain pages and their non-empty projected cells changed.
        FALCOR_PROFILE(pRenderContext, "sunPages");
        // Sun pages: the changed tiles and every tile whose sun rays cross them (up to the layer height along the sun).
        const float3 sun = normalize(mParams.sunDirection);
        const auto& seaDesc = mpCloudSea->getDesc();
        const float reach = std::abs(sun.y) > 1e-3f ? seaDesc.layerHeight * length(sun.xz()) / std::abs(sun.y) : 1e9f;
        const int32_t n = int32_t(seaDesc.tiles);
        const int32_t steps = std::min(n, int32_t(std::ceil(reach / seaDesc.tileWorld)) + 1);
        const float2 downwind = length(sun.xz()) > 1e-6f ? -normalize(sun.xz()) * (sun.y >= 0.f ? 1.f : -1.f) : float2(0.f);
        std::vector<uint8_t> stale(mCloudTileBatches.size(), 0);
        for (uint32_t slot : mSunPageSlots)
            stale[slot] = 1;
        for (uint32_t slot : changed)
            for (int32_t k = 0; k <= steps; ++k)
                for (int32_t side = -1; side <= 1; ++side)
                {
                    const float2 p = float2(float(slot % n), float(slot / n)) + downwind * float(k) + float2(-downwind.y, downwind.x) * float(side);
                    const int32_t x = ((int32_t(std::floor(p.x + 0.5f)) % n) + n) % n;
                    const int32_t z = ((int32_t(std::floor(p.y + 0.5f)) % n) + n) % n;
                    stale[x + n * z] = 1;
                }
        mSunPageSlots.clear();
        for (uint32_t slot = 0; slot < stale.size(); ++slot)
            if (stale[slot])
                mSunPageSlots.push_back(slot);
        mSunPagesDirty = true;
        mBeamReusable = false;
    }
    if (!mpCloudResidency)
        return;
    hstrcloud::CloudView view;
    view.position = camera->getPosition();
    view.viewProjection = camera->getViewProjMatrixNoJitter();
    mParams.cloudPixelAngle = camera->getFrameHeight() / camera->getFocalLength() / float(std::max(1u, mParams.frameDim.y));
    view.pixelAngle = mParams.cloudPixelAngle;
    view.lodBias = mParams.cloudLodBias;
    view.sunDirection = normalize(mParams.sunDirection);
    view.sunReach = (mParams.sunNearVoxels + 1.f) * mpCloudSea->getVoxelWorld();
    // Baked sun depth is valid for one sun direction, density scale and reach. Moving the sun more than cloudSunBakeAngle from the
    // bake direction starts a new generation that rebakes in place, most important bricks first, while the older bakes keep
    // answering; any other change starts one that nothing older answers for (the live march covers the bricks until they rebake).
    const float3 sun = normalize(mParams.sunDirection);
    const float densityScale = mpScene->getGridVolume(0)->getDensityScale() * mParams.densityScale;
    const bool sunReset = densityScale != mCloudSunBakeInputs.w || mParams.sunNearVoxels != mCloudSunBakeNear;
    if (sunReset || dot(sun, mCloudSunBakeInputs.xyz()) < std::cos(math::radians(mCloudSunBakeAngle)))
    {
        mCloudSunBakeInputs = float4(sun, densityScale);
        mCloudSunBakeNear = mParams.sunNearVoxels;
        mParams.cloudSunGeneration = (mParams.cloudSunGeneration + 1) & 0x0FFFFFFF;
        if (sunReset)
            mParams.cloudSunOldestGeneration = mParams.cloudSunGeneration;
    }
    view.sunNearVoxels = mParams.sunNearVoxels;
    view.sunGeneration = mParams.cloudSunGeneration;
    view.sunBakeDirection = mCloudSunBakeInputs.xyz();
    view.densityScale = mpScene->getGridVolume(0)->getDensityScale() * mParams.densityScale;
    view.maxDistance = mParams.seaViewDistance;
    view.cutMargin = mCloudCutMargin;
    view.cutTurn = mCloudCutTurn;
    view.cutAsync = mCloudCutAsync;
    bool densityChanged = false;
    // Frozen (benchmarks only): the resident set stays as it is. A moving camera re-runs the whole residency cut on the CPU - over
    // 100 ms a frame on the sea - and while the GPU waits it drops to a lower power state, so its moving-camera timings measured
    // the laptop's power management at five times the settled cost rather than the frame.
    if (!mCloudResidencyFrozen)
    {
        FALCOR_PROFILE(pRenderContext, "residency");
        densityChanged = mpCloudResidency->update(*mpCloudSea, view, changed);
        if (mpCloudResidency->fadesRunning())
        {
            FALCOR_PROFILE(pRenderContext, "fades");
            if (std::find(mResidencyPassesBound.begin(), mResidencyPassesBound.end(), mpAdvanceFadesPass.get()) == mResidencyPassesBound.end())
            {
                // Only the bricks and the fade frame, which live as long as the residency.
                ShaderVar var = mpAdvanceFadesPass->getRootVar()["CB"]["gHSTRCloud"];
                var["hstrCloudBricksOutput"] = mpCloudResidency->getBricks();
                var["hstrCloudFadeFrame"] = mpCloudResidency->getFadeFrame();
                mResidencyPassesBound.push_back(mpAdvanceFadesPass.get());
            }
            mpAdvanceFadesPass->execute(pRenderContext, uint3(mpCloudResidency->getBrickCapacity(), 1, 1));
        }
    }
    // The staged bricks' residuals decode from the payload pool in one dispatch; then one reconstruction dispatch per level,
    // coarsest first: every brick predicts from a parent already in the atlas.
    if (!mpCloudResidency->getCommitGroups().empty())
    {
        FALCOR_PROFILE(pRenderContext, "commitBricks");
        mParams.cloudStagedCount = mpCloudResidency->getStagedCount();
        bindResidencyPass(pRenderContext, mpDecodeCloudPass, false);
        mpDecodeCloudPass->execute(pRenderContext, uint3(mParams.cloudStagedCount, 1, 1));
        mParams.cloudStagedCount = 0;
        for (const auto& group : mpCloudResidency->getCommitGroups())
        {
            mParams.cloudCommitOffset = group.offset;
            mParams.cloudCommitCount = group.count;
            if (bindResidencyPass(pRenderContext, mpCommitCloudPass, false))
                bindOutput(mpCommitCloudPass, "hstrCloudAtlasOutput", mpCloudResidency->getAtlas(), "hstrCloudAtlas");
            mpCommitCloudPass->execute(pRenderContext, uint3(10, 10, 10 * group.count));
        }
        mParams.cloudCommitCount = 0;
        // Then their occupancy, from the reconstructed atlas texels, for the camera marches' empty-cell skipping.
        mParams.cloudStagedCount = mpCloudResidency->getStagedCount();
        if (bindResidencyPass(pRenderContext, mpOccupancyCloudPass, false))
        {
            ShaderVar occupancyVar = mpOccupancyCloudPass->getRootVar()["CB"]["gHSTRCloud"];
            occupancyVar["hstrCloudOccupancy"] = ref<Buffer>();
            occupancyVar["hstrCloudOccupancyOutput"] = mpCloudResidency->getOccupancy();
        }
        mpOccupancyCloudPass->execute(pRenderContext, uint3(mParams.cloudStagedCount, 1, 1));
        mParams.cloudStagedCount = 0;
    }
    // The CPU hierarchy is canonical. Regions touched by paint/replace are marked into a persistent bitmap, atomically compacted,
    // then resolved exactly once per unique page. Clearing follows last frame's compact queue, never the full virtual address space.
    {
        FALCOR_PROFILE(pRenderContext, "updateCloudPages");
        auto& residency = *mpCloudResidency;
        if (bindResidencyPass(pRenderContext, mpClearDirtyCloudPagesPass, false))
            residency.bindPageUpdates(mpClearDirtyCloudPagesPass->getRootVar()["CB"]["gHSTRCloud"]);
        mpClearDirtyCloudPagesPass->executeIndirect(pRenderContext, residency.getDirtyPageArgs().get(), 0);
        pRenderContext->clearUAV(residency.getDirtyPageCount()->getUAV().get(), uint4(0));

        mParams.cloudPageRegionCount = residency.getDirtyPageRegionCount();
        if (mParams.cloudPageRegionCount > 0)
        {
            if (bindResidencyPass(pRenderContext, mpMarkDirtyCloudPagesPass, false))
                residency.bindPageUpdates(mpMarkDirtyCloudPagesPass->getRootVar()["CB"]["gHSTRCloud"]);
            mpMarkDirtyCloudPagesPass->execute(pRenderContext, uint3(residency.getDirtyPageWorkCount(), 1, 1));
        }
        if (bindResidencyPass(pRenderContext, mpCloudPageArgsPass, false))
            residency.bindPageUpdates(mpCloudPageArgsPass->getRootVar()["CB"]["gHSTRCloud"]);
        mpCloudPageArgsPass->getRootVar()["CB"]["gHSTRCloud"]["hstrCloudDirtyPageArgs"] = residency.getDirtyPageArgs();
        mpCloudPageArgsPass->execute(pRenderContext, uint3(1));
        if (mParams.cloudPageRegionCount > 0)
        {
            if (bindResidencyPass(pRenderContext, mpResolveDirtyCloudPagesPass, false))
                residency.bindPageUpdates(mpResolveDirtyCloudPagesPass->getRootVar()["CB"]["gHSTRCloud"]);
            ShaderVar pageVar = mpResolveDirtyCloudPagesPass->getRootVar()["CB"]["gHSTRCloud"];
            pageVar["hstrCloudPages"] = ref<Buffer>();
            pageVar["hstrCloudPagesOutput"] = residency.getPages();
            mpResolveDirtyCloudPagesPass->executeIndirect(pRenderContext, residency.getDirtyPageArgs().get(), 0);
        }
        mParams.cloudPageRegionCount = 0;
        residency.pageUpdatesDispatched();
    }
    // Sun bakes last: they read the bricks and occupancy committed above.
    if (mCloudGpuSun)
    {
        dispatchSunScheduling(pRenderContext);
        mpCloudResidency->gpuSunDispatched();
    }
    else if (const uint32_t bakes = mpCloudResidency->getSunBakeCount(); bakes > 0)
    {
        FALCOR_PROFILE(pRenderContext, "bakeCloudSun");
        mParams.cloudCommitOffset = 0;
        mParams.cloudCommitCount = bakes;
        bindRenderer(pRenderContext, mpBakeCloudSunPass);
        bindOutput(mpBakeCloudSunPass, "hstrCloudSunAtlasOutput", mpCloudResidency->getSunAtlas(), "hstrCloudSunAtlas");
        mpBakeCloudSunPass->execute(pRenderContext, uint3(10, 10, 10 * bakes));
        mParams.cloudCommitCount = 0;
        mBeamReusable = false;
    }
    if (mpCloudResidency->getStats().sunBakesFrame > 0)
        ++mSunBakeFrames;
    mDensityChangedFrame = densityChanged;
    if (densityChanged)
    {
        mBeamReusable = false;
        ++mDensityChangedFrames;
    }
    if (++mCloudFrames % 240 == 0)
    {
        const auto& stats = mpCloudResidency->getStats();
        logInfo(
            "HSTRCloud: sea frame {}: {} desired, {} loaded, {} mapped, {} pending bricks, {} tiles pending, {:.0f} MB resident, cut {:.2f} ms.",
            mCloudFrames,
            stats.desired,
            stats.loaded,
            stats.mapped,
            stats.pending,
            mpCloudSea->pendingTiles(),
            stats.residentMB,
            stats.cutMilliseconds
        );
    }
}

bool HSTRCloud::bindResidencyPass(RenderContext* pRenderContext, const ref<ComputePass>& pPass, bool scene)
{
    // A residency pass reads the residency's buffers, the samplers and the params (and the scene, for its density scale). It binds
    // them once: every other binding bindRenderer makes would cost each dispatch its resource state tracking, and a params change
    // rebuilds all of a pass's bindings. A full bind per dispatch cost 0.1-0.3 ms of CPU each.
    ShaderVar var = pPass->getRootVar()["CB"]["gHSTRCloud"];
    const bool first = std::find(mResidencyPassesBound.begin(), mResidencyPassesBound.end(), pPass.get()) == mResidencyPassesBound.end();
    if (first)
    {
        if (scene)
            mpScene->bindShaderDataForRaytracing(pRenderContext, pPass->getRootVar()["gScene"]);
        mpCloudResidency->bind(var);
        var["hstrLinearClampSampler"] = mpLinearClampSampler;
        var["hstrLinearSampler"] = mpLinearSampler;
        mResidencyPassesBound.push_back(pPass.get());
    }
    var["params"].setBlob(mParams);
    return first;
}

void HSTRCloud::dispatchSunScheduling(RenderContext* pRenderContext)
{
    auto& residency = *mpCloudResidency;
    // The bake count of an earlier run, read back without waiting: a run that staged nothing lets the scheduler idle. The copy was
    // recorded in an earlier frame, whose end submitted it, so the fence is signalled now rather than by flushing then.
    if (mSunReadbackRecorded)
    {
        mSunReadbackPending = pRenderContext->signal(mpSunFence.get());
        mSunReadbackRecorded = false;
    }
    if (mSunReadbackPending != 0 && mpSunFence->getCurrentValue() >= mSunReadbackPending)
    {
        const uint32_t bakes = *static_cast<const uint32_t*>(mpSunReadback->map());
        mpSunReadback->unmap();
        residency.gpuSunRead(bakes);
        mSunReadbackPending = 0;
    }
    const auto& sun = residency.getGpuSunFrame();
    const HSTRCloudSunFrame& info = sun.info;
    FALCOR_PROFILE(pRenderContext, "scheduleSun");
    // The scheduler passes read only the residency's buffers (their frame included, HSTRCloudSunFrame, so no params), which live as
    // long as it: each pass binds them once (as bindResidencyPass does) and then dispatches with nothing changed.
    auto run = [&](const char* name, const ref<ComputePass>& pPass, uint3 threads)
    {
        FALCOR_PROFILE(pRenderContext, name);
        if (std::find(mResidencyPassesBound.begin(), mResidencyPassesBound.end(), pPass.get()) == mResidencyPassesBound.end())
        {
            // Only the residency's buffers: every other binding would still cost each dispatch its resource state tracking.
            ShaderVar var = pPass->getRootVar()["CB"]["gHSTRCloud"];
            residency.bind(var);
            residency.bindGpuSun(var);
            mResidencyPassesBound.push_back(pPass.get());
        }
        pPass->execute(pRenderContext, threads);
    };
    // The invalidation field ages every frame, idle or not.
    if (info.ageRows > 0)
        run("age", mpAgeSunPass, uint3(kCloudSunFieldWidth, info.ageRows, 1));
    if (!sun.run)
        return; // The counters keep the last run's (the stats read them).
    pRenderContext->clearUAV(residency.getGpuSunCounters()->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(residency.getGpuSunHistogram()->getUAV().get(), uint4(0));
    if (info.releaseCount > 0)
        run("release", mpReleaseSunPass, uint3(info.releaseCount, 1, 1));
    // New blocks and nodes restart before this frame's changes stamp over them.
    if (info.blockResets > 0)
        run("resetBlocks", mpResetSunBlocksPass, uint3(info.blockLength, info.blockResets, 1));
    if (info.nodeResets > 0)
        run("resetNodes", mpResetSunNodesPass, uint3(kCloudSunFineEntries, info.nodeResets, 1));
    if (info.changeCount > 0)
        run("stamp", mpStampSunPass, uint3(info.changeCount, 1, 1));
    run("scan", mpScanSunPass, uint3(info.capacity, 1, 1));
    run("select", mpSelectSunPass, uint3(1));
    run("emit", mpEmitSunPass, uint3(info.capacity, 1, 1));
    run("evict", mpEvictSunPass, uint3(info.bakeMax, 1, 1));
    run("assign", mpAssignSunPass, uint3(info.bakeMax, 1, 1));
    // The jobs assign wrote; entries without a slot return at once.
    {
        FALCOR_PROFILE(pRenderContext, "bakeCloudSun");
        mParams.cloudCommitOffset = 0;
        mParams.cloudCommitCount = info.bakeMax;
        if (bindResidencyPass(pRenderContext, mpBakeCloudSunPass, true))
            bindOutput(mpBakeCloudSunPass, "hstrCloudSunAtlasOutput", residency.getSunAtlas(), "hstrCloudSunAtlas");
        mpBakeCloudSunPass->execute(pRenderContext, uint3(10, 10, 10 * info.bakeMax));
        mParams.cloudCommitCount = 0;
    }
    mBeamReusable = false;
    if (mSunReadbackPending == 0 && !mSunReadbackRecorded)
    {
        if (!mpSunReadback)
        {
            mpSunReadback = mpDevice->createBuffer(sizeof(uint32_t), ResourceBindFlags::None, MemoryType::ReadBack);
            mpSunFence = mpDevice->createFence();
        }
        pRenderContext->copyBufferRegion(mpSunReadback.get(), 0, residency.getGpuSunCounters().get(), 2 * sizeof(uint32_t), sizeof(uint32_t));
        residency.gpuSunQueued();
        mSunReadbackRecorded = true;
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

void HSTRCloud::updateSunVoxelDirection()
{
    // One voxel step towards the sun for the sub-voxel sun reconstruction (the grid axes are world aligned).
    const float3 sunVoxels = normalize(mParams.sunDirection) / mVoxelSize;
    mParams.sunVoxelDirection = sunVoxels / length(sunVoxels);
    mParams.sunVoxelWorldLength = 1.f / length(sunVoxels);
}

void HSTRCloud::onLightingChanged(const HSTRCloudParams& previous)
{
    if (mParams.cloudDomain == 0 || !mpCloudSea)
    {
        solveLighting();
        return;
    }
    // The sea amortizes lighting changes. Its cache holds unit sun and sky fields coloured at lookup, so colours need nothing; a
    // new sun direction decays the cache by the angle turned (new photons replace the old estimate over the next frames) and
    // refreshes the sun pages tile by tile, nearest first.
    updateSunVoxelDirection();
    const float cosine = dot(normalize(previous.sunDirection), normalize(mParams.sunDirection));
    if (cosine > 0.999999f)
        return;
    const float angle = std::acos(std::clamp(cosine, -1.f, 1.f));
    mCloudCacheKeep = std::min(mCloudCacheKeep, std::exp(-angle / 0.05f));
    const auto& desc = mpCloudSea->getDesc();
    const int32_t n = int32_t(desc.tiles);
    const float3 camera = mpScene->getCamera()->getPosition();
    const int2 cameraSlot(
        ((int32_t(std::floor((camera.x - desc.origin.x) / desc.tileWorld)) % n) + n) % n,
        ((int32_t(std::floor((camera.z - desc.origin.z) / desc.tileWorld)) % n) + n) % n
    );
    mSunPageQueue.resize(size_t(n) * n);
    std::iota(mSunPageQueue.begin(), mSunPageQueue.end(), 0u);
    auto torusDistance = [&](uint32_t slot)
    {
        const int32_t dx = std::abs(int32_t(slot % n) - cameraSlot.x);
        const int32_t dz = std::abs(int32_t(slot / n) - cameraSlot.y);
        return std::max(std::min(dx, n - dx), std::min(dz, n - dz));
    };
    std::stable_sort(mSunPageQueue.begin(), mSunPageQueue.end(), [&](uint32_t a, uint32_t b) { return torusDistance(a) < torusDistance(b); });
}

void HSTRCloud::solveLighting()
{
    if (mParams.cloudDomain != 0)
    {
        // The sea has no Schur hierarchy: its transport is the world cache and the sun pages, rebuilt for the new lights.
        //
        // updateSunVoxelDirection has just queued EVERY tile, nearest to the camera first, for the throttle at the top of execute
        // to spread over mCloudSunTilesPerFrame frames. Setting mResidualDirty here threw that queue away (see the else branch
        // beside the throttle) and rebuilt the whole fine-sun volume in ONE dispatch on every frame the sun moved: residualPages
        // 1.08 ms through a 0.57 deg/frame sweep, against nothing at all for a static sun, which was most of the 0.91 -> 2.60 ms
        // a moving sun cost parked. A sun move invalidates lighting, not density, and the beam basis carries through it
        // unharmed - refresh 16 and refresh 256 score the same 0.024% under the sweep - so it does not have to be paid at once.
        updateSunVoxelDirection();
        return;
    }
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
    updateSunVoxelDirection();
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
    static bool layoutChecked = false;
    if (!layoutChecked)
    {
        layoutChecked = true;
        // The params upload is a raw copy, so the host and shader layouts must agree to the byte.
        const size_t shaderSize = var["params"].getType()->getByteSize();
        const size_t shaderMatrix = var["params"]["beamPrevViewProj"].getByteOffset() - var["params"].getByteOffset();
        const size_t hostMatrix = offsetof(HSTRCloudParams, beamPrevViewProj);
        if (shaderSize != sizeof(mParams) || shaderMatrix != hostMatrix)
            logError(
                "HSTRCloud: params layout differs: {} bytes in the shader and {} on the host, beamPrevViewProj at {} and {}.", shaderSize,
                sizeof(mParams), shaderMatrix, hostMatrix
            );
    }
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
    var["hstrTightMajorant"] = mpTightMajorant;
    var["hstrOccupancy"] = mpOccupancy;
    var["hstrMajorantZero"] = mpMajorantZero;
    var["hstrTransferProbeOutput"] = mpTransferProbe;
    if (mpCloudResidency)
        mpCloudResidency->bind(var);
    if (mpCloudTileBatches)
    {
        var["hstrCloudTileBatches"] = mpCloudTileBatches;
        var["hstrCloudTileReset"] = mpCloudTileReset;
    }
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
    var["hstrBeamPageTable"] = mpBeamPageTable;
    var["hstrBeamGuardDepth"] = mpBeamGuardDepth;
    var["hstrBeamGuardCamera"] = mpBeamGuardCamera;
    var["hstrBeamLevel"] = mpBeamLevel;
    var["hstrBeamLatticePrev"] = mpBeamLatticePrev;
    var["hstrBeamLevelPrev"] = mpBeamLevelPrev;
    var["hstrBeamPixelPrev"] = mpBeamPixels[mBeamRefFrame ? 0u : (mBeamParity ^ 1u)];
    var["hstrBeamPixels"] = mpBeamPixels[mBeamRefFrame ? 0u : mBeamParity];
    var["hstrBeamQueue"] = mpBeamQueue;
    var["hstrBeamQueueCounts"] = mpBeamQueueCounts;
    var["hstrBeamLists"] = mpBeamLists[mBeamParity];
    var["hstrBeamCounts"] = mpBeamCounts[mBeamParity];
    var["hstrBeamSparseCandidates"] = mpBeamSparseCandidates;
    var["hstrBeamResults"] = mpBeamResults;
    var["hstrBeamFinalTiles"] = mpBeamFinalTiles;
    var["hstrBeamQueryMap"] = mpBeamQueryMap;
    var["hstrBeamTileMap"] = mpBeamTileMap;
    var["hstrBeamPreviousLists"] = mpBeamLists[mBeamParity ^ 1u];
    var["hstrBeamPreviousCounts"] = mpBeamCounts[mBeamParity ^ 1u];
    var["hstrBeamHistory"] = mpBeamHistory[mBeamParity ^ 1u];
    var["hstrBeamGuide"] = mpBeamGuide;
    var["hstrExactFrame"] = mpExactFrame;
    var["hstrExtinctionSampler"] = mpExtinctionSampler;
    var["hstrLinearSampler"] = mpLinearSampler;
    var["hstrLinearClampSampler"] = mpLinearClampSampler;
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

    FALCOR_PROFILE(pRenderContext, "fineSun");
    if (mParams.cloudDomain != 0 && !mResidualDirty)
    {
        // Sea tiles whose cloud changed, and the tiles in their shadow, recompute their sun pages; the rest stays valid. The residual
        // mask is an optimisation of the HST camera path, which the sea does not use.
        const uint32_t r = mParams.cloudTileVoxels;
        for (uint32_t slot : mSunPageSlots)
        {
            mParams.fineSunOffset = uint3(slot % mParams.cloudTiles.x * r, 0, slot / mParams.cloudTiles.x * r);
            mParams.fineSunSize = uint3(r, fineDims.y, r);
            bindRenderer(pRenderContext, mpFineSunPass);
            bindOutput(mpFineSunPass, "hstrFineSunOutput", mpFineSun, "hstrFineSun");
            mpFineSunPass->execute(pRenderContext, mParams.fineSunSize);
        }
        mParams.fineSunOffset = uint3(0);
        mParams.fineSunSize = uint3(0);
    }
    else
    {
        // Ballistic sun transmittance per residual cell, with per-leaf activation, then non-resident pages are cleared.
        pRenderContext->clearUAV(mpLeafResidual->getUAV().get(), uint4(0));
        bindRenderer(pRenderContext, mpFineSunPass);
        bindOutput(mpFineSunPass, "hstrFineSunOutput", mpFineSun, "hstrFineSun");
        mpFineSunPass->execute(pRenderContext, fineDims);
        if (mParams.cloudDomain == 0)
        {
            bindRenderer(pRenderContext, mpResidualMaskPass);
            bindOutput(mpResidualMaskPass, "hstrFineSunOutput", mpFineSun, "hstrFineSun");
            mpResidualMaskPass->execute(pRenderContext, fineDims);
        }
    }
    mSunPageSlots.clear();

    // Octaves 1-3: scatterer-weighted sun arrival on a 2-voxel grid, spread by a separable Gaussian. This runs over the WHOLE
    // volume however few tiles were dirty, so throttling the fine-sun tiles above does not reduce it - which is why the sea's
    // per-tile path did not make residualPages any cheaper under a moving sun. Scoped separately to price the two halves.
    FALCOR_PROFILE(pRenderContext, "sunOctaves");
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

/// Places the beam image for this build.
///
/// With beamRefFrame off the beam image is the screen and nothing here changes. With it on the image is anchored to a world
/// orientation: the anchor pose's camera basis, widened by beamRefMargin so an ordinary turn stays inside it. A basis point is then
/// a world direction, so the previous build's value for it sits at the same index and carries with no reprojection and no
/// resampling - the resampling is what made a turning camera the most expensive motion there is. The frame is re-anchored only when
/// the screen would leave it, and that one build rebuilds from nothing.
void HSTRCloud::updateBeamReferenceFrame(const uint2& frameDim)
{
    mParams.beamRefFrame = mBeamRefFrame ? 1u : 0u;
    mParams.beamOctNdcPerTexel = 0.f;
    mParams.beamOct = mBeamOct && mBeamRefFrame && mpScene ? 1u : 0u;
    mParams.beamOctFull = mBeamOctFull ? 1u : 0u;
    mParams.beamOctAxis = mBeamOctAxis;
    mParams.beamGuard = mBeamGuard && mBeamRefFrame && mpScene ? 1u : 0u;
    mParams.beamScreenResidual = mBeamOct && mBeamScreenResidual && mBeamRefFrame && mpScene ? 1u : 0u;
    if (!mBeamRefFrame || !mpScene)
    {
        mParams.beamFrameDim = frameDim;
        mParams.beamRefValid = 0;
        mParams.beamRefAligned = 0;
        mParams.beamScreenBounds = float4(0.f, 0.f, float(frameDim.x), float(frameDim.y));
        mBeamRefAnchored = false;
        return;
    }
    const auto& camera = mpScene->getCamera()->getData();
    if (mBeamOct)
    {
        updateBeamOctFrame(frameDim, camera);
        return;
    }
    const float widen = 1.f + 2.f * mBeamRefMargin;
    const uint2 beamDim = uint2(float2(frameDim) * widen + 0.5f);
    const uint32_t guard = std::max(1u, mParams.beamTileSize);
    // The inverse of the matrix whose columns are the reference basis: it sends a world direction back to a beam position.
    auto inverseRows = [](const float3& u, const float3& v, const float3& w, float3 rows[3])
    {
        const float3 r0 = cross(v, w);
        const float3 r1 = cross(w, u);
        const float3 r2 = cross(u, v);
        const float determinant = dot(u, r0);
        const float scale = std::abs(determinant) > 1e-20f ? 1.f / determinant : 0.f;
        rows[0] = r0 * scale;
        rows[1] = r1 * scale;
        rows[2] = r2 * scale;
    };
    float3 rows[3];
    inverseRows(mBeamRefU, mBeamRefV, mBeamRefW, rows);
    auto beamPosition = [&](const float3& direction, float2& position)
    {
        const float3 c(dot(rows[0], direction), dot(rows[1], direction), dot(rows[2], direction));
        if (c.z <= 1e-6f)
            return false;
        position = (float2(c.x, c.y) / c.z * float2(0.5f, -0.5f) + 0.5f) * float2(beamDim);
        return true;
    };
    // Both images are pinhole projections from the same point, so the screen rectangle maps to a quadrilateral: its four corners
    // bound it exactly.
    bool anchor = !mBeamRefAnchored || any(beamDim != mParams.beamFrameDim);
    float2 low(std::numeric_limits<float>::max());
    float2 high(std::numeric_limits<float>::lowest());
    for (uint32_t corner = 0; corner < 4 && !anchor; ++corner)
    {
        const float2 ndc((corner & 1u) ? 1.f : -1.f, (corner & 2u) ? -1.f : 1.f);
        const float3 direction = normalize(ndc.x * camera.cameraU + ndc.y * camera.cameraV + camera.cameraW);
        float2 position;
        if (!beamPosition(direction, position) || any(position < float2(float(guard))) ||
            any(position > float2(beamDim) - float2(float(guard))))
        {
            anchor = true;
            break;
        }
        low = min(low, position);
        high = max(high, position);
    }
    if (anchor)
    {
        // Match the basis to the integer image size we actually allocated. Together with the centred shader projection this makes
        // an anchor-frame screen ray and its reference-image ray numerically identical instead of merely algebraically identical.
        mBeamRefU = camera.cameraU * (float(beamDim.x) / float(frameDim.x));
        mBeamRefV = camera.cameraV * (float(beamDim.y) / float(frameDim.y));
        mBeamRefW = camera.cameraW;
        mBeamRefCameraU = camera.cameraU;
        mBeamRefCameraV = camera.cameraV;
        mBeamRefCameraW = camera.cameraW;
        mBeamRefAnchored = true;
        ++mBeamRefAnchors;
        // Nothing of the previous basis is addressable in the new frame, so this build starts from nothing.
        mBeamHistoryValid = false;
        mParams.beamHistoryValid = 0;
        mBeamReusable = false;
        low = float2(0.f);
        high = float2(beamDim);
    }
    // See mBeamRefPrebuild: the whole image is the build region, so nothing is ever "off screen" to the basis, the residual or
    // the carry, and a turn only changes which already-built directions are resolved.
    if (mBeamRefPrebuild)
    {
        low = float2(0.f);
        high = float2(beamDim);
    }
    mParams.beamFrameDim = beamDim;
    mParams.beamRefU = mBeamRefU;
    mParams.beamRefV = mBeamRefV;
    mParams.beamRefW = mBeamRefW;
    mParams.beamRefAligned = all(camera.cameraU == mBeamRefCameraU) && all(camera.cameraV == mBeamRefCameraV) &&
                                     all(camera.cameraW == mBeamRefCameraW)
                                 ? 1u
                                 : 0u;
    inverseRows(mBeamRefU, mBeamRefV, mBeamRefW, rows);
    mParams.beamRefInverse0 = rows[0];
    mParams.beamRefInverse1 = rows[1];
    mParams.beamRefInverse2 = rows[2];
    mParams.beamRefValid = anchor ? 0u : 1u;
    // A block's angular reach: its own points, plus the two tiles a reconstruction can read past them. The beam image widens the
    // screen's angle and its pixel count by the same factor, so a beam pixel spans the angle a screen pixel does, and the centre
    // value is the largest one a perspective frame has.
    mParams.beamGuardAngle = beamGuardReach(frameDim, camera) * (2.f * length(camera.cameraU)) /
                             (std::max(length(camera.cameraW), 1e-6f) * float(frameDim.x));
    // Two tiles of guard each way: at level 0 a tile test reads the corners and centres of all eight neighbours.
    const float slack = 2.f * float(guard);
    mParams.beamScreenBounds = float4(
        std::max(0.f, low.x - slack), std::max(0.f, low.y - slack), std::min(float(beamDim.x), high.x + slack),
        std::min(float(beamDim.y), high.y + slack)
    );
}

/// Beam pixels a translation guard block reaches across: its own lattice points, plus the two tiles a level-0 tile test reads past
/// them. The bound the guard certifies is conservative in this reach, so it is taken at the outside.
float HSTRCloud::beamGuardReach(const uint2& frameDim, const CameraData& camera) const
{
    const uint32_t block = std::max(mParams.beamRefreshBlock, 1u) * std::max(mParams.beamLatticeStep, 1u);
    return float(block + 2u * std::max(mParams.beamTileSize, 1u));
}

/// Places the octahedral beam image. There is no anchor and so no re-anchor: the image is the sphere, a direction's texel is
/// fixed for the life of the renderer, and a turn only moves which part of it the screen resolves.
///
/// The image is sized so a texel spans the angle a screen pixel does at the view centre: the sphere is 4 pi sr, so (2 sqrt(pi) f)^2
/// texels cover it at f pixels per radian. At 4K that is about 114 M texels, 13.7x the screen - which is why the residual has to
/// be paged before this ships at 4K, and why the first measurements of it are taken at lower resolutions where it fits outright.
///
/// beamScreenBounds is the screen's footprint in octahedral space. The footprint is curved, so it is bounded by sampling a grid of
/// screen directions and padding by the largest gap between neighbouring samples: near the square's border - the two half great
/// circles x = 0 and y = 0 below the horizon, where the map wraps - that gap is large and the bound grows to stay conservative,
/// which costs dispatch breadth but never correctness. Everywhere else it is tight.
void HSTRCloud::updateBeamOctFrame(const uint2& frameDim, const CameraData& camera)
{
    const float focal = 0.5f * float(frameDim.x) * length(camera.cameraW) / std::max(length(camera.cameraU), 1e-6f);
    const uint32_t guard = std::max(1u, mParams.beamTileSize);
    uint32_t dim = uint32_t(2.f * std::sqrt(3.14159265f) * focal * std::max(mBeamOctScale, 0.01f) + 0.5f);
    dim = std::max(((dim + guard - 1u) / guard) * guard, 4u * guard);
    const float2 imageDim = float2(float(dim), float(dim));
    auto octOf = [&](const float3& direction)
    {
        const float3 d = mBeamOctAxis == 1 ? float3(direction.y, direction.z, direction.x)
                         : mBeamOctAxis == 2 ? direction
                                             : float3(direction.x, direction.z, direction.y); // Matches octFrame in the shader.
        const float3 n = d / (std::abs(d.x) + std::abs(d.y) + std::abs(d.z));
        float2 p(n.x, n.y);
        if (n.z < 0.f)
            p = float2((1.f - std::abs(n.y)) * (n.x >= 0.f ? 1.f : -1.f), (1.f - std::abs(n.x)) * (n.y >= 0.f ? 1.f : -1.f));
        return (p * 0.5f + 0.5f) * imageDim;
    };
    const uint32_t samples = 33;
    std::vector<float2> grid(samples * samples);
    float2 low(std::numeric_limits<float>::max());
    float2 high(std::numeric_limits<float>::lowest());
    for (uint32_t j = 0; j < samples; ++j)
        for (uint32_t i = 0; i < samples; ++i)
        {
            const float2 ndc(2.f * float(i) / float(samples - 1u) - 1.f, 1.f - 2.f * float(j) / float(samples - 1u));
            const float2 position = octOf(normalize(ndc.x * camera.cameraU + ndc.y * camera.cameraV + camera.cameraW));
            grid[j * samples + i] = position;
            low = min(low, position);
            high = max(high, position);
        }
    float pad = 2.f * float(guard);
    for (uint32_t j = 0; j < samples; ++j)
        for (uint32_t i = 0; i < samples; ++i)
        {
            const float2 a = grid[j * samples + i];
            if (i + 1 < samples)
            {
                const float2 d = abs(grid[j * samples + i + 1] - a);
                pad = std::max(pad, std::max(d.x, d.y));
            }
            if (j + 1 < samples)
            {
                const float2 d = abs(grid[(j + 1) * samples + i] - a);
                pad = std::max(pad, std::max(d.x, d.y));
            }
        }
    if (mBeamOctFull)
    {
        low = float2(0.f);
        high = imageDim;
        pad = 0.f;
    }
    mParams.beamFrameDim = uint2(dim, dim);
    // An octahedral texel is anisotropic - at the diamond mid-edge the map's singular values are 2.36 and 1.20 - so the reach has
    // to be taken along the worst axis, not on average, or the guard would certify blocks it has no right to.
    mParams.beamGuardAngle = beamGuardReach(frameDim, camera) * 2.36f * 2.f / float(dim);
    // The on-screen test projects a direction through the camera basis instead of comparing against a padded bounding box, so it
    // needs that basis inverted, and a conversion from a box's radius in texels to the normalised device units it can span. Both
    // are taken at their largest: the map's worst-axis texel angle, and the steeper of the two screen axes, where a radian buys
    // the most normalised device units.
    const float3 u = camera.cameraU, v = camera.cameraV, w = camera.cameraW;
    const float3 r0 = cross(v, w), r1 = cross(w, u), r2 = cross(u, v);
    const float determinant = dot(u, r0);
    const float inverseScale = std::abs(determinant) > 1e-20f ? 1.f / determinant : 0.f;
    mParams.beamRefInverse0 = r0 * inverseScale;
    mParams.beamRefInverse1 = r1 * inverseScale;
    mParams.beamRefInverse2 = r2 * inverseScale;
    const float tanHalf = std::max(length(v) / std::max(length(w), 1e-6f), 1e-6f);
    mParams.beamOctNdcPerTexel = (2.36f * 2.f / float(dim)) * (1.f + tanHalf * tanHalf) / tanHalf;
    mParams.beamRefU = camera.cameraU;
    mParams.beamRefV = camera.cameraV;
    mParams.beamRefW = camera.cameraW;
    mParams.beamRefAligned = 0;
    // The image cannot go stale, so the only build that starts from nothing is the one that allocates it.
    const bool fresh = !mBeamRefAnchored || any(mParams.beamFrameDim != mBeamOctDim);
    if (fresh)
    {
        mBeamRefAnchored = true;
        mBeamOctDim = mParams.beamFrameDim;
        ++mBeamRefAnchors;
        mBeamHistoryValid = false;
        mParams.beamHistoryValid = 0;
        mBeamReusable = false;
    }
    mParams.beamRefValid = fresh ? 0u : 1u;
    mParams.beamScreenBounds = float4(
        std::max(0.f, low.x - pad), std::max(0.f, low.y - pad), std::min(imageDim.x, high.x + pad),
        std::min(imageDim.y, high.y + pad)
    );
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
    if (nodeCount > 0 && (!mpNodeProjection || mpNodeProjection->getElementCount() != nodeCount))
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
    if (!mpScene || !mpPass || (!mpLeafRadiance && !mpCloudSea))
    {
        pRenderContext->clearUAV(color->getUAV().get(), float4(0.f));
        pRenderContext->clearUAV(error->getUAV().get(), float4(0.f));
        pRenderContext->clearUAV(cutStats->getUAV().get(), float4(0.f));
        return;
    }
    if (mpCloudSea)
        updateCloudDomain(pRenderContext);

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
        mpReferenceRowHistogram = mpDevice->createStructuredBuffer(
            sizeof(float),
            frameDim.y * kCompareBins,
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
        // In bands of mReferenceBandRows, each submitted on its own: one 4K sample of the sea runs longer than the driver's
        // timeout (DXGI_ERROR_DEVICE_REMOVED).
        const uint32_t bandRows = mReferenceBandRows > 0 ? mReferenceBandRows : mParams.frameDim.y;
        for (uint32_t row = 0; row < mParams.frameDim.y; row += bandRows)
        {
            mParams.referenceRowOffset = row;
            bindRenderer(pRenderContext, mpReferencePass);
            ShaderVar var = mpReferencePass->getRootVar()["CB"]["gHSTRCloud"];
            var["hstrReferenceSum"] = mpReferenceSum;
            var["color"] = color;
            var["transportError"] = error;
            var["cutStats"] = cutStats;
            mpReferencePass->execute(pRenderContext, uint3(mParams.frameDim.x, std::min(bandRows, mParams.frameDim.y - row), 1));
            if (bandRows < mParams.frameDim.y)
                pRenderContext->submit(true);
        }
        mParams.referenceRowOffset = 0;
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
    if (mParams.cloudDomain != 0 && !mSunPageQueue.empty() && !mResidualDirty)
    {
        const size_t count = std::min<size_t>(mCloudSunTilesPerFrame, mSunPageQueue.size());
        for (size_t i = 0; i < count; ++i)
            if (std::find(mSunPageSlots.begin(), mSunPageSlots.end(), mSunPageQueue[i]) == mSunPageSlots.end())
                mSunPageSlots.push_back(mSunPageQueue[i]);
        mSunPageQueue.erase(mSunPageQueue.begin(), mSunPageQueue.begin() + count);
        mSunPagesDirty = true;
    }
    else if (mResidualDirty)
        mSunPageQueue.clear();
    if (mResidualDirty || mSunPagesDirty)
    {
        FALCOR_PROFILE(pRenderContext, "residualPages");
        dispatchResidualPages(pRenderContext);
        mResidualDirty = false;
        mSunPagesDirty = false;
        mBasisDirty = true;
        mBeamReusable = false; // The beam queries read the sun pages.
    }

    // The cache modulation's default is the diffusion attenuation per unit optical depth, sqrt(3 (1 - albedo) (1 - albedo g)):
    // multiply scattered light fades into the cloud at that rate, so the stored ratio is nearly flat at cell scale.
    {
        const float albedo = dot(mpScene->getGridVolume(0)->getAlbedo(), float3(1.f / 3.f));
        const float diffusion = std::sqrt(std::max(0.f, 3.f * (1.f - albedo) * (1.f - albedo * mParams.anisotropy)));
        mParams.worldCacheModulation = mWorldCacheModulation < 0.f ? diffusion : mWorldCacheModulation;
    }

    // World cache experiment: camera-independent gather passes accumulated while its view is shown.
    const bool worldCacheFrame = mParams.debugView == kWorldCacheView || mParams.debugView == kBeamView || mParams.cloudDomain != 0;
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
        const size_t depositCount = size_t(dims.x) * dims.y * dims.z * kWorldCacheSlots;
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
        // Measured (frame_cost.py, 4K laptop GPU): one photon update has a ~0.8 ms floor independent of the photon count (64
        // photons cost as much as 16k, an empty dispatch 0.02 ms): the tracer is latency bound at low occupancy. Not caused
        // by path length (roulette from 8 bounces: 1k photons 1.78 -> 1.66 ms), the modulation read (off: 1.66 ms), or the
        // deposit walk. The persistent pool (worldCacheSegments > 0) does not remove it: 64k slots x 4 flights 2.12 ms against
        // 1.96 ms for 16k whole paths, and no error gain at equal time (photon_study.py).
        // Async compute would hide it, but slang-gfx 2024.1.34 exposes only a graphics queue (ICommandQueue::QueueType), and
        // ParameterBlock::prepareResource puts a UAV barrier on every bound UAV, so passes serialize: overlapping it needs a
        // native D3D12 compute queue with fences. Amortize instead: stop updating once converged, or fewer, larger updates.
        FALCOR_PROFILE(pRenderContext, "worldCache");
        if (mCloudCacheKeep < 1.f && mpWorldCacheDeposit && mParams.worldCacheSamples > 0)
        {
            mParams.worldCacheKeep = mCloudCacheKeep;
            bindRenderer(pRenderContext, mpDecayWorldCachePass);
            mpDecayWorldCachePass->getRootVar()["CB"]["gHSTRCloud"]["hstrWorldCacheDeposit"] = mpWorldCacheDeposit;
            mpDecayWorldCachePass->execute(pRenderContext, dims);
            for (float& batches : mCloudTileBatches)
                batches *= mCloudCacheKeep;
            mpCloudTileBatches->setBlob(mCloudTileBatches.data(), 0, mCloudTileBatches.size() * sizeof(float));
            mWorldCacheBakeDirty = true;
        }
        mCloudCacheKeep = 1.f;
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
                if (mParams.worldCacheSamples == 0)
                    std::fill(mCloudTileBatches.begin(), mCloudTileBatches.end(), 0.f);
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
            if (!mCloudTileBatches.empty())
            {
                for (float& batches : mCloudTileBatches)
                    batches += 1.f;
                mpCloudTileBatches->setBlob(mCloudTileBatches.data(), 0, mCloudTileBatches.size() * sizeof(float));
            }
            ++mWorldCacheBakes; // Buffer lookups see every update.
            mWorldCacheBakeDirty = true;
        }

        // Bake the means into hardware-filtered textures for the camera whenever the sums changed.
        if (!mpWorldCacheTextures[0] || mpWorldCacheTextures[0]->getWidth() != dims.x || mpWorldCacheTextures[0]->getHeight() != dims.y ||
            mpWorldCacheTextures[0]->getDepth() != dims.z)
        {
            for (auto& texture : mpWorldCacheTextures)
            {
                texture = mpDevice->createTexture3D(
                    dims.x,
                    dims.y,
                    dims.z,
                    ResourceFormat::RGBA16Float,
                    1,
                    nullptr,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
                );
                // The bake skips cells that never receive deposits; they stay at these zeros.
                pRenderContext->clearUAV(texture->getUAV().get(), float4(0.f));
            }
            mWorldCacheBakeDirty = true;
        }
        // A full bake cost 0.72 ms; skipping cells without medium (bakeWorldCacheCell) made it 0.24 ms, and baking every
        // mWorldCacheBakeInterval (8) updates makes it negligible per frame (16k photons per frame, side: 2.07 ms baking every
        // update, 1.95 ms every 8).
        const bool bakeDue = mWorldCacheUpdates == 0 || mParams.worldCacheSamples % mWorldCacheBakeInterval == 0;
        if (mWorldCacheBakeDirty && mParams.worldCacheTextured != 0 && mParams.worldCacheEstimator != 0 && bakeDue)
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

    // Camera space: nothing below re-solves transport; it reprojects the solved field. The sparse beam cut evaluator needs the
    // projected antichain, but not the HST camera lighting field (the sea's source is the world cache) or the fixed tile basis.
    const bool hstCamera = !worldCacheFrame;
    const bool projectedCamera = hstCamera || (mParams.debugView == kBeamView && mBeamSparse && mBeamSparseCut);
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
    if (mParams.hstrCutFlat != 0)
    {
        // The window of domain cells that carries the cut follows the camera in whole cells. Its content is the same 2048 cells
        // whatever the offset - the torus only decides which cell sits where - so a step re-places the boxes and re-pages them,
        // which is one dispatch over the cells, not a rebuild of anything the sea owns.
        const float3 cameraVoxel = (mpScene->getCamera()->getPosition() - mParams.seaOrigin) / mParams.seaVoxelSize - 0.5f;
        const int3 cameraLeaf = int3(floor(cameraVoxel / float(mParams.hstrCellWidth)));
        const int3 origin(
            cameraLeaf.x - int32_t(mParams.hstrLeafDims.x / 2u), 0, cameraLeaf.z - int32_t(mParams.hstrLeafDims.z / 2u)
        );
        if (!mDomainCutOriginValid || any(origin != mParams.hstrCutOriginLeaf))
        {
            mParams.hstrCutOriginLeaf = origin;
            mDomainCutOriginValid = true;
            mCutDirty = true;
        }
    }
    if (mCutDirty && projectedCamera)
    {
        FALCOR_PROFILE(pRenderContext, "cut");
        const uint32_t nodeCount = mParams.hstrNodeCount;
        if (mParams.hstrCutFlat != 0)
        {
            FALCOR_PROFILE(pRenderContext, "domainCut");
            bindRenderer(pRenderContext, mpDomainCutPass);
            // The cut buffer is the only resource this pass writes, so its read view goes away for the dispatch.
            mpDomainCutPass->getRootVar()["CB"]["gHSTRCloud"]["hstrCutNodes"] = ref<Buffer>();
            mpDomainCutPass->getRootVar()["CB"]["gHSTRCloud"]["hstrDomainCutNodes"] = mpCutNodes;
            mpDomainCutPass->execute(pRenderContext, uint3(nodeCount, 1, 1));
        }
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

    // Transfer-reuse probe: two entry-cell mask words per (asset brick, direction class), then a distinct-entry counter and a
    // crossing counter. Cleared every frame, so what it reports is one frame's worth of what a transfer cache would face.
    if (mParams.cloudTransferClasses > 0 && mpCloudResidency && mpCloudResidency->getOccupancy())
    {
        const uint32_t bricks = uint32_t(mpCloudResidency->getOccupancy()->getElementCount()) / kCloudCellWords;
        const uint32_t classes = mParams.cloudTransferClasses * mParams.cloudTransferClasses;
        const uint32_t words = 2u * bricks * classes + 2u;
        if (!mpTransferProbe || mpTransferProbe->getElementCount() != words)
            mpTransferProbe = mpDevice->createStructuredBuffer(
                sizeof(uint32_t), words, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal, nullptr, false
            );
        mParams.cloudTransferWords = words;
        pRenderContext->clearUAV(mpTransferProbe->getUAV().get(), uint4(0));
    }
    else
        mParams.cloudTransferWords = 0;

    // Beam view: hierarchical tiles. Level 0 queries every coarsest tile corner and centre and tests every tile; each finer
    // level queries and tests only the children of the tiles refined above it, through compacted lists and indirect
    // dispatches. Tiles failing the finest level are marched per pixel.
    if (mParams.debugView == kBeamView)
    {
        mBeamSparseBuilt = false;
        uint32_t maximumLevels = 0;
        while ((mParams.beamTileSize >> (maximumLevels + 1)) >= 2u && maximumLevels + 1 < kBeamMaxLevels)
            ++maximumLevels;
        mParams.beamLevels = std::clamp(mParams.beamLevels, 1u, maximumLevels + 1);
        const uint32_t finest = mParams.beamTileSize >> (mParams.beamLevels - 1);
        mParams.beamLatticeStep = std::max(1u, finest / 2);
        // Everything below is sized by the BEAM image, which is the screen unless the rotation-invariant frame is on.
        updateBeamReferenceFrame(frameDim);
        const uint2 beamDim = mParams.beamFrameDim;
        mParams.beamTileDims = (beamDim + mParams.beamTileSize - 1u) / mParams.beamTileSize;
        mParams.beamLatticeDims = mParams.beamTileDims * mParams.beamTileSize / mParams.beamLatticeStep + 1u;
        mParams.beamLevelDims = (beamDim + finest - 1u) / finest;
        const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
        const uint2 latticeDims = mParams.beamLatticeDims;
        const uint2 levelDims = mParams.beamLevelDims;
        const uint32_t tileCount = mParams.beamTileDims.x * mParams.beamTileDims.y;
        if (!mpBeamLattice || mpBeamLattice->getWidth() != latticeDims.x || mpBeamLattice->getHeight() != latticeDims.y)
            mpBeamLattice = mpDevice->createTexture2D(latticeDims.x, latticeDims.y, ResourceFormat::RGBA16Float, 3, 1, nullptr, flags);
        // Identity page table for the indirection probe: one entry per page of the lattice, entry i = i. Rebuilt only when the
        // lattice is resized, so it costs nothing per frame; what it buys is the ability to pay a sparse atlas's addressing cost
        // without building a sparse atlas.
        {
            const uint32_t shift = std::max(mParams.beamPageShift, 1u);
            const uint2 pageDims = (latticeDims + ((1u << shift) - 1u)) >> shift;
            mParams.beamPageDims = pageDims;
            const uint32_t pages = pageDims.x * pageDims.y;
            if (pages > 0 && (!mpBeamPageTable || mpBeamPageTable->getElementCount() != pages))
            {
                std::vector<uint32_t> identity(pages);
                for (uint32_t i = 0; i < pages; ++i)
                    identity[i] = i;
                mpBeamPageTable = mpDevice->createStructuredBuffer(
                    sizeof(uint32_t), pages, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, identity.data(), false
                );
            }
        }
        // Translation guard: one entry per block of beamRefreshBlock lattice points, the same blocks the refresh phase uses.
        {
            const uint2 guardDims = (latticeDims + (std::max(mParams.beamRefreshBlock, 1u) - 1u)) / std::max(mParams.beamRefreshBlock, 1u);
            if (!mpBeamGuardDepth || mpBeamGuardDepth->getWidth() != guardDims.x || mpBeamGuardDepth->getHeight() != guardDims.y)
            {
                mpBeamGuardDepth =
                    mpDevice->createTexture2D(guardDims.x, guardDims.y, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
                mpBeamGuardCamera =
                    mpDevice->createTexture2D(guardDims.x, guardDims.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
                mBeamGuardCleared = false;
            }
        }
        if (!mpBeamLevel || mpBeamLevel->getWidth() != levelDims.x || mpBeamLevel->getHeight() != levelDims.y)
            mpBeamLevel = mpDevice->createTexture2D(levelDims.x, levelDims.y, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
        // Full-resolution guide for the tile tests. Near view at 4K (one segment per query, moving camera), 4x1 tiles, mean 8-bit
        // display error against the per-pixel march / GPU ms: off 0.16 / 175, beamGuide 1 at 0.2 0.14 / 244, 0.1 0.13 / 275,
        // 0.05 0.10 / 315 (the guide itself ~38 ms). Off by default: what it refines is marched per pixel at the full lighting cost.
        if (mParams.beamGuide != 0)
        {
            if (!mpBeamGuide || mpBeamGuide->getWidth() != frameDim.x || mpBeamGuide->getHeight() != frameDim.y)
                mpBeamGuide = mpDevice->createTexture2D(frameDim.x, frameDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
            FALCOR_PROFILE(pRenderContext, "beamGuide");
            bindRenderer(pRenderContext, mpBeamGuidePass);
            bindOutput(mpBeamGuidePass, "hstrBeamGuideOutput", mpBeamGuide, "hstrBeamGuide");
            mpBeamGuidePass->execute(pRenderContext, uint3(frameDim, 1));
        }
        else
            mpBeamGuide = nullptr;
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
                    mpDevice->createStructuredBuffer(sizeof(uint32_t), kBeamCountSlots, flags, MemoryType::DeviceLocal, nullptr, false);
            if (!mpBeamHistory[i] || mpBeamHistory[i]->getWidth() != levelDims.x || mpBeamHistory[i]->getHeight() != levelDims.y)
            {
                mpBeamHistory[i] = mpDevice->createTexture2D(levelDims.x, levelDims.y, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
                layoutChanged = true;
            }
        }
        // Sparse query compiler. The coordinate map provides stable virtual IDs;
        // results and final tiles are authoritative for verification and resolve.
        const uint32_t sparseTileCapacity = tileCount * 85u;
        const uint32_t sparseQueryCapacity = latticeDims.x * latticeDims.y;
        const uint32_t sparseFinalCapacity = tileCount * (1u << (2u * (mParams.beamLevels - 1u)));
        if (!mpBeamSparseCandidates || mpBeamSparseCandidates->getElementCount() != sparseTileCapacity)
            mpBeamSparseCandidates = mpDevice->createStructuredBuffer(sizeof(uint32_t), sparseTileCapacity, flags, MemoryType::DeviceLocal, nullptr, false);
        if (!mpBeamSparseArgs)
            mpBeamSparseArgs = mpDevice->createStructuredBuffer(
                sizeof(uint32_t), 9, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::IndirectArg, MemoryType::DeviceLocal, nullptr, false
            );
        if (!mpBeamResults || mpBeamResults->getElementCount() != sparseQueryCapacity || !mpBeamQueryMap ||
            mpBeamQueryMap->getWidth() != latticeDims.x || mpBeamQueryMap->getHeight() != latticeDims.y)
        {
            mpBeamResults = mpDevice->createStructuredBuffer(sizeof(BeamResult), sparseQueryCapacity, flags, MemoryType::DeviceLocal, nullptr, false);
            mpBeamQueryMap = mpDevice->createTexture2D(latticeDims.x, latticeDims.y, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
        }
        if (!mpBeamFinalTiles || mpBeamFinalTiles->getElementCount() != sparseFinalCapacity)
            mpBeamFinalTiles = mpDevice->createStructuredBuffer(sizeof(BeamTile), sparseFinalCapacity, flags, MemoryType::DeviceLocal, nullptr, false);
        if (!mpBeamTileMap || mpBeamTileMap->getWidth() != levelDims.x || mpBeamTileMap->getHeight() != levelDims.y)
            mpBeamTileMap = mpDevice->createTexture2D(levelDims.x, levelDims.y, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
        const uint4 layout(frameDim, mParams.beamTileSize, mParams.beamLevels);
        if (layoutChanged || any(layout != mBeamLayout))
        {
            mBeamLayout = layout;
            mBeamHistoryValid = false;
            mBeamReusable = false;
            mBeamRefreshValid = false;
        }
        // beamRefresh and beamQueue need one level, the per-level build and single-segment queries; otherwise the view marches
        // everything in place. Both keep the previous build's lattice and levels: the refresh carries from them, the queue predicts
        // each point's cost from them.
        const bool rootOnly = mParams.beamLevels == 1 && mParams.beamTemporal == 0 && mParams.beamSegments == 1;
        const bool refresh = mBeamRefresh != 0 && rootOnly;
        const bool queue = mBeamQueue && rootOnly;
        const bool history = refresh || queue;
        mParams.beamRefresh = refresh ? mBeamRefresh : 0u;
        mParams.beamQueue = queue ? 1u : 0u;
        if (history)
        {
            // The reference frame's lattice is persistent: a basis point's index is its world direction, so there is nothing for a
            // second image to hold and the query pass updates the one in place. That halves the lattice's memory, which is most of
            // what beamRefMargin costs, and lets a direction that turns off screen keep its basis until it turns back.
            if (mBeamRefFrame)
                mpBeamLatticePrev = nullptr;
            else if (!mpBeamLatticePrev || mpBeamLatticePrev->getWidth() != latticeDims.x ||
                     mpBeamLatticePrev->getHeight() != latticeDims.y)
            {
                mpBeamLatticePrev = mpDevice->createTexture2D(latticeDims.x, latticeDims.y, ResourceFormat::RGBA16Float, 3, 1, nullptr, flags);
                mBeamRefreshValid = false;
            }
            if (mBeamRefFrame)
                mpBeamLevelPrev = nullptr; // Persistent: never read, never swapped.
            else if (!mpBeamLevelPrev || mpBeamLevelPrev->getWidth() != levelDims.x || mpBeamLevelPrev->getHeight() != levelDims.y)
            {
                mpBeamLevelPrev = mpDevice->createTexture2D(levelDims.x, levelDims.y, ResourceFormat::R32Uint, 1, 1, nullptr, flags);
                mBeamRefreshValid = false;
            }
        }
        else
        {
            mpBeamLatticePrev = nullptr;
            mpBeamLevelPrev = nullptr;
            mBeamRefreshValid = false;
        }
        if (refresh || mBeamRefFrame)
        {
            // In the reference frame the residual lives in the beam image beside the basis, so it is sized by that image.
            for (ref<Texture>& pPixels : mpBeamPixels)
                if (!pPixels || pPixels->getWidth() != beamDim.x || pPixels->getHeight() != beamDim.y)
                {
                    pPixels = mpDevice->createTexture2D(beamDim.x, beamDim.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, flags);
                    mBeamRefreshValid = false;
                }
        }
        else
            mpBeamPixels = {};
        if (queue)
        {
            // Each list can hold every root point: corners and centres.
            const uint32_t capacity = (mParams.beamTileDims.x + 1) * (mParams.beamTileDims.y + 1) + tileCount;
            mParams.beamQueueCapacity = capacity;
            if (!mpBeamQueue || mpBeamQueue->getElementCount() != capacity * kBeamQueueBuckets)
                mpBeamQueue = mpDevice->createStructuredBuffer(sizeof(uint32_t), capacity * kBeamQueueBuckets, flags, MemoryType::DeviceLocal, nullptr, false);
            if (!mpBeamQueueCounts)
                mpBeamQueueCounts = mpDevice->createStructuredBuffer(sizeof(uint32_t), kBeamQueueBuckets, flags, MemoryType::DeviceLocal, nullptr, false);
            if (!mpBeamQueueArgs)
                mpBeamQueueArgs = mpDevice->createStructuredBuffer(
                    sizeof(uint32_t), 3 * kBeamQueueBuckets, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::IndirectArg,
                    MemoryType::DeviceLocal, nullptr, false
                );
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
        // MEASURED (4K, cloud sea library, minStepVoxels 2, baked sun): a frozen camera takes this reuse path and pays no query
        // cost at all, which is what the settled A/B timings of the beam view were measuring - 5.7 ms near, 4.4 ms sea. The moment
        // the camera or the sun moves the queries come back and the frame costs 11-13 ms (near: 12.3 static, 11.3 and 12.7 through
        // slow and fast sun sweeps, 13.1 and 27.8 flying at 4 and 1 units a frame). Quality holds through all of it, p99.9 of the
        // log error against the per-pixel march staying between 0.018 and 0.036. So the beam view's real budget is split about
        // evenly between the query rays and the per-pixel refinement of the tiles that fail, and any timing of it taken with the
        // camera parked is a best case, not the frame cost.
        // The query rays run the same march as the refinement, so they need the same permutation, and they need it before they are
        // dispatched: setting it with the march pass below left a frame's queries compiled against the previous value of
        // cloudSunLiveMarch whenever it changed.
        mpBeamQueryPass->getProgram()->addDefine("HSTR_SUN_LIVE", mCloudSunLiveMarch ? "1" : "0");
        mpBeamQueryPass->getProgram()->addDefine("HSTR_SHIP", std::to_string(beamShipping() ? mBeamShipMask : 0u));
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
                const bool sparse = mBeamSparse && !history && mParams.beamSegments == 1;
                if (sparse)
                {
                    mBeamSparseBuilt = true;
                    // Final sparse work-generation path: metadata produces candidate leaves, each level emits and globally
                    // deduplicates only its five basis positions, exact marches fill BeamResult[], and the verifier either records a
                    // final BeamTile or appends four children. Stable virtual query IDs address BeamResult directly; the legacy
                    // lattice is neither written nor read by this path.
                    mBeamParity ^= 1u;
                    pRenderContext->clearUAV(mpBeamCounts[mBeamParity]->getUAV().get(), uint4(0));
                    pRenderContext->clearUAV(mpBeamQueryMap->getUAV().get(), uint4(0xffffffffu));
                    pRenderContext->clearUAV(mpBeamTileMap->getUAV().get(), uint4(0xffffffffu));
                    pRenderContext->clearUAV(mpBeamLevel->getUAV().get(), uint4(mParams.beamLevels));

                    {
                        FALCOR_PROFILE(pRenderContext, "beamSparseClassify");
                        bindRenderer(pRenderContext, mpBeamClassifyPass);
                        mpBeamClassifyPass->getProgram()->addDefine("HSTR_BEAM_SPARSE_MIN_LEVEL", std::to_string(mBeamSparseMinLevel));
                        bindOutput(mpBeamClassifyPass, "hstrBeamLevelOutput", mpBeamLevel, "hstrBeamLevel");
                        mpBeamClassifyPass->execute(pRenderContext, uint3(tileCount, 1, 1));
                    }

                    mpBeamSparseEmitPass->getProgram()->addDefine("HSTR_SUN_LIVE", mCloudSunLiveMarch ? "1" : "0");
                    mpBeamSparseEmitPass->getProgram()->addDefine("HSTR_SHIP", std::to_string(beamShipping() ? mBeamShipMask : 0u));
                    mpBeamSparseEmitPass->getProgram()->addDefine("HSTR_BEAM_SPARSE_CUT", mBeamSparseCut ? "1" : "0");
                    for (uint32_t level = 0; level < mParams.beamLevels; ++level)
                    {
                        FALCOR_PROFILE(pRenderContext, "beamSparseLevel" + std::to_string(level));
                        mParams.beamPassLevel = level;
                        auto writeSparseArgs = [&]()
                        {
                            bindRenderer(pRenderContext, mpBeamSparseArgsPass);
                            mpBeamSparseArgsPass->getRootVar()["CB"]["gHSTRCloud"]["hstrBeamSparseArgs"] = mpBeamSparseArgs;
                            mpBeamSparseArgsPass->execute(pRenderContext, uint3(1));
                        };
                        {
                            FALCOR_PROFILE(pRenderContext, "args");
                            writeSparseArgs();
                        }
                        {
                            FALCOR_PROFILE(pRenderContext, "emitMarch");
                            bindRenderer(pRenderContext, mpBeamSparseEmitPass);
                            mpBeamSparseEmitPass->executeIndirect(pRenderContext, mpBeamSparseArgs.get(), 0);
                        }
                        {
                            FALCOR_PROFILE(pRenderContext, "verify");
                            bindRenderer(pRenderContext, mpBeamSparseVerifyPass);
                            bindOutput(mpBeamSparseVerifyPass, "hstrBeamLevelOutput", mpBeamLevel, "hstrBeamLevel");
                            mpBeamSparseVerifyPass->executeIndirect(pRenderContext, mpBeamSparseArgs.get(), 12);
                        }
                    }
                    mBeamHistoryValid = false;
                    mParams.beamHistoryValid = 0;
                }
                else
                {
                // The per-level build keeps the refinement history too: beamAdaptiveRoot reads the previous frame's finest-level
                // bits to decide which root tiles to skip, and that prediction has to exist whether or not the queries are temporal.
                mBeamParity ^= 1u;
                if (!mBeamHistoryValid)
                    pRenderContext->clearUAV(mpBeamHistory[mBeamParity ^ 1u]->getUAV().get(), uint4(0));
                pRenderContext->clearUAV(mpBeamHistory[mBeamParity]->getUAV().get(), uint4(0));
                pRenderContext->clearUAV(mpBeamCounts[mBeamParity]->getUAV().get(), uint4(0));
                if (history)
                {
                    // Last build's lattice and levels become the history; this build writes every root point and tile again.
                    // Nothing in the reference frame is swapped: lattice, level map and residual are all addressed by world
                    // direction, so each IS its own history and the build updates it in place. The residual carries its validity
                    // in its own alpha (negative until marched since the anchor), which is what replaced asking the previous
                    // level map which tiles failed and the previous screen box which units it covered.
                    if (!mBeamRefFrame)
                    {
                        std::swap(mpBeamLattice, mpBeamLatticePrev);
                        std::swap(mpBeamLevel, mpBeamLevelPrev);
                    }
                    mParams.beamHistoryValid = mBeamRefreshValid ? 1u : 0u;
                    mParams.beamPrevViewProj = mBeamPrevViewProj;
                    mParams.beamPrevCamera = mBeamPrevCamera;
                    // What the build that wrote the history covered: in the reference frame the image reaches past the screen and
                    // only the covered part holds marched values, so a unit that has just turned on screen has nothing to carry.
                    mParams.beamPrevScreenBounds = mBeamBuiltScreenBounds;
                    mBeamBuiltScreenBounds = mParams.beamScreenBounds;
                    // A build the camera did not translate for cannot have moved any persistent point, so its carries touch memory
                    // only to re-measure a distance that has not changed - which is to say, not at all.
                    mParams.beamStationary = all(cameraPosition == mBeamPrevCamera) ? 1u : 0u;
                    ++mParams.beamFrame;
                }
                else
                {
                    mParams.beamHistoryValid = 0;
                    mParams.beamStationary = 0;
                }
                // The guard claims a block was verified from a camera; a rebuilt image makes every such claim false, and an
                // uncleared one holds whatever the allocation left behind. Either way it has to be emptied before it is believed.
                if (mpBeamGuardDepth && (!mBeamGuardCleared || (mBeamRefFrame && mParams.beamRefValid == 0)))
                {
                    pRenderContext->clearUAV(mpBeamGuardDepth->getUAV().get(), uint4(0xFFFFFFFFu));
                    pRenderContext->clearUAV(mpBeamGuardCamera->getUAV().get(), float4(0.f));
                    mBeamGuardCleared = true;
                }
                // A re-anchored frame shares no index with the one before it, so every mark in the persistent lattice goes.
                if (mBeamRefFrame && mParams.beamRefValid == 0)
                {
                    if (mpBeamLattice)
                        pRenderContext->clearUAV(mpBeamLattice->getUAV().get(), float4(0.f));
                    if (mpBeamPixels[0])
                        pRenderContext->clearUAV(mpBeamPixels[0]->getUAV().get(), float4(0.f, 0.f, 0.f, -1.f));
                }
                const bool gridDispatch = (mBeamGridDispatch || history) && mParams.beamSegments == 1;
                for (uint32_t level = 0; level < mParams.beamLevels; ++level)
                {
                    FALCOR_PROFILE(pRenderContext, "beamLevel" + std::to_string(level));
                    if (level > 0)
                        writeArgs(level);
                    mParams.beamPassLevel = level;
                    {
                        FALCOR_PROFILE(pRenderContext, "queries" + std::to_string(level));
                        if (level == 0 && gridDispatch)
                        {
                            // The corner grid, then the centre grid, each as a 2D dispatch at the lattice spacing. With the queue these
                            // only carry and queue; the queued points are marched below, one list of similar cost per dispatch.
                            mpBeamGridQueryPass->getProgram()->addDefine("HSTR_SUN_LIVE", mCloudSunLiveMarch ? "1" : "0");
                            mpBeamGridQueryPass->getProgram()->addDefine("HSTR_SHIP", std::to_string(beamShipping() ? mBeamShipMask : 0u));
                            if (queue)
                                pRenderContext->clearUAV(mpBeamQueueCounts->getUAV().get(), uint4(0));
                            // Launch over the screen box, not the whole beam image. beamPointOnScreen keeps a point while any
                            // tile that could read it is on screen, and its reach is two tiles, so a point matters only where
                            // point * tileSize + 2 * tileSize >= bounds.min and point * tileSize - 2 * tileSize <= bounds.max.
                            // Outside that the thread was launched solely to fail that test - 1.69x the screen at the shipping
                            // margin, and 4x at beamRefMargin 0.5, which is why enlarging the frame kept costing more than it saved.
                            const float tile = float(std::max(1u, mParams.beamTileSize));
                            const float4 box = mParams.beamScreenBounds;
                            const int2 first = int2(std::floor(box.x / tile), std::floor(box.y / tile)) - 2;
                            const int2 last = int2(std::ceil(box.z / tile), std::ceil(box.w / tile)) + 2;
                            const uint2 origin = uint2(std::max(0, first.x), std::max(0, first.y));
                            // Nothing in the image can need work except the blocks this build's refresh phase selects, WHEN the
                            // camera has not translated (so no parallax can have failed), the frame is not re-anchoring, the
                            // history is valid, and the screen covers the same box as the build that wrote it (so no direction
                            // has entered). Then the refresh set is a closed form: blocks whose index is congruent to the phase
                            // cell, which at beamRefresh 256 over 208 x 117 blocks is one block of 16 tiles, or none at all.
                            const uint32_t phases = mBeamRefresh;
                            // A camera that has not translated cannot have failed a parallax test anywhere, so the only points
                            // that can need work are the refresh phase's blocks plus whatever the screen box newly covers. The
                            // second part is exact rectangle subtraction: current tile box minus the box the last build wrote,
                            // which for a rotation is one or two thin strips. Nothing else in the image can have changed.
                            const bool carrying = mBeamRefFrame && mParams.beamRefValid != 0 && mParams.beamHistoryValid != 0 &&
                                                  mParams.beamStationary != 0 && phases > 2;
                            const bool sameBox = all(mParams.beamScreenBounds == mParams.beamPrevScreenBounds);
                            const bool generated = carrying;
                            const uint32_t refreshBlock = std::max(mParams.beamRefreshBlock, 1u);
                            const uint32_t cell = generated ? (mParams.beamFrame % (phases * phases)) : 0u;
                            const uint2 firstBlock(cell % phases, cell / phases);
                            mParams.beamGridBlocks = generated ? 1u : 0u;
                            mParams.beamGridBlockStride = phases;
                            for (uint32_t centres = 0; centres < (mParams.beamCentreless != 0 ? 1u : 2u); ++centres)
                            {
                                const uint2 dims = mParams.beamTileDims + (centres ? 0u : 1u);
                                uint2 threads;
                                if (generated)
                                {
                                    // Blocks firstBlock + k * phases that are still inside the grid, each contributing its tiles.
                                    const uint2 blocks = (dims + refreshBlock - 1u) / refreshBlock;
                                    const uint2 count(
                                        firstBlock.x < blocks.x ? (blocks.x - firstBlock.x + phases - 1u) / phases : 0u,
                                        firstBlock.y < blocks.y ? (blocks.y - firstBlock.y + phases - 1u) / phases : 0u
                                    );
                                    mParams.beamGridOrigin = firstBlock;
                                    threads = count * refreshBlock;
                                }
                                else
                                {
                                    const uint2 end = uint2(
                                        std::min<int32_t>(int32_t(dims.x), std::max(0, last.x + 1)),
                                        std::min<int32_t>(int32_t(dims.y), std::max(0, last.y + 1))
                                    );
                                    if (any(end <= origin))
                                        continue;
                                    mParams.beamGridOrigin = origin;
                                    threads = end - origin;
                                }
                                mParams.beamGridCentres = centres;
                                if (centres == 0)
                                {
                                    mBeamGridRegions.clear();
                                    mBeamGridRegionMode.clear();
                                }
                                auto dispatch = [&](uint2 at, uint2 size, uint32_t blocks)
                                {
                                    if (size.x == 0 || size.y == 0)
                                        return;
                                    if (centres == 0 && generated)
                                    {
                                        mBeamGridRegions.push_back(uint4(at.x, at.y, size.x, size.y));
                                        mBeamGridRegionMode.push_back(blocks);
                                    }
                                    mParams.beamGridOrigin = at;
                                    mParams.beamGridBlocks = blocks;
                                    mBeamGridThreads += size.x * size.y;
                                    bindRenderer(pRenderContext, mpBeamGridQueryPass);
                                    bindOutput(mpBeamGridQueryPass, "hstrBeamLatticeOutput", mpBeamLattice, "hstrBeamLattice");
                                    mpBeamGridQueryPass->execute(pRenderContext, uint3(size, 1));
                                };
                                if (!generated)
                                {
                                    dispatch(origin, threads, 0u);
                                    if (centres == 0)
                                        ++mBeamGridSweeps;
                                    continue;
                                }
                                dispatch(firstBlock, threads, 1u); // The refresh phase, wherever the camera is looking.
                                if (sameBox)
                                {
                                    if (centres == 0)
                                        ++mBeamGridGenerated;
                                    continue;
                                }
                                // The strips the screen newly covers, as current box minus previous box. Clamped to the grid,
                                // and taken with the same two-tile reach beamPointOnScreen uses so a strip carries its guard.
                                const int2 pf = int2(std::floor(mParams.beamPrevScreenBounds.x / tile), std::floor(mParams.beamPrevScreenBounds.y / tile)) - 2;
                                const int2 pl = int2(std::ceil(mParams.beamPrevScreenBounds.z / tile), std::ceil(mParams.beamPrevScreenBounds.w / tile)) + 3;
                                const int2 cf(std::max(0, first.x), std::max(0, first.y));
                                const int2 cl(std::min(int32_t(dims.x), last.x + 1), std::min(int32_t(dims.y), last.y + 1));
                                const int2 pfc(std::max(cf.x, pf.x), std::max(cf.y, pf.y));
                                const int2 plc(std::min(cl.x, pl.x), std::min(cl.y, pl.y));
                                auto strip = [&](int2 lo, int2 hi)
                                {
                                    if (hi.x > lo.x && hi.y > lo.y)
                                        dispatch(uint2(lo), uint2(hi - lo), 0u);
                                };
                                strip(cf, int2(std::min(cl.x, pfc.x), cl.y));                 // left of the old box
                                strip(int2(std::max(cf.x, plc.x), cf.y), cl);                 // right of it
                                const int2 midLow(std::max(cf.x, pfc.x), cf.y);
                                const int2 midHigh(std::min(cl.x, plc.x), cl.y);
                                strip(midLow, int2(midHigh.x, std::min(cl.y, pfc.y)));        // above it
                                strip(int2(midLow.x, std::max(cf.y, plc.y)), midHigh);        // below it
                                if (centres == 0)
                                    ++mBeamGridStrips;
                            }
                            mParams.beamGridCentres = 0;
                            mParams.beamGridOrigin = uint2(0);
                            mParams.beamGridBlocks = 0;
                            if (queue)
                            {
                                writeBeamQueueArgs(pRenderContext, 1);
                                mpBeamQueueMarchPass->getProgram()->addDefine("HSTR_SUN_LIVE", mCloudSunLiveMarch ? "1" : "0");
                                mpBeamQueueMarchPass->getProgram()->addDefine("HSTR_SHIP", std::to_string(beamShipping() ? mBeamShipMask : 0u));
                                for (uint32_t bucket = 0; bucket < kBeamQueueBuckets; ++bucket)
                                {
                                    FALCOR_PROFILE(pRenderContext, "bucket" + std::to_string(bucket));
                                    mParams.beamQueueBucket = bucket;
                                    bindRenderer(pRenderContext, mpBeamQueueMarchPass);
                                    bindOutput(mpBeamQueueMarchPass, "hstrBeamLatticeOutput", mpBeamLattice, "hstrBeamLattice");
                                    // The offset is in bytes: three uint32 per list.
                                    mpBeamQueueMarchPass->executeIndirect(pRenderContext, mpBeamQueueArgs.get(), 12 * bucket);
                                }
                            }
                        }
                        else if (level == 0)
                        {
                            bindRenderer(pRenderContext, mpBeamQueryPass);
                            bindOutput(mpBeamQueryPass, "hstrBeamLatticeOutput", mpBeamLattice, "hstrBeamLattice");
                            // Corners and centres in blocks of 8 x 4 lattice points (see beamQueryPosition).
                            auto blockQueries = [](uint2 grid) { return ((grid.x + 7u) / 8u) * ((grid.y + 3u) / 4u) * 32u; };
                            // A centreless root queries the corners only; the centre threads would all return at once.
                            const uint32_t queries = blockQueries(mParams.beamTileDims + 1u) +
                                                     (mParams.beamCentreless != 0 ? 0u : blockQueries(mParams.beamTileDims));
                            mpBeamQueryPass->execute(pRenderContext, uint3(queries * mParams.beamSegments, 1, 1));
                        }
                        else
                        {
                            bindRenderer(pRenderContext, mpBeamQueryPass);
                            bindOutput(mpBeamQueryPass, "hstrBeamLatticeOutput", mpBeamLattice, "hstrBeamLattice");
                            mpBeamQueryPass->executeIndirect(pRenderContext, mpBeamArgs.get(), 24 * level);
                        }
                    }
                    bindRenderer(pRenderContext, mpBeamTilePass);
                    bindOutput(mpBeamTilePass, "hstrBeamLevelOutput", mpBeamLevel, "hstrBeamLevel");
                    mpBeamTilePass->getRootVar()["CB"]["gHSTRCloud"]["hstrBeamHistoryOutput"] = mpBeamHistory[mBeamParity];
                    if (level == 0 && !mBeamGridRegions.empty())
                    {
                        // Only the tiles whose basis points this build touched can have changed classification.
                        mParams.beamTileGenerated = 1;
                        for (size_t r = 0; r < mBeamGridRegions.size(); ++r)
                        {
                            const uint4& region = mBeamGridRegions[r];
                            mParams.beamGridOrigin = uint2(region.x, region.y);
                            mParams.beamGridBlocks = mBeamGridRegionMode[r];
                            bindRenderer(pRenderContext, mpBeamTilePass);
                            bindOutput(mpBeamTilePass, "hstrBeamLevelOutput", mpBeamLevel, "hstrBeamLevel");
                            mpBeamTilePass->getRootVar()["CB"]["gHSTRCloud"]["hstrBeamHistoryOutput"] = mpBeamHistory[mBeamParity];
                            mpBeamTilePass->execute(pRenderContext, uint3(region.z, region.w, 1));
                        }
                        mParams.beamTileGenerated = 0;
                        mParams.beamGridBlocks = 0;
                        mParams.beamGridOrigin = uint2(0);
                    }
                    else if (level == 0)
                        mpBeamTilePass->execute(pRenderContext, uint3(tileCount, 1, 1));
                    else
                        mpBeamTilePass->executeIndirect(pRenderContext, mpBeamArgs.get(), 24 * level + 12);
                }
                mBeamHistoryValid = true;
                if (history)
                {
                    mBeamPrevViewProj = mpScene->getCamera()->getViewProjMatrixNoJitter();
                    mBeamPrevCamera = cameraPosition;
                    mBeamRefreshValid = true;
                }
                }
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
        // Reconstruction of every accepted pixel in a light kernel, then the compacted per-pixel march of failed finest tiles. In
        // the reference frame the order reverses: the refinement writes the beam image and the resolve reads it, so that every
        // pixel is produced by one pass over the frame whichever way its tile went.
        auto resolve = [&]()
        {
            FALCOR_PROFILE(pRenderContext, "resolve");
            const ref<ComputePass>& pResolve = mBeamSparseBuilt ? mpBeamSparseResolvePass : mpBeamResolvePass;
            bindRenderer(pRenderContext, pResolve);
            pResolve->getRootVar()["CB"]["gHSTRCloud"]["color"] = color;
            pResolve->execute(pRenderContext, uint3(mParams.frameDim, 1));
        };
        if (mParams.beamRefFrame == 0)
            resolve();
        {
        FALCOR_PROFILE(pRenderContext, "march");
        // The beam's per-pixel refinement runs the same camera march, so it drops the live sun march with it.
        // A four-bucket sparse fallback queue was slower at 4K near (16.81 ms
        // versus 16.65 ms), so sparse failures keep the single compact dispatch.
        // The compacted fallback list enumerates BEAM tiles and marches their own units as if they were pixels, which is only true
        // while the beam image is the screen. In the reference frame the fallback runs over the beam image instead, where it is a
        // function of world direction and carries exactly through a turn; the resolve samples it.
        // beamScreenResidual sends the failed-tile march back to screen space, where a pixel marches its own ray and the resolve
        // never resamples. It costs the residual's carry, so it is a quality-for-marching trade rather than a free win.
        const bool screenResidual = mBeamOct && mBeamScreenResidual && mParams.beamRefFrame != 0;
        const bool unitMarch = mParams.beamRefFrame != 0 && !screenResidual;
        const bool gridMarch = (mBeamGridDispatch && !unitMarch) || screenResidual;
        const bool queued = mParams.beamQueue != 0 && !gridMarch && !unitMarch;
        const ref<ComputePass>& pMarch =
            unitMarch ? mpBeamUnitMarchPass : (queued ? mpBeamQueuePixelPass : (gridMarch ? mpBeamGridMarchPass : mpBeamMarchPass));
        pMarch->getProgram()->addDefine("HSTR_SUN_LIVE", mCloudSunLiveMarch ? "1" : "0");
        pMarch->getProgram()->addDefine("HSTR_SHIP", std::to_string(beamShipping() ? mBeamShipMask : 0u));
        auto bindMarch = [&]()
        {
            bindRenderer(pRenderContext, pMarch);
            pMarch->getRootVar()["CB"]["gHSTRCloud"]["color"] = color;
            pMarch->getRootVar()["CB"]["gHSTRCloud"]["hstrBeamPixelOutput"] =
                (mParams.beamRefresh != 0 || unitMarch) ? mpBeamPixels[unitMarch ? 0u : mBeamParity] : ref<Texture>();
            // The unit march writes the same texture the resolve reads, so its read view goes away for this dispatch.
            if (unitMarch)
                pMarch->getRootVar()["CB"]["gHSTRCloud"]["hstrBeamPixels"] = ref<Texture>();
        };
        if (queued)
        {
            // The refined tiles by predicted cost, then one march dispatch per list (see beamQueue).
            const uint32_t finest = mParams.beamTileSize >> (mParams.beamLevels - 1);
            pRenderContext->clearUAV(mpBeamQueueCounts->getUAV().get(), uint4(0));
            bindRenderer(pRenderContext, mpBeamQueueTilePass);
            mpBeamQueueTilePass->execute(pRenderContext, uint3(mParams.beamTileDims.x * mParams.beamTileDims.y, 1, 1));
            writeBeamQueueArgs(pRenderContext, finest * finest);
            for (uint32_t bucket = 0; bucket < kBeamQueueBuckets; ++bucket)
            {
                mParams.beamQueueBucket = bucket;
                bindMarch();
                pMarch->executeIndirect(pRenderContext, mpBeamQueueArgs.get(), 12 * bucket); // Bytes: three uint32 per list.
            }
        }
        else
        {
            bindMarch();
            if (unitMarch && !mBeamGridRegions.empty())
            {
                // Only the units whose tile this build could have reclassified, or whose own carry could have expired: the same
                // regions the query and tile passes covered. Everything else already holds a residual for its own direction.
                const uint32_t ts = std::max(mParams.beamTileSize, 1u);
                mParams.beamUnitGenerated = 1;
                for (size_t r = 0; r < mBeamGridRegions.size(); ++r)
                {
                    const uint4& region = mBeamGridRegions[r];
                    mParams.beamGridOrigin = uint2(region.x, region.y);
                    mParams.beamGridBlocks = mBeamGridRegionMode[r];
                    bindMarch();
                    pMarch->execute(pRenderContext, uint3(region.z * ts, region.w * ts, 1));
                }
                mParams.beamUnitGenerated = 0;
                mParams.beamGridBlocks = 0;
                mParams.beamGridOrigin = uint2(0);
            }
            else if (unitMarch)
                pMarch->execute(pRenderContext, uint3(mParams.beamFrameDim, 1));
            else if (gridMarch)
                pMarch->execute(pRenderContext, uint3(mParams.frameDim, 1));
            else
                pMarch->executeIndirect(pRenderContext, mpBeamArgs.get(), 24 * mParams.beamLevels);
        }
        }
        if (mParams.beamRefFrame != 0)
            resolve();
    }
    else
    {
        FALCOR_PROFILE(pRenderContext, "resolve");
        // The steady-state camera pixel has its own entry point, which the other views, the probes and the reference display are not
        // part of; both programs also drop sunDepthAt's live march when the bakes are to answer every query.
        const bool cameraOnly = mCloudCameraKernel && mParams.debugView == kWorldCacheView && mParams.marchProbe == 0 &&
                                mParams.cloudDomain != 0 && !mpScene->getGridVolumes().empty();
        const ref<ComputePass>& pPass = cameraOnly ? mpCameraPass : mpPass;
        pPass->getProgram()->addDefine("HSTR_SUN_LIVE", mCloudSunLiveMarch ? "1" : "0");
        bindRenderer(pRenderContext, pPass);
        ShaderVar var = pPass->getRootVar()["CB"]["gHSTRCloud"];
        var["color"] = color;
        var["transportError"] = error;
        var["cutStats"] = cutStats;
        if (cameraOnly)
        {
            pRenderContext->clearUAV(error->getUAV().get(), float4(0.f));
            pRenderContext->clearUAV(cutStats->getUAV().get(), float4(0.f));
        }
        pPass->execute(pRenderContext, uint3(mParams.frameDim, 1));
    }

    if (mParams.cloudTransferWords > 0 && mpTransferProbe)
    {
        mTransferEntries = mpTransferProbe->getElement<uint32_t>(mParams.cloudTransferWords - 2u);
        mTransferCrossings = mpTransferProbe->getElement<uint32_t>(mParams.cloudTransferWords - 1u);
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
            for (uint32_t level = 0; level < kBeamCountSlots; ++level)
                mBeamLevelCounts[level] = mpBeamCounts[mBeamParity]->getElement<uint32_t>(level);
        }
        bindRenderer(pRenderContext, mpCompareReferencePass);
        ShaderVar var = mpCompareReferencePass->getRootVar()["CB"]["gHSTRCloud"];
        var["hstrReferenceSum"] = mpReferenceSum;
        var["hstrReferenceRowError"] = mpReferenceRowError;
        var["hstrReferenceRowHistogram"] = mpReferenceRowHistogram;
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
        mReferenceLogP999 = -1.f;
        mReferenceLogMax = -1.f;
        if (mParams.compareExact != 0 && mParams.compareBlock <= 1)
        {
            const std::vector<float> histogram = mpReferenceRowHistogram->getElements<float>(0, rowCount * kCompareBins);
            std::vector<double> bins(kCompareBins - 1, 0.0);
            mReferenceLogMax = 0.f;
            for (uint32_t row = 0; row < rowCount; ++row)
            {
                for (uint32_t b = 0; b + 1 < kCompareBins; ++b)
                    bins[b] += histogram[row * kCompareBins + b];
                mReferenceLogMax = std::max(mReferenceLogMax, histogram[row * kCompareBins + kCompareBins - 1]);
            }
            double pixels = 0.0;
            for (double v : bins)
                pixels += v;
            double above = 0.0;
            for (uint32_t b = kCompareBins - 1; b-- > 0;)
            {
                above += bins[b];
                if (above > 0.001 * pixels)
                {
                    mReferenceLogP999 = std::min(mReferenceLogMax, kCompareBinFloor * std::pow(2.f, float(b) / 4.f));
                    break;
                }
            }
            if (mReferenceLogP999 < 0.f)
                mReferenceLogP999 = 0.f;
        }
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
    const HSTRCloudParams previousUI = mParams;
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
        onLightingChanged(previousUI);
    if (operatorChanged || lightingChanged)
        mParams.referenceSamples = 0;
    if (operatorChanged || (lightingChanged && mParams.cloudDomain == 0))
        mParams.worldCacheSamples = 0;
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
