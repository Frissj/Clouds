/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
import Utils.Geometry.IntersectionHelpers;
import Utils.Math.Ray;
import Scene.Scene;
import HSTRCloudTypes;

static const uint kMaxTraceDofs = 24;
static const uint kMaxTileNodes = 128;
static const uint kSortGroupSize = 64;
static const float kMinTransmittance = 1e-3f;
static const float kPi = 3.14159265f;

struct HSTRCorrectionAtom
{
    uint row;
    uint col;
    float value;
    float goalError;
};

/// One reference path's radiance, split into the kComponent* components.
struct ReferenceSample
{
    float3 background;
    float3 sunSingle;
    float3 sunMultiple;
    float3 sky;

    /// Component by bit position.
    float3 component(uint c) { return c == 0 ? background : (c == 1 ? sunSingle : (c == 2 ? sunMultiple : sky)); }
};

/// One photon of the persistent light-tracing pool.
struct Photon
{
    float4 positionWeight;      ///< Voxel position and emission weight.
    float4 directionThroughput; ///< World direction and Russian-roulette throughput.
    uint rng;
    uint state; ///< Bounce count, kPhotonSky, kPhotonAlive.
    uint2 pad;
};
static const uint kPhotonBounceMask = 0xFFFF;
static const uint kPhotonSky = 1u << 16;
static const uint kPhotonAlive = 1u << 17;

/// Reference averages of a set of components over the even-frame half, the odd-frame half, and both.
struct ReferenceHalves
{
    float3 first;
    float3 second;
    float3 both;
};

/// Split-frame terms along one camera ray, before albedo: world-cache radiance, single-scattered sun without its phase factor,
/// and optical depth (capped where the march terminates).
struct BeamSample
{
    float3 cache;
    float3 single;
    float depth;
};

/// A primary ray restricted to the volume bounds, expressed in leaf space.
/// position is the entry point (t = 0), direction the entry-to-exit vector (t = 1).
struct CutRay
{
    float3 position;
    float3 direction;
    float worldLength;
    float4 sunOctaves;    ///< Per-octave sun weight a^n p(theta; c^n g) for this ray's direction.
    float3 viewDirection; ///< Unit world-space direction of the camera ray.
};

struct HSTRCloudRenderer
{
    HSTRCloudParams params;
    RWStructuredBuffer<float4> hstrRadiance;
    StructuredBuffer<float> hstrLeafBasisLeft;
    StructuredBuffer<float> hstrLeafBasisRight;
    StructuredBuffer<float> hstrLeafBallistic;
    StructuredBuffer<float> hstrLeafNearScatter;
    StructuredBuffer<float> hstrLeafDiffuseLeft;
    StructuredBuffer<float> hstrLeafDiffuseRight;
    StructuredBuffer<uint> hstrLeafDiffuseRanks;
    StructuredBuffer<float> hstrLeafResidualBounds;
    StructuredBuffer<float4> hstrRootIncident;
    StructuredBuffer<float> hstrLeafAdjointResponses;
    StructuredBuffer<uint2> hstrCorrectionRanges;
    StructuredBuffer<HSTRCorrectionAtom> hstrCorrectionAtoms;
    StructuredBuffer<HSTRCutNode> hstrCutNodes;
    StructuredBuffer<uint> hstrCutNodeParents;
    RWStructuredBuffer<float4> hstrNodeProjection;
    RWStructuredBuffer<float2> hstrNodeDepth;
    RWStructuredBuffer<uint> hstrNodeOrder;
    StructuredBuffer<uint> hstrCutStatePrevious;
    RWStructuredBuffer<uint> hstrCutState;
    RWStructuredBuffer<uint> hstrTileNodeCounts;
    RWStructuredBuffer<uint> hstrTileNodes;
    Texture3D<float> hstrExtinction;
    Texture3D<float> hstrMajorant;
    Texture3D<float4> hstrCameraLighting;
    RWTexture3D<float4> hstrCameraLightingOutput;

    // World-space residual pages: sun optical depth at every voxel, resident per leaf.
    Texture3D<float> hstrFineSun;
    RWTexture3D<float> hstrFineSunOutput;
    RWStructuredBuffer<uint> hstrLeafResidual;
    // Scatterer-weighted sun arrival for octaves 1-3 (rgb numerators, a weight), spread laterally.
    Texture3D<float4> hstrSunOctaves;
    RWTexture3D<float4> hstrSunOctavesOutput;

    // Camera basis.
    Texture2D<float4> hstrCameraQueries;
    RWTexture2D<float4> hstrCameraQueryOutput;
    Texture2D<float4> hstrTileCenters;
    RWTexture2D<float4> hstrTileCenterOutput;
    RWStructuredBuffer<HSTRTileBasis> hstrTileBasis;
    RWStructuredBuffer<uint> hstrTileState;

    // Path-traced reference: running sums per radiance component, in two halves that alternate by frame (slice
    // 2 * component + half, rgb sum and sample count), and per-row comparison against the HST frame.
    RWTexture2DArray<float4> hstrReferenceSum;
    RWStructuredBuffer<float4> hstrReferenceRowError;
    // World-space radiance cache experiment: 9 SH coefficients x RGB running sums per cell, and the light-tracing
    // batch's fixed-point deposits (9 SH coefficients for a unit sun, then 9 for a unit-colour sky, per cell).
    RWStructuredBuffer<float> hstrWorldCache;
    RWStructuredBuffer<int> hstrWorldCacheDeposit;
    // Persistent light-tracing pool and the number of photons it has emitted since the last restart.
    RWStructuredBuffer<Photon> hstrPhotonPool;
    RWStructuredBuffer<uint> hstrPhotonEmitted;
    // The cache means baked for the camera: slot s = 3k + c of the 27 in texture s / 4, channel s % 4, trilinear by hardware.
    Texture3D<float4> hstrWorldCacheTexture[kWorldCacheTextures];
    RWTexture3D<float4> hstrWorldCacheTextureOutput[kWorldCacheTextures];
    // Beam view: exact queries on the lattice of all tile corners and centres, slice 0 (cache rgb, optical depth), slice 1
    // (single scatter without phase, unused); the level that finalised every finest tile (beamLevels: marched per pixel);
    // the per-level lists of refined tiles (list beamLevels: finest tiles marched per pixel), their counts and indirect
    // arguments; and the stored exact frame for comparisons.
    Texture2DArray<float4> hstrBeamLattice;
    RWTexture2DArray<float4> hstrBeamLatticeOutput;
    Texture2D<uint> hstrBeamLevel;
    RWTexture2D<uint> hstrBeamLevelOutput;
    RWStructuredBuffer<uint> hstrBeamLists;
    RWStructuredBuffer<uint> hstrBeamCounts;
    RWStructuredBuffer<uint> hstrBeamArgs;
    // Temporal beam tiles: last frame's lists and counts, and per finest cell the levels at which the tile starting there was
    // refined (bit per level), last frame's and this frame's.
    StructuredBuffer<uint> hstrBeamPreviousLists;
    StructuredBuffer<uint> hstrBeamPreviousCounts;
    Texture2D<uint> hstrBeamHistory;
    RWTexture2D<uint> hstrBeamHistoryOutput;
    Texture2D<float4> hstrExactFrame;

    SamplerState hstrExtinctionSampler;
    SamplerState hstrLinearSampler;
    RWTexture2D<float4> color;
    RWTexture2D<float> transportError;
    RWTexture2D<float4> cutStats;

    float phaseHG(float cosine) { return phaseHG(cosine, params.anisotropy); }

    float phaseHG(float cosine, float g)
    {
        const float denominator = max(1e-4f, 1.f + g * g - 2.f * g * cosine);
        return (1.f - g * g) / (4.f * kPi * denominator * sqrt(denominator));
    }

    float densityAt(GridVolume volume, Grid grid, float3 worldPosition, inout Grid::Accessor accessor)
    {
        const float3 localPosition = mul(volume.data.invTransform, float4(worldPosition, 1.f)).xyz;
        return max(0.f, grid.lookupLinearIndex(localPosition, accessor)) * volume.data.densityScale * params.densityScale;
    }

    uint leafIndex(uint3 p) { return p.x + params.hstrLeafDims.x * (p.y + params.hstrLeafDims.y * p.z); }

    float3 sampleLeafFace(uint leaf, uint face, float3 localPosition)
    {
        const uint order = params.traceSpatialOrder;
        const uint faceDofs = params.hstrFaceDofs;
        const uint base = params.hstrTraceDofs * leaf + faceDofs * face;
        if (order == 1)
            return hstrRadiance[base].rgb;

        const uint faceAxis = face / 2;
        const uint tangent0 = faceAxis == 0 ? 1 : 0;
        const uint tangent1 = faceAxis == 2 ? 1 : 2;
        const float2 samplePosition = saturate(float2(localPosition[tangent0], localPosition[tangent1])) * order - 0.5f;
        const int2 sampleBase = int2(floor(samplePosition));
        const float2 f = frac(samplePosition);
        float3 result = 0.f;
        for (uint v = 0; v < 2; ++v)
            for (uint u = 0; u < 2; ++u)
            {
                const uint2 q = uint2(clamp(sampleBase + int2(u, v), int2(0), int2(order) - 1));
                const float2 w = select(uint2(u, v) == 0, 1.f - f, f);
                result += hstrRadiance[base + q.x + order * q.y].rgb * w.x * w.y;
            }
        return result;
    }

    float3 hstrField(float3 p, uint face)
    {
        const int3 base = int3(floor(p));
        const float3 f = frac(p);
        float3 result = 0.f;
        for (uint z = 0; z < 2; ++z)
            for (uint y = 0; y < 2; ++y)
                for (uint x = 0; x < 2; ++x)
                {
                    const int3 q = clamp(base + int3(x, y, z), int3(0), int3(params.hstrLeafDims) - 1);
                    const float3 w = select(uint3(x, y, z) == 0, 1.f - f, f);
                    const float3 localPosition = p - float3(q) + 0.5f;
                    result += sampleLeafFace(leafIndex(uint3(q)), face, localPosition) * w.x * w.y * w.z;
                }
        return result;
    }

    float hstrResidualBound(float3 indexPosition)
    {
        const float3 p = (indexPosition - float3(params.hstrGridMin)) / float(params.hstrCellWidth) - 0.5f;
        const int3 base = int3(floor(p));
        const float3 f = frac(p);
        float result = 0.f;
        for (uint z = 0; z < 2; ++z)
            for (uint y = 0; y < 2; ++y)
                for (uint x = 0; x < 2; ++x)
                {
                    const int3 q = clamp(base + int3(x, y, z), int3(0), int3(params.hstrLeafDims) - 1);
                    const float3 w = select(uint3(x, y, z) == 0, 1.f - f, f);
                    result += hstrRadiance[params.hstrTraceDofs * leafIndex(uint3(q))].a * w.x * w.y * w.z;
                }
        return result;
    }

    float3 hstrLighting(float3 indexPosition, float3 viewDirection)
    {
        const float3 p = (indexPosition - float3(params.hstrGridMin)) / float(params.hstrCellWidth) - 0.5f;
        const float3 outgoing = -normalize(viewDirection);
        float3 result = 0.f;
        result += abs(outgoing.x) * hstrField(p, outgoing.x < 0.f ? 0u : 1u);
        result += abs(outgoing.y) * hstrField(p, outgoing.y < 0.f ? 2u : 3u);
        result += abs(outgoing.z) * hstrField(p, outgoing.z < 0.f ? 4u : 5u);
        return result / max(1e-4f, abs(outgoing.x) + abs(outgoing.y) + abs(outgoing.z));
    }

    // ---------------------------------------------------------------------------------------------
    // Field lookups. v denotes voxel coordinates relative to the grid minimum (voxel i is centred at
    // v = i) and leaf-space positions are v / hstrCellWidth.

    float extinctionAtVoxel(float3 v)
    {
        return hstrExtinction.SampleLevel(hstrExtinctionSampler, (v + 0.5f) / float3(params.hstrExtinctionDims), 0.f);
    }

    float extinctionAt(float3 leafPosition) { return extinctionAtVoxel(leafPosition * float(params.hstrCellWidth)); }

    /// Upper bound on the trilinear extinction within one majorant block of v in any direction.
    float majorantAt(float3 v)
    {
        const int3 block = clamp(int3(floor(v * 0.25f)), int3(0), int3(params.hstrMajorantDims) - 1);
        return hstrMajorant.Load(int4(block, 0));
    }

    float3 cameraLightingAt(float3 leafPosition)
    {
        return hstrCameraLighting.SampleLevel(hstrLinearSampler, saturate(leafPosition / float3(params.hstrLeafDims)), 0.f).rgb;
    }

    /// Camera-basis source radiance towards the camera (before albedo).
    float3 sourceAtVoxel(float3 v) { return cameraLightingAt(v / float(params.hstrCellWidth)); }

    /// Camera factor of the sun pages for one ray direction: octave n contributes a^n p(theta; c^n g).
    float4 sunOctaveWeights(float3 direction)
    {
        const float cosine = dot(normalize(params.sunDirection), normalize(direction));
        const float a = params.octaveEnergy;
        const float c = params.octavePhase;
        const float g = params.anisotropy;
        return float4(phaseHG(cosine, g), a * phaseHG(cosine, c * g), a * a * phaseHG(cosine, c * c * g), a * a * a * phaseHG(cosine, c * c * c * g));
    }

    /// Rasterized sun residual at v (before albedo): the page's sun optical depth attenuated per octave.
    /// The stored depth is read one voxel upstream towards the sun and the last voxel is integrated from the trilinear density
    /// at v itself, so the sub-voxel lit shell of thick surfaces follows the density instead of the voxel grid.
    float3 residualAtVoxel(float3 v, float4 octaves)
    {
        const float3 upstream = v + params.sunVoxelDirection;
        const float depth = hstrFineSun.SampleLevel(hstrLinearSampler, (upstream + 0.5f) / float3(params.hstrFineDims), 0.f) +
                            extinctionAtVoxel(v + 0.5f * params.sunVoxelDirection) * params.sunVoxelWorldLength;
        if (all(octaves.yzw == 0.f))
            return params.residualStrength * params.sunRadiance * octaves.x * exp(-depth);
        // Octaves 1-3 are multiply scattered: they arrive from the scatterers around v, not along v's own sun ray.
        const float4 spread = hstrSunOctaves.SampleLevel(hstrLinearSampler, (v + 0.5f) / (2.f * float3(params.hstrOctaveDims)), 0.f);
        const float b = params.octaveExtinction;
        const float3 local = exp(-depth * float3(b, b * b, b * b * b));
        const float3 multiple = (spread.rgb + 1e-3f * local) / (spread.a + 1e-3f);
        const float4 transmittance = float4(exp(-depth), multiple);
        return params.residualStrength * params.sunRadiance * dot(octaves, transmittance);
    }

    /// Scatterer-weighted sun arrival of octaves 1-3 over one 2-voxel cell: sum of sigma * exp(-b^n tau) and sum of sigma.
    void gatherSunOctaves(uint3 cell)
    {
        if (any(cell >= params.hstrOctaveDims))
            return;
        const float b = params.octaveExtinction;
        float4 sum = 0.f;
        for (uint z = 0; z < 2; ++z)
            for (uint y = 0; y < 2; ++y)
                for (uint x = 0; x < 2; ++x)
                {
                    const uint3 voxel = 2u * cell + uint3(x, y, z);
                    if (any(voxel >= params.hstrFineDims))
                        continue;
                    const float extinction = hstrExtinction.Load(int4(voxel, 0));
                    const float depth = hstrFineSun[voxel];
                    sum += extinction * float4(exp(-depth * float3(b, b * b, b * b * b)), 1.f);
                }
        hstrSunOctavesOutput[cell] = sum;
    }

    /// One axis of the separable Gaussian that spreads multiply scattered sun light between scatterers.
    void blurSunOctaves(uint3 cell)
    {
        if (any(cell >= params.hstrOctaveDims))
            return;
        const int radius = int(ceil(2.5f * params.octaveBlurSigma));
        int3 axis = 0;
        axis[params.octaveBlurAxis] = 1;
        float4 sum = 0.f;
        for (int i = -radius; i <= radius; ++i)
        {
            const int3 p = int3(cell) + i * axis;
            if (any(p < 0) || any(p >= int3(params.hstrOctaveDims)))
                continue;
            sum += exp(-0.5f * float(i * i) / (params.octaveBlurSigma * params.octaveBlurSigma)) * hstrSunOctaves[uint3(p)];
        }
        hstrSunOctavesOutput[cell] = sum;
    }

    bool leafResidualResident(uint3 leaf) { return asfloat(hstrLeafResidual[leafIndex(leaf)]) > params.residualTolerance; }

    // ---------------------------------------------------------------------------------------------
    // Camera-side basis update: evaluated on camera motion only.

    void writeCameraLighting(uint3 samplePosition)
    {
        if (any(samplePosition >= params.hstrCameraLightingDims) || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        // Two samples per leaf axis resolve the 2x2 spatial modes of every retained face trace.
        const float3 leafPosition = (float3(samplePosition) + 0.5f) * float3(params.hstrLeafDims) / float3(params.hstrCameraLightingDims);
        const float3 indexPosition = float3(params.hstrGridMin) + float(params.hstrCellWidth) * leafPosition;
        const float3 worldPosition = mul(volume.data.transform, float4(indexPosition, 1.f)).xyz;
        hstrCameraLightingOutput[samplePosition] = float4(hstrLighting(indexPosition, worldPosition - gScene.camera.data.posW), 0.f);
    }

    float3 cutNodeWorldPosition(uint3 leafPosition, GridVolume volume)
    {
        const float3 indexPosition = float3(params.hstrGridMin) + float(params.hstrCellWidth) * float3(leafPosition);
        return mul(volume.data.transform, float4(indexPosition, 1.f)).xyz;
    }

    void projectCutNode(uint nodeID)
    {
        if (nodeID >= params.hstrNodeCount || gScene.getGridVolumeCount() == 0)
            return;
        const HSTRCutNode node = hstrCutNodes[nodeID];
        const GridVolume volume = gScene.getGridVolume(0);
        float2 minimumPixel = float2(params.frameDim);
        float2 maximumPixel = 0.f;
        float minimumDepth = 1e30f;
        float maximumDepth = -1e30f;
        bool crossesCamera = false;
        [unroll]
        for (uint corner = 0; corner < 8; ++corner)
        {
            const uint3 leafPosition = uint3(
                (corner & 1u) ? node.maxRight.x : node.minLeft.x,
                (corner & 2u) ? node.maxRight.y : node.minLeft.y,
                (corner & 4u) ? node.maxRight.z : node.minLeft.z
            );
            const float3 worldPosition = cutNodeWorldPosition(leafPosition, volume);
            const float4 clip = mul(gScene.camera.data.viewProjMatNoJitter, float4(worldPosition, 1.f));
            crossesCamera |= clip.w <= 1e-5f;
            const float2 ndc = clip.xy / max(1e-5f, clip.w);
            const float2 pixel = (ndc * float2(0.5f, -0.5f) + 0.5f) * float2(params.frameDim);
            minimumPixel = min(minimumPixel, pixel);
            maximumPixel = max(maximumPixel, pixel);
            minimumDepth = min(minimumDepth, dot(worldPosition - gScene.camera.data.posW, gScene.camera.data.cameraW));
            maximumDepth = max(maximumDepth, dot(worldPosition - gScene.camera.data.posW, gScene.camera.data.cameraW));
        }
        if (crossesCamera)
        {
            minimumPixel = 0.f;
            maximumPixel = float2(params.frameDim);
        }
        hstrNodeProjection[nodeID] = float4(clamp(minimumPixel, 0.f, float2(params.frameDim)), clamp(maximumPixel, 0.f, float2(params.frameDim)));
        hstrNodeDepth[nodeID] = float2(minimumDepth, maximumDepth);
        hstrNodeOrder[nodeID] = bspOrderKey(nodeID, volume);
    }

    /// Front-to-back key of a node in the HST's binary space partition seen from the camera: one bit per ancestor split,
    /// set when the node lies in the far child, most significant bit at the root. For disjoint nodes the key order is the
    /// painter's order of every ray leaving the eye, so a tile's cut sorted by it composites correctly for all its pixels.
    uint bspOrderKey(uint nodeID, GridVolume volume)
    {
        const float3 cameraIndex = mul(volume.data.invTransform, float4(gScene.camera.data.posW, 1.f)).xyz;
        const float3 camera = (cameraIndex - float3(params.hstrGridMin)) / float(params.hstrCellWidth);
        uint path = 0;
        uint depth = 0;
        uint child = nodeID;
        uint parent = hstrCutNodeParents[nodeID];
        [loop]
        while (parent != ~0u && depth < 31)
        {
            const HSTRCutNode node = hstrCutNodes[parent];
            const HSTRCutNode left = hstrCutNodes[node.minLeft.w];
            // The children differ along the split axis only; the left child is the lower half.
            const uint3 lowerMax = left.maxRight.xyz;
            const uint axis = lowerMax.x != node.maxRight.x ? 0u : (lowerMax.y != node.maxRight.y ? 1u : 2u);
            const bool leftNear = camera[axis] < float(lowerMax[axis]);
            const bool isLeft = child == node.minLeft.w;
            if (isLeft != leftNear)
                path |= 1u << depth;
            ++depth;
            child = parent;
            parent = hstrCutNodeParents[parent];
        }
        // path holds the deepest split in bit 0 and the root split in bit depth - 1. Left-aligning puts the root split in the
        // most significant bit, so nodes at different depths compare lexicographically.
        return depth == 0 ? 0u : path << (32u - depth);
    }

    bool acceptTileCutNode(HSTRCutNode node, uint nodeID, GridVolume volume)
    {
        if (node.minLeft.w == ~0u)
            return true;
        const float3 leafExtent = float3(node.maxRight.xyz - node.minLeft.xyz) * float(params.hstrCellWidth);
        const float3 axisX = mul(volume.data.transform, float4(leafExtent.x, 0.f, 0.f, 0.f)).xyz;
        const float3 axisY = mul(volume.data.transform, float4(0.f, leafExtent.y, 0.f, 0.f)).xyz;
        const float3 axisZ = mul(volume.data.transform, float4(0.f, 0.f, leafExtent.z, 0.f)).xyz;
        const float maximumWorldLength = length(axisX) + length(axisY) + length(axisZ);
        const float pageError = 1.f - exp(-node.transmittancePage.x * maximumWorldLength);
        if (pageError <= params.cutTransmittanceTolerance)
            return true;
        const float variation = (node.statistics.y - node.statistics.x) / max(1e-6f, node.statistics.y);
        const float4 bounds = hstrNodeProjection[nodeID];
        const float projectedPixels = max(bounds.z - bounds.x, bounds.w - bounds.y);
        // Split/merge hysteresis: a node that was accepted keeps its page until it grows past the upper band, and a split node
        // only merges once it shrinks below the lower band, so an orbiting camera does not flicker between cuts.
        const bool wasAccepted = hstrCutStatePrevious[nodeID] != 0;
        const float band = wasAccepted ? 1.f + params.cutHysteresis : 1.f - params.cutHysteresis;
        return variation <= params.mixedFidelityThreshold && node.statistics.z <= params.mixedFidelityThreshold &&
               projectedPixels <= 12.f * band;
    }

    void binCutNode(uint nodeID)
    {
        if (nodeID >= params.hstrNodeCount || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        const HSTRCutNode node = hstrCutNodes[nodeID];
        const bool accepted = acceptTileCutNode(node, nodeID, volume);
        hstrCutState[nodeID] = accepted ? 1u : 0u;
        if (node.statistics.y <= 1e-7f || !accepted)
            return;

        // Every node tests its ancestor chain, producing one scene-wide antichain
        // without repeating a hierarchy traversal for every screen tile.
        uint parent = hstrCutNodeParents[nodeID];
        [loop]
        while (parent != ~0u)
        {
            if (acceptTileCutNode(hstrCutNodes[parent], parent, volume))
                return;
            parent = hstrCutNodeParents[parent];
        }

        const float4 bounds = hstrNodeProjection[nodeID];
        const uint2 tileMinimum = min(uint2(floor(bounds.xy / float(params.hstrTileSize))), params.hstrTileDims);
        const uint2 tileMaximum = min(uint2(ceil(bounds.zw / float(params.hstrTileSize))), params.hstrTileDims);
        [loop]
        for (uint y = tileMinimum.y; y < tileMaximum.y; ++y)
            [loop]
            for (uint x = tileMinimum.x; x < tileMaximum.x; ++x)
            {
                const uint tileID = x + params.hstrTileDims.x * y;
                uint slot;
                InterlockedAdd(hstrTileNodeCounts[tileID], 1u, slot);
                if (slot < params.hstrMaxTileNodes)
                    hstrTileNodes[tileID * params.hstrMaxTileNodes + slot] = nodeID;
            }
    }

    bool intersectCutNode(HSTRCutNode node, float3 origin, float3 direction, out float2 interval)
    {
        const bool3 parallel = abs(direction) < 1e-8f;
        if (any(parallel && (origin < float3(node.minLeft.xyz) || origin >= float3(node.maxRight.xyz))))
        {
            interval = 0.f;
            return false;
        }
        const float3 inverseDirection = 1.f / select(parallel, float3(1.f), direction);
        const float3 a = (float3(node.minLeft.xyz) - origin) * inverseDirection;
        const float3 b = (float3(node.maxRight.xyz) - origin) * inverseDirection;
        const float3 lo = select(parallel, float3(-1e30f), min(a, b));
        const float3 hi = select(parallel, float3(1e30f), max(a, b));
        interval = float2(max(lo.x, max(lo.y, lo.z)), min(hi.x, min(hi.y, hi.z)));
        interval = clamp(interval, 0.f, 1.f);
        return interval.x < interval.y;
    }

    float3 cameraRayDirection(float2 pixelPosition)
    {
        const float2 p = pixelPosition / float2(params.frameDim);
        const float2 ndc = float2(2.f, -2.f) * p + float2(-1.f, 1.f);
        return normalize(ndc.x * gScene.camera.data.cameraU + ndc.y * gScene.camera.data.cameraV + gScene.camera.data.cameraW);
    }

    bool makeCutRay(float3 origin, float3 direction, GridVolume volume, out CutRay ray)
    {
        ray.position = 0.f;
        ray.direction = 0.f;
        ray.worldLength = 0.f;
        ray.sunOctaves = 0.f;
        ray.viewDirection = normalize(direction);
        float2 interval;
        const AABB bounds = volume.getBounds();
        if (!intersectRayAABB(origin, direction, bounds.minPoint, bounds.maxPoint, interval))
            return false;
        const float entry = max(0.f, interval.x);
        if (interval.y <= entry)
            return false;
        const float3 entryIndex = mul(volume.data.invTransform, float4(origin + entry * direction, 1.f)).xyz;
        const float3 exitIndex = mul(volume.data.invTransform, float4(origin + interval.y * direction, 1.f)).xyz;
        ray.position = (entryIndex - float3(params.hstrGridMin)) / float(params.hstrCellWidth);
        ray.direction = (exitIndex - entryIndex) / float(params.hstrCellWidth);
        ray.worldLength = length(direction) * (interval.y - entry);
        const float single = (params.hstComponents & kComponentSunSingle) != 0 ? 1.f : 0.f;
        const float multiple = (params.hstComponents & kComponentSunMultiple) != 0 ? 1.f : 0.f;
        ray.sunOctaves = sunOctaveWeights(direction) * float4(single, multiple, multiple, multiple);
        return true;
    }

    // ---------------------------------------------------------------------------------------------
    // Transmittance and source integration along the projected cut.

    /// Majorant-bounded march of [segment.x, segment.y]: every step spans at most stepOpticalDepth at the local extinction
    /// bound, so thick shells are resolved at sub-voxel spacing while empty blocks are crossed in one step.
    /// radiance accumulates the camera-basis source when withSource is set; residualRadiance accumulates the rasterized residual
    /// whenever the segment lies in a resident page's footprint.
    void marchSegment(
        CutRay ray,
        float2 segment,
        bool withSource,
        bool residual,
        inout float3 radiance,
        inout float3 residualRadiance,
        inout float transmittance,
        inout uint steps
    )
    {
        const float cellWidth = float(params.hstrCellWidth);
        const float voxelsPerUnit = length(ray.direction) * cellWidth;
        if (voxelsPerUnit <= 1e-6f)
            return;
        const float worldPerVoxel = ray.worldLength / voxelsPerUnit;
        const bool cached = params.debugView == kWorldCacheView || params.debugView == kBeamView;
        float3 held = 0.f;
        float3 heldAt = 0.f;
        uint sinceLighting = ~0u;
        float t = segment.x;
        [loop]
        while (t < segment.y && transmittance > kMinTransmittance && steps < params.maxMarchSteps)
        {
            const float3 v = (ray.position + t * ray.direction) * cellWidth;
            const float majorant = majorantAt(v);
            const float stepVoxels = clamp(
                params.stepOpticalDepth / max(1e-6f, majorant * worldPerVoxel), params.minStepVoxels, params.maxStepVoxels
            );
            const float dt = min(stepVoxels / voxelsPerUnit, segment.y - t);
            if (majorant > 0.f)
            {
                const float3 midpoint = (ray.position + (t + 0.5f * dt) * ray.direction) * cellWidth;
                const float alpha = 1.f - exp(-extinctionAtVoxel(midpoint) * dt * ray.worldLength);
                if ((withSource || residual) && alpha > 1e-6f)
                {
                    if (withSource)
                    {
                        // The cache is smooth at cell scale: it is re-evaluated every lightingStride contributing steps, and
                        // whenever the march has moved a cache cell since, so a held value never crosses a gap between puffs.
                        const float3 moved = midpoint - heldAt;
                        const float cellVoxels = float(params.worldCacheCellVoxels);
                        if (!cached || sinceLighting >= params.lightingStride || dot(moved, moved) > cellVoxels * cellVoxels)
                        {
                            held = cached ? worldCacheRadiance(midpoint, ray.viewDirection) : sourceAtVoxel(midpoint);
                            heldAt = midpoint;
                            sinceLighting = 0;
                        }
                        ++sinceLighting;
                        radiance += transmittance * alpha * held;
                    }
                    if (residual)
                        residualRadiance += transmittance * alpha * residualAtVoxel(midpoint, ray.sunOctaves);
                }
                transmittance *= 1.f - alpha;
            }
            t += dt;
            ++steps;
        }
    }

    float transmittancePageAt(HSTRCutNode node, float3 leafPosition)
    {
        const float3 local = saturate((leafPosition - float3(node.minLeft.xyz)) / float3(node.maxRight.xyz - node.minLeft.xyz));
        const float x00 = lerp(node.transmittanceCorners0.x, node.transmittanceCorners0.y, local.x);
        const float x10 = lerp(node.transmittanceCorners0.z, node.transmittanceCorners0.w, local.x);
        const float x01 = lerp(node.transmittanceCorners1.x, node.transmittanceCorners1.y, local.x);
        const float x11 = lerp(node.transmittanceCorners1.z, node.transmittanceCorners1.w, local.x);
        return lerp(lerp(x00, x10, local.y), lerp(x01, x11, local.y), local.z);
    }

    /// Accepted HST node: two-point Gauss quadrature of its trilinear transmittance page.
    void integratePage(HSTRCutNode node, CutRay ray, float2 segment, bool withSource, inout float3 radiance, inout float transmittance)
    {
        static const float kGaussOffset = 0.2886751345948129f;
        const float midpoint = 0.5f * (segment.x + segment.y);
        const float segmentLength = segment.y - segment.x;
        const float3 p0 = ray.position + (midpoint - kGaussOffset * segmentLength) * ray.direction;
        const float3 p1 = ray.position + (midpoint + kGaussOffset * segmentLength) * ray.direction;
        const float extinction0 = transmittancePageAt(node, p0);
        const float extinction1 = transmittancePageAt(node, p1);
        const float opticalDepth = 0.5f * ray.worldLength * segmentLength * (extinction0 + extinction1);
        if (opticalDepth <= 1e-8f)
            return;
        const float segmentTransmittance = exp(-opticalDepth);
        if (withSource)
        {
            const float weight0 = extinction0 * exp(-opticalDepth * (0.5f - kGaussOffset));
            const float weight1 = extinction1 * exp(-opticalDepth * (0.5f + kGaussOffset));
            const float3 source = (weight0 * cameraLightingAt(p0) + weight1 * cameraLightingAt(p1)) / max(1e-8f, weight0 + weight1);
            radiance += transmittance * (1.f - segmentTransmittance) * source;
        }
        transmittance *= segmentTransmittance;
    }

    /// Composites one tile's front-to-back sorted cut. Leaves whose page error exceeds the tolerance, or that hold a
    /// resident residual page, are marched; accepted nodes use their transmittance pages. Returns radiance before albedo.
    float3 traverseCut(
        CutRay ray,
        uint tileID,
        bool withSource,
        bool withResidual,
        inout float3 residualRadiance,
        inout float transmittance,
        inout uint visited,
        inout uint steps
    )
    {
        float3 radiance = 0.f;
        const uint count = hstrTileNodeCounts[tileID];
        if (count > params.hstrMaxTileNodes)
        {
            // The tile's cut overflowed its list: march the whole volume segment on the majorant grid.
            marchSegment(ray, float2(0.f, 1.f), withSource, withResidual, radiance, residualRadiance, transmittance, steps);
            visited = count;
            return radiance;
        }
        const uint base = tileID * params.hstrMaxTileNodes;
        [loop]
        for (uint i = 0; i < count && transmittance > kMinTransmittance; ++i)
        {
            const HSTRCutNode node = hstrCutNodes[hstrTileNodes[base + i]];
            float2 segment;
            if (!intersectCutNode(node, ray.position, ray.direction, segment))
                continue;
            ++visited;
            const bool leaf = node.minLeft.w == ~0u;
            const float pageError = transmittance * (1.f - exp(-node.transmittancePage.x * ray.worldLength * (segment.y - segment.x)));
            // Resident residual pages are the leaves whose footprint this tile's cut list rasterized; their segments are marched
            // in every pass so that transmittance stays identical between camera queries and pixels.
            const bool resident = leaf && leafResidualResident(node.minLeft.xyz);
            if (leaf && (resident || pageError > params.cutTransmittanceTolerance))
                marchSegment(ray, segment, withSource, withResidual && resident, radiance, residualRadiance, transmittance, steps);
            else
                integratePage(node, ray, segment, withSource, radiance, transmittance);
        }
        return radiance;
    }

    /// Premultiplied cloud radiance (rgb, background excluded) and opacity (a) along one camera ray.
    float4 cameraQuery(float2 pixelPosition, uint tileID)
    {
        const GridVolume volume = gScene.getGridVolume(0);
        CutRay ray;
        if (!makeCutRay(gScene.camera.data.posW, cameraRayDirection(pixelPosition), volume, ray))
            return 0.f;
        float transmittance = 1.f;
        uint visited = 0;
        uint steps = 0;
        float3 residual = 0.f;
        // The tile basis carries only the low-frequency camera basis; residuals are rasterized per pixel.
        const float3 radiance = traverseCut(ray, tileID, true, false, residual, transmittance, visited, steps);
        return float4(radiance * volume.data.albedo, 1.f - transmittance);
    }

    /// Tile-corner lattice plus one centre query per tile. Deterministic, and only rebuilt when the camera, cut or lighting change.
    void writeCameraQuery(uint2 query)
    {
        if (any(query >= params.hstrLatticeDims) || gScene.getGridVolumeCount() == 0)
            return;
        const uint2 tile = min(query, params.hstrTileDims - 1u);
        const uint tileID = tile.x + params.hstrTileDims.x * tile.y;
        const float tileSize = float(params.hstrTileSize);
        hstrCameraQueryOutput[query] = cameraQuery(float2(query) * tileSize, tileID);
        if (all(query < params.hstrTileDims))
            hstrTileCenterOutput[query] = cameraQuery((float2(query) + 0.5f) * tileSize, tileID);
    }

    float maxComponent(float3 v) { return max(v.x, max(v.y, v.z)); }

    /// Fits the tile camera basis and decides, with split/merge hysteresis, whether the tile descends to per-pixel integration.
    void buildTileBasis(uint2 tile)
    {
        if (any(tile >= params.hstrTileDims) || gScene.getGridVolumeCount() == 0)
            return;
        const uint tileID = tile.x + params.hstrTileDims.x * tile.y;
        const float4 q00 = hstrCameraQueries[tile];
        const float4 q10 = hstrCameraQueries[tile + uint2(1, 0)];
        const float4 q01 = hstrCameraQueries[tile + uint2(0, 1)];
        const float4 q11 = hstrCameraQueries[tile + uint2(1, 1)];
        const float4 qc = hstrTileCenters[tile];
        HSTRTileBasis basis;
        basis.c0 = q00;
        basis.cx = q10 - q00;
        basis.cy = q01 - q00;
        basis.cxy = q11 - q10 - q01 + q00;
        const float4 bilinearCentre = 0.25f * (q00 + q10 + q01 + q11);
        // Hierarchical surplus: the centre query's departure from the bilinear corner basis. It becomes the coefficient of a
        // bubble function that vanishes on the tile boundary, and is the tile's a-posteriori error indicator.
        basis.bubble = qc - bilinearCentre;
        const float3 exactSource = qc.rgb / max(qc.a, 1e-4f);
        const float3 predictedSource = bilinearCentre.rgb / max(bilinearCentre.a, 1e-4f);
        float surplus = qc.a * maxComponent(abs(exactSource - predictedSource)) / (maxComponent(exactSource) + 1e-2f);
        if ((qc.a > 1e-3f) != (bilinearCentre.a > 1e-3f))
            surplus = 1.f;

        const bool wasRefined = hstrTileState[tileID] != 0;
        const bool refine = surplus > params.basisTolerance * (wasRefined ? params.basisHysteresis : 1.f);
        hstrTileState[tileID] = refine ? 1u : 0u;
        basis.info = float4(surplus, refine ? 1.f : 0.f, min(min(q00.a, q10.a), min(q01.a, q11.a)), 0.f);
        hstrTileBasis[tileID] = basis;
    }

    float4 evaluateTileBasis(HSTRTileBasis basis, float2 f)
    {
        return basis.c0 + basis.cx * f.x + basis.cy * f.y + basis.cxy * (f.x * f.y) +
               basis.bubble * (16.f * f.x * (1.f - f.x) * f.y * (1.f - f.y));
    }

    // ---------------------------------------------------------------------------------------------
    // World-space residual pages: rebuilt when the sun or the density changes, never on camera motion.

    float voxelWorldLength(GridVolume volume)
    {
        return (length(mul(volume.data.transform, float4(1.f, 0.f, 0.f, 0.f)).xyz) +
                length(mul(volume.data.transform, float4(0.f, 1.f, 0.f, 0.f)).xyz) +
                length(mul(volume.data.transform, float4(0.f, 0.f, 1.f, 0.f)).xyz)) /
               3.f;
    }

    float3 fineCellVoxel(uint3 cell) { return 2.f * float3(cell) + 1.f; }

    /// Sun optical depth from the centre of every voxel to the grid boundary, and page activation per leaf.
    void computeFineSun(uint3 voxel)
    {
        if (any(voxel >= params.hstrFineDims) || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        const float3 sunIndex = mul(volume.data.invTransform, float4(normalize(params.sunDirection), 0.f)).xyz;
        const float voxelsPerWorld = length(sunIndex);
        const float3 direction = sunIndex / voxelsPerWorld;
        const float3 origin = float3(voxel);
        const float3 inverseDirection = 1.f / select(abs(direction) < 1e-8f, float3(1e-8f), direction);
        const float3 a = (-0.5f - origin) * inverseDirection;
        const float3 b = (float3(params.hstrExtinctionDims) - 0.5f - origin) * inverseDirection;
        const float3 far = max(a, b);
        const float exitDistance = max(0.f, min(far.x, min(far.y, far.z)));
        float opticalDepth = 0.f;
        float t = 0.f;
        uint steps = 0;
        [loop]
        while (t < exitDistance && opticalDepth < 60.f && steps < 4096)
        {
            const float3 v = origin + t * direction;
            const float majorant = majorantAt(v);
            const float stepVoxels = min(
                clamp(params.stepOpticalDepth * voxelsPerWorld / max(1e-6f, majorant), params.minStepVoxels * 0.5f, params.maxStepVoxels),
                exitDistance - t
            );
            if (majorant > 0.f)
                opticalDepth += extinctionAtVoxel(v + 0.5f * stepVoxels * direction) * stepVoxels / voxelsPerWorld;
            t += stepVoxels;
            ++steps;
        }
        opticalDepth = min(opticalDepth, 60.f);
        hstrFineSunOutput[voxel] = opticalDepth;
        // Residency: the most penetrating octave's sun arrival, weighted by the voxel's opacity.
        const float bb = params.octaveExtinction;
        const float opacity = 1.f - exp(-extinctionAtVoxel(origin) * voxelWorldLength(volume));
        const uint3 leaf = min(voxel / params.hstrCellWidth, params.hstrLeafDims - 1u);
        InterlockedMax(hstrLeafResidual[leafIndex(leaf)], asuint(opacity * exp(-bb * bb * bb * opticalDepth)));
    }

    /// Pages below the activation tolerance are not resident: their cells contribute nothing.
    void maskResidual(uint3 cell)
    {
        if (any(cell >= params.hstrFineDims))
            return;
        const uint3 leaf = min(cell / params.hstrCellWidth, params.hstrLeafDims - 1u);
        if (!leafResidualResident(leaf))
            hstrFineSunOutput[cell] = 60.f;
    }

    // ---------------------------------------------------------------------------------------------
    // Inside-cloud path: the local high-resolution working set.

    float3 deterministicResidual(GridVolume volume, Grid grid, float3 indexPosition, float3 viewDirection, inout Grid::Accessor accessor)
    {
        const float h = 1.5f;
        const float3 gradient = float3(
            grid.lookupLinearIndex(indexPosition + float3(h, 0.f, 0.f), accessor) -
                grid.lookupLinearIndex(indexPosition - float3(h, 0.f, 0.f), accessor),
            grid.lookupLinearIndex(indexPosition + float3(0.f, h, 0.f), accessor) -
                grid.lookupLinearIndex(indexPosition - float3(0.f, h, 0.f), accessor),
            grid.lookupLinearIndex(indexPosition + float3(0.f, 0.f, h), accessor) -
                grid.lookupLinearIndex(indexPosition - float3(0.f, 0.f, h), accessor)
        );
        const float density = max(0.f, grid.lookupLinearIndex(indexPosition, accessor));
        const float edge = saturate(2.f * length(gradient) / max(0.02f, density));
        if (edge <= 0.f)
            return 0.f;

        const float3 outwardIndex = -normalize(gradient + 1e-6f);
        const float3 sunIndex = normalize(mul(volume.data.invTransform, float4(normalize(params.sunDirection), 0.f)).xyz);
        const float facing = saturate(dot(outwardIndex, sunIndex));
        const float silver = facing * facing * edge;
        const float phase = 4.f * kPi * phaseHG(dot(normalize(params.sunDirection), viewDirection));
        return params.sunRadiance * (0.01f + 0.075f * silver) * (0.25f + phase);
    }

    float pixelJitter(uint2 pixel)
    {
        const uint seed = pixel.x * 1973u + pixel.y * 9277u + params.frameIndex * 26699u + 911u;
        uint value = seed;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        return (value & 0x00ffffffu) / 16777216.f;
    }

    float4 integrateVolume(Ray ray, GridVolume volume, uint stepCount, uint2 pixel, out float residualError)
    {
        residualError = 0.f;
        const float3 sky = params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * ray.dir.y));
        float2 interval;
        AABB bounds = volume.getBounds();
        if (!intersectRayAABB(ray.origin, ray.dir, bounds.minPoint, bounds.maxPoint, interval))
            return float4(sky, 1.f);
        interval.x = max(interval.x, 0.f);
        if (interval.x >= interval.y || !volume.hasDensityGrid())
            return float4(sky, 1.f);

        Grid grid;
        gScene.getGrid(volume.getDensityGrid(), grid);
        Grid::Accessor accessor = grid.createAccessor();

        const float stepLength = (interval.y - interval.x) / stepCount;
        float3 radiance = 0.f;
        float transmittance = 1.f;
        const float jitter = pixelJitter(pixel + uint2(stepCount, stepCount * 7));
        for (uint step = 0; step < stepCount && transmittance > 1e-3f; ++step)
        {
            const float t = interval.x + (step + jitter) * stepLength;
            const float3 position = ray.origin + t * ray.dir;
            const float extinction = densityAt(volume, grid, position, accessor);
            if (extinction <= 0.f)
                continue;

            const float alpha = 1.f - exp(-extinction * stepLength);
            const float3 indexPosition = mul(volume.data.invTransform, float4(position, 1.f)).xyz;
            const float3 base = hstrLighting(indexPosition, ray.dir);
            const int3 leafPosition = int3((indexPosition - float3(params.hstrGridMin)) / float(params.hstrCellWidth));
            const int3 windowOffset = abs(leafPosition - int3(params.schurWindowCenter));
            const bool insideWindow = all(windowOffset <= int(params.schurWindowRadius));
            const float3 correction = insideWindow ? deterministicResidual(volume, grid, indexPosition, ray.dir, accessor) : 0.f;
            const float3 source = volume.data.albedo * (0.72f * base + params.residualBlend * correction);
            radiance += transmittance * alpha * source;
            residualError += transmittance * alpha * hstrResidualBound(indexPosition);
            transmittance *= 1.f - alpha;
        }
        return float4(radiance + transmittance * sky, transmittance);
    }

    // ---------------------------------------------------------------------------------------------
    // Reference: unbiased volumetric path tracer on the same extinction field, sun and sky.

    float3 skyRadianceAt(float3 direction) { return params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * direction.y)); }

    bool insideExtinctionGrid(float3 v) { return all(v >= -1.f) && all(v <= float3(params.hstrExtinctionDims)); }

    /// Distance (world units along the ray) to the exit of v's majorant block; d is the voxel displacement per world unit.
    float majorantBlockExit(float3 v, float3 d)
    {
        const float3 lower = floor(v * 0.25f) * 4.f;
        const float3 inverse = 1.f / select(abs(d) < 1e-12f, float3(1e-12f), d);
        const float3 t0 = (lower - v) * inverse;
        const float3 t1 = (lower + 4.f - v) * inverse;
        // A ray lying on a block face with a tiny direction component cannot leave that face in float precision; ignore
        // zero-length exits (the dilated majorant still bounds the neighbouring block) so delta tracking cannot stall.
        const float3 far = select(max(t0, t1) > 1e-5f, max(t0, t1), float3(1e30f));
        return min(far.x, min(far.y, far.z));
    }

    /// Delta tracking with the piecewise-constant block majorant. Returns true at a real collision (v moved there).
    bool sampleCollision(inout float3 v, float3 d, inout uint rng)
    {
        const float epsilon = 1e-3f;
        for (uint i = 0; i < 16384; ++i)
        {
            if (!insideExtinctionGrid(v))
                return false;
            const float majorant = majorantAt(v);
            const float exit = majorantBlockExit(v, d);
            if (majorant <= 0.f)
            {
                v += (exit + epsilon) * d;
                continue;
            }
            const float flight = -log(1.f - randomFloat(rng)) / majorant;
            if (flight >= exit)
            {
                v += (exit + epsilon) * d;
                continue;
            }
            v += flight * d;
            if (randomFloat(rng) * majorant < extinctionAtVoxel(v))
                return true;
        }
        return false;
    }

    /// Ratio-tracking transmittance estimate from v to the grid boundary.
    float ratioTrack(float3 v, float3 d, inout uint rng)
    {
        const float epsilon = 1e-3f;
        float transmittance = 1.f;
        for (uint i = 0; i < 16384 && transmittance > 1e-4f; ++i)
        {
            if (!insideExtinctionGrid(v))
                break;
            const float majorant = majorantAt(v);
            const float exit = majorantBlockExit(v, d);
            if (majorant <= 0.f)
            {
                v += (exit + epsilon) * d;
                continue;
            }
            const float flight = -log(1.f - randomFloat(rng)) / majorant;
            if (flight >= exit)
            {
                v += (exit + epsilon) * d;
                continue;
            }
            v += flight * d;
            transmittance *= max(0.f, 1.f - extinctionAtVoxel(v) / majorant);
        }
        return transmittance;
    }

    uint hashUint(uint value)
    {
        const uint state = value * 747796405u + 2891336453u;
        const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    float randomFloat(inout uint rng)
    {
        rng = hashUint(rng);
        return float(rng >> 8) / 16777216.f;
    }

    /// Henyey-Greenstein sample around the propagation direction.
    float3 sampleHG(float3 forward, float2 u)
    {
        const float g = params.anisotropy;
        float cosTheta;
        if (abs(g) < 1e-3f)
            cosTheta = 1.f - 2.f * u.x;
        else
        {
            const float s = (1.f - g * g) / (1.f - g + 2.f * g * u.x);
            cosTheta = clamp((1.f + g * g - s * s) / (2.f * g), -1.f, 1.f);
        }
        const float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
        const float phi = 2.f * kPi * u.y;
        const float3 up = abs(forward.y) < 0.999f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
        const float3 tangent = normalize(cross(up, forward));
        const float3 bitangent = cross(forward, tangent);
        return normalize(sinTheta * cos(phi) * tangent + sinTheta * sin(phi) * bitangent + cosTheta * forward);
    }

    /// One reference path, split into the kComponent* radiance components.
    ReferenceSample referencePath(float3 origin, float3 direction, GridVolume volume, inout uint rng)
    {
        float2 interval;
        const AABB bounds = volume.getBounds();
        if (!intersectRayAABB(origin, direction, bounds.minPoint, bounds.maxPoint, interval) || interval.y <= max(0.f, interval.x))
        {
            ReferenceSample miss = {};
            miss.background = skyRadianceAt(direction);
            return miss;
        }
        const float3 entryIndex = mul(volume.data.invTransform, float4(origin + max(0.f, interval.x) * direction, 1.f)).xyz;
        return tracePath(entryIndex - float3(params.hstrGridMin), direction, volume, rng);
    }

    /// Radiance arriving back along -direction at voxel position v, split into components by the number of scattering events
    /// after v. Direct sunlight at v itself is not included.
    ReferenceSample tracePath(float3 v, float3 direction, GridVolume volume, inout uint rng)
    {
        ReferenceSample result = {};
        const float3 sunDirection = normalize(params.sunDirection);
        const float3 sunVoxel = mul(volume.data.invTransform, float4(sunDirection, 0.f)).xyz;
        float3 throughput = 1.f;
        for (uint bounce = 0; bounce < 4096; ++bounce)
        {
            const float3 d = mul(volume.data.invTransform, float4(direction, 0.f)).xyz;
            if (!sampleCollision(v, d, rng))
            {
                // Escaping after `bounce` scattering events: unscattered background or scattered sky.
                const float3 sky = throughput * skyRadianceAt(direction);
                if (bounce == 0)
                    result.background += sky;
                else
                    result.sky += sky;
                break;
            }
            throughput *= volume.data.albedo;
            // Sun: directional irradiance through the ratio-tracked transmittance; this is scattering order bounce + 1.
            const float3 sun = throughput * phaseHG(dot(sunDirection, direction)) * params.sunRadiance * ratioTrack(v, sunVoxel, rng);
            if (bounce == 0)
                result.sunSingle += sun;
            else
                result.sunMultiple += sun;
            direction = sampleHG(direction, float2(randomFloat(rng), randomFloat(rng)));
            // Russian roulette keeps the weight at or below one. A survival probability capped below the albedo (the former
            // min(0.95, throughput)) grows the weight by albedo / cap per bounce and gives infinite variance at albedo 0.99.
            if (bounce >= 8)
            {
                const float survive = min(1.f, maxComponent(throughput));
                if (randomFloat(rng) >= survive)
                    break;
                throughput /= survive;
            }
        }
        return result;
    }

    // ---------------------------------------------------------------------------------------------
    // World-space radiance cache (experiment): per cell, order-2 SH of all incident radiance except direct sunlight,
    // gathered by paths started inside the cell. Scattered towards the camera it is the sun-multiple plus scattered-sky term.

    /// Real SH basis function k (0-8, bands 0-2) at unit direction d.
    float shBasis(uint k, float3 d)
    {
        switch (k)
        {
        case 0:
            return 0.282095f;
        case 1:
            return 0.488603f * d.y;
        case 2:
            return 0.488603f * d.z;
        case 3:
            return 0.488603f * d.x;
        case 4:
            return 1.092548f * d.x * d.y;
        case 5:
            return 1.092548f * d.y * d.z;
        case 6:
            return 0.315392f * (3.f * d.z * d.z - 1.f);
        case 7:
            return 1.092548f * d.x * d.z;
        default:
            return 0.546274f * (d.x * d.x - d.y * d.y);
        }
    }

    /// Real SH basis function m (0-6) of band 3 at unit direction d.
    float shBasis3(uint m, float3 d)
    {
        switch (m)
        {
        case 0:
            return 0.590044f * d.y * (3.f * d.x * d.x - d.y * d.y);
        case 1:
            return 2.890611f * d.x * d.y * d.z;
        case 2:
            return 0.457046f * d.y * (5.f * d.z * d.z - 1.f);
        case 3:
            return 0.373176f * d.z * (5.f * d.z * d.z - 3.f);
        case 4:
            return 0.457046f * d.x * (5.f * d.z * d.z - 1.f);
        case 5:
            return 1.445306f * d.z * (d.x * d.x - d.y * d.y);
        default:
            return 0.590044f * d.x * (d.x * d.x - 3.f * d.y * d.y);
        }
    }

    /// Sun extension slots in use: all of band 3, or zonal bands 3-worldCacheZonalBands about the sun direction.
    uint worldCacheSunExtension()
    {
        if (params.worldCacheSunOrder >= 3)
            return 7u;
        return params.worldCacheZonalBands > 2 ? min(params.worldCacheZonalBands, 8u) - 2u : 0u;
    }

    /// Basis of sun extension slot j at unit direction d: band 3 function j, or the zonal Y_l0 of band l = 3 + j about the sun,
    /// sqrt((2 l + 1) / (4 pi)) P_l(d . sun).
    float worldCacheExtensionBasis(uint j, float3 d)
    {
        if (params.worldCacheSunOrder >= 3)
            return shBasis3(j, d);
        const float mu = dot(d, normalize(params.sunDirection));
        float previous = 1.f;
        float current = mu;
        for (uint l = 2; l <= j + 3; ++l)
        {
            const float next = ((2.f * l - 1.f) * mu * current - (l - 1.f) * previous) / float(l);
            previous = current;
            current = next;
        }
        return sqrt((2.f * (j + 3) + 1.f) / (4.f * kPi)) * current;
    }

    /// Hanning window of SH band l for a truncation at band order (1 without windowing).
    float shWindow(uint l, uint order)
    {
        return params.worldCacheWindow != 0 ? 0.5f + 0.5f * cos(kPi * float(l) / float(order + 1)) : 1.f;
    }

    uint worldCacheIndex(uint3 cell) { return cell.x + params.worldCacheDims.x * (cell.y + params.worldCacheDims.y * cell.z); }

    /// Whether a cell holds an estimate. The gather skips cells with no density within a majorant block of their centre;
    /// light tracing deposits into every cell a photon crosses.
    bool worldCacheCellValid(uint3 cell)
    {
        if (params.worldCacheEstimator != 0 || params.worldCacheCellVoxels > 4)
            return true;
        return majorantAt((float3(cell) + 0.5f) * float(params.worldCacheCellVoxels)) > 0.f;
    }

    /// Light-tracing deposit of one straight flight from v0 to v1 (voxel coordinates): track length per crossed cell times
    /// the photon weight, projected on SH bands 0-worldCacheBands of the direction the light arrives from, plus the sun
    /// extension slots for the sun field. field 0: sun, 1: sky.
    void depositSegment(float3 v0, float3 v1, float3 look, float weight, uint field, inout uint rng)
    {
        const uint coefficients = params.worldCacheBands >= 2 ? 9u : 4u;
        const uint extension = field == 0 ? worldCacheSunExtension() : 0u;
        float extensionBasis[7];
        [unroll]
        for (uint j = 0; j < 7; ++j)
            extensionBasis[j] = j < extension ? worldCacheExtensionBasis(j, look) : 0.f;
        const float cellVoxels = float(params.worldCacheCellVoxels);
        const float worldLength = length((v1 - v0) * params.worldVoxelSize);
        if (worldLength <= 0.f || weight <= 0.f)
            return;
        const float cellVolume = cellVoxels * cellVoxels * cellVoxels * params.worldVoxelSize.x * params.worldVoxelSize.y *
                                 params.worldVoxelSize.z;
        // Fixed-point deposit per unit of segment parameter.
        const float perUnit = weight * worldLength / cellVolume * params.worldCacheDepositScale;

        const float3 p0 = v0 / cellVoxels;
        const float3 delta = (v1 - v0) / cellVoxels;
        int3 cell = int3(floor(p0));
        const int3 stepDirection = int3(sign(delta));
        const float3 inverse = 1.f / max(abs(delta), 1e-12f);
        float3 tMax = float3(
            delta.x > 0.f ? (floor(p0.x) + 1.f - p0.x) * inverse.x : (delta.x < 0.f ? (p0.x - floor(p0.x)) * inverse.x : 1e30f),
            delta.y > 0.f ? (floor(p0.y) + 1.f - p0.y) * inverse.y : (delta.y < 0.f ? (p0.y - floor(p0.y)) * inverse.y : 1e30f),
            delta.z > 0.f ? (floor(p0.z) + 1.f - p0.z) * inverse.z : (delta.z < 0.f ? (p0.z - floor(p0.z)) * inverse.z : 1e30f)
        );
        const float3 tDelta = float3(delta.x != 0.f ? inverse.x : 1e30f, delta.y != 0.f ? inverse.y : 1e30f, delta.z != 0.f ? inverse.z : 1e30f);
        float t = 0.f;
        for (uint i = 0; i < 1024; ++i)
        {
            const float tNext = min(min(tMax.x, tMax.y), min(tMax.z, 1.f));
            const float fraction = tNext - t;
            if (fraction > 0.f && all(cell >= 0) && all(cell < int3(params.worldCacheDims)))
            {
                const uint index = kWorldCacheSlots * worldCacheIndex(uint3(cell));
                const uint base = index + 9 * field;
                // Deposits are divided by the modulation at the middle of this cell's part of the flight.
                const float inverseModulation = 1.f / worldCacheModulation((p0 + (t + 0.5f * fraction) * delta) * cellVoxels);
                for (uint k = 0; k < coefficients; ++k)
                {
                    // Stochastic rounding keeps the fixed-point sum unbiased.
                    const float value = shBasis(k, look) * perUnit * fraction * inverseModulation;
                    const int quantized = int(floor(value + randomFloat(rng)));
                    if (quantized != 0)
                        InterlockedAdd(hstrWorldCacheDeposit[base + k], quantized);
                }
                for (uint j = 0; j < extension; ++j)
                {
                    const float value = extensionBasis[j] * perUnit * fraction * inverseModulation;
                    const int quantized = int(floor(value + randomFloat(rng)));
                    if (quantized != 0)
                        InterlockedAdd(hstrWorldCacheDeposit[index + 18 + j], quantized);
                }
            }
            if (tNext >= 1.f)
                break;
            t = tNext;
            if (tMax.x <= tMax.y && tMax.x <= tMax.z)
            {
                cell.x += stepDirection.x;
                tMax.x += tDelta.x;
            }
            else if (tMax.y <= tMax.z)
            {
                cell.y += stepDirection.y;
                tMax.y += tDelta.y;
            }
            else
            {
                cell.z += stepDirection.z;
                tMax.z += tDelta.z;
            }
        }
    }

    /// Emitted power of photon source i: the sun through the side facing it on axis i (0-2), or the unit-colour sky
    /// L(d) = 0.675 + 0.325 d.y through side i - 3 (outward normal n), whose inward flux is A (0.675 pi + 0.325 (2 pi / 3) n.y).
    float photonSourcePower(uint i, float3 sun, float3 faceArea, float sunMean, float skyMean)
    {
        if (i < 3)
            return sunMean * abs(sun[i]) * faceArea[i];
        const uint axis = (i - 3) >> 1;
        const float ny = axis == 1 ? (((i - 3) & 1) != 0 ? 1.f : -1.f) : 0.f;
        return skyMean * faceArea[axis] * (0.675f * kPi + 0.325f * 2.f * kPi / 3.f * ny);
    }

    /// One light-tracing photon: emitted from the sun through a sun-facing side of the grid box, or from the unit-colour sky
    /// through any side, chosen by power. Every flight segment after the first sun segment (direct sunlight) is deposited.
    void tracePhoton(uint index)
    {
        if (index >= params.worldCachePhotons || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        uint rng = hashUint(index * 2654435761u + 1013u) ^ hashUint(params.worldCacheSamples * 9781u + 6271u);
        Photon photon = spawnPhoton(rng);
        float3 v = photon.positionWeight.xyz;
        const float weight = photon.positionWeight.w;
        float3 direction = photon.directionThroughput.xyz;
        const bool fromSun = (photon.state & kPhotonSky) == 0;

        // The medium is grey: one scalar albedo.
        const float albedo = dot(volume.data.albedo, 1.f / 3.f);
        float throughput = 1.f;
        for (uint bounce = 0; bounce < 4096; ++bounce)
        {
            const float3 d = mul(volume.data.invTransform, float4(direction, 0.f)).xyz;
            const float3 start = v;
            const bool collided = sampleCollision(v, d, rng);
            if (!fromSun || bounce > 0)
                depositSegment(start, v, -direction, weight * throughput, fromSun ? 0u : 1u, rng);
            if (!collided)
                break;
            throughput *= albedo;
            direction = sampleHG(direction, float2(randomFloat(rng), randomFloat(rng)));
            // Weight-bounded Russian roulette, as in tracePath.
            if (bounce >= 8)
            {
                const float survive = min(1.f, throughput);
                if (randomFloat(rng) >= survive)
                    break;
                throughput /= survive;
            }
        }
    }

    /// Persistent pool: advances one photon by worldCacheSegments flight segments, respawning it whenever it dies, so every
    /// update costs the same and every path is still deposited in full across updates. Counts the photons emitted.
    void advancePhoton(uint slot)
    {
        if (slot >= params.worldCachePhotons || gScene.getGridVolumeCount() == 0)
            return;
        const GridVolume volume = gScene.getGridVolume(0);
        const float albedo = dot(volume.data.albedo, 1.f / 3.f);
        Photon photon = hstrPhotonPool[slot];
        uint rng = photon.rng != 0 ? photon.rng : hashUint(slot * 2654435761u + 1013u) ^ hashUint(params.worldCacheSamples * 9781u + 7919u);
        uint emitted = 0;
        for (uint segment = 0; segment < params.worldCacheSegments; ++segment)
        {
            if ((photon.state & kPhotonAlive) == 0)
            {
                photon = spawnPhoton(rng);
                ++emitted;
            }
            float3 v = photon.positionWeight.xyz;
            float3 direction = photon.directionThroughput.xyz;
            float throughput = photon.directionThroughput.w;
            const uint bounce = photon.state & kPhotonBounceMask;
            const bool fromSun = (photon.state & kPhotonSky) == 0;
            const float3 d = mul(volume.data.invTransform, float4(direction, 0.f)).xyz;
            const float3 start = v;
            const bool collided = sampleCollision(v, d, rng);
            if (!fromSun || bounce > 0)
                depositSegment(start, v, -direction, photon.positionWeight.w * throughput, fromSun ? 0u : 1u, rng);
            bool alive = collided;
            if (collided)
            {
                throughput *= albedo;
                direction = sampleHG(direction, float2(randomFloat(rng), randomFloat(rng)));
                if (bounce >= 8)
                {
                    const float survive = min(1.f, throughput);
                    alive = randomFloat(rng) < survive;
                    throughput /= survive;
                }
            }
            photon.positionWeight.xyz = v;
            photon.directionThroughput = float4(direction, throughput);
            photon.state = (photon.state & ~(kPhotonAlive | kPhotonBounceMask)) | (alive ? kPhotonAlive : 0u) | min(bounce + 1, kPhotonBounceMask);
        }
        photon.rng = rng;
        hstrPhotonPool[slot] = photon;
        const uint waveEmitted = WaveActiveSum(emitted);
        if (WaveIsFirstLane() && waveEmitted > 0)
            InterlockedAdd(hstrPhotonEmitted[0], waveEmitted);
    }

    /// A new photon at the grid box boundary (see tracePhoton), with its weight normalised by worldCachePhotons.
    Photon spawnPhoton(inout uint rng)
    {
        const float3 dims = float3(params.hstrExtinctionDims);
        const float3 extent = dims * params.worldVoxelSize;
        const float3 faceArea = float3(extent.y * extent.z, extent.x * extent.z, extent.x * extent.y);
        const float3 sun = normalize(params.sunDirection);
        const float sunMean = dot(params.sunRadiance, 1.f / 3.f);
        const float skyMean = dot(params.skyRadiance, 1.f / 3.f);

        // Sources by power: three sun-facing sides (0-2), then six sky sides (3-8: -x, +x, -y, +y, -z, +z).
        float total = 0.f;
        for (uint i = 0; i < 9; ++i)
            total += photonSourcePower(i, sun, faceArea, sunMean, skyMean);
        float u = randomFloat(rng) * total;
        uint choice = 8;
        for (uint i = 0; i < 9; ++i)
        {
            const float power = photonSourcePower(i, sun, faceArea, sunMean, skyMean);
            if (u < power)
            {
                choice = i;
                break;
            }
            u -= power;
        }

        const bool fromSun = choice < 3;
        const uint axis = fromSun ? choice : (choice - 3) >> 1;
        // Outward normal of the entry side: the sun side faces the sun; sky sides alternate -/+.
        const float side = fromSun ? (sun[axis] >= 0.f ? 1.f : -1.f) : (((choice - 3) & 1) != 0 ? 1.f : -1.f);
        float3 normal = 0.f;
        normal[axis] = side;
        float3 v = float3(randomFloat(rng), randomFloat(rng), randomFloat(rng)) * dims;
        v[axis] = side > 0.f ? dims[axis] - 1e-3f : 1e-3f;

        float3 direction;
        float weight;
        if (fromSun)
        {
            direction = -sun;
            weight = total / (sunMean * float(params.worldCachePhotons));
        }
        else
        {
            // Cosine-weighted direction d towards the sky about the outward normal; the photon travels along -d.
            const float r = sqrt(randomFloat(rng));
            const float phi = 2.f * kPi * randomFloat(rng);
            const float3 up = abs(normal.y) < 0.999f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
            const float3 tangent = normalize(cross(up, normal));
            const float3 bitangent = cross(normal, tangent);
            const float3 d = normalize(r * cos(phi) * tangent + r * sin(phi) * bitangent + sqrt(max(0.f, 1.f - r * r)) * normal);
            direction = -d;
            // f / pdf with pdf = (power / total) (1 / area) (cos / pi); power / (skyMean area) is the side's unit flux.
            const float unitFlux = photonSourcePower(choice, sun, faceArea, sunMean, skyMean) / (skyMean * faceArea[axis]);
            weight = (0.675f + 0.325f * d.y) * kPi * total / (skyMean * unitFlux * float(params.worldCachePhotons));
        }

        Photon photon;
        photon.positionWeight = float4(v, weight);
        photon.directionThroughput = float4(direction, 1.f);
        photon.rng = rng;
        photon.state = kPhotonAlive | (fromSun ? 0u : kPhotonSky);
        photon.pad = 0;
        return photon;
    }

    /// Adds one light-tracing batch to the running sums: deposits times the sun and sky colours.
    void resolveWorldCacheCell(uint3 cell)
    {
        if (any(cell >= params.worldCacheDims))
            return;
        const uint index = worldCacheIndex(cell);
        const bool restart = params.worldCacheSamples == 0;
        const float scale = 1.f / params.worldCacheDepositScale;
        [unroll]
        for (uint k = 0; k < 9; ++k)
        {
            const float3 value = scale * (float(hstrWorldCacheDeposit[kWorldCacheSlots * index + k]) * params.sunRadiance +
                                          float(hstrWorldCacheDeposit[kWorldCacheSlots * index + 9 + k]) * params.skyRadiance);
            [unroll]
            for (uint c = 0; c < 3; ++c)
            {
                const uint slot = 27 * index + 3 * k + c;
                hstrWorldCache[slot] = (restart ? 0.f : hstrWorldCache[slot]) + value[c];
            }
        }
    }

    /// One gather pass: a single path per cell, from a uniform point in the cell along a uniform direction.
    void updateWorldCacheCell(uint3 cell)
    {
        if (any(cell >= params.worldCacheDims) || gScene.getGridVolumeCount() == 0)
            return;
        const uint index = worldCacheIndex(cell);
        const float cellVoxels = float(params.worldCacheCellVoxels);
        const bool needed = worldCacheCellValid(cell);
        float3 value = 0.f;
        float3 d = float3(0.f, 0.f, 1.f);
        if (needed)
        {
            uint rng = hashUint(index * 7919u + 17u) ^ hashUint(params.worldCacheSamples * 9781u + 6271u);
            const float3 v = (float3(cell) + float3(randomFloat(rng), randomFloat(rng), randomFloat(rng))) * cellVoxels;
            const float z = 1.f - 2.f * randomFloat(rng);
            const float phi = 2.f * kPi * randomFloat(rng);
            const float r = sqrt(max(0.f, 1.f - z * z));
            d = float3(r * cos(phi), r * sin(phi), z);
            const ReferenceSample path = tracePath(v, d, gScene.getGridVolume(0), rng);
            value = path.background + path.sunSingle + path.sunMultiple + path.sky;
            if (any(isnan(value) || isinf(value)))
                value = 0.f;
        }
        const bool restart = params.worldCacheSamples == 0;
        [unroll]
        for (uint k = 0; k < 9; ++k)
        {
            const float3 projected = 4.f * kPi * shBasis(k, d) * value;
            [unroll]
            for (uint c = 0; c < 3; ++c)
            {
                const uint slot = 27 * index + 3 * k + c;
                hstrWorldCache[slot] = (restart ? 0.f : hstrWorldCache[slot]) + projected[c];
            }
        }
    }

    /// Mean of scalar cache slot s of a cell (unit-colour sun and sky fields), straight from the light tracer's fixed-point
    /// deposits, which accumulate over every batch since the last restart.
    float worldCacheSlotMean(uint index, uint slot)
    {
        // Photon weights are normalised by worldCachePhotons; a pool has emitted hstrPhotonEmitted[0] of them in total.
        const float batches = params.worldCacheSegments > 0 ? float(max(hstrPhotonEmitted[0], 1u)) / float(params.worldCachePhotons)
                                                            : float(max(params.worldCacheSamples, 1u));
        return float(hstrWorldCacheDeposit[kWorldCacheSlots * index + slot]) / (batches * params.worldCacheDepositScale);
    }

    /// Writes one cell's cache means into the packed camera textures (light tracing only; the gather is read untextured).
    void bakeWorldCacheCell(uint3 cell)
    {
        if (any(cell >= params.worldCacheDims))
            return;
        const uint index = worldCacheIndex(cell);
        [unroll]
        for (uint t = 0; t < kWorldCacheTextures; ++t)
        {
            float4 texel = 0.f;
            [unroll]
            for (uint c = 0; c < 4; ++c)
            {
                const uint slot = 4 * t + c;
                if (slot < kWorldCacheSlots)
                    texel[c] = worldCacheSlotMean(index, slot);
            }
            hstrWorldCacheTextureOutput[t][cell] = texel;
        }
    }

    /// Textured form of worldCacheRadiance: one trilinear fetch per four slots, only for the evaluated bands. The unit sun and
    /// sky fields take their colours here. With worldCacheZonalBands, each sun band splits into its zonal part about the sun
    /// direction s, f_l(s) P_l(w . s), windowed at the zonal order and continued by the zonal extension bands, and the rest,
    /// windowed at order 2. With worldCacheSunOrder 3 the sun field is plain SH to band 3.
    float3 worldCacheRadianceTextured(float3 v, float3 w)
    {
        const float3 uvw = v / (float(params.worldCacheCellVoxels) * float3(params.worldCacheDims));
        const float g = params.anisotropy;
        const uint order = params.worldCacheOrder;
        const uint extension = order == 2 ? worldCacheSunExtension() : 0u;
        const uint sunOrder = extension == 7 ? 3u : order;
        const bool zonal = order == 2 && params.worldCacheSunOrder < 3 && params.worldCacheZonalBands > 0;
        const uint zonalOrder = params.worldCacheZonalWindow != 0 ? params.worldCacheZonalWindow : max(params.worldCacheZonalBands, 2u);

        float4 texels[kWorldCacheTextures];
        // Sun slots 0-8 and sky slots 9-17 of bands 0, 0-1 or 0-2 lie in textures {0, 2}, {0, 2, 3} or 0-4; the sun extension
        // (slots 18-24) in textures 4-6.
        const uint textures = (order == 0 ? 0x5u : (order == 1 ? 0xDu : 0x1Fu)) | (extension > 0 ? 0x70u : 0u);
        [unroll]
        for (uint t = 0; t < kWorldCacheTextures; ++t)
            texels[t] = (textures & (1u << t)) != 0 ? hstrWorldCacheTexture[t].SampleLevel(hstrLinearSampler, uvw, 0.f) : 0.f;

        const float3 s = normalize(params.sunDirection);
        float sun = 0.f;
        float sky = 0.f;
        float3 bandAtView = 0.f; // Sun band l at w and at s, for the zonal split.
        float3 bandAtSun = 0.f;
        [unroll]
        for (uint k = 0; k < 9; ++k)
        {
            const uint l = k == 0 ? 0u : (k < 4 ? 1u : 2u);
            if (l > order)
                continue;
            const float gl = l == 0 ? 1.f : (l == 1 ? g : g * g);
            const float basis = shBasis(k, w);
            sky += shWindow(l, order) * gl * basis * texels[(9 + k) / 4][(9 + k) % 4];
            const float c = texels[k / 4][k % 4];
            if (zonal)
            {
                bandAtView[l] += c * basis;
                bandAtSun[l] += c * shBasis(k, s);
            }
            else
                sun += shWindow(l, sunOrder) * gl * basis * c;
        }
        if (zonal)
        {
            // Legendre recurrence in mu = w . s through band 8.
            const float mu = dot(w, s);
            float previous = 1.f;
            float current = mu;
            float gl = g;
            [unroll]
            for (uint l = 0; l <= 8; ++l)
            {
                const float legendre = l == 0 ? 1.f : current;
                const float bandPower = l == 0 ? 1.f : gl;
                if (l <= 2)
                {
                    const float zonalPart = bandAtSun[l] * legendre;
                    sun += bandPower * (shWindow(l, 2) * (bandAtView[l] - zonalPart) + shWindow(l, zonalOrder) * zonalPart);
                }
                else if (l - 3 < extension)
                    sun += bandPower * shWindow(l, zonalOrder) * sqrt((2.f * l + 1.f) / (4.f * kPi)) * legendre * texels[(15 + l) / 4][(15 + l) % 4];
                if (l >= 1)
                {
                    const float next = ((2.f * (l + 1) - 1.f) * mu * current - float(l) * previous) / float(l + 1);
                    previous = current;
                    current = next;
                    gl *= g;
                }
            }
        }
        else if (extension == 7)
        {
            [unroll]
            for (uint m = 0; m < 7; ++m)
                sun += shWindow(3, 3) * g * g * g * shBasis3(m, w) * texels[(18 + m) / 4][(18 + m) % 4];
        }
        return max(0.f, sun * params.sunRadiance + sky * params.skyRadiance);
    }

    /// In-scattered radiance towards the camera (before albedo) from the cache at voxel position v, for camera ray direction w:
    /// the HG phase convolves SH band l by g^l. Trilinear over valid cell centres (cells without an estimate get no weight
    /// rather than black light); bands above worldCacheOrder are dropped.
    float3 worldCacheRadiance(float3 v, float3 w) { return worldCacheModulation(v) * worldCacheRadianceStored(v, w); }

    /// Voxel-resolution factor the cache is divided by when deposited and multiplied by when read: exp(-b tau_sun), with tau_sun
    /// the fine sun optical depth (capped so deep interiors do not inflate the fixed-point deposits). Multiple scattering falls
    /// off into the cloud with the sun depth, so the stored ratio is much smoother than the radiance at cell scale, and the
    /// camera recovers the voxel-scale structure near lit surfaces from the fine sun field. b = 0 disables it.
    float worldCacheModulation(float3 v)
    {
        if (params.worldCacheModulation <= 0.f)
            return 1.f;
        const float depth = hstrFineSun.SampleLevel(hstrLinearSampler, (v + 0.5f) / float3(params.hstrFineDims), 0.f);
        return exp(-params.worldCacheModulation * min(depth, params.worldCacheModulationDepth));
    }

    /// The cache as stored (divided by worldCacheModulation). The float sums (gather, or untextured light tracing) carry bands
    /// 0-2 of the coloured total only.
    float3 worldCacheRadianceStored(float3 v, float3 w)
    {
        if (params.worldCacheTextured != 0 && params.worldCacheEstimator != 0)
            return worldCacheRadianceTextured(v, w);
        const float3 p = v / float(params.worldCacheCellVoxels) - 0.5f;
        const float3 base = floor(p);
        const float3 f = p - base;
        const float g = params.anisotropy;
        const uint coefficients = (params.worldCacheOrder + 1) * (params.worldCacheOrder + 1);
        float3 result = 0.f;
        float weightSum = 0.f;
        [unroll]
        for (uint corner = 0; corner < 8; ++corner)
        {
            const uint3 offset = uint3(corner & 1, (corner >> 1) & 1, corner >> 2);
            const float3 weights = lerp(1.f - f, f, float3(offset));
            const uint3 cell = uint3(clamp(int3(base) + int3(offset), int3(0), int3(params.worldCacheDims) - 1));
            const float weight = worldCacheCellValid(cell) ? weights.x * weights.y * weights.z : 0.f;
            weightSum += weight;
            const uint index = worldCacheIndex(cell);
            for (uint k = 0; k < coefficients; ++k)
            {
                const uint l = k == 0 ? 0u : (k < 4 ? 1u : 2u);
                const float window = params.worldCacheWindow != 0 ? 0.5f + 0.5f * cos(kPi * float(l) / float(params.worldCacheOrder + 1)) : 1.f;
                const float band = window * (l == 0 ? 1.f : (l == 1 ? g : g * g));
                const float3 c = float3(hstrWorldCache[27 * index + 3 * k], hstrWorldCache[27 * index + 3 * k + 1], hstrWorldCache[27 * index + 3 * k + 2]);
                result += weight * band * shBasis(k, w) * c;
            }
        }
        return max(0.f, result / (float(max(params.worldCacheSamples, 1u)) * max(weightSum, 1e-6f)));
    }

    /// Reference averages of the selected components: the even-frame half, the odd-frame half, and both.
    ReferenceHalves referenceHalves(uint2 pixel, uint mask)
    {
        float3 evenSum = 0.f;
        float3 oddSum = 0.f;
        [unroll]
        for (uint c = 0; c < kComponentCount; ++c)
        {
            const float weight = (mask & (1u << c)) != 0 ? 1.f : 0.f;
            evenSum += weight * hstrReferenceSum[uint3(pixel, 2 * c)].rgb;
            oddSum += weight * hstrReferenceSum[uint3(pixel, 2 * c + 1)].rgb;
        }
        const float evenCount = hstrReferenceSum[uint3(pixel, 0)].a;
        const float oddCount = hstrReferenceSum[uint3(pixel, 1)].a;
        ReferenceHalves result;
        result.first = evenSum / max(evenCount, 1.f);
        result.second = oddSum / max(oddCount, 1.f);
        result.both = (evenSum + oddSum) / max(evenCount + oddCount, 1.f);
        return result;
    }

    void referencePixel(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        // Seeded by the sample index, not frameIndex: property changes reset frameIndex and would repeat earlier samples.
        uint rng = hashUint(pixel.x + params.frameDim.x * pixel.y) ^ hashUint(params.referenceSamples * 9781u + 6271u);
        const float2 jitter = float2(randomFloat(rng), randomFloat(rng));
        const float3 direction = cameraRayDirection(float2(pixel) + jitter);
        ReferenceSample path = {};
        path.background = skyRadianceAt(direction);
        if (gScene.getGridVolumeCount() > 0)
            path = referencePath(gScene.camera.data.posW, direction, gScene.getGridVolume(0), rng);
        bool valid = true;
        for (uint c = 0; c < kComponentCount; ++c)
            valid = valid && !any(isnan(path.component(c)) || isinf(path.component(c)));
        // Frames alternate between two independent halves, whose disagreement estimates the reference's own noise.
        const uint halfIndex = params.referenceSamples & 1u;
        const bool restart = params.referenceSamples < 2;
        for (uint c = 0; c < kComponentCount; ++c)
        {
            const uint3 slice = uint3(pixel, 2 * c + halfIndex);
            const float4 previous = restart ? float4(0.f) : hstrReferenceSum[slice];
            const float3 value = valid ? path.component(c) : float3(0.f);
            hstrReferenceSum[slice] = previous + float4(value, 1.f);
        }
        // referenceShow bits 4-5 pick one half (1: even frames, 2: odd frames); 0 shows both.
        const ReferenceHalves shown = referenceHalves(pixel, params.referenceShow & 0xFu);
        const uint shownHalves = (params.referenceShow >> 4) & 3u;
        color[pixel] = float4(shownHalves == 1 ? shown.first : (shownHalves == 2 ? shown.second : shown.both), 1.f);
        transportError[pixel] = 0.f;
        cutStats[pixel] = 0.f;
    }

    /// Mean absolute difference of one image row against the reference average, in linear radiance (x) and log(1 + radiance) (y).
    /// The compared value is the HST frame plus the substituted reference components; the reference is the target components.
    /// zw hold the same measures between the two reference halves, halved: the expected noise of the full reference average.
    void compareReferenceRow(uint row)
    {
        if (params.compareBlock > 1)
        {
            compareReferenceBlockRow(row);
            return;
        }
        if (row >= params.frameDim.y)
            return;
        float4 error = 0.f;
        uint count = 0;
        for (uint x = 0; x < params.frameDim.x; ++x)
        {
            const uint2 pixel = uint2(x, row);
            if (params.compareExact != 0)
            {
                // Against the stored exact frame: zw become the fractions of pixels whose log error exceeds 0.02 and 0.1.
                const float3 value = color[pixel].rgb;
                const float3 reference = hstrExactFrame[pixel].rgb;
                if (any(isnan(value) || isinf(value)))
                    continue;
                const float logError = dot(abs(log(1.f + max(value, 0.f)) - log(1.f + max(reference, 0.f))), 1.f / 3.f);
                error += float4(dot(abs(value - reference), 1.f / 3.f), logError, logError > 0.02f ? 1.f : 0.f, logError > 0.1f ? 1.f : 0.f);
                ++count;
                continue;
            }
            const ReferenceHalves target = referenceHalves(pixel, params.compareTarget);
            const float3 reference = target.both;
            const float3 value = color[pixel].rgb + referenceHalves(pixel, params.compareSubstitute).both;
            if (any(isnan(value) || isinf(value) || isnan(reference) || isinf(reference)))
                continue;
            const float3 halfDifference = abs(target.first - target.second);
            const float3 halfLogDifference = abs(log(1.f + max(target.first, 0.f)) - log(1.f + max(target.second, 0.f)));
            error += float4(
                dot(abs(value - reference), 1.f / 3.f),
                dot(abs(log(1.f + max(value, 0.f)) - log(1.f + max(reference, 0.f))), 1.f / 3.f),
                0.5f * dot(halfDifference, 1.f / 3.f),
                0.5f * dot(halfLogDifference, 1.f / 3.f)
            );
            ++count;
        }
        hstrReferenceRowError[row] = error / float(max(count, 1u));
    }

    // ---------------------------------------------------------------------------------------------
    // Beam view: the split frame's terms are marched exactly only at tile corners and centres. Pixels reconstruct them from a
    // bilinear-plus-bubble tile basis and apply their own phase, transmittance and background.

    /// Split-frame terms along the [segment.x, segment.y] part of one camera ray's volume interval, starting from unit
    /// transmittance. The phase is applied per pixel: octave 0 carries a unit weight.
    BeamSample marchBeam(float3 direction, float2 segment)
    {
        BeamSample result = {};
        CutRay ray;
        if (!makeCutRay(gScene.camera.data.posW, direction, gScene.getGridVolume(0), ray))
            return result;
        ray.sunOctaves = float4(1.f, 0.f, 0.f, 0.f);
        float transmittance = 1.f;
        uint steps = 0;
        marchSegment(ray, segment, beamWithCache(), beamWithSingle(), result.cache, result.single, transmittance, steps);
        result.depth = -log(max(transmittance, 1e-12f));
        return result;
    }

    bool beamWithCache() { return (params.hstComponents & (kComponentSunMultiple | kComponentSkyScattered)) != 0; }

    bool beamWithSingle() { return (params.hstComponents & kComponentSunSingle) != 0; }

    uint beamTileEdge(uint level) { return params.beamTileSize >> level; }

    /// Queries covering a grid of lattice points in blocks of 8 x 4.
    uint beamBlockQueries(uint2 grid) { return ((grid.x + 7u) / 8u) * ((grid.y + 3u) / 4u) * 32u; }

    /// Start of the list of tiles refined at level - 1, whose children are processed at level (>= 1).
    uint beamListOffset(uint level)
    {
        const uint tiles = params.beamTileDims.x * params.beamTileDims.y;
        return tiles * (((1u << (2u * (level - 1u))) - 1u) / 3u);
    }

    uint packBeamTile(uint2 tile) { return tile.x | (tile.y << 16); }

    uint2 unpackBeamTile(uint value) { return uint2(value & 0xFFFFu, value >> 16); }

    /// Lattice query at a pixel-corner position (a multiple of beamLatticeStep).
    float4 beamLatticeAt(uint2 position, uint slice) { return hstrBeamLattice[uint3(position / params.beamLatticeStep, slice)]; }

    /// Pixel-corner position of query q. Per level: the current pass level, over the lists refined this frame. Temporal: every
    /// level in one dispatch, level 0 first, then each finer level over the tiles refined last frame.
    bool beamQueryPosition(uint q, out uint2 position)
    {
        if (params.beamTemporal == 0)
            return beamQueryAt(params.beamPassLevel, q, false, position);
        const uint levelZero = beamBlockQueries(params.beamTileDims + 1u) + beamBlockQueries(params.beamTileDims);
        if (q < levelZero)
            return beamQueryAt(0, q, true, position);
        q -= levelZero;
        for (uint level = 1; level < params.beamLevels; ++level)
        {
            const uint queries = 8u * hstrBeamPreviousCounts[level];
            if (q < queries)
                return beamQueryAt(level, q, true, position);
            q -= queries;
        }
        position = 0;
        return false;
    }

    /// Query q of one level. Level 0 queries every coarsest corner, then every coarsest centre; a finer level queries eight
    /// points per tile refined at the level above (this frame's list, or last frame's when previous is set): its edge
    /// midpoints, then its children's centres (its corners and centre are already on the lattice).
    bool beamQueryAt(uint level, uint q, bool previous, out uint2 position)
    {
        position = 0;
        if (level == 0)
        {
            // Corners, then centres, each in blocks of 8 x 4 lattice points per 32 consecutive queries so that a wave's rays
            // are neighbours.
            const uint s = params.beamTileSize;
            const uint2 corners = params.beamTileDims + 1u;
            const uint cornerQueries = beamBlockQueries(corners);
            const bool centre = q >= cornerQueries;
            const uint2 grid = centre ? params.beamTileDims : corners;
            const uint local = centre ? q - cornerQueries : q;
            const uint blocksX = (grid.x + 7u) / 8u;
            const uint block = local / 32u;
            const uint lane = local % 32u;
            const uint2 point = uint2(block % blocksX, block / blocksX) * uint2(8u, 4u) + uint2(lane % 8u, lane / 8u);
            if (any(point >= grid))
                return false;
            position = point * s + (centre ? s / 2 : 0u);
            return true;
        }
        const uint entry = q / 8;
        if (entry >= (previous ? hstrBeamPreviousCounts[level] : hstrBeamCounts[level]))
            return false;
        const uint s = beamTileEdge(level - 1);
        const uint h = s / 2;
        const uint k = s / 4;
        uint2 offset;
        switch (q % 8)
        {
        case 0:
            offset = uint2(h, 0);
            break;
        case 1:
            offset = uint2(0, h);
            break;
        case 2:
            offset = uint2(s, h);
            break;
        case 3:
            offset = uint2(h, s);
            break;
        case 4:
            offset = uint2(k, k);
            break;
        case 5:
            offset = uint2(h + k, k);
            break;
        case 6:
            offset = uint2(k, h + k);
            break;
        default:
            offset = uint2(h + k, h + k);
            break;
        }
        const uint index = beamListOffset(level) + entry;
        position = unpackBeamTile(previous ? hstrBeamPreviousLists[index] : hstrBeamLists[index]) * s + offset;
        return true;
    }

    void writeBeamQuery(uint2 position, float4 depthCache, float4 single)
    {
        const uint2 texel = position / params.beamLatticeStep;
        hstrBeamLatticeOutput[uint3(texel, 0)] = depthCache;
        hstrBeamLatticeOutput[uint3(texel, 1)] = single;
    }

    float4 bilinear(float4 q00, float4 q10, float4 q01, float4 q11, float2 f) { return lerp(lerp(q00, q10, f.x), lerp(q01, q11, f.x), f.y); }

    /// Split-frame radiance from its terms (before albedo) for view direction w.
    float3 beamRadiance(float3 cache, float3 single, float depth, float3 w, float3 albedo)
    {
        const float phase = phaseHG(dot(normalize(params.sunDirection), w));
        const float3 background = (params.hstComponents & kComponentBackground) != 0 ? exp(-depth) * skyRadianceAt(w) : 0.f;
        return albedo * (phase * single + cache) + background;
    }

    /// Surplus test of one tile: the centre as the bilinear corner basis predicts it, against the exact centre query, in
    /// log(1 + radiance); optionally also a straddled beamEdgeDepth.
    bool beamTileRefines(uint level, uint2 tile)
    {
        const uint s = beamTileEdge(level);
        const uint2 origin = tile * s;
        const float4 a00 = beamLatticeAt(origin, 0);
        const float4 a10 = beamLatticeAt(origin + uint2(s, 0), 0);
        const float4 a01 = beamLatticeAt(origin + uint2(0, s), 0);
        const float4 a11 = beamLatticeAt(origin + uint2(s, s), 0);
        const float4 ac = beamLatticeAt(origin + s / 2, 0);
        const float4 bp = 0.25f * (beamLatticeAt(origin, 1) + beamLatticeAt(origin + uint2(s, 0), 1) + beamLatticeAt(origin + uint2(0, s), 1) +
                                   beamLatticeAt(origin + uint2(s, s), 1));
        const float4 bc = beamLatticeAt(origin + s / 2, 1);
        const float4 ap = 0.25f * (a00 + a10 + a01 + a11);
        const float3 albedo = gScene.getGridVolume(0).data.albedo;
        const float3 wc = cameraRayDirection(float2(origin + s / 2));
        const float3 exactCentre = beamRadiance(ac.rgb, bc.rgb, ac.a, wc, albedo);
        const float3 predictedCentre = beamRadiance(ap.rgb, bp.rgb, ap.a, wc, albedo);
        bool refine = maxComponent(abs(log(1.f + max(predictedCentre, 0.f)) - log(1.f + exactCentre))) > params.beamTolerance;
        if (params.beamEdgeDepth > 0.f)
        {
            const float minimumDepth = min(min(min(a00.a, a10.a), min(a01.a, a11.a)), ac.a);
            const float maximumDepth = max(max(max(a00.a, a10.a), max(a01.a, a11.a)), ac.a);
            refine = refine || (minimumDepth < params.beamEdgeDepth && maximumDepth >= params.beamEdgeDepth);
        }
        return refine;
    }

    /// Appends a refined tile to the list of the level below it; past the finest level, that list is the per-pixel march list.
    /// The tile's origin cell in the history map records the refinement for the next frame's temporal queries.
    void recordBeamRefinement(uint level, uint2 tile)
    {
        uint slot;
        InterlockedAdd(hstrBeamCounts[level + 1], 1u, slot);
        hstrBeamLists[beamListOffset(level + 1) + slot] = packBeamTile(tile);
        if (params.beamTemporal != 0)
        {
            const uint2 cell = tile * beamTileEdge(level) / beamTileEdge(params.beamLevels - 1);
            if (all(cell < params.beamLevelDims))
                InterlockedOr(hstrBeamHistoryOutput[cell], 1u << level);
        }
    }

    /// Whether this frame queried the children of a level tile: it was refined last frame.
    bool beamChildrenQueried(uint level, uint2 tile)
    {
        const uint2 cell = tile * beamTileEdge(level) / beamTileEdge(params.beamLevels - 1);
        return all(cell < params.beamLevelDims) && (hstrBeamHistory[cell] & (1u << level)) != 0;
    }

    /// Writes the level that finalised a tile into all its level-map cells.
    void writeBeamLevel(uint level, uint2 tile, uint value)
    {
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint cells = beamTileEdge(level) / finest;
        const uint2 first = tile * cells;
        for (uint y = 0; y < cells; ++y)
            for (uint x = 0; x < cells; ++x)
            {
                const uint2 cell = first + uint2(x, y);
                if (all(cell < params.beamLevelDims))
                    hstrBeamLevelOutput[cell] = value;
            }
    }

    /// Tile test of the current pass level. Accepted tiles write their level into the level map. Refined tiles join the next
    /// level's list, or at the finest level are marked for per-pixel marching (level beamLevels).
    void testBeamTile(uint thread)
    {
        const uint level = params.beamPassLevel;
        uint2 tile;
        if (level == 0)
        {
            if (thread >= params.beamTileDims.x * params.beamTileDims.y)
                return;
            tile = uint2(thread % params.beamTileDims.x, thread / params.beamTileDims.x);
        }
        else
        {
            const uint entry = thread / 4;
            if (entry >= hstrBeamCounts[level])
                return;
            const uint child = thread % 4;
            tile = 2u * unpackBeamTile(hstrBeamLists[beamListOffset(level) + entry]) + uint2(child & 1u, child >> 1);
        }
        if (any(tile * beamTileEdge(level) >= params.frameDim))
            return;
        if (!beamTileRefines(level, tile))
        {
            writeBeamLevel(level, tile, level);
            return;
        }
        recordBeamRefinement(level, tile);
        if (level + 1 >= params.beamLevels)
            writeBeamLevel(level, tile, params.beamLevels);
    }

    /// Temporal tile tests: one thread walks one coarsest tile down through the levels whose queries exist this frame (children
    /// of tiles refined last frame). A tile that needs refining but whose children were not queried keeps its own basis for
    /// this frame; its refinement is recorded, so the next frame queries them.
    void testBeamTileTemporal(uint thread)
    {
        if (thread >= params.beamTileDims.x * params.beamTileDims.y)
            return;
        // Depth-first stack of (tile, level); each refined tile pushes four children, so it never exceeds 1 + 3 levels.
        uint stackTile[1 + 3 * kBeamMaxLevels];
        uint stackLevel[1 + 3 * kBeamMaxLevels];
        uint depth = 1;
        stackTile[0] = packBeamTile(uint2(thread % params.beamTileDims.x, thread / params.beamTileDims.x));
        stackLevel[0] = 0;
        while (depth > 0)
        {
            --depth;
            const uint2 tile = unpackBeamTile(stackTile[depth]);
            const uint level = stackLevel[depth];
            if (any(tile * beamTileEdge(level) >= params.frameDim))
                continue;
            if (!beamTileRefines(level, tile))
            {
                writeBeamLevel(level, tile, level);
                continue;
            }
            const bool childrenQueried = level + 1 < params.beamLevels && beamChildrenQueried(level, tile);
            recordBeamRefinement(level, tile);
            if (level + 1 >= params.beamLevels)
                writeBeamLevel(level, tile, params.beamLevels);
            else if (!childrenQueried)
                writeBeamLevel(level, tile, level);
            else
                for (uint child = 0; child < 4; ++child)
                {
                    stackTile[depth] = packBeamTile(2u * tile + uint2(child & 1u, child >> 1));
                    stackLevel[depth] = level + 1;
                    ++depth;
                }
        }
    }

    /// Indirect arguments of the query and tile passes of the current pass level, from its list count; at level beamLevels,
    /// of the per-pixel march pass. Temporal level 0: the single query pass over every level (see beamQueryPosition).
    void writeBeamArgs()
    {
        const uint level = params.beamPassLevel;
        if (params.beamTemporal != 0 && level == 0)
        {
            uint queries = beamBlockQueries(params.beamTileDims + 1u) + beamBlockQueries(params.beamTileDims);
            for (uint l = 1; l < params.beamLevels; ++l)
                queries += 8u * hstrBeamPreviousCounts[l];
            hstrBeamArgs[0] = (queries * params.beamSegments + 63u) / 64u;
            hstrBeamArgs[1] = 1;
            hstrBeamArgs[2] = 1;
            return;
        }
        const uint count = hstrBeamCounts[level];
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint threadsPerEntry = level < params.beamLevels ? 8u * params.beamSegments : finest * finest;
        hstrBeamArgs[6 * level + 0] = (count * threadsPerEntry + 63u) / 64u;
        hstrBeamArgs[6 * level + 1] = 1;
        hstrBeamArgs[6 * level + 2] = 1;
        hstrBeamArgs[6 * level + 3] = (count * 4u + 63u) / 64u;
        hstrBeamArgs[6 * level + 4] = 1;
        hstrBeamArgs[6 * level + 5] = 1;
    }

    /// Beam view pixel reconstructed from the tile level that accepted it: bilinear corners plus the centre's bubble. No voxel
    /// is touched. Pixels of tiles that failed the finest level are left to marchBeamPixel.
    void resolveBeamPixel(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint level = hstrBeamLevel[pixel / finest];
        if (level >= params.beamLevels)
            return;
        const uint s = beamTileEdge(level);
        const uint2 origin = pixel / s * s;
        const float4 a00 = beamLatticeAt(origin, 0);
        const float4 a10 = beamLatticeAt(origin + uint2(s, 0), 0);
        const float4 a01 = beamLatticeAt(origin + uint2(0, s), 0);
        const float4 a11 = beamLatticeAt(origin + uint2(s, s), 0);
        const float4 ac = beamLatticeAt(origin + s / 2, 0);
        const float4 b00 = beamLatticeAt(origin, 1);
        const float4 b10 = beamLatticeAt(origin + uint2(s, 0), 1);
        const float4 b01 = beamLatticeAt(origin + uint2(0, s), 1);
        const float4 b11 = beamLatticeAt(origin + uint2(s, s), 1);
        const float4 bc = beamLatticeAt(origin + s / 2, 1);
        const float2 f = (float2(pixel - origin) + 0.5f) / float(s);
        const float bubble = 16.f * f.x * (1.f - f.x) * f.y * (1.f - f.y);
        const float4 a = bilinear(a00, a10, a01, a11, f) + (ac - 0.25f * (a00 + a10 + a01 + a11)) * bubble;
        const float4 b = bilinear(b00, b10, b01, b11, f) + (bc - 0.25f * (b00 + b10 + b01 + b11)) * bubble;
        const float3 w = cameraRayDirection(float2(pixel) + 0.5f);
        color[pixel] = float4(beamRadiance(max(a.rgb, 0.f), max(b.rgb, 0.f), max(a.a, 0.f), w, gScene.getGridVolume(0).data.albedo), 1.f);
    }

    /// One pixel of a finest tile that failed its surplus test, marched exactly.
    void marchBeamPixel(uint thread)
    {
        const uint finest = beamTileEdge(params.beamLevels - 1);
        const uint entry = thread / (finest * finest);
        if (entry >= hstrBeamCounts[params.beamLevels])
            return;
        const uint local = thread % (finest * finest);
        const uint2 pixel = unpackBeamTile(hstrBeamLists[beamListOffset(params.beamLevels) + entry]) * finest + uint2(local % finest, local / finest);
        if (any(pixel >= params.frameDim))
            return;
        const float3 w = cameraRayDirection(float2(pixel) + 0.5f);
        const BeamSample exact = marchBeam(w, float2(0.f, 1.f));
        color[pixel] = float4(beamRadiance(exact.cache, exact.single, exact.depth, w, gScene.getGridVolume(0).data.albedo), 1.f);
    }

    /// compareReferenceRow on compareBlock x compareBlock pixel means: one row of blocks. Averaging removes most of the
    /// reference's per-pixel noise (and the halves' estimate of it shrinks accordingly), so what remains is systematic error.
    void compareReferenceBlockRow(uint blockRow)
    {
        const uint block = params.compareBlock;
        const uint2 blocks = params.frameDim / block;
        if (blockRow >= blocks.y)
            return;
        float4 error = 0.f;
        uint count = 0;
        for (uint bx = 0; bx < blocks.x; ++bx)
        {
            float3 value = 0.f;
            float3 reference = 0.f;
            float3 first = 0.f;
            float3 second = 0.f;
            for (uint y = 0; y < block; ++y)
                for (uint x = 0; x < block; ++x)
                {
                    const uint2 pixel = uint2(bx, blockRow) * block + uint2(x, y);
                    value += params.compareExact != 0 ? color[pixel].rgb
                                                      : color[pixel].rgb + referenceHalves(pixel, params.compareSubstitute).both;
                    if (params.compareExact != 0)
                    {
                        reference += hstrExactFrame[pixel].rgb;
                        continue;
                    }
                    const ReferenceHalves target = referenceHalves(pixel, params.compareTarget);
                    reference += target.both;
                    first += target.first;
                    second += target.second;
                }
            const float inverse = 1.f / float(block * block);
            value *= inverse;
            reference *= inverse;
            first *= inverse;
            second *= inverse;
            if (any(isnan(value) || isinf(value) || isnan(reference) || isinf(reference)))
                continue;
            error += float4(
                dot(abs(value - reference), 1.f / 3.f),
                dot(abs(log(1.f + max(value, 0.f)) - log(1.f + max(reference, 0.f))), 1.f / 3.f),
                0.5f * dot(abs(first - second), 1.f / 3.f),
                0.5f * dot(abs(log(1.f + max(first, 0.f)) - log(1.f + max(second, 0.f))), 1.f / 3.f)
            );
            ++count;
            if (params.compareMapScale > 0.f)
            {
                // Error map: the signed block log error, red where the frame is brighter than the reference and blue where
                // darker, over the reference's log luminance in grey.
                const float signedError = dot(log(1.f + max(value, 0.f)) - log(1.f + max(reference, 0.f)), 1.f / 3.f);
                const float grey = 0.4f * saturate(log(1.f + dot(max(reference, 0.f), 1.f / 3.f)) / 3.f);
                const float red = saturate(signedError / params.compareMapScale);
                const float blue = saturate(-signedError / params.compareMapScale);
                const float3 map = float3(grey + (1.f - grey) * red, grey * (1.f - max(red, blue)), grey + (1.f - grey) * blue);
                for (uint y = 0; y < block; ++y)
                    for (uint x = 0; x < block; ++x)
                        color[uint2(bx, blockRow) * block + uint2(x, y)] = float4(map, 1.f);
            }
        }
        hstrReferenceRowError[blockRow] = error / float(max(count, 1u));
    }

    // ---------------------------------------------------------------------------------------------
    // Full-resolution resolve: L = (1 - tau) * basis source + tau * background, or the exact per-pixel integral in refined tiles.

    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        const float3 direction = cameraRayDirection(float2(pixel) + 0.5f);
        const float3 sky = params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * direction.y));
        if (gScene.getGridVolumeCount() == 0)
        {
            color[pixel] = float4(sky, 1.f);
            transportError[pixel] = 0.f;
            cutStats[pixel] = 0.f;
            return;
        }

        const GridVolume volume = gScene.getGridVolume(0);
        const AABB bounds = volume.getBounds();
        const float3 origin = gScene.camera.data.posW;
        if (all(origin > bounds.minPoint) && all(origin < bounds.maxPoint))
        {
            Ray ray;
            ray.origin = origin;
            ray.dir = direction;
            ray.tMin = 0.f;
            ray.tMax = 1e30f;
            float residualError;
            const uint stepCount = min(1024u, (params.baseSteps << params.refinementLevel) << 1);
            color[pixel] = float4(integrateVolume(ray, volume, stepCount, pixel, residualError).rgb, 1.f);
            transportError[pixel] = residualError;
            cutStats[pixel] = 0.f;
            return;
        }

        CutRay ray;
        if (!makeCutRay(origin, direction, volume, ray))
        {
            color[pixel] = float4((params.hstComponents & kComponentBackground) != 0 ? sky : 0.f, 1.f);
            transportError[pixel] = 0.f;
            cutStats[pixel] = 0.f;
            return;
        }

        if (params.debugView == kWorldCacheView)
        {
            // World cache experiment, marched exactly per pixel: the cache supplies sun-multiple plus scattered sky, the fine sun
            // field supplies single scattering (octave 0 only), and the background is transmitted; each gated by hstComponents.
            ray.sunOctaves.yzw = 0.f;
            float3 cached = 0.f;
            float3 single = 0.f;
            float cachedTransmittance = 1.f;
            uint cachedSteps = 0;
            const bool withCache = (params.hstComponents & (kComponentSunMultiple | kComponentSkyScattered)) != 0;
            const bool withSingle = (params.hstComponents & kComponentSunSingle) != 0;
            marchSegment(ray, float2(0.f, 1.f), withCache, withSingle, cached, single, cachedTransmittance, cachedSteps);
            const float3 background = (params.hstComponents & kComponentBackground) != 0 ? cachedTransmittance * sky : 0.f;
            color[pixel] = float4((cached + single) * volume.data.albedo + background, 1.f);
            transportError[pixel] = 0.f;
            cutStats[pixel] = 0.f;
            return;
        }

        if (params.debugView == 5)
        {
            // Resident residual page value (cell-averaged sun transmittance) where the ray's opacity reaches one half.
            const float cellWidth = float(params.hstrCellWidth);
            float T = 1.f;
            float t = 0.f;
            float3 debug = 0.f;
            for (uint i = 0; i < 2048 && t < 1.f && T > 0.5f; ++i)
            {
                const float dt = 0.5f / max(1.f, length(ray.direction) * cellWidth);
                const float3 v = (ray.position + (t + 0.5f * dt) * ray.direction) * cellWidth;
                T *= exp(-extinctionAtVoxel(v) * dt * ray.worldLength);
                if (T <= 0.5f)
                {
                    debug = exp(-hstrFineSun.SampleLevel(hstrLinearSampler, (v + 0.5f) / float3(params.hstrFineDims), 0.f));
                }
                t += dt;
            }
            color[pixel] = float4(debug, 1.f);
            return;
        }

        const uint2 tile = pixel / params.hstrTileSize;
        const uint tileID = tile.x + params.hstrTileDims.x * tile.y;
        const HSTRTileBasis basis = hstrTileBasis[tileID];
        bool refine = basis.info.y != 0.f || params.debugView == 4;
        float transmittance = 1.f;
        uint visited = 0;
        uint steps = 0;
        float3 residual = 0.f;
        float3 radiance = traverseCut(ray, tileID, refine, true, residual, transmittance, visited, steps);
        if (!refine)
        {
            const float2 f = (float2(pixel - tile * params.hstrTileSize) + 0.5f) / float(params.hstrTileSize);
            const float4 fitted = evaluateTileBasis(basis, f);
            const float opacity = 1.f - transmittance;
            if (fitted.a > 1e-3f || opacity <= 1e-3f)
                radiance = max(0.f, fitted.rgb) / max(fitted.a, 1e-4f) * opacity;
            else
            {
                // The tile basis carries no source where this pixel has opacity: exact fallback for this pixel only.
                refine = true;
                transmittance = 1.f;
                residual = 0.f;
                radiance = traverseCut(ray, tileID, true, true, residual, transmittance, visited, steps) * volume.data.albedo;
            }
        }
        else
            radiance *= volume.data.albedo;

        // L = L_base + sum of rasterized residuals + tau * background.
        residual *= volume.data.albedo;
        // Baseline for the HST sky: a constant source at the sphere-averaged sky radiance, weighted by this pixel's opacity.
        if (params.skyAmbient > 0.f)
            radiance = params.skyAmbient * volume.data.albedo * params.skyRadiance * 0.675f * (1.f - transmittance);
        radiance *= params.skyScale;
        if ((params.hstComponents & kComponentSkyScattered) == 0)
            radiance = 0.f;
        const float3 background = (params.hstComponents & kComponentBackground) != 0 ? transmittance * sky : 0.f;
        float3 result = max(0.f, radiance + residual) + background;
        if (params.debugView == 7)
            result = 0.5f + 4.f * residual;
        if (params.debugView == 1 && refine)
            result = lerp(result, float3(1.f, 0.1f, 0.1f), 0.35f);
        else if (params.debugView == 2)
            result = float3(steps / 64.f, visited / 32.f, 0.f);
        else if (params.debugView == 3)
            result = 1.f - transmittance;
        color[pixel] = float4(result, 1.f);
        transportError[pixel] = basis.info.x;
        cutStats[pixel] = float4(refine ? 1.f : 0.f, visited, steps, transmittance);
    }
};

cbuffer CB
{
    HSTRCloudRenderer gHSTRCloud;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.execute(dispatchThreadId.xy);
}

groupshared float4 gBeamDepthCache[64];
groupshared float4 gBeamSingle[64];

/// Beam queries of one level: beamSegments consecutive threads march consecutive segments of one query ray from unit
/// transmittance, and the first composes them front to back, (C, T) = (C_a + T_a C_b, T_a T_b).
[numthreads(64, 1, 1)]
void buildBeamQueries(uint3 dispatchThreadId: SV_DispatchThreadID, uint3 groupThreadId: SV_GroupThreadID)
{
    const uint segments = gHSTRCloud.params.beamSegments;
    const uint segment = dispatchThreadId.x % segments;
    uint2 position;
    const bool valid = gHSTRCloud.beamQueryPosition(dispatchThreadId.x / segments, position);
    BeamSample sample = {};
    if (valid)
        sample = gHSTRCloud.marchBeam(
            gHSTRCloud.cameraRayDirection(float2(position)), float2(float(segment), float(segment + 1)) / float(segments)
        );
    gBeamDepthCache[groupThreadId.x] = float4(sample.cache, sample.depth);
    gBeamSingle[groupThreadId.x] = float4(sample.single, 0.f);
    GroupMemoryBarrierWithGroupSync();
    if (!valid || segment != 0)
        return;
    float depth = 0.f;
    float3 cache = 0.f;
    float3 single = 0.f;
    for (uint k = 0; k < segments; ++k)
    {
        const float transmittance = exp(-depth);
        cache += transmittance * gBeamDepthCache[groupThreadId.x + k].rgb;
        single += transmittance * gBeamSingle[groupThreadId.x + k].rgb;
        depth += gBeamDepthCache[groupThreadId.x + k].a;
    }
    gHSTRCloud.writeBeamQuery(position, float4(cache, depth), float4(single, 0.f));
}

[numthreads(64, 1, 1)]
void testBeamTiles(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.testBeamTile(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void testBeamTilesTemporal(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.testBeamTileTemporal(dispatchThreadId.x);
}

[numthreads(8, 8, 1)]
void resolveBeam(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.resolveBeamPixel(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void marchBeamPixels(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.marchBeamPixel(dispatchThreadId.x);
}

[numthreads(1, 1, 1)]
void writeBeamArgs(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.writeBeamArgs();
}

[numthreads(8, 8, 1)]
void referencePathTrace(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.referencePixel(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void compareReference(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.compareReferenceRow(dispatchThreadId.x);
}

[numthreads(4, 4, 4)]
void updateWorldCache(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.updateWorldCacheCell(dispatchThreadId);
}

[numthreads(64, 1, 1)]
void traceWorldCachePhotons(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.tracePhoton(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void advanceWorldCachePhotons(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.advancePhoton(dispatchThreadId.x);
}

[numthreads(4, 4, 4)]
void resolveWorldCache(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.resolveWorldCacheCell(dispatchThreadId);
}

[numthreads(4, 4, 4)]
void bakeWorldCache(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.bakeWorldCacheCell(dispatchThreadId);
}

[numthreads(4, 4, 4)]
void updateCameraLighting(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.writeCameraLighting(dispatchThreadId);
}

[numthreads(8, 8, 1)]
void buildCameraQueries(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.writeCameraQuery(dispatchThreadId.xy);
}

[numthreads(8, 8, 1)]
void buildTileBases(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.buildTileBasis(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void projectCutNodes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.projectCutNode(dispatchThreadId.x);
}

[numthreads(64, 1, 1)]
void binCutNodes(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.binCutNode(dispatchThreadId.x);
}

groupshared uint gSortKeys[kMaxTileNodes];
groupshared uint gSortIDs[kMaxTileNodes];

/// One group per tile: a bitonic sort orders the tile's cut by BSP front-to-back key.
[numthreads(kSortGroupSize, 1, 1)]
void sortTileCuts(uint3 groupId: SV_GroupID, uint3 groupThreadId: SV_GroupThreadID)
{
    const HSTRCloudParams params = gHSTRCloud.params;
    const uint2 tile = groupId.xy;
    const uint lane = groupThreadId.x;
    const bool valid = all(tile < params.hstrTileDims) && gScene.getGridVolumeCount() > 0;
    const uint tileID = tile.x + params.hstrTileDims.x * tile.y;
    const uint count = valid ? min(gHSTRCloud.hstrTileNodeCounts[tileID], params.hstrMaxTileNodes) : 0u;
    const uint base = tileID * params.hstrMaxTileNodes;
    // count is uniform across the group, so returning here cannot split a barrier.
    if (count < 2u)
        return;
    uint sortSize = 2u;
    while (sortSize < count)
        sortSize <<= 1;

    for (uint i = lane; i < sortSize; i += kSortGroupSize)
    {
        const uint id = i < count ? gHSTRCloud.hstrTileNodes[base + i] : ~0u;
        gSortKeys[i] = i < count ? gHSTRCloud.hstrNodeOrder[id] : ~0u;
        gSortIDs[i] = id;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 2; k <= sortSize; k <<= 1)
    {
        for (uint j = k >> 1; j > 0; j >>= 1)
        {
            for (uint i = lane; i < sortSize; i += kSortGroupSize)
            {
                const uint partner = i ^ j;
                if (partner > i)
                {
                    const bool ascending = (i & k) == 0;
                    const uint a = gSortKeys[i];
                    const uint b = gSortKeys[partner];
                    if ((a > b) == ascending)
                    {
                        gSortKeys[i] = b;
                        gSortKeys[partner] = a;
                        const uint id = gSortIDs[i];
                        gSortIDs[i] = gSortIDs[partner];
                        gSortIDs[partner] = id;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for (uint i = lane; i < count; i += kSortGroupSize)
        gHSTRCloud.hstrTileNodes[base + i] = gSortIDs[i];
}

[numthreads(4, 4, 4)]
void computeFineSun(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.computeFineSun(dispatchThreadId);
}

[numthreads(4, 4, 4)]
void gatherSunOctaves(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.gatherSunOctaves(dispatchThreadId);
}

[numthreads(4, 4, 4)]
void blurSunOctaves(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.blurSunOctaves(dispatchThreadId);
}

[numthreads(4, 4, 4)]
void maskResidual(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.maskResidual(dispatchThreadId);
}

[numthreads(64, 1, 1)]
void solveLeaves(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint leaf = dispatchThreadId.x;
    const uint leafCount = gHSTRCloud.params.hstrLeafDims.x * gHSTRCloud.params.hstrLeafDims.y * gHSTRCloud.params.hstrLeafDims.z;
    if (leaf >= leafCount)
        return;
    const HSTRCloudParams params = gHSTRCloud.params;

    float3 incident[kMaxTraceDofs];
    for (uint dof = 0; dof < params.hstrTraceDofs; ++dof)
        incident[dof] = 0.f;
    float omittedError = 0.f;
    const uint matrixSize = params.hstrTraceDofs * params.hstrStorageRank;
    for (uint mode = 0; mode < params.hstrStorageRank; ++mode)
    {
        float3 coefficient = 0.f;
        for (uint rootDof = 0; rootDof < params.hstrTraceDofs; ++rootDof)
            coefficient += gHSTRCloud.hstrLeafBasisRight[matrixSize * leaf + params.hstrStorageRank * rootDof + mode] *
                           gHSTRCloud.hstrRootIncident[rootDof].rgb;

        const float adjointResponse = gHSTRCloud.hstrLeafAdjointResponses[leaf * params.hstrStorageRank + mode];
        const float importance = adjointResponse * length(coefficient);
        const bool active = mode < params.activeRank && importance >= params.activeThreshold;
        if (active)
            for (uint dof = 0; dof < params.hstrTraceDofs; ++dof)
                incident[dof] += gHSTRCloud.hstrLeafBasisLeft[matrixSize * leaf + params.hstrStorageRank * dof + mode] * coefficient;
        else
            omittedError += importance;
    }
    float3 outgoing[kMaxTraceDofs];
    for (uint outputDof = 0; outputDof < params.hstrTraceDofs; ++outputDof)
        outgoing[outputDof] = 0.f;
    for (uint inputDof = 0; inputDof < params.hstrTraceDofs; ++inputDof)
    {
        const uint inputFace = inputDof / params.hstrFaceDofs;
        const uint spatialMode = inputDof % params.hstrFaceDofs;
        const uint ballisticOutput = (inputFace ^ 1u) * params.hstrFaceDofs + spatialMode;
        outgoing[ballisticOutput] += gHSTRCloud.hstrLeafBallistic[leaf * params.hstrTraceDofs + inputDof] * incident[inputDof];
        for (uint outputFace = 0; outputFace < 6; ++outputFace)
            outgoing[outputFace * params.hstrFaceDofs + spatialMode] +=
                gHSTRCloud.hstrLeafNearScatter[(leaf * params.hstrTraceDofs + inputDof) * 6 + outputFace] * incident[inputDof];
    }
    for (uint mode = 0; mode < gHSTRCloud.hstrLeafDiffuseRanks[leaf]; ++mode)
    {
        float3 coefficient = 0.f;
        for (uint inputDof = 0; inputDof < params.hstrTraceDofs; ++inputDof)
            coefficient +=
                gHSTRCloud
                    .hstrLeafDiffuseRight[params.hstrTraceDofs * params.hstrTraceDofs * leaf + params.hstrTraceDofs * inputDof + mode] *
                incident[inputDof];
        for (uint outputDof = 0; outputDof < params.hstrTraceDofs; ++outputDof)
            outgoing[outputDof] +=
                gHSTRCloud
                    .hstrLeafDiffuseLeft[params.hstrTraceDofs * params.hstrTraceDofs * leaf + params.hstrTraceDofs * outputDof + mode] *
                coefficient;
    }
    const float correctionFade = saturate(float(params.frameIndex + 1) / max(1.f, float(params.correctionFadeFrames)));
    const uint2 correctionRange = gHSTRCloud.hstrCorrectionRanges[leaf];
    float admittedError = 0.f;
    for (uint i = 0; i < correctionRange.y; ++i)
    {
        const HSTRCorrectionAtom atom = gHSTRCloud.hstrCorrectionAtoms[correctionRange.x + i];
        outgoing[atom.row] += correctionFade * atom.value * incident[atom.col];
        admittedError += correctionFade * atom.goalError;
    }
    float3 incidentFlux = 0.f;
    float3 outgoingFlux = 0.f;
    for (uint dof = 0; dof < params.hstrTraceDofs; ++dof)
    {
        incidentFlux += max(0.f, incident[dof]);
        outgoing[dof] = max(0.f, outgoing[dof]);
        outgoingFlux += outgoing[dof];
    }
    const float3 passiveScale = min(1.f, incidentFlux / max(float3(1e-8f), outgoingFlux));
    for (uint dof = 0; dof < params.hstrTraceDofs; ++dof)
        outgoing[dof] *= passiveScale;
    const float omittedCorrectionError = max(0.f, gHSTRCloud.hstrLeafResidualBounds[leaf] - admittedError);
    for (uint outputDof = 0; outputDof < params.hstrTraceDofs; ++outputDof)
        gHSTRCloud.hstrRadiance[params.hstrTraceDofs * leaf + outputDof] = float4(outgoing[outputDof], omittedCorrectionError + omittedError);
}
