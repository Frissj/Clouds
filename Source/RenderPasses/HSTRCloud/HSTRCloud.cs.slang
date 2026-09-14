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

/// A primary ray restricted to the volume bounds, expressed in leaf space.
/// position is the entry point (t = 0), direction the entry-to-exit vector (t = 1).
struct CutRay
{
    float3 position;
    float3 direction;
    float worldLength;
    float4 sunOctaves; ///< Per-octave sun weight a^n p(theta; c^n g) for this ray's direction.
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

    // Path-traced reference: running sum and per-row comparison against the HST frame.
    RWTexture2D<float4> hstrReferenceSum;
    RWStructuredBuffer<float2> hstrReferenceRowError;

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
        ray.sunOctaves = sunOctaveWeights(direction);
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
                    const float3 base = sourceAtVoxel(midpoint);
                    if (withSource)
                        radiance += transmittance * alpha * base;
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

    float3 referencePath(float3 origin, float3 direction, GridVolume volume, inout uint rng)
    {
        float2 interval;
        const AABB bounds = volume.getBounds();
        if (!intersectRayAABB(origin, direction, bounds.minPoint, bounds.maxPoint, interval) || interval.y <= max(0.f, interval.x))
            return skyRadianceAt(direction);
        const float3 sunDirection = normalize(params.sunDirection);
        const float3 sunVoxel = mul(volume.data.invTransform, float4(sunDirection, 0.f)).xyz;
        const float3 entryIndex = mul(volume.data.invTransform, float4(origin + max(0.f, interval.x) * direction, 1.f)).xyz;
        float3 v = entryIndex - float3(params.hstrGridMin);
        float3 radiance = 0.f;
        float3 throughput = 1.f;
        for (uint bounce = 0; bounce < 4096; ++bounce)
        {
            const float3 d = mul(volume.data.invTransform, float4(direction, 0.f)).xyz;
            if (!sampleCollision(v, d, rng))
            {
                radiance += throughput * skyRadianceAt(direction);
                break;
            }
            throughput *= volume.data.albedo;
            // Sun: directional irradiance through the ratio-tracked transmittance.
            radiance += throughput * phaseHG(dot(sunDirection, direction)) * params.sunRadiance * ratioTrack(v, sunVoxel, rng);
            direction = sampleHG(direction, float2(randomFloat(rng), randomFloat(rng)));
            if (bounce >= 8)
            {
                const float survive = min(0.95f, maxComponent(throughput));
                if (randomFloat(rng) >= survive)
                    break;
                throughput /= survive;
            }
        }
        return radiance;
    }

    void referencePixel(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        uint rng = hashUint(pixel.x + params.frameDim.x * pixel.y) ^ hashUint(params.frameIndex * 9781u + 6271u);
        const float2 jitter = float2(randomFloat(rng), randomFloat(rng));
        const float3 direction = cameraRayDirection(float2(pixel) + jitter);
        float3 result = skyRadianceAt(direction);
        if (gScene.getGridVolumeCount() > 0)
            result = referencePath(gScene.camera.data.posW, direction, gScene.getGridVolume(0), rng);
        if (any(isnan(result) || isinf(result)))
            result = 0.f;
        const float4 sum = (params.referenceSamples == 0 ? 0.f : hstrReferenceSum[pixel]) + float4(result, 1.f);
        hstrReferenceSum[pixel] = sum;
        color[pixel] = float4(sum.rgb / sum.a, 1.f);
        transportError[pixel] = 0.f;
        cutStats[pixel] = 0.f;
    }

    /// Mean absolute difference of one image row against the reference average: linear radiance and log(1 + radiance).
    void compareReferenceRow(uint row)
    {
        if (row >= params.frameDim.y)
            return;
        float2 error = 0.f;
        uint count = 0;
        for (uint x = 0; x < params.frameDim.x; ++x)
        {
            const float4 sum = hstrReferenceSum[uint2(x, row)];
            const float3 reference = sum.rgb / max(sum.a, 1.f);
            const float3 value = color[uint2(x, row)].rgb;
            if (any(isnan(value) || isinf(value) || isnan(reference) || isinf(reference)))
                continue;
            error.x += dot(abs(value - reference), 1.f / 3.f);
            error.y += dot(abs(log(1.f + max(value, 0.f)) - log(1.f + max(reference, 0.f))), 1.f / 3.f);
            ++count;
        }
        hstrReferenceRowError[row] = error / float(max(count, 1u));
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
            color[pixel] = float4(sky, 1.f);
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
        float3 result = max(0.f, radiance + residual) + transmittance * sky;
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
