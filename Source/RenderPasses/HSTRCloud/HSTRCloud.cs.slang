/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
import Utils.Geometry.IntersectionHelpers;
import Utils.Math.Ray;
import Scene.Scene;
import HSTRCloudTypes;

static const uint kMaxTraceDofs = 24;

struct HSTRCorrectionAtom
{
    uint row;
    uint col;
    float value;
    float goalError;
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
    Texture3D<float> hstrExtinction;
    SamplerState hstrExtinctionSampler;
    RWTexture2D<float4> color;
    RWTexture2D<float> transportError;

    float phaseHG(float cosine)
    {
        const float g = params.anisotropy;
        const float denominator = max(1e-4f, 1.f + g * g - 2.f * g * cosine);
        return (1.f - g * g) / (4.f * 3.14159265f * denominator * sqrt(denominator));
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

    float extinctionAt(float3 leafPosition)
    {
        const float3 voxelPosition = leafPosition * float(params.hstrCellWidth);
        return hstrExtinction.SampleLevel(hstrExtinctionSampler, saturate(voxelPosition / float3(params.hstrExtinctionDims)), 0.f);
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

    float4 integrateTransportCut(Ray ray, GridVolume volume, float2 interval, out float residualError)
    {
        residualError = 0.f;
        const float3 sky = params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * ray.dir.y));
        const float3 entryIndex = mul(volume.data.invTransform, float4(ray.origin + max(0.f, interval.x) * ray.dir, 1.f)).xyz;
        const float3 exitIndex = mul(volume.data.invTransform, float4(ray.origin + interval.y * ray.dir, 1.f)).xyz;
        float3 position = (entryIndex - float3(params.hstrGridMin)) / float(params.hstrCellWidth);
        const float3 endPosition = (exitIndex - float3(params.hstrGridMin)) / float(params.hstrCellWidth);
        const float3 direction = endPosition - position;
        float3 radiance = 0.f;
        float transmittance = 1.f;
        uint stack[32];
        uint stackSize = 1;
        stack[0] = params.hstrRootNode;

        [loop]
        for (uint visit = 0; visit < 512 && stackSize > 0 && transmittance > 1e-3f; ++visit)
        {
            const uint nodeID = stack[--stackSize];
            const HSTRCutNode node = hstrCutNodes[nodeID];
            float2 nodeInterval;
            if (node.statistics.y <= 1e-7f || !intersectCutNode(node, position, direction, nodeInterval))
                continue;
            const bool leaf = node.minLeft.w == ~0u;
            const float variation = (node.statistics.y - node.statistics.x) / max(1e-6f, node.statistics.y);
            const float3 midpointPosition = position + (0.5f * (nodeInterval.x + nodeInterval.y)) * direction;
            const float projectedPixels =
                params.frameDim.y *
                max(node.maxRight.x - node.minLeft.x, max(node.maxRight.y - node.minLeft.y, node.maxRight.z - node.minLeft.z)) /
                max(1.f, length(midpointPosition - position));
            const bool accept = leaf || (variation <= params.mixedFidelityThreshold && node.statistics.z <= params.mixedFidelityThreshold &&
                                         projectedPixels <= 12.f);
            if (!accept)
            {
                float2 leftInterval;
                float2 rightInterval;
                const bool hitLeft = intersectCutNode(hstrCutNodes[node.minLeft.w], position, direction, leftInterval);
                const bool hitRight = intersectCutNode(hstrCutNodes[node.maxRight.w], position, direction, rightInterval);
                if (hitLeft && hitRight)
                {
                    const bool leftFirst = leftInterval.x < rightInterval.x;
                    stack[stackSize++] = leftFirst ? node.maxRight.w : node.minLeft.w;
                    stack[stackSize++] = leftFirst ? node.minLeft.w : node.maxRight.w;
                }
                else if (hitLeft)
                    stack[stackSize++] = node.minLeft.w;
                else if (hitRight)
                    stack[stackSize++] = node.maxRight.w;
                continue;
            }

            const float segmentParameterLength = nodeInterval.y - nodeInterval.x;
            const float voxelLength = segmentParameterLength * length(direction) * params.hstrCellWidth;
            const uint sampleCount =
                leaf && variation > params.mixedFidelityThreshold ? clamp(uint(ceil(0.5f * voxelLength)), 2u, 12u) : 2u;
            const float dt = segmentParameterLength / sampleCount;
            [loop]
            for (uint sample = 0; sample < sampleCount && transmittance > 1e-3f; ++sample)
            {
                const float t = nodeInterval.x + (sample + 0.5f) * dt;
                const float3 samplePosition = position + t * direction;
                const float extinction = extinctionAt(samplePosition);
                if (extinction <= 1e-7f)
                    continue;
                const float worldLength = (interval.y - max(0.f, interval.x)) * dt;
                const float sampleTransmittance = exp(-extinction * worldLength);
                const float alpha = 1.f - sampleTransmittance;
                const float3 indexPosition = float3(params.hstrGridMin) + float(params.hstrCellWidth) * samplePosition;
                radiance += transmittance * alpha * volume.data.albedo * hstrLighting(indexPosition, ray.dir);
                residualError += transmittance * alpha * hstrResidualBound(indexPosition);
                transmittance *= sampleTransmittance;
            }
        }
        return float4(radiance + transmittance * sky, transmittance);
    }

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
        const float phase = 4.f * 3.14159265f * phaseHG(dot(normalize(params.sunDirection), viewDirection));
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

    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        const Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        const float3 sky = params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * ray.dir.y));
        if (gScene.getGridVolumeCount() == 0)
        {
            color[pixel] = float4(sky, 1.f);
            transportError[pixel] = 0.f;
            return;
        }

        const GridVolume volume = gScene.getGridVolume(0);
        const AABB bounds = volume.getBounds();
        float2 interval;
        if (!intersectRayAABB(ray.origin, ray.dir, bounds.minPoint, bounds.maxPoint, interval) || interval.y <= max(0.f, interval.x))
        {
            color[pixel] = float4(sky, 1.f);
            transportError[pixel] = 0.f;
            return;
        }

        const bool insideCloudBounds = all(ray.origin > bounds.minPoint) && all(ray.origin < bounds.maxPoint);
        if (insideCloudBounds)
        {
            float residualError;
            color[pixel] = float4(
                integrateVolume(ray, volume, min(1024u, (params.baseSteps << params.refinementLevel) << 1), pixel, residualError).rgb, 1.f
            );
            transportError[pixel] = residualError;
            return;
        }

        // HST-Cut: traverse preintegrated transport nodes, not NanoVDB samples.
        float residualError;
        color[pixel] = float4(integrateTransportCut(ray, volume, interval, residualError).rgb, 1.f);
        transportError[pixel] = residualError;
    }
};

cbuffer CB
{
    HSTRCloudRenderer gHSTRCloud;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gHSTRCloud.execute(dispatchThreadId.xy);
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
        gHSTRCloud.hstrRadiance[params.hstrTraceDofs * leaf + outputDof] =
            float4(outgoing[outputDof], omittedCorrectionError + omittedError);
}
