/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
import Utils.Geometry.IntersectionHelpers;
import Utils.Math.Ray;
import Scene.Scene;
import HSTRCloudTypes;

struct HSTRCloudRenderer
{
    HSTRCloudParams params;
    RWStructuredBuffer<float4> hstrRadiance;
    StructuredBuffer<float> hstrLeafBasisLeft;
    StructuredBuffer<float> hstrLeafBasisRight;
    StructuredBuffer<float> hstrLeafTransport;
    StructuredBuffer<float> hstrLeafResidualBounds;
    StructuredBuffer<float4> hstrRootIncident;
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

    uint leafIndex(uint3 p)
    {
        return p.x + params.hstrLeafDims.x * (p.y + params.hstrLeafDims.y * p.z);
    }

    float3 hstrField(float3 p, uint field)
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
                    result += hstrRadiance[6 * leafIndex(uint3(q)) + field].rgb * w.x * w.y * w.z;
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
                    result += hstrRadiance[6 * leafIndex(uint3(q))].a * w.x * w.y * w.z;
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

    float3 deterministicResidual(GridVolume volume, Grid grid, float3 indexPosition, float3 viewDirection, inout Grid::Accessor accessor)
    {
        const float h = 1.5f;
        const float3 gradient = float3(
            grid.lookupLinearIndex(indexPosition + float3(h, 0.f, 0.f), accessor) - grid.lookupLinearIndex(indexPosition - float3(h, 0.f, 0.f), accessor),
            grid.lookupLinearIndex(indexPosition + float3(0.f, h, 0.f), accessor) - grid.lookupLinearIndex(indexPosition - float3(0.f, h, 0.f), accessor),
            grid.lookupLinearIndex(indexPosition + float3(0.f, 0.f, h), accessor) - grid.lookupLinearIndex(indexPosition - float3(0.f, 0.f, h), accessor)
        );
        const float density = max(0.f, grid.lookupLinearIndex(indexPosition, accessor));
        const float edge = saturate(2.f * length(gradient) / max(0.02f, density));
        if (edge <= 0.f) return 0.f;

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
        if (!intersectRayAABB(ray.origin, ray.dir, bounds.minPoint, bounds.maxPoint, interval)) return float4(sky, 1.f);
        interval.x = max(interval.x, 0.f);
        if (interval.x >= interval.y || !volume.hasDensityGrid()) return float4(sky, 1.f);

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
            if (extinction <= 0.f) continue;

            const float alpha = 1.f - exp(-extinction * stepLength);
            const float3 indexPosition = mul(volume.data.invTransform, float4(position, 1.f)).xyz;
            const float3 base = hstrLighting(indexPosition, ray.dir);
            const float3 correction = deterministicResidual(volume, grid, indexPosition, ray.dir, accessor);
            const float3 source = volume.data.albedo * (0.72f * base + params.residualBlend * correction);
            radiance += transmittance * alpha * source;
            residualError += transmittance * alpha * hstrResidualBound(indexPosition);
            transmittance *= 1.f - alpha;
        }
        return float4(radiance + transmittance * sky, transmittance);
    }

    void execute(uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;
        const Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        const float3 sky = params.skyRadiance * (0.35f + 0.65f * saturate(0.5f + 0.5f * ray.dir.y));
        if (gScene.getGridVolumeCount() == 0)
        {
            color[pixel] = float4(sky, 1.f);
            transportError[pixel] = 0.f;
            return;
        }

        const uint coarseSteps = params.baseSteps << params.refinementLevel;
        const uint fineSteps = min(1024u, coarseSteps << 1);
        const GridVolume volume = gScene.getGridVolume(0);
        float coarseResidual;
        float fineResidual;
        const float4 coarse = integrateVolume(ray, volume, coarseSteps, pixel, coarseResidual);
        const float4 fine = integrateVolume(ray, volume, fineSteps, pixel, fineResidual);
        color[pixel] = float4(lerp(coarse.rgb, fine.rgb, params.residualBlend), 1.f);
        transportError[pixel] = length(fine.rgb - coarse.rgb) + fineResidual;
    }
};

cbuffer CB
{
    HSTRCloudRenderer gHSTRCloud;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gHSTRCloud.execute(dispatchThreadId.xy);
}

[numthreads(64, 1, 1)]
void solveLeaves(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint leaf = dispatchThreadId.x;
    const uint leafCount = gHSTRCloud.params.hstrLeafDims.x * gHSTRCloud.params.hstrLeafDims.y * gHSTRCloud.params.hstrLeafDims.z;
    if (leaf >= leafCount) return;

    float3 incident[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float omittedError = 0.f;
    for (uint mode = 0; mode < 6; ++mode)
    {
        float3 coefficient = 0.f;
        for (uint rootFace = 0; rootFace < 6; ++rootFace)
            coefficient += gHSTRCloud.hstrLeafBasisRight[36 * leaf + 6 * rootFace + mode] * gHSTRCloud.hstrRootIncident[rootFace].rgb;

        float adjointResponse = 0.f;
        for (uint inputFace = 0; inputFace < 6; ++inputFace)
            adjointResponse += gHSTRCloud.hstrLeafTransport[
                36 * leaf + 6 * gHSTRCloud.params.goalFace + inputFace
            ] * gHSTRCloud.hstrLeafBasisLeft[36 * leaf + 6 * inputFace + mode];
        const float importance = abs(adjointResponse) * length(coefficient);
        const bool active = mode < gHSTRCloud.params.activeRank && importance >= gHSTRCloud.params.activeThreshold;
        if (active)
            for (uint face = 0; face < 6; ++face)
                incident[face] += gHSTRCloud.hstrLeafBasisLeft[36 * leaf + 6 * face + mode] * coefficient;
        else
            omittedError += importance;
    }
    for (uint outputFace = 0; outputFace < 6; ++outputFace)
    {
        float3 outgoing = 0.f;
        for (uint inputFace = 0; inputFace < 6; ++inputFace)
            outgoing += gHSTRCloud.hstrLeafTransport[36 * leaf + 6 * outputFace + inputFace] * incident[inputFace];
        gHSTRCloud.hstrRadiance[6 * leaf + outputFace] = float4(outgoing, gHSTRCloud.hstrLeafResidualBounds[leaf] + omittedError);
    }
}
