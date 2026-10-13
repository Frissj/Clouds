/// The cut walk's brick visibility (CloudResidency::brickPriority, CloudView::cutGpuVisibility): the transmittance from the camera
/// through the sea domain's proxy to a brick's nearest point and, until one passes 0.99, its corners; the most visible counts. The
/// same samples as CloudResidency::transmittance - up to 128 steps a ray, voxel centres of the wrapped domain, stopped at depth 6 -
/// read from the domain's extinction (the mean density times the density scale, in float16), so not bit-identical to the CPU's.
/// A corner ray stops once it cannot raise the best so far.
/// It runs on the residency's compute queue, which may not touch the extinction texture while the main queue uses it (D3D12 1047 on
/// a non-simultaneous-access texture, visdbg2, even read beside read): it reads a buffer copy of the same float16 values, two to a
/// word, x + dims.x * (y + dims.y * z), which packVisibilityProxy refreshes for each slot the domain uploads.

cbuffer CB
{
    float3 gCamera;
    uint gCount;
    float3 gOrigin; ///< The sea's world corner (CloudSeaDesc::origin).
    float gVoxel;   ///< Domain voxel world size.
    uint3 gDims;    ///< Domain voxels.
    uint gTileVoxels; ///< packVisibilityProxy: the slot's voxels a side ...
    uint2 gSlotOrigin; ///< ... and its first voxel's x and z.
};

StructuredBuffer<float> gBoxes; ///< Per request: lo.xyz, hi.xyz.
StructuredBuffer<uint> gProxy;
RWStructuredBuffer<float> gResults;
Texture3D<float> gExtinction;
RWStructuredBuffer<uint> gProxyOutput;

float proxyAt(int3 v)
{
    const uint index = uint(v.x) + gDims.x * (uint(v.y) + gDims.y * uint(v.z));
    return f16tof32(gProxy[index >> 1] >> ((index & 1u) * 16u));
}

/// A slot's extinction into the proxy buffer: a thread per two voxels along x.
[numthreads(8, 4, 8)]
void packVisibilityProxy(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= gTileVoxels / 2 || id.y >= gDims.y || id.z >= gTileVoxels)
        return;
    const uint3 v = uint3(gSlotOrigin.x + 2 * id.x, id.y, gSlotOrigin.y + id.z);
    const float a = gExtinction.Load(int4(v, 0));
    const float b = gExtinction.Load(int4(v + uint3(1, 0, 0), 0));
    gProxyOutput[(v.x + gDims.x * (v.y + gDims.y * v.z)) >> 1] = f32tof16(a) | (f32tof16(b) << 16);
}

float transmittance(float3 target, float maxDepth)
{
    const float3 delta = target - gCamera;
    const float len = length(delta);
    if (len <= gVoxel)
        return 1.f;
    const uint steps = min(128u, uint(ceil(len / gVoxel)));
    const float dt = len / float(steps);
    float depth = 0.f;
    for (uint i = 0; i < steps && depth < maxDepth; ++i)
    {
        const float3 p = gCamera + delta * ((float(i) + 0.5f) / float(steps));
        const int3 v = int3(floor((p - gOrigin) / gVoxel));
        if (v.y < 0 || v.y >= int(gDims.y))
            continue;
        const int x = ((v.x % int(gDims.x)) + int(gDims.x)) % int(gDims.x);
        const int z = ((v.z % int(gDims.z)) + int(gDims.z)) % int(gDims.z);
        depth += proxyAt(int3(x, v.y, z)) * dt;
    }
    return exp(-depth);
}

[numthreads(64, 1, 1)]
void cutVisibility(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= gCount)
        return;
    const uint b = id.x * 6;
    const float3 lo = float3(gBoxes[b], gBoxes[b + 1], gBoxes[b + 2]);
    const float3 hi = float3(gBoxes[b + 3], gBoxes[b + 4], gBoxes[b + 5]);
    float best = transmittance(clamp(gCamera, lo, hi), 6.f);
    for (uint corner = 0; corner < 8 && best < 0.99f; ++corner)
    {
        const float3 p = float3((corner & 1) != 0 ? hi.x : lo.x, (corner & 2) != 0 ? hi.y : lo.y, (corner & 4) != 0 ? hi.z : lo.z);
        const float cutoff = best > 0.f ? min(6.f, -log(best) * (1.f + 1e-5f) + 1e-6f) : 6.f;
        best = max(best, transmittance(p, cutoff));
    }
    gResults[id.x] = best;
}
