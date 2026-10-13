/// The cut walk's brick visibility (CloudResidency::brickPriority, CloudView::cutGpuVisibility): the transmittance from the camera
/// through the sea domain's proxy to a brick's nearest point and, until one passes 0.99, its corners; the most visible counts. The
/// same samples as CloudResidency::transmittance - up to 128 steps a ray, voxel centres of the wrapped domain, stopped at depth 6 -
/// read from the domain's extinction (the mean density times the density scale, in float16), so not bit-identical to the CPU's.
/// A corner ray stops once it cannot raise the best so far.

cbuffer CB
{
    float3 gCamera;
    uint gCount;
    float3 gOrigin; ///< The sea's world corner (CloudSeaDesc::origin).
    float gVoxel;   ///< Domain voxel world size.
    uint3 gDims;    ///< Domain voxels.
    uint _pad0;
};

StructuredBuffer<float> gBoxes; ///< Per request: lo.xyz, hi.xyz.
Texture3D<float> gExtinction;
RWStructuredBuffer<float> gResults;

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
        depth += gExtinction.Load(int4(x, v.y, z, 0)) * dt;
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
