/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

/// rtSpanProbe (DIAGNOSTIC, off by default): what it costs to find a camera ray's occupied regions with the RT cores instead of
/// marching through empty space and walking the page / brick chain per step.
///
/// buildRtBoxes writes one procedural box per present child octant of every mapped brick of an asset (the region its page table
/// sends samples to: a level-0 brick's page, a level-L brick's present octant of 2^(L-1) pages per axis), and an inactive box
/// (NaN) in every other page slot, so a bottom-level structure per asset spans a fixed range of the box buffer. A missing octant
/// whose pages carry skirt bits (cloudOctantSkirt) keeps density near its faces and counts as present.
/// traceRtProbe follows every pixel's camera ray through the top-level structure over the sea instances (HSTRCloud.cpp places
/// each at its tile copy nearest the camera), and gathers every candidate box's interval in depth order, as a span march would
/// before sampling: the timed work. It writes only its own pixel. reduceRtProbe, untimed, counts what it found.
///
/// MEASURED (rtprobe1, 4K sunset walk, 8.3M pixel rays a frame; branch rt-span-oracle): rtProbe/trace gather-all 33.9 ms (720p
/// smoke: ~78 boxes a ray, 53% of rays 64+), ordered nearest-box queries 4 / 8 / 16 spans 6.0 / 11.7 / 23.2 ms - linear, ~1.45 ms
/// per query per 8.3M rays, so ~0.13 ms per span for the walk's ~740k dirty rays: one box (8 source voxels) costs about what ~1.3
/// march steps do, before any sample inside it. The stop rule (dirty-ray trace under 0.4 ms) fails at 4 spans (~0.54 ms). Shipping
/// units + query in the same arms 3.9-4.1 ms. Rebuilding the structures: boxes 77 ms, bottom-level 22.6 ms, top-level 0.09 ms
/// (14.4M page slots, 775k active boxes, 513 MB). With span evaluation's measured ceiling (~2.2 ms, 09baeced) the RT route saves at
/// most the traversal share less ~1 ms of queries: not the 2x it was meant to find. REJECTED by its stop rule; off by default.
///
/// Approximations of a cost probe: boxes follow the asset's mapped (finest) cut, so an instance reading a coarser level sees the
/// finer boxes; the ray is the pixel's camera ray over the content slab and view distance (mediumInterval), not a beam texel's.

import HSTRCloudTypes;

cbuffer RtProbeCB
{
    uint3 gPageDims;     ///< Level-0 pages per axis of the asset (buildRtBoxes).
    uint gPageOffset;    ///< The asset's first virtual page.
    uint gBoxOffset;     ///< The asset's first box slot.
    uint2 gFrameDim;     ///< traceRtProbe / reduceRtProbe.
    float3 gCameraVoxel; ///< Camera position in sea voxels (worldToVoxel).
    float3 gCameraU;     ///< Camera basis (Falcor pinhole: direction = ndc.x U + ndc.y V + W), world units.
    float3 gCameraV;
    float3 gCameraW;
    float3 gInvVoxelSize; ///< 1 / seaVoxelSize: a world direction in voxels a world unit.
    float2 gSlabY;       ///< The content slab (seaContentY clamped to the grid), world y.
    float gViewDistance;  ///< seaViewDistance, world units.
    float gCameraY;       ///< Camera world y.
    /// 0: one query gathers every box on the ray (no early out). n > 0: up to n ordered queries, each committing its nearest box so
    /// the traversal prunes behind it, then restarting at that box's exit - the next span on demand, as a march that stops when the
    /// ray turns opaque would ask for them.
    uint gTraceSpans;
};

struct RtBox
{
    float3 lo;
    float3 hi;
};

StructuredBuffer<HSTRCloudVirtualPage> gPages;
StructuredBuffer<HSTRCloudBrick> gBricks;
StructuredBuffer<uint> gSkirtMasks;
RWStructuredBuffer<RtBox> gBoxes;
StructuredBuffer<RtBox> gBoxesRead;
RaytracingAccelerationStructure gTlas;
RWTexture2D<uint> gProbeCount;   ///< Candidate boxes along the pixel's ray.
RWTexture2D<float2> gProbeSpan;  ///< First box entry and summed box length (world units), kept so the gather is not dead code.
RWStructuredBuffer<uint> gProbeStats;

static const uint kRtProbeSpans = 32; ///< Intervals a ray keeps sorted (the rest are counted).
// gProbeStats slots.
static const uint kRtStatBoxes = 0;      ///< Active boxes written (buildRtBoxes, all assets).
static const uint kRtStatRays = 1;       ///< Rays with an interval in the slab.
static const uint kRtStatHitRays = 2;    ///< Rays with at least one candidate.
static const uint kRtStatCandidates = 3; ///< Candidate boxes, all rays.
static const uint kRtStatMax = 4;        ///< Most candidates on one ray.
static const uint kRtStatOverflow = 5;   ///< Rays with more than kRtProbeSpans candidates.
static const uint kRtStatCovered = 6;    ///< Summed length of the kept boxes on rays, whole world units.
static const uint kRtStatHistogram = 8;  ///< 8 bins of candidates a ray: 0, 1, 2-3, 4-7, 8-15, 16-31, 32-63, 64+.

[numthreads(4, 4, 4)]
void buildRtBoxes(uint3 p: SV_DispatchThreadID)
{
    if (any(p >= gPageDims))
        return;
    const uint index = p.x + gPageDims.x * (p.y + gPageDims.y * p.z);
    RtBox box;
    box.lo = float3(asfloat(0x7FC00000u), 0.f, 0.f); // Inactive (NaN min.x).
    box.hi = float3(0.f);
    const HSTRCloudVirtualPage page = gPages[gPageOffset + index];
    bool active = false;
    uint size = 1;
    if (page.physicalRef != kCloudRefNone)
    {
        const uint level = page.mappedLevel;
        if (level == 0)
            active = true;
        else
        {
            // Pages of a level-L brick's octant (2^(L-1) per axis) all map that brick: the octant's first page speaks for them.
            size = 1u << (level - 1u);
            if (all((p & (size - 1u)) == 0u))
            {
                const uint3 child = (p >> (level - 1u)) & 1u;
                active = ((gBricks[page.physicalRef].info >> 4) & (1u << (child.x + 2u * child.y + 4u * child.z))) != 0;
                for (uint z = 0; z < size && !active; ++z)
                    for (uint y = 0; y < size && !active; ++y)
                        for (uint x = 0; x < size && !active; ++x)
                        {
                            const uint3 q = min(p + uint3(x, y, z), gPageDims - 1u);
                            active = gSkirtMasks[gPageOffset + q.x + gPageDims.x * (q.y + gPageDims.y * q.z)] != 0u;
                        }
            }
        }
    }
    if (active)
    {
        // Samples whose nearest voxel lies in the pages: [8 p - 0.5, 8 (p + size) - 0.5) source voxels.
        box.lo = float3(p * 8u) - 0.5f;
        box.hi = float3((p + size) * 8u) - 0.5f;
        uint unused;
        InterlockedAdd(gProbeStats[kRtStatBoxes], 1u, unused);
    }
    gBoxes[gBoxOffset + index] = box;
}

[numthreads(8, 8, 1)]
void traceRtProbe(uint2 pixel: SV_DispatchThreadID)
{
    if (any(pixel >= gFrameDim))
        return;
    const float2 ndc = float2(2.f, -2.f) * (float2(pixel) + 0.5f) / float2(gFrameDim) + float2(-1.f, 1.f);
    const float3 world = normalize(ndc.x * gCameraU + ndc.y * gCameraV + gCameraW);
    // mediumInterval for the sea: the content slab, capped at the view distance from the camera (unit-speed world direction).
    float2 interval = float2(0.f, gViewDistance);
    if (abs(world.y) > 1e-8f)
    {
        const float2 y = (gSlabY - gCameraY) / world.y;
        interval = float2(max(min(y.x, y.y), 0.f), min(max(y.x, y.y), gViewDistance));
    }
    else if (gCameraY < gSlabY.x || gCameraY > gSlabY.y)
        interval = float2(1.f, 0.f);
    uint count = 0;
    float2 span = float2(0.f);
    if (interval.y > interval.x && gTraceSpans > 0)
    {
        RayDesc ray;
        ray.Origin = gCameraVoxel;
        ray.Direction = world * gInvVoxelSize;
        ray.TMin = interval.x;
        ray.TMax = interval.y;
        span.x = -1.f;
        for (uint n = 0; n < gTraceSpans && ray.TMin < interval.y; ++n)
        {
            RayQuery<RAY_FLAG_NONE> query;
            query.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
            float exit = 0.f;
            while (query.Proceed())
            {
                if (query.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE)
                    continue;
                const RtBox box = gBoxesRead[query.CandidateInstanceID() + query.CandidatePrimitiveIndex()];
                const float3 o = query.CandidateObjectRayOrigin();
                const float3 inv = 1.f / query.CandidateObjectRayDirection();
                const float3 a = (box.lo - o) * inv;
                const float3 b = (box.hi - o) * inv;
                const float t0 = max(max(min(a.x, b.x), min(a.y, b.y)), max(min(a.z, b.z), ray.TMin));
                const float t1 = min(min(max(a.x, b.x), max(a.y, b.y)), min(max(a.z, b.z), interval.y));
                // Committing at the entry shrinks TMax, so nodes behind this box are not visited any more.
                if (t1 > t0 && t0 < query.CommittedRayT())
                {
                    query.CommitProceduralPrimitiveHit(t0);
                    exit = t1;
                }
            }
            if (query.CommittedStatus() != COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
                break;
            ++count;
            span.x = span.x < 0.f ? query.CommittedRayT() : span.x;
            span.y += exit - query.CommittedRayT();
            ray.TMin = max(exit, ray.TMin + 1e-4f);
        }
        span.x = max(span.x, 0.f);
    }
    else if (interval.y > interval.x)
    {
        RayDesc ray;
        ray.Origin = gCameraVoxel;
        ray.Direction = world * gInvVoxelSize; // t stays in world units.
        ray.TMin = interval.x;
        ray.TMax = interval.y;
        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
        float2 spans[kRtProbeSpans];
        while (query.Proceed())
        {
            if (query.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE)
                continue;
            const RtBox box = gBoxesRead[query.CandidateInstanceID() + query.CandidatePrimitiveIndex()];
            const float3 o = query.CandidateObjectRayOrigin();
            const float3 inv = 1.f / query.CandidateObjectRayDirection();
            const float3 a = (box.lo - o) * inv;
            const float3 b = (box.hi - o) * inv;
            const float3 near = min(a, b);
            const float3 far = max(a, b);
            const float t0 = max(max(near.x, near.y), max(near.z, ray.TMin));
            const float t1 = min(min(far.x, far.y), min(far.z, ray.TMax));
            if (!(t1 > t0))
                continue;
            // Kept in depth order, as the span march would read them (insertion: a handful a ray).
            if (count < kRtProbeSpans)
            {
                uint k = count;
                for (; k > 0 && spans[k - 1].x > t0; --k)
                    spans[k] = spans[k - 1];
                spans[k] = float2(t0, t1);
            }
            ++count;
        }
        const uint kept = min(count, kRtProbeSpans);
        for (uint k = 0; k < kept; ++k)
            span.y += spans[k].y - spans[k].x;
        span.x = kept > 0 ? spans[0].x : 0.f;
    }
    gProbeCount[pixel] = interval.y > interval.x ? count + 1u : 0u; // 0: no interval; else candidates + 1.
    gProbeSpan[pixel] = span;
}

[numthreads(8, 8, 1)]
void reduceRtProbe(uint2 pixel: SV_DispatchThreadID)
{
    const bool inside = all(pixel < gFrameDim);
    const uint coded = inside ? gProbeCount[pixel] : 0u;
    const bool rayed = coded != 0u;
    const uint count = rayed ? coded - 1u : 0u;
    const uint covered = rayed ? uint(gProbeSpan[pixel].y * 1024.f) : 0u;
    const uint bin = count == 0 ? 0u : min(firstbithigh(count) + 1u, 7u);
    const uint rays = WaveActiveCountBits(rayed);
    const uint hits = WaveActiveCountBits(count > 0);
    const uint total = WaveActiveSum(count);
    const uint most = WaveActiveMax(count);
    const uint overflow = WaveActiveCountBits(count > kRtProbeSpans);
    const uint length = WaveActiveSum(covered >> 10); // Whole world units.
    if (WaveIsFirstLane())
    {
        uint unused;
        InterlockedAdd(gProbeStats[kRtStatRays], rays, unused);
        InterlockedAdd(gProbeStats[kRtStatHitRays], hits, unused);
        InterlockedAdd(gProbeStats[kRtStatCandidates], total, unused);
        InterlockedMax(gProbeStats[kRtStatMax], most, unused);
        InterlockedAdd(gProbeStats[kRtStatOverflow], overflow, unused);
        InterlockedAdd(gProbeStats[kRtStatCovered], length, unused);
    }
    for (uint b = 0; b < 8; ++b)
    {
        const uint n = WaveActiveCountBits(rayed && bin == b);
        if (WaveIsFirstLane() && n > 0)
        {
            uint unused;
            InterlockedAdd(gProbeStats[kRtStatHistogram + b], n, unused);
        }
    }
}
