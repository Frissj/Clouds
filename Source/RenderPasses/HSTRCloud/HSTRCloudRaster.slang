/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

/// rasterProbe (DIAGNOSTIC, off by default): the timing ceiling of drawing the cloud sea with the rasterizer instead of marching it.
///
/// The dirty march finds every sample again through the instance / level page / brick chain, one step at a time in one thread per
/// ray, so a frame pays traversal per sample and its longest rays per wave (db20cf3b: traversal ~70% of the dirty march; the query is
/// bound by its dispatch tail). Here the chain is resolved once per brick and frame, and fixed-function hardware does the rest:
/// - rasterLevel walks each sea instance's resolved level pages from a coarse level down, and keeps per region the brick a sample
///   there reads: the finest resident at or above the footprint level of the region's nearest point (cloudAssetBrickAtFlat's word);
/// - each kept box is cut by view-aligned planes at the march's step length for that brick (z = k step along the view axis), and the
///   slices are put in depth order by a counting sort over 2^16 depth bins (rasterLevel counts, rasterScan, rasterScatter);
/// - one indirect draw rasterizes the slices front to back with hardware under-blending (src x (1 - dst alpha) + dst). A fragment
///   reads the density atlas and the baked sun once: no loop, no traversal, the brick's addressing in flat interpolants.
///
/// A cost ceiling, not the march's image: density and single sun only (hstComponents 1 | 2, as the span probe's arms), and no
/// proxy, skirts, fades, far field, cache, octaves, view-distance fade or transmittance-scaled steps; every slice takes the shortest
/// step (the march's majorant term lengthens steps in thin regions, so it takes fewer samples than this draws); a region takes the
/// footprint level of its nearest point (finer than the march's per-sample level). Each of these makes the probe dearer, not
/// cheaper.
///
/// MEASURED (raster2, sunset_motion 3840x2160 walk, matched leaves; units / query in the same frames ~1.7-2.0 / 1.4-1.6 ms):
/// boxes ~1.0 ms (2.5-2.8M boxes, 3.1-3.4M slices, ~44M fragments at half scale), sort ~0.4 ms, draw at scale 1/4 3.52 / 1/2
/// 4.51 (again 4.70) / full 6.41 ms; 1/2 without fetches 2.80 ms. The draw is ~3.3 ms fixed + ~0.37 ms per million target
/// pixels: primitive-bound (~1 ns a slice polygon), not fragment-bound; the fetches add ~1.7 ms at 1/2. Raster at 1/2 totals
/// ~5.9 ms against ~3.3 ms units + query: slices must drop ~10x (larger regions, the brick looked up per fragment) to compete.
/// MEASURED (raster3, same conditions, single profiled frames): prebuilt polygons (RASTER_PREBUILT) draw at 1/2 4.96 ms against
/// 4.71 / 4.96 for per-vertex polygons, sort 0.42 -> 1.47 ms (the scatter now builds them). Without fetches the prebuilt draw is
/// 1.94 ms (per-vertex 2.80 in raster2): the polygon work was real (~0.9 ms) but hidden under the fetches, which cost ~3.0 ms of
/// the 1/2 draw. 1/4 3.95, full 7.15 ms. The draw is bound by its atlas / bake reads, as the march is - not by geometry.
/// MEASURED (raster4, prebuilt, 1/2, matched leaves): draw both reads 4.80 / again 4.92, density only 3.47, sun only 2.38, none
/// 1.86 ms. The density read costs ~1.6 ms alone, the sun's ~0.5, both ~3.0 (more than the sum: they evict each other). The
/// density is the dear one although its slots are the small ones: workset2 has ~87k distinct density bricks a frame (~87 MB,
/// over the 48 MB L2) against ~25k sun slots.

import HSTRCloudTypes;

#ifndef RASTER_COUNT
#define RASTER_COUNT 0 // rasterProbeCount's second draw: count the fragments shaded.
#endif
#ifndef RASTER_FETCH
// rasterProbeFetch x rasterProbeFetchMask: 1 the density atlas read, 2 the sun bake read (cost split only); 4 reads the density from
// the apron-free 8^3 copy instead (the same values: the read never leaves a slot's core).
#define RASTER_FETCH 3
#endif
#ifndef RASTER_PREBUILT
// rasterProbePrebuilt: rasterScatter builds each slice's polygon once (gPolygons) and the draw is one indexed list over all of them;
// 0: every one of a slice's six vertices rebuilds its polygon (12 edge crossings, an angle sort) in a 12-index instance per slice.
#define RASTER_PREBUILT 0
#endif

/// One sea instance in range: its transforms and the words cloudAssetBrickAtFlat reads (HSTRCloud::runRasterProbe).
struct RasterInstance
{
    float4 toWorld0; ///< Asset source voxel position to world (affine rows: a scaled signed permutation, so boxes stay boxes).
    float4 toWorld1;
    float4 toWorld2;
    float4 toAsset0; ///< World to asset source voxel position.
    float4 toAsset1;
    float4 toAsset2;
    uint levelPageBase;
    uint pageDims; ///< Level-0 pages per axis, 10 bits each.
    uint sunClass;
    float sourceVoxelWorld;
    float scale;
    uint topLevel;
    uint padRaster0;
    uint padRaster1;
};

cbuffer RasterCB
{
    float4x4 gViewProj;
    float4 gFrustum[4];      ///< Side planes (n, d): dot(n, p) + d >= 0 inside.
    float3 gCameraPos;
    float gViewDistance;
    float3 gForward;         ///< Unit view axis.
    float gPixelAngle;
    float3 gCameraU;         ///< Falcor pinhole basis: direction = ndc.x U + ndc.y V + W.
    float gLodBias;
    float3 gCameraV;
    float gNear;
    float3 gCameraW;
    float gMinStepVoxels;
    float3 gSunColour;       ///< residualStrength x sunRadiance, as evaluateSpan lights.
    float gMaxStepVoxels;
    float3 gAtlasInvSize;
    float gSeaVoxel;         ///< World size of a domain voxel (x).
    float3 gSunAtlasInvSize;
    float gFineMinVoxels;
    float gDensityScale;     ///< The grid's density scale x densityScale.
    float gBinScale;         ///< Depth bins a world unit.
    uint gAtlasShift;
    uint gSun8;              ///< cloudSun8: the sun atlas holds the 8-bit code.
    uint2 gTargetDim;
    uint gLevel;             ///< rasterLevel: the level of the queue it reads.
    uint gQueueCapacity;
    uint gBoxCapacity;
    uint gItemCapacity;
    uint gArgsSource;        ///< rasterArgs: the counter it sizes a dispatch from, its cap and where the arguments go.
    uint gArgsCap;
    uint gArgsOffset;
    uint padRasterCB;
    float3 gCompactInvSize;  ///< RASTER_FETCH 4: the apron-free density copy's inverse size.
    float padRasterCB1;
};

// gCounters slots.
static const uint kRasterBoxes = 0;         ///< Boxes kept.
static const uint kRasterItems = 1;         ///< Slices (draw instances).
static const uint kRasterQueueOverflow = 2;
static const uint kRasterBoxOverflow = 3;
static const uint kRasterItemOverflow = 4;
static const uint kRasterCulled = 5;        ///< Regions past the view distance or outside the frustum.
static const uint kRasterEmpty = 6;         ///< Regions whose word is none or empty (no brick density there).
static const uint kRasterUnbaked = 7;       ///< Kept boxes without a sun bake (their light is unlit here).
static const uint kRasterFragments = 8;     ///< rasterProbeCount's draw: fragments shaded.
static const uint kRasterSubdivided = 9;    ///< Regions split into their eight children.
static const uint kRasterNoSlice = 10;      ///< Boxes no plane crosses (thinner than a step along the view axis).
static const uint kRasterLevelBoxes = 16;   ///< 16 slots: boxes kept per level of their region.
static const uint kRasterQueue = 32;        ///< 16 slots: regions queued per level.
static const uint kRasterCounters = 48;
static const uint kRasterBins = 65536;
static const uint kRasterDrawArgs = 4;      ///< gArgs: dispatch arguments at 0, DrawIndexedInstanced arguments at 4.

StructuredBuffer<RasterInstance> gInstances;
StructuredBuffer<uint> gLevelPages;
StructuredBuffer<HSTRCloudBrick> gBricks;
StructuredBuffer<uint2> gSunResolved;
StructuredBuffer<uint4> gQueueIn;
RWStructuredBuffer<uint4> gQueueOut;
RWStructuredBuffer<uint> gCounters;
RWStructuredBuffer<uint> gArgs;
RWStructuredBuffer<uint4> gBoxes;    ///< Two words a box: (instance, region, brick, levels), (sun slot, first plane, planes, step).
StructuredBuffer<uint4> gBoxesRead;
RWStructuredBuffer<uint> gBins;      ///< Slices per depth bin (rasterLevel), then each bin's next item (rasterScan, rasterScatter).
RWStructuredBuffer<uint2> gItems;    ///< (box, plane) in depth order.
StructuredBuffer<uint2> gItemsRead;
RWStructuredBuffer<float4> gPolygons; ///< RASTER_PREBUILT: six world vertices a slice, in fan order (short polygons repeat the last).
StructuredBuffer<float4> gPolygonsRead;
Texture3D<float> gAtlas;
Texture3D<float> gAtlasCompact; ///< RASTER_FETCH 4: the density atlas without aprons, 8^3 a slot (packSpanCompact).
Texture3D<float> gSunAtlas;
SamplerState gLinearClamp;

uint3 unpack10(uint w)
{
    return uint3(w & 1023u, (w >> 10) & 1023u, w >> 20);
}

uint pack10(uint3 c)
{
    return c.x | (c.y << 10) | (c.z << 20);
}

float3 affine(float4 r0, float4 r1, float4 r2, float3 p)
{
    return float3(dot(r0.xyz, p) + r0.w, dot(r1.xyz, p) + r1.w, dot(r2.xyz, p) + r2.w);
}

/// World box of an asset source voxel box (the instance maps boxes to boxes).
void worldBox(RasterInstance instance, float3 lo, float3 hi, out float3 worldLo, out float3 worldHi)
{
    const float3 a = affine(instance.toWorld0, instance.toWorld1, instance.toWorld2, lo);
    const float3 b = affine(instance.toWorld0, instance.toWorld1, instance.toWorld2, hi);
    worldLo = min(a, b);
    worldHi = max(a, b);
}

bool outsideFrustum(float3 lo, float3 hi)
{
    [unroll]
    for (uint k = 0; k < 4; ++k)
    {
        const float3 n = gFrustum[k].xyz;
        if (dot(n, select(n >= 0.f, hi, lo)) + gFrustum[k].w < 0.f)
            return true;
    }
    return false;
}

/// First word of an asset's level-L resolved pages (cloudAssetBrickAtFlat's base).
uint levelBase(RasterInstance instance, uint3 pages, uint level)
{
    uint base = instance.levelPageBase;
    for (uint l = 0; l < level; ++l)
    {
        const uint3 d = (pages + (1u << l) - 1u) >> l;
        base += d.x * d.y * d.z;
    }
    return base;
}

/// marchBeamLean's shortest step through a brick of the given level, in world units (its majorant term only lengthens it).
float stepWorld(RasterInstance instance, uint brickLevel)
{
    const float fineVoxels = float(1u << brickLevel) * instance.sourceVoxelWorld / gSeaVoxel;
    return min(clamp(fineVoxels, gFineMinVoxels, 1.f) * gMinStepVoxels, gMaxStepVoxels) * gSeaVoxel;
}

uint depthBin(int plane, float step)
{
    return min(uint(float(plane) * step * gBinScale), kRasterBins - 1u);
}

/// Dispatch arguments for 64-thread groups over min(gCounters[gArgsSource], gArgsCap) items.
[numthreads(1, 1, 1)]
void rasterArgs()
{
    const uint n = min(gCounters[gArgsSource], gArgsCap);
    gArgs[gArgsOffset] = (n + 63u) / 64u;
    gArgs[gArgsOffset + 1u] = 1u;
    gArgs[gArgsOffset + 2u] = 1u;
}

/// One region of an instance at level gLevel: culled, dropped where no brick holds density, split where a finer level is wanted and
/// this level's brick is resident (so its children may be), else kept as a box with its slices counted into the depth bins.
[numthreads(64, 1, 1)]
void rasterLevel(uint3 id: SV_DispatchThreadID)
{
    const uint level = gLevel;
    if (id.x >= min(gCounters[kRasterQueue + level], gQueueCapacity))
        return;
    const uint4 item = gQueueIn[id.x];
    const RasterInstance instance = gInstances[item.x];
    const uint3 cell = unpack10(item.y);
    const float size = float(8u << level);
    float3 lo, hi;
    worldBox(instance, float3(cell) * size - 0.5f, float3(cell + 1u) * size - 0.5f, lo, hi);
    const float distance = length(clamp(gCameraPos, lo, hi) - gCameraPos);
    uint unused;
    if (distance > gViewDistance || outsideFrustum(lo, hi))
    {
        InterlockedAdd(gCounters[kRasterCulled], 1u, unused);
        return;
    }
    const uint3 pages = unpack10(instance.pageDims);
    const uint3 levelDims = (pages + (1u << level) - 1u) >> level;
    const uint word = gLevelPages[levelBase(instance, pages, level) + cell.x + levelDims.x * (cell.y + levelDims.y * cell.z)];
    if (word == kCloudRefNone || (word & kCloudLevelEmpty) != 0u)
    {
        InterlockedAdd(gCounters[kRasterEmpty], 1u, unused);
        return;
    }
    const uint entry = word & 0xFFFFFFu;
    const uint brickLevel = (word >> 24) & 15u; // The finest resident at or above level: never finer.
    const uint desired = uint(max(log2(max(distance * gPixelAngle / instance.sourceVoxelWorld, 1.f)) + gLodBias, 0.f));
    if (level > 0u && desired < level && brickLevel == level)
    {
        InterlockedAdd(gCounters[kRasterSubdivided], 1u, unused);
        const uint3 childDims = (pages + (1u << (level - 1u)) - 1u) >> (level - 1u);
        for (uint o = 0; o < 8u; ++o)
        {
            const uint3 child = 2u * cell + uint3(o & 1u, (o >> 1) & 1u, o >> 2);
            if (any(child >= childDims))
                continue;
            uint slot;
            InterlockedAdd(gCounters[kRasterQueue + level - 1u], 1u, slot);
            if (slot < gQueueCapacity)
                gQueueOut[slot] = uint4(item.x, pack10(child), 0u, 0u);
            else
                InterlockedAdd(gCounters[kRasterQueueOverflow], 1u, unused);
        }
        return;
    }
    // Kept: the planes z = k step (view axis) that cross the box, inside the view distance and past the near plane.
    const float step = stepWorld(instance, brickLevel);
    const float centre = dot(0.5f * (lo + hi) - gCameraPos, gForward);
    const float reach = dot(0.5f * (hi - lo), abs(gForward));
    const int first = int(ceil(max(centre - reach, gNear) / step));
    const int last = int(floor(min(centre + reach, gViewDistance) / step));
    if (last < first)
    {
        InterlockedAdd(gCounters[kRasterNoSlice], 1u, unused);
        return;
    }
    uint box;
    InterlockedAdd(gCounters[kRasterBoxes], 1u, box);
    if (box >= gBoxCapacity)
    {
        InterlockedAdd(gCounters[kRasterBoxOverflow], 1u, unused);
        return;
    }
    InterlockedAdd(gCounters[kRasterLevelBoxes + level], 1u, unused);
    const uint2 resolved = gSunResolved[entry * kCloudSunClasses + instance.sunClass];
    if (resolved.x == kCloudRefNone)
        InterlockedAdd(gCounters[kRasterUnbaked], 1u, unused);
    gBoxes[2u * box] = uint4(item.x, item.y, entry, level | (brickLevel << 4) | (((resolved.y >> 24) & 15u) << 8));
    gBoxes[2u * box + 1u] = uint4(resolved.x, asuint(first), uint(last - first + 1), asuint(step));
    for (int k = first; k <= last; ++k)
        InterlockedAdd(gBins[depthBin(k, step)], 1u, unused);
    InterlockedAdd(gCounters[kRasterItems], uint(last - first + 1), unused);
}

/// Exclusive prefix over the depth bins (one group), so each bin's slices land after every nearer bin's; and the draw's arguments.
groupshared uint gsRasterSum[1024];

[numthreads(1024, 1, 1)]
void rasterScan(uint3 thread: SV_GroupThreadID)
{
    const uint per = kRasterBins / 1024u;
    const uint start = thread.x * per;
    uint sum = 0;
    for (uint i = 0; i < per; ++i)
        sum += gBins[start + i];
    gsRasterSum[thread.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < 1024u; offset <<= 1)
    {
        const uint add = thread.x >= offset ? gsRasterSum[thread.x - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsRasterSum[thread.x] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint running = gsRasterSum[thread.x] - sum;
    for (uint i = 0; i < per; ++i)
    {
        const uint count = gBins[start + i];
        gBins[start + i] = running;
        running += count;
    }
    if (thread.x == 1023u)
    {
        // DrawIndexedInstanced: the polygon's 12 indices (a fan of four triangles over six vertices), one instance a slice;
        // RASTER_PREBUILT: one instance of every slice's 12.
        const uint slices = min(gsRasterSum[1023], gItemCapacity);
        gArgs[kRasterDrawArgs] = RASTER_PREBUILT ? 12u * slices : 12u;
        gArgs[kRasterDrawArgs + 1u] = RASTER_PREBUILT ? 1u : slices;
        gArgs[kRasterDrawArgs + 2u] = 0u;
        gArgs[kRasterDrawArgs + 3u] = 0u;
        gArgs[kRasterDrawArgs + 4u] = 0u;
    }
}

static const uint2 kRasterEdges[12] = {
    uint2(0, 1), uint2(2, 3), uint2(4, 5), uint2(6, 7), uint2(0, 2), uint2(1, 3),
    uint2(4, 6), uint2(5, 7), uint2(0, 4), uint2(1, 5), uint2(2, 6), uint2(3, 7)
};

/// The plane z (view axis, from the camera) through a world box: its crossings of the box's 12 edges, ordered by angle about their
/// centroid; fewer than six repeat the last (the fan's extra triangles have no area). Returns the crossings (under 3: no polygon).
uint slicePolygon(float3 lo, float3 hi, float z, out float3 sorted[6])
{
    float3 points[6];
    uint count = 0;
    for (uint e = 0; e < 12; ++e)
    {
        const uint2 edge = kRasterEdges[e];
        const float3 p0 = select(bool3((edge.x & 1u) != 0u, (edge.x & 2u) != 0u, (edge.x & 4u) != 0u), hi, lo);
        const float3 p1 = select(bool3((edge.y & 1u) != 0u, (edge.y & 2u) != 0u, (edge.y & 4u) != 0u), hi, lo);
        const float d0 = dot(p0 - gCameraPos, gForward) - z;
        const float d1 = dot(p1 - gCameraPos, gForward) - z;
        if ((d0 < 0.f) != (d1 < 0.f) && count < 6u)
            points[count++] = lerp(p0, p1, d0 / (d0 - d1));
    }
    for (uint i = 0; i < 6u; ++i)
        sorted[i] = points[0];
    if (count < 3u)
        return count;
    float3 centroid = 0.f;
    for (uint i = 0; i < count; ++i)
        centroid += points[i];
    centroid /= float(count);
    const float3 u = normalize(cross(gForward, abs(gForward.y) < 0.9f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f)));
    const float3 v = cross(gForward, u);
    float angles[6];
    for (uint i = 0; i < count; ++i)
        angles[i] = atan2(dot(points[i] - centroid, v), dot(points[i] - centroid, u));
    for (uint i = 0; i < count; ++i)
    {
        uint rank = 0;
        for (uint j = 0; j < count; ++j)
            rank += angles[j] < angles[i] || (angles[j] == angles[i] && j < i) ? 1u : 0u;
        sorted[rank] = points[i];
    }
    for (uint i = count; i < 6u; ++i)
        sorted[i] = sorted[count - 1u];
    return count;
}

/// Each kept box's slices into their depth-ordered places (RASTER_PREBUILT: with their polygons).
[numthreads(64, 1, 1)]
void rasterScatter(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= min(gCounters[kRasterBoxes], gBoxCapacity))
        return;
    const uint4 b = gBoxes[2u * id.x + 1u];
    const int first = asint(b.y);
    const float step = asfloat(b.w);
#if RASTER_PREBUILT
    const uint4 a = gBoxes[2u * id.x];
    const RasterInstance instance = gInstances[a.x];
    const uint3 cell = unpack10(a.y);
    const float size = float(8u << (a.w & 15u));
    float3 lo, hi;
    worldBox(instance, float3(cell) * size - 0.5f, float3(cell + 1u) * size - 0.5f, lo, hi);
#endif
    for (uint j = 0; j < b.z; ++j)
    {
        const int k = first + int(j);
        uint slot;
        InterlockedAdd(gBins[depthBin(k, step)], 1u, slot);
        if (slot < gItemCapacity)
        {
            gItems[slot] = uint2(id.x, uint(k));
#if RASTER_PREBUILT
            float3 polygon[6];
            const uint count = slicePolygon(lo, hi, float(k) * step, polygon);
            for (uint i = 0; i < 6u; ++i)
                gPolygons[6u * slot + i] = float4(polygon[i], count >= 3u ? 1.f : 0.f);
#endif
        }
        else
        {
            uint unused;
            InterlockedAdd(gCounters[kRasterItemOverflow], 1u, unused);
        }
    }
}

struct RasterVertex
{
    float4 position : SV_Position;
    float3 x : ASSET;                              ///< Asset source voxel position.
    nointerpolation float4 density : DENSITY;      ///< Brick-local position = x w + xyz (cloudBrickDensity's levelPosition - brick).
    nointerpolation float4 densityBase : DENSITYBASE; ///< Atlas texel of the brick's first core voxel (xyz), scaled minValue (w).
    nointerpolation float4 sun : SUN;              ///< The bake's brick-local position, as density.
    nointerpolation float4 sunBase : SUNBASE;      ///< Sun atlas texel of its first core voxel (xyz), decode scale (w; 0: no bake).
    nointerpolation float2 brick : BRICK;          ///< Scaled range (0: a constant brick), step (world).
};

float3 atlasTexel(uint slot)
{
    const uint mask = (1u << gAtlasShift) - 1u;
    return float3(uint3(slot & mask, (slot >> gAtlasShift) & mask, slot >> (2u * gAtlasShift)) * kCloudBrickEdge) + 1.5f;
}

/// One vertex of a slice's polygon (slicePolygon). RASTER_PREBUILT: vertex id 6 slice + corner of one list over every slice, the
/// polygon read from gPolygons; else the slice is the instance and every vertex builds the polygon itself.
RasterVertex rasterVS(uint vertexId: SV_VertexID, uint instanceId: SV_InstanceID)
{
    RasterVertex o = {};
    o.position = float4(0.f, 0.f, 0.f, 1.f);
#if RASTER_PREBUILT
    const uint slice = vertexId / 6u;
    const float4 corner = gPolygonsRead[vertexId];
    if (!(corner.w > 0.f))
        return o;
    const float3 p = corner.xyz;
#else
    const uint slice = instanceId;
#endif
    const uint2 item = gItemsRead[slice];
    const uint4 a = gBoxesRead[2u * item.x];
    const uint4 b = gBoxesRead[2u * item.x + 1u];
    const RasterInstance instance = gInstances[a.x];
    const uint level = a.w & 15u;
    const uint brickLevel = (a.w >> 4) & 15u;
    const uint sunLevel = (a.w >> 8) & 15u;
    const uint3 cell = unpack10(a.y);
    const float step = asfloat(b.w);
#if !RASTER_PREBUILT
    const float size = float(8u << level);
    float3 lo, hi;
    worldBox(instance, float3(cell) * size - 0.5f, float3(cell + 1u) * size - 0.5f, lo, hi);
    float3 polygon[6];
    if (slicePolygon(lo, hi, float(int(item.y)) * step, polygon) < 3u)
        return o;
    const float3 p = polygon[min(vertexId, 5u)];
#endif
    o.position = mul(gViewProj, float4(p, 1.f));
    o.x = affine(instance.toAsset0, instance.toAsset1, instance.toAsset2, p);

    // The brick's addressing (cloudBrickDensity, cloudBrickSunAt): every voxel of the box lies in one brick of brickLevel and in
    // one of sunLevel, so the brick coordinates are the box's first voxel's.
    const HSTRCloudBrick brick = gBricks[a.z];
    const int3 first = int3(cell * (8u << level));
    const float inverse = 1.f / float(1u << brickLevel);
    o.density = float4(0.5f * inverse - 0.5f - float3(((first >> int(brickLevel)) >> 3) * 8), inverse);
#if (RASTER_FETCH & 4)
    const uint mask = (1u << gAtlasShift) - 1u;
    const uint3 slotCoord = uint3(brick.slot & mask, (brick.slot >> gAtlasShift) & mask, brick.slot >> (2u * gAtlasShift));
    o.densityBase = float4(float3(slotCoord * 8u) + 0.5f, brick.minValue * gDensityScale);
#else
    // The density atlas is core-only (kCloudDensityEdge), as the compact copy.
    const uint mask = (1u << gAtlasShift) - 1u;
    const uint3 slotCoord = uint3(brick.slot & mask, (brick.slot >> gAtlasShift) & mask, brick.slot >> (2u * gAtlasShift));
    o.densityBase = float4(float3(slotCoord * kCloudDensityEdge) + 0.5f, brick.minValue * gDensityScale);
#endif
    o.brick = float2((brick.info & kCloudBrickConstant) != 0u ? 0.f : brick.range * gDensityScale, step);
    if (b.x != kCloudRefNone)
    {
        const float sunInverse = 1.f / float(1u << sunLevel);
        o.sun = float4(0.5f * sunInverse - 0.5f - float3(((first >> int(sunLevel)) >> 3) * 8), sunInverse);
        o.sunBase = float4(atlasTexel(b.x), gSun8 != 0u ? instance.scale : instance.sourceVoxelWorld);
    }
    return o;
}

/// cloudSunDecode (HSTRCloud.cs.slang): the 8-bit sun code back to the asset's world units.
float rasterSunDecode(float code)
{
    return 0.5f * (exp(code * 3.6636f) - 1.f);
}

/// One sample: density from the atlas, its opacity over the slice spacing along the pixel's ray, its single sun from the bake.
float4 rasterPS(RasterVertex v) : SV_Target
{
#if RASTER_COUNT
    const uint lanes = WaveActiveCountBits(true);
    if (WaveIsFirstLane())
    {
        uint unused;
        InterlockedAdd(gCounters[kRasterFragments], lanes, unused);
    }
#endif
    float sigma = v.densityBase.w;
    if (v.brick.x > 0.f)
    {
#if (RASTER_FETCH & 1)
        const float3 local = clamp(v.x * v.density.w + v.density.xyz, 0.f, 7.f);
#if (RASTER_FETCH & 4)
        const float t = gAtlasCompact.SampleLevel(gLinearClamp, (v.densityBase.xyz + local) * gCompactInvSize, 0.f);
#else
        const float t = gAtlas.SampleLevel(gLinearClamp, (v.densityBase.xyz + local) * gAtlasInvSize, 0.f);
#endif
#else
        // rasterProbeFetch false or rasterProbeFetchMask without 1 (cost split only): no atlas read.
        const float t = frac(dot(v.x, float3(0.13f, 0.17f, 0.19f)));
#endif
        sigma += v.brick.x * t * t;
    }
    const float2 ndc = float2(2.f, -2.f) * (v.position.xy / float2(gTargetDim)) + float2(-1.f, 1.f);
    const float3 direction = normalize(ndc.x * gCameraU + ndc.y * gCameraV + gCameraW);
    const float alpha = 1.f - exp(-sigma * v.brick.y / max(dot(direction, gForward), 1e-3f));
    if (!(alpha > 1e-6f))
        discard;
    float depth = 0.f;
    if ((RASTER_FETCH & 2) != 0 && v.sunBase.w > 0.f)
    {
        const float3 local = clamp(v.x * v.sun.w + v.sun.xyz, -0.5f, 7.5f);
        const float stored = gSunAtlas.SampleLevel(gLinearClamp, (v.sunBase.xyz + local) * gSunAtlasInvSize, 0.f);
        depth = gSun8 != 0u ? rasterSunDecode(stored) * v.sunBase.w : stored * v.sunBase.w;
    }
    return float4(alpha * gSunColour * exp(-depth), alpha);
}
