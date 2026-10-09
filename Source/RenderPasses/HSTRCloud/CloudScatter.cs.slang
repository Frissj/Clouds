/// CloudResidency's table uploads: the elements the frame changed, packed by the CPU (indices, then their words) and written into the
/// GPU tables here. Uploading the dirty 4096-brick blocks as runs re-sent whole tables: a walk changes a few thousand bricks a frame,
/// scattered over every block (upload1: ~11 MB in ~36 copies a frame, the residency upload scope 1.42 ms).
import HSTRCloudTypes;

cbuffer CB
{
    uint gCount;     ///< Elements to write.
    uint gWords;     ///< Words per element.
    uint gIndexBase; ///< First of them in gIndex.
    uint gWordBase;  ///< Their first word in gData.
};

StructuredBuffer<uint> gIndex;
StructuredBuffer<uint> gData;
RWStructuredBuffer<uint> gUints;
RWStructuredBuffer<HSTRCloudBrick> gBricks;
RWStructuredBuffer<HSTRCloudSunSched> gSched;

/// Tables of words (page nodes of gWords entries, sun states): a thread per word.
[numthreads(256, 1, 1)]
void scatterUints(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= gCount * gWords)
        return;
    gUints[gIndex[gIndexBase + id.x / gWords] * gWords + id.x % gWords] = gData[gWordBase + id.x];
}

[numthreads(256, 1, 1)]
void scatterBricks(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= gCount)
        return;
    const uint w = gWordBase + id.x * 8;
    HSTRCloudBrick b;
    b.slot = gData[w];
    b.info = gData[w + 1];
    b.parent = gData[w + 2];
    b.fade = asfloat(gData[w + 3]);
    b.minValue = asfloat(gData[w + 4]);
    b.range = asfloat(gData[w + 5]);
    b.fadeStart = gData[w + 6];
    b.fadeBase = asfloat(gData[w + 7]);
    gBricks[gIndex[gIndexBase + id.x]] = b;
}

[numthreads(256, 1, 1)]
void scatterSunSched(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= gCount)
        return;
    const uint w = gWordBase + id.x * 2;
    HSTRCloudSunSched s;
    s.coord = gData[w];
    s.asset = gData[w + 1];
    gSched[gIndex[gIndexBase + id.x]] = s;
}
