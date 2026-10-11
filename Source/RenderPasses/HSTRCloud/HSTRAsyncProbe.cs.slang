/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

/// asyncProbe (DIAGNOSTIC, off by default): a fixed amount of independent ALU work, dispatched while the dirty query runs, to
/// measure whether a second command queue overlaps the query on this GPU (HSTRCloud.cpp, mAsyncProbe). It reads nothing the
/// renderer writes; gAsyncProbeOut counts the threads that ran (the arm's proof that it ran) and keeps the chain from being dead.

RWStructuredBuffer<uint> gAsyncProbeOut;

cbuffer AsyncProbeCB
{
    uint gIters; ///< Dependent multiply-adds a thread.
};

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    float x = 1.f + float(dispatchThreadId.x) * 1e-7f;
    [loop]
    for (uint i = 0; i < gIters; ++i)
        x = x * 0.99999f + 1e-5f;
    const uint ran = WaveActiveCountBits(true);
    if (WaveIsFirstLane())
    {
        uint unused;
        InterlockedAdd(gAsyncProbeOut[0], ran, unused);
        if (x == 1234.5f)
            gAsyncProbeOut[1] = 1u;
    }
}
