#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct HSTRCloudParams
{
    uint2 frameDim = {0, 0};
    uint baseSteps = 32;
    uint refinementLevel = 1;

    uint3 hstrLeafDims = {0, 0, 0};
    uint hstrCellWidth = 1;

    int3 hstrGridMin = {0, 0, 0};
    uint frameIndex = 0;

    uint activeRank = 24;
    float activeThreshold = 0.f;
    uint goalFace = 1;
    uint traceSpatialOrder = 2;

    uint hstrFaceDofs = 4;
    uint hstrTraceDofs = 24;
    uint hstrStorageRank = 24;
    uint padHSTR1 = 0;

    float residualBlend = 1.f;
    float densityScale = 1.f;
    float padHSTR2 = 0.f;
    float padHSTR3 = 0.f;

    float3 sunDirection = float3(0.4319f, 0.8639f, 0.2699f);
    float anisotropy = 0.85f;

    float3 sunRadiance = float3(12.f, 10.5f, 8.f);
    float pad0 = 0.f;

    float3 skyRadiance = float3(0.25f, 0.4f, 0.8f);
    float pad1 = 0.f;
};

END_NAMESPACE_FALCOR
