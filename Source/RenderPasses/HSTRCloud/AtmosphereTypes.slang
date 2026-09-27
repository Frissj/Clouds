#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/// pl-sky's plAtmosphereSettings (github.com/hoffstadt/pl-sky, shaders/sky_interop.inc), as float4s so the host and shader
/// layouts agree by construction. Kilometres; the planet is centred at the origin with +y up (pl-sky's frame).
///
/// Differences from pl-sky, all of which follow from where the LUTs are used, not from the model:
/// - sunDirection points towards the sun (HSTRCloud's convention); pl-sky stores the direction the light travels.
/// - The aerial perspective LUT is indexed by direction (the sky-view LUT's mapping) rather than by the camera frustum: the cloud
///   sea is resolved from a view-direction (octahedral) beam image, not the screen, and it has no shadow map.
/// - worldToKm and groundY place the scene: a world point is at altitude (y - groundY) * worldToKm km.
struct AtmosphereParams
{
    float3 scatteringRayleighGround = float3(0.0058f, 0.0135f, 0.0331f); ///< Per km, at the ground.
    float planetRadius = 6371.f;
    float3 extinctionRayleighGround = float3(0.0058f, 0.0135f, 0.0331f);
    float atmosphereHeight = 100.f;
    float3 ozoneExtinction = float3(0.00065f, 0.00188f, 0.00008f);
    float scatteringMieGround = 0.006f;
    float extinctionMieGround = 0.00666f;
    float mieScatteringExponent = 0.76f; ///< The Mie phase function's asymmetry g.
    float viewRadius = 6371.f;           ///< Distance from the planet centre that the sky-view and aerial LUTs are built for.
    uint multiScatter = 1;
    float3 sunDirection = float3(0.f, 1.f, 0.f); ///< Towards the sun.
    float worldToKm = 0.001f;
    float3 sunRadiance = float3(3.f, 2.85f, 2.55f); ///< Outside the atmosphere: pl-sky's sun colour times its intensity.
    float sunAngularRadius = 0.00465f;
    float aerialDistance = 10.f; ///< Aerial LUT's farthest distance (km).
    float aerialDepthExponent = 2.f;
    float aerialSamplesPerSlice = 8.f;
    float groundY = 0.f; ///< World y of the ground.
};

END_NAMESPACE_FALCOR
