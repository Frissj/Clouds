/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

/** pl-sky's shaders/sky.inc (github.com/hoffstadt/pl-sky) translated from GLSL to Slang, function for function: Hillaire, "A Scalable
    and Production Ready Sky and Atmosphere Rendering Technique" (EGSR 2020). Shared by the LUT kernels (Atmosphere.cs.slang) and
    the renderer, which samples the LUTs. The unused pl-sky helpers (Henyey-Greenstein, Schlick) are left out.
*/
__exported import AtmosphereTypes;

static const float PL_PI = 3.14159265358979f;

// Converts a texel-center UV into a unit UV whose first and last texel centers represent exactly 0 and 1.
float skyLutSubUvToUnit(float subUv, float resolution)
{
    if (resolution <= 1.f)
        return 0.f;
    return saturate((subUv * resolution - 0.5f) / (resolution - 1.f));
}

// Converts a unit UV back into the valid texel-center range.
float skyLutUnitToSubUv(float unitUv, float resolution)
{
    if (resolution <= 1.f)
        return 0.5f;
    return (saturate(unitUv) * (resolution - 1.f) + 0.5f) / resolution;
}

// The sky-view LUT's direction for a texel. viewHeight is the camera's distance from the planet center. The upper half of the LUT
// runs from the zenith to the geometric horizon and the lower half from the horizon to the nadir, both squeezed towards the horizon.
float3 fromSkyLut(float2 subUv, uint2 lutSize, float viewHeight, float planetRadius)
{
    const float2 uv = float2(skyLutSubUvToUnit(subUv.x, float(lutSize.x)), skyLutSubUvToUnit(subUv.y, float(lutSize.y)));
    viewHeight = max(viewHeight, planetRadius);
    // Geometry of the tangent ray from the camera to the planet: beta is the angle between it and the inward radial direction.
    const float horizonDistance = sqrt(max(viewHeight * viewHeight - planetRadius * planetRadius, 0.f));
    const float cosBeta = saturate(horizonDistance / max(viewHeight, 1e-6f));
    const float beta = acos(cosBeta);
    const float zenithToHorizonAngle = PL_PI - beta;
    float viewZenithAngle;
    if (uv.y < 0.5f)
    {
        float coord = uv.y * 2.f;
        coord = 1.f - coord;
        coord *= coord;
        coord = 1.f - coord;
        viewZenithAngle = zenithToHorizonAngle * coord;
    }
    else
    {
        float coord = uv.y * 2.f - 1.f;
        coord *= coord;
        viewZenithAngle = zenithToHorizonAngle + beta * coord;
    }
    const float phi = (0.5f - uv.x) * 2.f * PL_PI;
    const float sinZenith = sin(viewZenithAngle);
    const float cosZenith = cos(viewZenithAngle);
    return float3(sinZenith * cos(phi), cosZenith, sinZenith * sin(phi));
}

float pl_phase_cornette_shanks(float VoL, float g)
{
    const float nominator = 3.f / (8.f * PL_PI) * (1.f - g * g) * (1.f + VoL * VoL);
    const float denominator = (2.f + g * g) * pow(1.f + g * g - 2.f * g * VoL, 1.5f);
    return nominator / denominator;
}

float pl_phase_rayleigh(float VoL) { return 3.f / (16.f * PL_PI) * (1.f + VoL * VoL); }

struct plIntersection
{
    float3 tPos;
    float fDistance;
    bool bHitEarth;
};

plIntersection pl_ray_intersection_planet(float3 P, float3 D, float3 C, float planetRadius, float atmosphere)
{
    const float3 L = C - P;
    const float t_ca = dot(L, D);
    const float d = sqrt(max(dot(L, L) - t_ca * t_ca, 0.f));
    // Planet intersection, from outside. pl-sky takes the root unguarded; a miss is NaN there, which the comparison below reads as
    // no hit, as it does here.
    const float t_hc_planet = sqrt(planetRadius * planetRadius - d * d);
    const float t_planet = t_ca - t_hc_planet;
    // Atmosphere intersection.
    const float r = planetRadius + atmosphere;
    const float t_hc_atmosphere = sqrt(max(r * r - d * d, 0.f));
    const float t_atmosphere = t_ca + abs(t_hc_atmosphere);
    // If the ray meets the planet in front, the view is blocked by it.
    plIntersection tResult;
    tResult.bHitEarth = t_planet >= 0.f;
    const float t = tResult.bHitEarth ? t_planet : t_atmosphere;
    tResult.fDistance = t;
    tResult.tPos = P + t * D;
    return tResult;
}

// The nearest nonnegative ray/sphere intersection, or -1 when the ray misses the sphere.
float pl_ray_intersect_sphere_nearest(float3 tOrigin, float3 tDirection, float3 tCenter, float fRadius)
{
    const float3 tRelativeOrigin = tOrigin - tCenter;
    const float a = dot(tDirection, tDirection);
    const float b = 2.f * dot(tDirection, tRelativeOrigin);
    const float c = dot(tRelativeOrigin, tRelativeOrigin) - fRadius * fRadius;
    const float fDiscriminant = b * b - 4.f * a * c;
    if (fDiscriminant < 0.f || a <= 0.f)
        return -1.f;
    const float fRoot = sqrt(max(fDiscriminant, 0.f));
    const float fInverseDenominator = 0.5f / a;
    const float t0 = (-b - fRoot) * fInverseDenominator;
    const float t1 = (-b + fRoot) * fInverseDenominator;
    if (t0 < 0.f && t1 < 0.f)
        return -1.f;
    if (t0 < 0.f)
        return max(t1, 0.f);
    if (t1 < 0.f)
        return max(t0, 0.f);
    return max(min(t0, t1), 0.f);
}

float3 pl_safe_normalize(float3 tValue) { return tValue * rsqrt(max(dot(tValue, tValue), 1e-20f)); }

// 1 when the sun is visible and 0 when the planet blocks it.
float pl_planet_visibility(float3 tPosition, float3 tDirectionToSun, float3 tEarthCenter, float fEarthRadius)
{
    const float3 tRadialDirection = pl_safe_normalize(tPosition - tEarthCenter);
    // Scale-aware offset: roughly sub-metre to metre for an Earth-sized planet.
    const float fSurfaceEpsilon = max(fEarthRadius * 1e-7f, 1e-6f);
    const float3 tBiasedPosition = tPosition + tRadialDirection * fSurfaceEpsilon;
    const float bEarthHit = pl_ray_intersect_sphere_nearest(tBiasedPosition, tDirectionToSun, tEarthCenter, fEarthRadius);
    return bEarthHit >= 0.f ? 0.f : 1.f;
}

float pl_final_ray_sphere_nearest(float3 rayOrigin, float3 rayDirection, float sphereRadius)
{
    const float b = dot(rayOrigin, rayDirection);
    const float c = dot(rayOrigin, rayOrigin) - sphereRadius * sphereRadius;
    const float discriminant = b * b - c;
    if (discriminant < 0.f)
        return -1.f;
    const float root = sqrt(max(discriminant, 0.f));
    const float t0 = -b - root;
    const float t1 = -b + root;
    if (t0 >= 0.f)
        return t0;
    if (t1 >= 0.f)
        return t1;
    return -1.f;
}

// Keeps bilinear samples inside actual texel centers instead of filtering against an exact LUT boundary.
float2 pl_clamp_lut_uv(float2 tUv, uint2 tLutSize)
{
    const float2 tHalfTexel = 0.5f / float2(tLutSize);
    return clamp(tUv, tHalfTexel, 1.f - tHalfTexel);
}

struct plAtmosphereCoefficients
{
    float3 tScatterRayleigh;
    float3 tScatterMie;
    float3 tExtinction;
};

float pl_height_factor_rayleigh(float fHeight) { return exp(-fHeight * (1.f / 8.f)); }
float pl_height_factor_mie(float fHeight) { return exp(-fHeight * (1.f / 1.2f)); }
float pl_height_factor_ozone(float fHeight) { return max(0.f, 1.f - abs(fHeight - 25.f) / 15.f); }

plAtmosphereCoefficients pl_calculate_coefficients(float fHeight, AtmosphereParams tAtmosphere)
{
    const float fRayleighFactor = pl_height_factor_rayleigh(fHeight);
    const float fMieFactor = pl_height_factor_mie(fHeight);
    const float fOzoneFactor = pl_height_factor_ozone(fHeight);
    plAtmosphereCoefficients coefficients;
    coefficients.tScatterRayleigh = fRayleighFactor * tAtmosphere.scatteringRayleighGround;
    coefficients.tScatterMie = float3(fMieFactor * tAtmosphere.scatteringMieGround);
    coefficients.tExtinction = fRayleighFactor * tAtmosphere.extinctionRayleighGround + fMieFactor * tAtmosphere.extinctionMieGround +
                               fOzoneFactor * tAtmosphere.ozoneExtinction;
    return coefficients;
}

/// Transmittance and multiple-scattering LUT coordinates: x the height in the atmosphere, y the cosine to the local zenith.
/// pl-sky samples these unit coordinates directly, half a texel off the texels its kernels write (texel i holds i / (size - 1)):
/// pl_lut_uv maps them onto the texel centres.
float2 pl_compute_lut_uv(float height, float atmoshphereHeight, float3 up, float3 direction)
{
    float2 lutUV;
    lutUV.x = height / atmoshphereHeight;
    lutUV.y = dot(up, direction) * 0.5f + 0.5f;
    return lutUV;
}

float2 pl_lut_uv(float2 unitUv, uint2 lutSize)
{
    return float2(skyLutUnitToSubUv(unitUv.x, float(lutSize.x)), skyLutUnitToSubUv(unitUv.y, float(lutSize.y)));
}

// Reference: "Physically Based and Unified Volumetric Rendering in Frostbite", page 29.
float3 pl_integrate_inscattering(float3 inscattering, float3 extinctionCoefficients, float length)
{
    return (inscattering - inscattering * exp(-extinctionCoefficients * length)) / max(extinctionCoefficients, 0.00001f);
}

// Analytically integrates a constant source over one homogeneous segment, integral S exp(-sigma_t x) dx; the linear fallback avoids
// division problems for nearly zero extinction (sky_lut.comp's pl__integrate_source, sky_aerial_lut.comp's pl_integrate_source).
float3 pl_integrate_source(float3 source, float3 extinction, float segmentLength)
{
    const float3 segmentTransmittance = exp(-extinction * segmentLength);
    const float3 safeExtinction = max(extinction, 1e-7f);
    const float3 analytic = source * (1.f - segmentTransmittance) / safeExtinction;
    const float3 linear = source * segmentLength;
    return select(abs(extinction) < 1e-7f, linear, analytic);
}

float2 pl_to_sky_lut(float3 direction, uint2 lutSize, float viewHeight, float planetRadius)
{
    direction = normalize(direction);
    viewHeight = max(viewHeight, planetRadius);
    const float horizonDistance = sqrt(max(viewHeight * viewHeight - planetRadius * planetRadius, 0.f));
    const float cosBeta = saturate(horizonDistance / max(viewHeight, 1e-6f));
    const float beta = acos(cosBeta);
    // Zenith angle at which the geometric horizon occurs.
    const float zenithToHorizonAngle = PL_PI - beta;
    const float viewZenithAngle = acos(clamp(direction.y, -1.f, 1.f));
    float2 uv;
    if (viewZenithAngle <= zenithToHorizonAngle)
    {
        // Zenith to geometric horizon.
        float coord = viewZenithAngle / max(zenithToHorizonAngle, 1e-6f);
        coord = 1.f - coord;
        coord = sqrt(max(coord, 0.f));
        coord = 1.f - coord;
        uv.y = coord * 0.5f;
    }
    else
    {
        // Geometric horizon to nadir.
        float coord = (viewZenithAngle - zenithToHorizonAngle) / max(beta, 1e-6f);
        coord = sqrt(saturate(coord));
        uv.y = 0.5f + coord * 0.5f;
    }
    const float phi = atan2(direction.z, direction.x);
    uv.x = frac(0.5f - phi / (2.f * PL_PI));
    return float2(skyLutUnitToSubUv(uv.x, float(lutSize.x)), skyLutUnitToSubUv(uv.y, float(lutSize.y)));
}

// sky.frag's pl_sun_disk, with the edge a fixed width in cosine: fwidth has no meaning in a compute kernel.
float pl_sun_disk(float3 viewDirection, float3 sunDirection, float angularRadius)
{
    const float viewSunCos = dot(normalize(viewDirection), normalize(sunDirection));
    const float radiusCos = cos(angularRadius);
    const float edgeWidth = max(0.02f * (1.f - radiusCos), 1e-7f);
    return smoothstep(radiusCos - edgeWidth, radiusCos + edgeWidth, viewSunCos);
}

/// Aerial LUT depth of a slice boundary (sky_aerial_lut.comp's pl_aerial_depth).
float pl_aerial_depth(float sliceBoundary, float sliceCount, float maxDistance, float depthExponent)
{
    const float u = sliceBoundary / sliceCount;
    return maxDistance * pow(u, depthExponent);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Lookups, as pl-sky's sky.frag and postprocess.comp make them.

/// The camera's distance from the planet centre for a world position.
float atmosphereViewHeight(AtmosphereParams a, float worldY) { return a.planetRadius + max((worldY - a.groundY) * a.worldToKm, 0.f); }

/// Sky radiance along a view direction: the sky-view LUT (built for a.viewRadius) and the sun disk, attenuated by the transmittance
/// to the top of the atmosphere (sky.frag; pl-sky has that attenuation commented out, which leaves a white sun at the horizon).
float3 atmosphereSky(AtmosphereParams a, Texture2D<float4> skyLut, Texture2D<float4> transmissionLut, SamplerState s, float3 direction)
{
    uint2 skySize;
    skyLut.GetDimensions(skySize.x, skySize.y);
    float3 color = skyLut.SampleLevel(s, pl_to_sky_lut(direction, skySize, a.viewRadius, a.planetRadius), 0.f).rgb;
    float disk = pl_sun_disk(direction, a.sunDirection, a.sunAngularRadius);
    const float3 camera = float3(0.f, a.viewRadius, 0.f);
    if (pl_final_ray_sphere_nearest(camera, normalize(direction), a.planetRadius) >= 0.f)
        disk = 0.f;
    if (disk > 0.f)
    {
        uint2 size;
        transmissionLut.GetDimensions(size.x, size.y);
        const float altitude = clamp(a.viewRadius - a.planetRadius, 0.f, a.atmosphereHeight);
        const float2 uv = pl_lut_uv(pl_compute_lut_uv(altitude, a.atmosphereHeight, float3(0.f, 1.f, 0.f), normalize(direction)), size);
        color += disk * a.sunRadiance * transmissionLut.SampleLevel(s, uv, 0.f).rgb;
    }
    return color;
}

/// Aerial perspective over a world distance along a direction: rgb the in-scattered radiance, a the transmittance
/// (postprocess.comp's lookup, including its first-slice blend towards no atmosphere).
float4 atmosphereAerial(AtmosphereParams a, Texture3D<float4> aerialLut, SamplerState s, float3 direction, float worldDistance)
{
    uint3 size;
    aerialLut.GetDimensions(size.x, size.y, size.z);
    const float normalizedDistance = saturate(worldDistance * a.worldToKm / a.aerialDistance);
    const float boundaryU = pow(normalizedDistance, 1.f / a.aerialDepthExponent);
    const float aerialZ = clamp(boundaryU - 0.5f / float(size.z), 0.5f / float(size.z), 1.f - 0.5f / float(size.z));
    const float2 xy = pl_to_sky_lut(direction, size.xy, a.viewRadius, a.planetRadius);
    const float4 aerial = aerialLut.SampleLevel(s, float3(xy, aerialZ), 0.f);
    const float firstSliceBlend = saturate(boundaryU * float(size.z));
    return lerp(float4(0.f, 0.f, 0.f, 1.f), aerial, firstSliceBlend);
}
