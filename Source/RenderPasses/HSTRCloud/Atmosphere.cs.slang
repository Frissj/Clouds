/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

/** pl-sky's LUT kernels (github.com/hoffstadt/pl-sky, shaders/sky_transmission_lut.comp, sky_multiscatter_lut.comp, sky_lut.comp
    and sky_aerial_lut.comp) translated from GLSL to Slang, plus the cloud lighting the renderer takes from them (cloudLighting).
    Where this differs from pl-sky, a comment says so and why.
*/
import Atmosphere;

cbuffer CB
{
    AtmosphereParams gAtmosphere;
    float gLightingRadius; ///< cloudLighting: distance from the planet centre of the clouds it lights.
}

SamplerState gLinearClamp;
RWTexture2D<float4> gOutput2D;
RWTexture3D<float4> gOutput3D;
Texture2D<float4> gTransmissionLut;
Texture2D<float4> gMultiscatterLut;
Texture2D<float4> gSkyLut;
RWStructuredBuffer<float4> gLighting;

// ---------------------------------------------------------------------------------------------------------------------------------
// sky_transmission_lut.comp: if I start at altitude h and look in direction V, how much light survives travelling through the
// atmosphere before it exits or hits the ground.

static const int kTransmissionSampleCount = 40;

[numthreads(8, 8, 1)]
void transmissionLut(uint3 id: SV_DispatchThreadID)
{
    uint2 size;
    gOutput2D.GetDimensions(size.x, size.y);
    const uint2 uv = id.xy;
    if (any(uv >= size))
        return;
    const float x = uv.x / float(size.x - 1); // height in atmosphere
    const float y = uv.y / float(size.y - 1); // direction relative to "up"
    const float fHeight = lerp(0.f, gAtmosphere.atmosphereHeight, x);
    float fUpDot = y * 2.f - 1.f;
    fUpDot = max(fUpDot, -0.999f);
    const float3 V = float3(0.f, fUpDot, sqrt(1.f - fUpDot * fUpDot)); // assuming spherical symmetry
    const float3 P = float3(0.f, fHeight + gAtmosphere.planetRadius, 0.f);
    const float3 tEarthCenter = 0.f;
    const float3 localUp = normalize(P - tEarthCenter);
    const plIntersection tIntersection =
        pl_ray_intersection_planet(P + localUp * 0.01f, V, tEarthCenter, gAtmosphere.planetRadius, gAtmosphere.atmosphereHeight);
    const float fPathLength = max(distance(tIntersection.tPos, P), 0.01f);
    const float fStepLength = fPathLength / kTransmissionSampleCount;
    float3 tCurrentPos = P;
    float3 tTransmittance = 1.f;
    const float3 tStep = V * fStepLength;
    for (int i = 0; i < kTransmissionSampleCount; i++)
    {
        tCurrentPos += tStep;
        const float fCurrentHeight = max(distance(tEarthCenter, tCurrentPos) - gAtmosphere.planetRadius, 0.f);
        const plAtmosphereCoefficients tCoefficients = pl_calculate_coefficients(fCurrentHeight, gAtmosphere);
        tTransmittance *= exp(-tCoefficients.tExtinction * fStepLength); // Beer's law
    }
    tTransmittance = tIntersection.bHitEarth ? 0.f : tTransmittance;
    gOutput2D[uv] = float4(tTransmittance, 0.f);
}

float3 sampleTransmission(float2 unitUv)
{
    uint2 size;
    gTransmissionLut.GetDimensions(size.x, size.y);
    return gTransmissionLut.SampleLevel(gLinearClamp, pl_lut_uv(unitUv, size), 0.f).rgb;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// sky_multiscatter_lut.comp: at this altitude, with the sun at this zenith angle, how much radiance comes from second- and
// higher-order scattering.
//
// Differences from pl-sky, which are bugs there:
// - Its sphere directions are (sin t cos p, sin t, sin t sin p): not unit, and never below the horizon. Here (.., cos t, ..).
// - Its polar angles start at the pole (i / 8 of pi), whose sin t weight is 0; here at the cell centres ((i + 0.5) / 8).
// - It divides the sin t weighted sums by the sample count, which is the sphere mean times 2 / pi; here by the weights' sum.
// - Its texels hold x = i / size, while its readers take texel i as i / (size - 1); here i / (size - 1), as the transmission LUT.

static const int kMultiscatterInnerSampleCount = 20;

[numthreads(8, 8, 1)]
void multiscatterLut(uint3 id: SV_DispatchThreadID)
{
    uint2 size;
    gOutput2D.GetDimensions(size.x, size.y);
    const uint2 uv = id.xy;
    if (any(uv >= size))
        return;
    const float x = uv.x / float(size.x - 1); // altitude
    const float y = uv.y / float(size.y - 1); // solar zenith angle
    const float fHeight = lerp(0.f, gAtmosphere.atmosphereHeight, x);
    const float3 P = float3(0.f, fHeight + gAtmosphere.planetRadius, 0.f);
    const float3 tEarthCenter = 0.f;
    float fUpDot = y * 2.f - 1.f;
    fUpDot = max(fUpDot, -0.999f);
    const float3 L = float3(0.f, fUpDot, sqrt(1.f - fUpDot * fUpDot)); // direction towards sun

    float3 L_2nd = 0.f; // radiance available from second scattering order
    float3 f_ms = 0.f;  // fraction of light that may be scattered again
    float weightSum = 0.f;
    const float fIsotropicPhase = 1.f / (4.f * PL_PI);
    const int iSampleCountSqrt = 8;
    const float fSampleCountSqrtRcp = 1.f / float(iSampleCountSqrt);
    for (int i = 0; i < iSampleCountSqrt; i++)
    {
        for (int j = 0; j < iSampleCountSqrt; j++)
        {
            const float fTheta = PL_PI * (float(i) + 0.5f) * fSampleCountSqrtRcp;
            const float fPhi = 2.f * PL_PI * (float(j) + 0.5f) * fSampleCountSqrtRcp;
            const float fSinTheta = sin(fTheta);
            const float fCosTheta = cos(fTheta);
            float3 V = float3(fSinTheta * cos(fPhi), fCosTheta, fSinTheta * sin(fPhi));

            float3 inscattered = 0.f;
            const plIntersection intersection =
                pl_ray_intersection_planet(P, V, tEarthCenter, gAtmosphere.planetRadius, gAtmosphere.atmosphereHeight);
            float3 tCurrentPosition = P;
            const float fStepSize = intersection.fDistance / kMultiscatterInnerSampleCount;
            V *= fStepSize;
            float3 L_f = 0.f;

            // Light reflected from the planet surface if the sampled direction hits the ground.
            const float3 tEarthAlbedo = 0.3f;
            const float3 tEarthHitNormal = normalize(intersection.tPos - tEarthCenter);
            const float fEarthNoL = saturate(dot(tEarthHitNormal, L));
            const float3 transmissionToIntersection =
                sampleTransmission(pl_compute_lut_uv(0.f, gAtmosphere.atmosphereHeight, tEarthHitNormal, L));
            const float3 planetLit = tEarthAlbedo / PL_PI * transmissionToIntersection * fEarthNoL;
            const float3 direct = intersection.bHitEarth ? planetLit : 0.f;

            float3 transmission = 1.f;
            for (int k = 0; k < kMultiscatterInnerSampleCount; k++)
            {
                tCurrentPosition += V;
                float3 tUp = tCurrentPosition - tEarthCenter;
                const float tUpLength = length(tUp);
                const float currentHeight = tUpLength - gAtmosphere.planetRadius;
                tUp /= tUpLength;
                const plAtmosphereCoefficients tCoefficients = pl_calculate_coefficients(currentHeight, gAtmosphere);
                const float3 tScatteringCo = tCoefficients.tScatterRayleigh + tCoefficients.tScatterMie;
                // Light arriving at the point.
                const float3 transmissionSun = sampleTransmission(pl_compute_lut_uv(currentHeight, gAtmosphere.atmosphereHeight, tUp, L));
                // In-scattering from the light at the point; the integral is linear in the source, so it is reused.
                const float3 coefficientIntegral = pl_integrate_inscattering(tScatteringCo, tCoefficients.tExtinction, fStepSize);
                L_f += coefficientIntegral * transmission;
                const float3 scatterIntegral = coefficientIntegral * transmissionSun * fIsotropicPhase;
                inscattered = inscattered + scatterIntegral * transmission;
                transmission *= exp(-tCoefficients.tExtinction * fStepSize);
            }
            f_ms += L_f * fSinTheta;
            L_2nd += (direct * transmission + inscattered) * fSinTheta;
            weightSum += fSinTheta;
        }
    }
    f_ms /= weightSum;
    L_2nd /= weightSum;
    const float3 F_ms = 1.f / (1.f - f_ms);
    gOutput2D[uv] = float4(L_2nd * F_ms, 0.f);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// sky_lut.comp: the sky's radiance by direction, for a camera at gAtmosphere.viewRadius from the planet centre (pl-sky derives it
// from the camera's y; here the host sets it, so the same kernel also builds the sky seen from the clouds for cloudLighting).

static const float kSkyViewMinSampleCount = 32.f;
static const int kSkyViewMaxSampleCount = 64;
static const float kSkyViewSampleSegmentT = 0.3f; // position inside each ray-march interval; Hillaire's implementation uses 0.3

[numthreads(8, 8, 1)]
void skyViewLut(uint3 id: SV_DispatchThreadID)
{
    uint2 tOutputSize;
    gOutput2D.GetDimensions(tOutputSize.x, tOutputSize.y);
    const uint2 tPixel = id.xy;
    if (any(tPixel >= tOutputSize))
        return;
    const float2 tSkyUV = (float2(tPixel) + 0.5f) / float2(tOutputSize);
    const float fCameraSurfaceOffset = max(gAtmosphere.planetRadius * 1e-7f, 1e-6f);
    const float fViewHeight = max(gAtmosphere.viewRadius, gAtmosphere.planetRadius) + fCameraSurfaceOffset;
    const float3 rayOrigin = float3(0.f, fViewHeight, 0.f);
    const float3 viewDirection = fromSkyLut(tSkyUV, tOutputSize, fViewHeight, gAtmosphere.planetRadius);
    const float3 planetCenter = 0.f;
    const plIntersection intersection =
        pl_ray_intersection_planet(rayOrigin, viewDirection, planetCenter, gAtmosphere.planetRadius, gAtmosphere.atmosphereHeight);
    const float fRayLength = intersection.fDistance;
    if (!(fRayLength > 0.f))
    {
        gOutput2D[tPixel] = float4(0.f, 0.f, 0.f, 1.f);
        return;
    }
    const float3 tDirectionToSun = pl_safe_normalize(gAtmosphere.sunDirection);
    const float fViewDotLight = clamp(dot(viewDirection, tDirectionToSun), -1.f, 1.f);
    const float fRayleighPhaseValue = pl_phase_rayleigh(fViewDotLight);
    const float fMiePhaseValue = pl_phase_cornette_shanks(fViewDotLight, gAtmosphere.mieScatteringExponent);
    const float3 tSunRadiance = gAtmosphere.sunRadiance;
    float3 tAccumulatedTransmittance = 1.f;
    float3 tAccumulatedLuminance = 0.f;

    // More samples as the ray becomes long relative to the atmosphere's thickness.
    const float fSampleCountFactor = saturate(fRayLength / max(2.f * gAtmosphere.atmosphereHeight, 1e-6f));
    const float fSampleCount = lerp(kSkyViewMinSampleCount, float(kSkyViewMaxSampleCount), fSampleCountFactor);
    // The partial final segment avoids a jump whenever the fractional sample count crosses an integer.
    const float fSampleCountFloor = max(floor(fSampleCount), 1.f);
    const float fCompleteSegmentsRayLength = fRayLength * fSampleCountFloor / fSampleCount;
    uint2 tTransmissionLutSize, tMultiscatterLutSize;
    gTransmissionLut.GetDimensions(tTransmissionLutSize.x, tTransmissionLutSize.y);
    gMultiscatterLut.GetDimensions(tMultiscatterLutSize.x, tMultiscatterLutSize.y);

    for (int sampleIndex = 0; sampleIndex < kSkyViewMaxSampleCount; sampleIndex++)
    {
        const float fSampleNumber = float(sampleIndex);
        if (fSampleNumber >= fSampleCount)
            break;
        float fInterval0 = fSampleNumber / fSampleCountFloor;
        float fInterval1 = (fSampleNumber + 1.f) / fSampleCountFloor;
        // The quadratic distribution concentrates samples near the camera.
        fInterval0 *= fInterval0;
        fInterval1 *= fInterval1;
        const float fDistance0 = fCompleteSegmentsRayLength * fInterval0;
        const float fDistance1 = fInterval1 > 1.f ? fRayLength : fCompleteSegmentsRayLength * fInterval1;
        const float fSegmentLength = max(fDistance1 - fDistance0, 0.f);
        if (fSegmentLength <= 0.f)
            continue;
        const float fSampleDistance = lerp(fDistance0, fDistance1, kSkyViewSampleSegmentT);
        const float3 tCurrentPosition = rayOrigin + viewDirection * fSampleDistance;
        const float3 tRadialVector = tCurrentPosition - planetCenter;
        const float fRadialLength = length(tRadialVector);
        const float3 tUpDirection = tRadialVector / max(fRadialLength, 1e-20f);
        const float fCurrentHeight = clamp(fRadialLength - gAtmosphere.planetRadius, 0.f, gAtmosphere.atmosphereHeight);
        const float2 atmosphereUV = pl_compute_lut_uv(fCurrentHeight, gAtmosphere.atmosphereHeight, tUpDirection, tDirectionToSun);
        const float3 transmittanceToSun =
            gTransmissionLut.SampleLevel(gLinearClamp, pl_lut_uv(atmosphereUV, tTransmissionLutSize), 0.f).rgb;
        const float3 multipleScatteredLuminance =
            gMultiscatterLut.SampleLevel(gLinearClamp, pl_lut_uv(atmosphereUV, tMultiscatterLutSize), 0.f).rgb;
        const plAtmosphereCoefficients coefficients = pl_calculate_coefficients(fCurrentHeight, gAtmosphere);
        const float planetVisibility = pl_planet_visibility(tCurrentPosition, tDirectionToSun, planetCenter, gAtmosphere.planetRadius);
        // Direct sunlight scattered towards the camera.
        const float3 phaseTimesScattering =
            coefficients.tScatterRayleigh * fRayleighPhaseValue + coefficients.tScatterMie * fMiePhaseValue;
        const float3 directSource = planetVisibility * transmittanceToSun * phaseTimesScattering;
        // The multiple-scattering LUT is incoming multiply scattered radiance: times the local scattering, not the sun's
        // transmittance or the planet's shadow again.
        const float3 totalScattering = coefficients.tScatterRayleigh + coefficients.tScatterMie;
        const float3 multipleScatteringSource = gAtmosphere.multiScatter != 0 ? multipleScatteredLuminance * totalScattering : 0.f;
        const float3 source = tSunRadiance * (directSource + multipleScatteringSource);
        const float3 segmentLuminance = pl_integrate_source(source, coefficients.tExtinction, fSegmentLength);
        tAccumulatedLuminance += tAccumulatedTransmittance * segmentLuminance;
        tAccumulatedTransmittance *= exp(-coefficients.tExtinction * fSegmentLength);
    }
    gOutput2D[tPixel] = float4(tAccumulatedLuminance, 1.f);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// sky_aerial_lut.comp: the radiance scattered towards the camera and the transmittance over each distance slice.
//
// Differences from pl-sky:
// - x and y are the sky-view LUT's direction mapping (pl_to_sky_lut), not the camera frustum: the cloud sea resolves from a
//   direction-indexed beam image, so its haze has to hold for every direction, and it stays valid when the camera turns.
// - No shadow map term: there is no terrain, and the clouds' shadows on the air are not modelled.
// - The sun's radiance is its colour times intensity, as in the sky-view LUT; pl-sky uses the intensity alone here.
// - The stored transmittance is the channel mean; pl-sky divides the sum by the sun intensity, which is the mean only at 3.

static const int kAerialMaxSamplesPerSlice = 8;

[numthreads(8, 8, 1)]
void aerialLut(uint3 id: SV_DispatchThreadID)
{
    uint3 volumeSize;
    gOutput3D.GetDimensions(volumeSize.x, volumeSize.y, volumeSize.z);
    const uint2 froxelXY = id.xy;
    if (any(froxelXY >= volumeSize.xy))
        return;
    const float cameraSurfaceOffset = max(gAtmosphere.planetRadius * 1e-7f, 1e-6f);
    const float viewHeight = max(gAtmosphere.viewRadius, gAtmosphere.planetRadius) + cameraSurfaceOffset;
    const float3 cameraPosition = float3(0.f, viewHeight, 0.f);
    const float3 viewDirection = fromSkyLut((float2(froxelXY) + 0.5f) / float2(volumeSize.xy), volumeSize.xy, viewHeight,
                                            gAtmosphere.planetRadius);
    const float3 sunDirection = pl_safe_normalize(gAtmosphere.sunDirection);
    const float viewSunCos = clamp(dot(viewDirection, sunDirection), -1.f, 1.f);
    const float rayleighPhase = pl_phase_rayleigh(viewSunCos);
    const float miePhase = pl_phase_cornette_shanks(viewSunCos, gAtmosphere.mieScatteringExponent);
    uint2 transmissionSize, multiscatterSize;
    gTransmissionLut.GetDimensions(transmissionSize.x, transmissionSize.y);
    gMultiscatterLut.GetDimensions(multiscatterSize.x, multiscatterSize.y);
    const float maxDistance = gAtmosphere.aerialDistance;
    const float depthExponent = max(gAtmosphere.aerialDepthExponent, 1.f);
    const int samplesPerSlice = clamp(int(gAtmosphere.aerialSamplesPerSlice), 1, kAerialMaxSamplesPerSlice);
    float3 accumulatedLuminance = 0.f;
    float3 accumulatedTransmittance = 1.f;
    bool belowGround = false;
    for (uint slice = 0; slice < volumeSize.z; slice++)
    {
        const float distance0 = pl_aerial_depth(float(slice), float(volumeSize.z), maxDistance, depthExponent);
        const float distance1 = pl_aerial_depth(float(slice + 1), float(volumeSize.z), maxDistance, depthExponent);
        const float sliceLength = distance1 - distance0;
        for (int sampleIndex = 0; sampleIndex < kAerialMaxSamplesPerSlice && !belowGround; sampleIndex++)
        {
            if (sampleIndex >= samplesPerSlice)
                break;
            const float sampleU = (float(sampleIndex) + 0.5f) / float(samplesPerSlice);
            const float sampleDistance = lerp(distance0, distance1, sampleU);
            const float segmentLength = sliceLength / float(samplesPerSlice);
            const float3 position = cameraPosition + viewDirection * sampleDistance;
            const float radialDistance = length(position);
            const float3 localUp = position / max(radialDistance, 1e-20f);
            float currentHeight = radialDistance - gAtmosphere.planetRadius;
            // Once the ray is below the planet surface, the rest of the aerial volume is occluded.
            if (currentHeight < 0.f)
            {
                belowGround = true;
                break;
            }
            currentHeight = clamp(currentHeight, 0.f, gAtmosphere.atmosphereHeight);
            const plAtmosphereCoefficients coefficients = pl_calculate_coefficients(currentHeight, gAtmosphere);
            const float2 atmosphereUV = pl_compute_lut_uv(currentHeight, gAtmosphere.atmosphereHeight, localUp, sunDirection);
            const float3 transmittanceToSun =
                gTransmissionLut.SampleLevel(gLinearClamp, pl_lut_uv(atmosphereUV, transmissionSize), 0.f).rgb;
            const float3 multipleScatteredLuminance =
                gMultiscatterLut.SampleLevel(gLinearClamp, pl_lut_uv(atmosphereUV, multiscatterSize), 0.f).rgb;
            const float planetVisibility = pl_final_ray_sphere_nearest(position, sunDirection, gAtmosphere.planetRadius) >= 0.f ? 0.f : 1.f;
            const float3 phaseTimesScattering = coefficients.tScatterRayleigh * rayleighPhase + coefficients.tScatterMie * miePhase;
            const float3 directSource = planetVisibility * transmittanceToSun * phaseTimesScattering;
            const float3 totalScattering = coefficients.tScatterRayleigh + coefficients.tScatterMie;
            const float3 multipleSource = gAtmosphere.multiScatter != 0 ? multipleScatteredLuminance * totalScattering : 0.f;
            const float3 source = gAtmosphere.sunRadiance * (directSource + multipleSource);
            const float3 segmentLuminance = pl_integrate_source(source, coefficients.tExtinction, segmentLength);
            accumulatedLuminance += accumulatedTransmittance * segmentLuminance;
            accumulatedTransmittance *= exp(-coefficients.tExtinction * segmentLength);
        }
        gOutput3D[uint3(froxelXY, slice)] = float4(accumulatedLuminance, dot(accumulatedTransmittance, 1.f / 3.f));
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The clouds' lighting from the atmosphere (not in pl-sky, whose terrain takes the sun's colour unattenuated): gLighting[0] the
// sun's radiance at the clouds (outside radiance times the transmittance to the top of the atmosphere), gLighting[1] the mean
// radiance of the sky above their horizon (gSkyLut built for gAtmosphere.viewRadius = gLightingRadius), which lights the renderer's
// uniform-sky term (skyRadiance). One group: 256 threads reduce the LUT's upper half by solid angle.

groupshared float4 gSkySum[256];

[numthreads(256, 1, 1)]
void cloudLighting(uint3 id: SV_GroupThreadID)
{
    uint2 size;
    gSkyLut.GetDimensions(size.x, size.y);
    float4 sum = 0.f;
    for (uint texel = id.x; texel < size.x * size.y; texel += 256)
    {
        const uint2 pixel = uint2(texel % size.x, texel / size.x);
        const float2 uv = (float2(pixel) + 0.5f) / float2(size);
        const float3 direction = fromSkyLut(uv, size, gLightingRadius, gAtmosphere.planetRadius);
        if (direction.y <= 0.f)
            continue;
        // The texel's solid angle: the mapping's zenith angle is not uniform in v, so each texel is weighted by its own d(theta) and
        // sin(theta) (d(phi) is the same for all).
        const float3 below = fromSkyLut(uv + float2(0.f, 0.5f / size.y), size, gLightingRadius, gAtmosphere.planetRadius);
        const float3 above = fromSkyLut(uv - float2(0.f, 0.5f / size.y), size, gLightingRadius, gAtmosphere.planetRadius);
        const float weight = abs(acos(clamp(below.y, -1.f, 1.f)) - acos(clamp(above.y, -1.f, 1.f))) * length(direction.xz);
        sum += float4(gSkyLut[pixel].rgb * weight, weight);
    }
    gSkySum[id.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride /= 2)
    {
        if (id.x < stride)
            gSkySum[id.x] += gSkySum[id.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (id.x == 0)
    {
        const float altitude = clamp(gLightingRadius - gAtmosphere.planetRadius, 0.f, gAtmosphere.atmosphereHeight);
        const float3 transmittance =
            sampleTransmission(pl_compute_lut_uv(altitude, gAtmosphere.atmosphereHeight, float3(0.f, 1.f, 0.f), pl_safe_normalize(gAtmosphere.sunDirection)));
        gLighting[0] = float4(gAtmosphere.sunRadiance * transmittance, 0.f);
        gLighting[1] = float4(gSkySum[0].rgb / max(gSkySum[0].a, 1e-12f), 0.f);
    }
}
