/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "AtmosphereTypes.slang"

namespace hstrcloud
{
using namespace Falcor;

/** pl-sky's atmosphere (github.com/hoffstadt/pl-sky; kernels translated in Atmosphere.cs.slang): its four LUTs, rebuilt when what
    they depend on changes, plus the sun and sky radiance they give the clouds (skyModel 2).
*/
class Atmosphere
{
public:
    explicit Atmosphere(ref<Device> pDevice);

    /// Rebuilds the LUTs that params changes (the transmittance and multiple scattering only with the air itself; the sky view and
    /// aerial perspective with the sun or the camera's altitude). When the air, the sun or lightingRadius (the clouds' distance
    /// from the planet centre) changed, reads back the clouds' lighting into sun and sky and returns true.
    bool update(RenderContext* pRenderContext, const AtmosphereParams& params, float lightingRadius, float3& sun, float3& sky);

    const ref<Texture>& transmission() const { return mpTransmission; }
    const ref<Texture>& skyView() const { return mpSkyView; }
    const ref<Texture>& aerial() const { return mpAerial; }

private:
    ref<ComputePass> createPass(const char* entry);
    void bind(const ref<ComputePass>& pPass, const AtmosphereParams& params);

    ref<Device> mpDevice;
    ref<Sampler> mpLinearClamp;
    ref<ComputePass> mpTransmissionPass;
    ref<ComputePass> mpMultiscatterPass;
    ref<ComputePass> mpSkyViewPass;
    ref<ComputePass> mpAerialPass;
    ref<ComputePass> mpLightingPass;
    ref<Texture> mpTransmission; ///< pl-sky's resolutions: 256 x 256.
    ref<Texture> mpMultiscatter; ///< 64 x 64.
    ref<Texture> mpSkyView;      ///< 200 x 100, for the camera.
    ref<Texture> mpSkyViewCloud; ///< 200 x 100, for the clouds (cloudLighting).
    ref<Texture> mpAerial;       ///< 128 x 128 x 32.
    ref<Buffer> mpLighting;
    bool mValid = false;
    AtmosphereParams mAir{};  ///< What the transmittance and multiple-scattering LUTs were built for.
    AtmosphereParams mView{}; ///< What the sky-view and aerial LUTs were built for.
    AtmosphereParams mLit{};  ///< What the lighting was read back for.
    float mLitRadius = -1.f;
};
} // namespace hstrcloud
