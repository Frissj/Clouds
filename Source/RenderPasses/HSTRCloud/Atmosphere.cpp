/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "Atmosphere.h"
#include <cstring>

namespace hstrcloud
{
namespace
{
const char kShaderFile[] = "RenderPasses/HSTRCloud/Atmosphere.cs.slang";

bool same(const AtmosphereParams& a, const AtmosphereParams& b)
{
    return std::memcmp(&a, &b, sizeof(AtmosphereParams)) == 0;
}

/// The part of the parameters the transmittance and multiple-scattering LUTs depend on: the air, not the sun or the view.
AtmosphereParams air(AtmosphereParams p)
{
    p.viewRadius = 0.f;
    p.multiScatter = 0;
    p.sunDirection = float3(0.f);
    p.worldToKm = 0.f;
    p.sunRadiance = float3(0.f);
    p.sunAngularRadius = 0.f;
    p.aerialDistance = 0.f;
    p.aerialDepthExponent = 0.f;
    p.aerialSamplesPerSlice = 0.f;
    p.groundY = 0.f;
    return p;
}
} // namespace

Atmosphere::Atmosphere(ref<Device> pDevice) : mpDevice(std::move(pDevice))
{
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
    samplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpLinearClamp = mpDevice->createSampler(samplerDesc);
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    mpTransmission = mpDevice->createTexture2D(256, 256, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
    mpMultiscatter = mpDevice->createTexture2D(64, 64, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
    mpSkyView = mpDevice->createTexture2D(200, 100, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
    mpSkyViewCloud = mpDevice->createTexture2D(200, 100, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
    mpAerial = mpDevice->createTexture3D(128, 128, 32, ResourceFormat::RGBA16Float, 1, nullptr, flags);
    mpLighting = mpDevice->createStructuredBuffer(sizeof(float4), 2);
    mpTransmissionPass = createPass("transmissionLut");
    mpMultiscatterPass = createPass("multiscatterLut");
    mpSkyViewPass = createPass("skyViewLut");
    mpAerialPass = createPass("aerialLut");
    mpLightingPass = createPass("cloudLighting");
}

ref<ComputePass> Atmosphere::createPass(const char* entry)
{
    ProgramDesc desc;
    desc.addShaderLibrary(kShaderFile).csEntry(entry);
    desc.shareFrontEnd = true;
    return ComputePass::create(mpDevice, desc);
}

void Atmosphere::bind(const ref<ComputePass>& pPass, const AtmosphereParams& params)
{
    auto var = pPass->getRootVar();
    var["CB"]["gAtmosphere"].setBlob(params);
    var["gLinearClamp"] = mpLinearClamp;
    var["gTransmissionLut"] = mpTransmission;
    var["gMultiscatterLut"] = mpMultiscatter;
}

bool Atmosphere::update(RenderContext* pRenderContext, const AtmosphereParams& params, float lightingRadius, float3& sun, float3& sky)
{
    FALCOR_PROFILE(pRenderContext, "atmosphere");
    const bool airChanged = !mValid || !same(air(params), air(mAir));
    if (airChanged)
    {
        bind(mpTransmissionPass, params);
        mpTransmissionPass->getRootVar()["gOutput2D"] = mpTransmission;
        mpTransmissionPass->execute(pRenderContext, uint3(mpTransmission->getWidth(), mpTransmission->getHeight(), 1));
        bind(mpMultiscatterPass, params);
        mpMultiscatterPass->getRootVar()["gOutput2D"] = mpMultiscatter;
        mpMultiscatterPass->execute(pRenderContext, uint3(mpMultiscatter->getWidth(), mpMultiscatter->getHeight(), 1));
        mAir = params;
    }
    if (airChanged || !same(params, mView))
    {
        bind(mpSkyViewPass, params);
        mpSkyViewPass->getRootVar()["gOutput2D"] = mpSkyView;
        mpSkyViewPass->execute(pRenderContext, uint3(mpSkyView->getWidth(), mpSkyView->getHeight(), 1));
        bind(mpAerialPass, params);
        mpAerialPass->getRootVar()["gOutput3D"] = mpAerial;
        mpAerialPass->execute(pRenderContext, uint3(mpAerial->getWidth(), mpAerial->getHeight(), 1));
        mView = params;
    }
    mValid = true;
    // The clouds' lighting does not depend on where the camera is.
    AtmosphereParams lit = params;
    lit.viewRadius = 0.f;
    lit.aerialDistance = lit.aerialDepthExponent = lit.aerialSamplesPerSlice = 0.f;
    if (same(lit, mLit) && lightingRadius == mLitRadius)
        return false;
    mLit = lit;
    mLitRadius = lightingRadius;
    AtmosphereParams cloud = params;
    cloud.viewRadius = lightingRadius;
    bind(mpSkyViewPass, cloud);
    mpSkyViewPass->getRootVar()["gOutput2D"] = mpSkyViewCloud;
    mpSkyViewPass->execute(pRenderContext, uint3(mpSkyViewCloud->getWidth(), mpSkyViewCloud->getHeight(), 1));
    bind(mpLightingPass, cloud);
    auto var = mpLightingPass->getRootVar();
    var["CB"]["gLightingRadius"] = lightingRadius;
    var["gSkyLut"] = mpSkyViewCloud;
    var["gLighting"] = mpLighting;
    mpLightingPass->execute(pRenderContext, uint3(256, 1, 1));
    // A readback, once per change of the sun or the air: the lighting is the renderer's sunRadiance and skyRadiance, which the host
    // compares and uploads with its other parameters.
    const std::vector<float4> lighting = mpLighting->getElements<float4>(0, 2);
    sun = float3(lighting[0].x, lighting[0].y, lighting[0].z);
    sky = float3(lighting[1].x, lighting[1].y, lighting[1].z);
    return true;
}
} // namespace hstrcloud
