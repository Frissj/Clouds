/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "Utils/Math/Vector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

/** HSTR cloud package (.hstrlib): one cloud compiled offline (HSTRCloudCompiler), so clouds are distributed individually; the
 * runtime loads a directory of them as its library.
 *
 * Each cloud is a sparse mip pyramid of 8^3 bricks (level L voxels cover 2^L source voxels). A brick stores only the residual
 * between its true mean densities and the trilinear prediction from its parent's reconstruction, as quantised 3D wavelet
 * coefficients whose step the compiler chooses per brick by rate-distortion optimisation; bricks the prediction already serves
 * are dropped, so smooth regions cost nothing and detail goes where it changes the image. Bricks are grouped into pages for
 * streaming:
 *
 *   header | per asset: proxies, coarse page (levels >= 4), chunk table
 *   chunk page (one per 128^3 chunk): level-3 bricks and the directory of its level-2 pages
 *   level-2 page: one level-2 brick with its level-1 and level-0 descendants
 *
 * Every page is two GDeflate blobs: its meta (brick headers and page directory), which the CPU reads for the residency cut, and its
 * payload (the packed coefficients), which DirectStorage decompresses straight into GPU memory (RTX IO), or the CPU when GPU
 * decompression is unavailable. The GPU decodes the coefficients and reconstructs the brick from its parent in the atlas
 * (reconstructBrick documents the exact rule); the runtime never needs OpenVDB.
 */
namespace hstrcloud
{
using namespace Falcor;

constexpr uint64_t kLibraryMagic = 0x3442494C52545348ull; // HSTRLIB4
constexpr uint32_t kLibraryVersion = 2;
constexpr uint32_t kCoreValues = 512;   ///< 8^3 residual values per brick.
constexpr uint32_t kBrickValues = 1000; ///< 10^3 reconstructed values per brick (with a one-voxel apron) in the atlas.
constexpr uint32_t kChunkLevel = 4;     ///< Chunks are level-4 bricks.
constexpr uint32_t kChunkVoxels = 128;
constexpr uint32_t kPageLevel = 2; ///< Streaming pages are level-2 subtrees.

/// A GDeflate blob in the package (compressed 0: absent).
struct BlobRef
{
    uint64_t offset = 0;
    uint32_t compressed = 0;
    uint32_t raw = 0;
};

/// A page: meta for the CPU, payload for the GPU (raw size a multiple of 4: it is read as 32-bit words).
struct PageBlobs
{
    BlobRef meta;
    BlobRef payload;
};

struct LibraryHeader
{
    uint64_t magic = kLibraryMagic;
    uint32_t version = kLibraryVersion;
    uint32_t assetCount = 0;
    uint64_t assetTableOffset = 0;
    uint64_t packageBytes = 0;
    uint64_t sourceBytes = 0;
    float lambda = 0.f;              ///< Rate-distortion trade-off the package was compiled with (weighted error per byte).
    float transmittanceError = 0.f;  ///< 99th percentile of the close-up transmittance error the compiler measured.
    uint64_t packKey = 0;            ///< Hash of the source density and every compiler setting: an up-to-date package is not repacked.
};

struct AssetEntry
{
    char name[64] = {};
    int32_t sourceMin[3] = {};
    uint32_t dims[3] = {}; ///< Source voxels, multiples of 128.
    float voxelWorld = 1.f;
    uint32_t topLevel = 0;
    uint32_t proxyLevel = 0;
    uint32_t proxyDims[3] = {};
    uint32_t pad = 0;
    uint64_t densityHash = 0; ///< Of the source density: the runtime loads identical clouds once.
    BlobRef proxy;            ///< proxyDims^3 mean then maximum, float16.
    PageBlobs coarse;            ///< Bricks of levels >= 4.
    uint64_t chunkTableOffset = 0; ///< One PageBlobs per chunk.
    uint64_t level0Bricks = 0;     ///< Non-empty level-0 bricks of the source.
    uint64_t storedBricks = 0;     ///< Bricks in the package.
    uint64_t storedBytes = 0;
};

/// Brick flag: no residual; reconstructed from its parent only (kept because stored descendants need it in the atlas).
constexpr uint8_t kBrickPredicted = 1;

/// Residual transforms (BrickHeader::transform).
constexpr uint8_t kTransformIdentity = 0; ///< Coefficients are the residual voxels.
constexpr uint8_t kTransformHaar = 1;     ///< Orthonormal 3-level 3D Haar wavelet (haarInverse).

struct BrickHeader
{
    uint32_t coord = 0; ///< Level-local brick coordinates, 10 bits per axis.
    uint8_t level = 0;
    uint8_t childMask = 0; ///< Children with non-zero density, bit x + 2 y + 4 z. Children absent from the package are predicted.
    uint8_t transform = kTransformHaar;
    uint8_t flags = 0;
    float step = 0.f;          ///< Coefficient = quantised value * step.
    uint32_t payloadBytes = 0; ///< Packed coefficients (packCoefficients), a multiple of 4; 0 when predicted.
    float valueMin = 0.f;      ///< Atlas quantisation of the reconstructed 10^3 values: v = valueMin + valueRange * (code / 255)^2.
    float valueRange = 0.f;
    uint32_t payloadOffset = 0; ///< Within the page payload, a multiple of 4.

    uint3 brick() const { return uint3(coord & 1023u, (coord >> 10) & 1023u, coord >> 20); }
    static uint32_t pack(uint3 b) { return b.x | (b.y << 10) | (b.z << 20); }
};

/// Header of a page's meta, followed by its bricks, then (chunk pages) its page directory.
struct PageHeader
{
    uint32_t brickCount = 0;
    uint32_t pageCount = 0; ///< Level-2 pages of the chunk (chunk pages only).
};

struct PageEntry
{
    uint32_t coord = 0; ///< Level-2 brick coordinates.
    uint32_t pad = 0;
    PageBlobs blobs;
};

static_assert(sizeof(BlobRef) == 16 && sizeof(PageBlobs) == 32 && sizeof(BrickHeader) == 28 && sizeof(PageEntry) == 40);

inline uint8_t atlasCode(float value, float valueMin, float valueRange)
{
    if (valueRange <= 0.f)
        return 0;
    const float t = std::sqrt(std::clamp((value - valueMin) / valueRange, 0.f, 1.f));
    return uint8_t(std::lround(t * 255.f));
}

inline float atlasValue(uint8_t code, float valueMin, float valueRange)
{
    const float t = float(code) / 255.f;
    return valueMin + valueRange * t * t;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Residual codec. The GPU mirrors these exactly (decodeCloudResiduals in HSTRCloud.cs.slang).

/// Orthonormal Haar step of values v[offset + k * stride], k < n (n even): the first half becomes averages, the second details.
inline void haarStep(float* v, uint32_t offset, uint32_t stride, uint32_t n)
{
    float temp[8];
    const float s = 0.70710678f;
    for (uint32_t k = 0; k < n / 2; ++k)
    {
        const float a = v[offset + 2 * k * stride];
        const float b = v[offset + (2 * k + 1) * stride];
        temp[k] = (a + b) * s;
        temp[n / 2 + k] = (a - b) * s;
    }
    for (uint32_t k = 0; k < n; ++k)
        v[offset + k * stride] = temp[k];
}

inline void haarStepInverse(float* v, uint32_t offset, uint32_t stride, uint32_t n)
{
    float temp[8];
    const float s = 0.70710678f;
    for (uint32_t k = 0; k < n / 2; ++k)
    {
        const float a = v[offset + k * stride];
        const float d = v[offset + (n / 2 + k) * stride];
        temp[2 * k] = (a + d) * s;
        temp[2 * k + 1] = (a - d) * s;
    }
    for (uint32_t k = 0; k < n; ++k)
        v[offset + k * stride] = temp[k];
}

/// Forward 3D Haar of an 8^3 block (index x + 8 (y + 8 z)): per level on the low cube of edge 8, 4, 2, along x, then y, then z.
inline void haarForward(float v[kCoreValues])
{
    for (uint32_t n = 8; n >= 2; n /= 2)
    {
        for (uint32_t z = 0; z < n; ++z)
            for (uint32_t y = 0; y < n; ++y)
                haarStep(v, 8 * (y + 8 * z), 1, n);
        for (uint32_t z = 0; z < n; ++z)
            for (uint32_t x = 0; x < n; ++x)
                haarStep(v, x + 64 * z, 8, n);
        for (uint32_t y = 0; y < n; ++y)
            for (uint32_t x = 0; x < n; ++x)
                haarStep(v, x + 8 * y, 64, n);
    }
}

/// Inverse of haarForward: levels from the coarsest, each along z, then y, then x.
inline void haarInverse(float v[kCoreValues])
{
    for (uint32_t n = 2; n <= 8; n *= 2)
    {
        for (uint32_t y = 0; y < n; ++y)
            for (uint32_t x = 0; x < n; ++x)
                haarStepInverse(v, x + 8 * y, 64, n);
        for (uint32_t z = 0; z < n; ++z)
            for (uint32_t x = 0; x < n; ++x)
                haarStepInverse(v, x + 64 * z, 8, n);
        for (uint32_t z = 0; z < n; ++z)
            for (uint32_t y = 0; y < n; ++y)
                haarStepInverse(v, 8 * (y + 8 * z), 1, n);
    }
}

/// Coefficient scan: coarse to fine subbands (largest coordinate 0, 1, 2-3, 4-7), each in raster order, so the zeros of fine
/// subbands form long runs for GDeflate.
inline const std::array<uint16_t, kCoreValues>& coefficientScan()
{
    static const std::array<uint16_t, kCoreValues> scan = []
    {
        std::array<uint16_t, kCoreValues> s;
        uint32_t n = 0;
        const uint32_t lo[4] = {0, 1, 2, 4};
        const uint32_t hi[4] = {1, 2, 4, 8};
        for (uint32_t band = 0; band < 4; ++band)
            for (uint32_t z = 0; z < 8; ++z)
                for (uint32_t y = 0; y < 8; ++y)
                    for (uint32_t x = 0; x < 8; ++x)
                    {
                        const uint32_t m = std::max(x, std::max(y, z));
                        if (m >= lo[band] && m < hi[band])
                            s[n++] = uint16_t(x + 8 * (y + 8 * z));
                    }
        return s;
    }();
    return scan;
}

/// Packed coefficients: a significance bitmap of 16 words (bit s of the scan), then each non-zero value in scan order as a
/// zigzag varint of 1-3 bytes (7 bits per byte, high bit: more follow), zero-padded to a multiple of 4 bytes. Values are clamped
/// to +-2^20.
inline void packCoefficients(const int32_t q[kCoreValues], std::vector<uint8_t>& out)
{
    const auto& scan = coefficientScan();
    const size_t start = out.size();
    out.resize(start + 64, 0);
    for (uint32_t s = 0; s < kCoreValues; ++s)
    {
        const int32_t value = std::clamp(q[scan[s]], -(1 << 20), 1 << 20);
        if (value == 0)
            continue;
        out[start + s / 8] |= uint8_t(1u << (s % 8));
        uint32_t u = (uint32_t(value) << 1) ^ uint32_t(value >> 31);
        while (u >= 0x80)
        {
            out.push_back(uint8_t(0x80 | (u & 0x7F)));
            u >>= 7;
        }
        out.push_back(uint8_t(u));
    }
    while ((out.size() - start) % 4 != 0)
        out.push_back(0);
}

/// Inverse of packCoefficients; returns the bytes read (with padding).
inline uint32_t unpackCoefficients(const uint8_t* data, int32_t q[kCoreValues])
{
    const auto& scan = coefficientScan();
    uint32_t p = 64;
    for (uint32_t s = 0; s < kCoreValues; ++s)
    {
        if (!(data[s / 8] & (1u << (s % 8))))
        {
            q[scan[s]] = 0;
            continue;
        }
        uint32_t u = 0;
        for (uint32_t shift = 0;; shift += 7)
        {
            const uint8_t byte = data[p++];
            u |= uint32_t(byte & 0x7F) << shift;
            if (!(byte & 0x80))
                break;
        }
        q[scan[s]] = int32_t(u >> 1) ^ -int32_t(u & 1);
    }
    return (p + 3) / 4 * 4;
}

/// Spatial residual of dequantised coefficients.
inline void decodeResidual(const int32_t q[kCoreValues], float step, uint8_t transform, float residual[kCoreValues])
{
    for (uint32_t i = 0; i < kCoreValues; ++i)
        residual[i] = float(q[i]) * step;
    if (transform == kTransformHaar)
        haarInverse(residual);
}

/// The reconstruction rule shared by the compiler and the GPU commit pass (commitCloudBricks). Value j = x + 10 (y + 10 z) of a
/// brick at local position (x, y, z) - 1 is the trilinear interpolation of its parent's dequantised atlas values at parent-local
/// position 4 parity + (local + 0.5) / 2 - 0.5 (parity = brick coordinate & 1), plus the residual on the 8^3 core (none when null),
/// clamped at zero. A brick without a parent predicts zero.
inline void reconstructBrick(
    const uint8_t* parentCodes,
    float parentMin,
    float parentRange,
    uint3 parity,
    const float* residual,
    float values[kBrickValues]
)
{
    for (uint32_t z = 0; z < 10; ++z)
        for (uint32_t y = 0; y < 10; ++y)
            for (uint32_t x = 0; x < 10; ++x)
            {
                float prediction = 0.f;
                if (parentCodes)
                {
                    const float3 local = float3(float(x), float(y), float(z)) - 1.f;
                    // Parent texel coordinates (apron included): parent-local + 1.
                    const float3 texel = 4.f * float3(parity) + (local + 0.5f) * 0.5f - 0.5f + 1.f;
                    const int3 base = int3(std::floor(texel.x), std::floor(texel.y), std::floor(texel.z));
                    const float3 f = texel - float3(base);
                    for (uint32_t corner = 0; corner < 8; ++corner)
                    {
                        const int3 q = base + int3(corner & 1, (corner >> 1) & 1, corner >> 2);
                        const float w = ((corner & 1) ? f.x : 1.f - f.x) * ((corner & 2) ? f.y : 1.f - f.y) * ((corner & 4) ? f.z : 1.f - f.z);
                        prediction += w * atlasValue(parentCodes[q.x + 10 * (q.y + 10 * q.z)], parentMin, parentRange);
                    }
                }
                const bool core = x >= 1 && x <= 8 && y >= 1 && y <= 8 && z >= 1 && z <= 8;
                const float r = core && residual ? residual[(x - 1) + 8 * ((y - 1) + 8 * (z - 1))] : 0.f;
                values[x + 10 * (y + 10 * z)] = std::max(0.f, prediction + r);
            }
}

/// GDeflate (DirectStorage's CPU codec): the package's blob format, which DirectStorage can also decompress on the GPU.
bool compressGDeflate(const uint8_t* data, size_t size, std::vector<uint8_t>& out);
bool decompressGDeflate(const uint8_t* data, size_t size, std::vector<uint8_t>& out, size_t rawSize);
/// HRESULT of the calling thread's last GDeflate call (or of creating its codec).
int32_t lastGDeflateResult();

/// Deflate helpers (zlib) for offline caches.
bool deflateBytes(const uint8_t* data, size_t size, std::vector<uint8_t>& out);
bool inflateBytes(const uint8_t* data, size_t size, std::vector<uint8_t>& out, size_t rawSize);

} // namespace hstrcloud
