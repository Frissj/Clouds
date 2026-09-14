/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "Utils/Math/Vector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

/** HSTR cloud library package (.hstrlib): every cloud of a library compiled offline (HSTRCloudCompiler) into one budgeted file.
 *
 * Each cloud is a sparse mip pyramid of 8^3 bricks (level L voxels cover 2^L source voxels). A brick stores only the residual
 * between its true mean densities and the trilinear prediction from its parent's reconstruction, quantised to 0-8 bits chosen
 * against a library-wide optical-depth tolerance; bricks the prediction already reconstructs within it are dropped, so smooth
 * regions cost nothing and detail goes where it changes the image. Bricks are grouped into deflated pages for streaming:
 *
 *   header | asset table | per asset: proxies, coarse page (levels >= 4), chunk table
 *   chunk page (one per 128^3 chunk): level-3 bricks and the directory of its level-2 pages
 *   level-2 page: one level-2 brick with its level-1 and level-0 descendants
 *
 * The runtime reconstructs a brick on the GPU from its parent in the atlas (reconstructBrick documents the exact rule) and never
 * needs OpenVDB.
 */
namespace hstrcloud
{
using namespace Falcor;

constexpr uint64_t kLibraryMagic = 0x3342494C52545348ull; // HSTRLIB3
constexpr uint32_t kLibraryVersion = 1;
constexpr uint32_t kCoreValues = 512;   ///< 8^3 stored residuals per brick.
constexpr uint32_t kBrickValues = 1000; ///< 10^3 reconstructed values per brick (with a one-voxel apron) in the atlas.
constexpr uint32_t kChunkLevel = 4;     ///< Chunks are level-4 bricks.
constexpr uint32_t kChunkVoxels = 128;
constexpr uint32_t kPageLevel = 2; ///< Streaming pages are level-2 subtrees.
constexpr uint32_t kNoAlias = 0xFFFFFFFF;

/// A deflated blob in the package (compressed 0: absent).
struct BlobRef
{
    uint64_t offset = 0;
    uint32_t compressed = 0;
    uint32_t raw = 0;
};

struct LibraryHeader
{
    uint64_t magic = kLibraryMagic;
    uint32_t version = kLibraryVersion;
    uint32_t assetCount = 0;
    uint64_t assetTableOffset = 0;
    uint64_t packageBytes = 0;
    uint64_t sourceBytes = 0;
    float tolerance = 0.f; ///< Optical-depth tolerance per brick span the library was compiled for.
    uint32_t pad = 0;
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
    uint32_t aliasOf = kNoAlias; ///< Index of an earlier asset with identical density: its data is shared.
    BlobRef proxy;               ///< proxyDims^3 mean then maximum, float16.
    BlobRef coarse;              ///< Bricks of levels >= 4.
    uint64_t chunkTableOffset = 0; ///< One BlobRef per chunk.
    uint64_t level0Bricks = 0;     ///< Non-empty level-0 bricks of the source.
    uint64_t storedBricks = 0;     ///< Bricks in the package.
    uint64_t storedBytes = 0;
};

/// Brick flag: no residual; reconstructed from its parent only (kept because stored descendants need it in the atlas).
constexpr uint8_t kBrickPredicted = 1;

struct BrickHeader
{
    uint32_t coord = 0; ///< Level-local brick coordinates, 10 bits per axis.
    uint8_t level = 0;
    uint8_t childMask = 0; ///< Children with non-zero density, bit x + 2 y + 4 z. Children absent from the package are predicted.
    uint8_t bits = 0;      ///< Residual bits per value (0: residualMin everywhere).
    uint8_t flags = 0;
    float residualMin = 0.f;
    float residualStep = 0.f;
    float valueMin = 0.f; ///< Atlas quantisation of the reconstructed 10^3 values: v = valueMin + valueRange * (code / 255)^2.
    float valueRange = 0.f;
    uint32_t payloadOffset = 0; ///< Packed residuals within the page.

    uint3 brick() const { return uint3(coord & 1023u, (coord >> 10) & 1023u, coord >> 20); }
    static uint32_t pack(uint3 b) { return b.x | (b.y << 10) | (b.z << 20); }
    uint32_t payloadBytes() const { return (flags & kBrickPredicted) || bits == 0 ? 0 : (kCoreValues * bits + 7) / 8; }
};

/// Header of a deflated chunk page or level-2 page, followed by its bricks, then (chunk pages) its page directory, then payloads.
struct PageHeader
{
    uint32_t brickCount = 0;
    uint32_t pageCount = 0; ///< Level-2 pages of the chunk (chunk pages only).
};

struct PageEntry
{
    uint32_t coord = 0; ///< Level-2 brick coordinates.
    BlobRef blob;
};

static_assert(sizeof(BlobRef) == 16 && sizeof(BrickHeader) == 28 && sizeof(PageEntry) == 24);

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

/// Residual code q of value i in a packed payload.
inline uint32_t residualCode(const uint8_t* payload, uint32_t bits, uint32_t i)
{
    const uint32_t bit = i * bits;
    const uint32_t word = uint32_t(payload[bit / 8]) | (bit / 8 + 1 < (kCoreValues * bits + 7) / 8 ? uint32_t(payload[bit / 8 + 1]) << 8 : 0u);
    return (word >> (bit % 8)) & ((1u << bits) - 1u);
}

/// The reconstruction rule shared by the compiler and the GPU commit pass (commitCloudBricks). Value j = x + 10 (y + 10 z) of a
/// brick at local position (x, y, z) - 1 is the trilinear interpolation of its parent's dequantised atlas values at parent-local
/// position 4 parity + (local + 0.5) / 2 - 0.5 (parity = brick coordinate & 1), plus the residual on the 8^3 core, clamped at zero.
/// A brick without a parent predicts zero.
inline void reconstructBrick(
    const uint8_t* parentCodes,
    float parentMin,
    float parentRange,
    uint3 parity,
    const BrickHeader& brick,
    const uint8_t* payload,
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
                float residual = 0.f;
                if (x >= 1 && x <= 8 && y >= 1 && y <= 8 && z >= 1 && z <= 8 && !(brick.flags & kBrickPredicted))
                {
                    residual = brick.residualMin;
                    if (brick.bits > 0)
                        residual += brick.residualStep * float(residualCode(payload, brick.bits, (x - 1) + 8 * ((y - 1) + 8 * (z - 1))));
                }
                values[x + 10 * (y + 10 * z)] = std::max(0.f, prediction + residual);
            }
}

/// Deflate helpers (zlib); the codec is per package, so GDeflate for DirectStorage can replace it.
bool deflateBytes(const uint8_t* data, size_t size, std::vector<uint8_t>& out);
bool inflateBytes(const uint8_t* data, size_t size, std::vector<uint8_t>& out, size_t rawSize);

} // namespace hstrcloud
