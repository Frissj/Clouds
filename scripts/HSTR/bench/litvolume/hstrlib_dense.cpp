// Offline decoder for HSTR cloud packages (.hstrlib, CloudFormat.h): writes one asset's density at one pyramid level as a dense
// float16 .npy, for the offline lit-volume studies (lit_volume_fit.py, lit_volume_budget.py). No Falcor. GDeflate comes from either
//   Windows: DirectStorage's own codec, as CloudFormat.cpp uses it (define HSTR_DSTORAGE; the package Falcor downloads into
//            external/directstorage-*), e.g. from a VS developer prompt:
//            cl /O2 /std:c++17 /EHsc /DHSTR_DSTORAGE hstrlib_dense.cpp /I <directstorage>\native\include
//               <directstorage>\native\lib\x64\dstorage.lib
//            with dstorage.dll and dstoragecore.dll beside the exe (build\windows-ninja-msvc\bin\Release has them);
//   Linux / WSL: the reference decoder (microsoft/DirectStorage GDeflate + NVIDIA/libdeflate), see build_hstrlib_dense.sh.
//
//   hstrlib_dense <package.hstrlib> <level> <out.npy> [x0 y0 z0 x1 y1 z1]
//
// The optional box (level voxels, half-open) crops the output. Every stored brick is reconstructed exactly as commitCloudBricks does
// (reconstructBrick: parent atlas codes, trilinear at the parity offset, plus the dequantised Haar residual, clamped at zero, then
// re-quantised to the brick's own 8-bit atlas codes). Where the level has no stored brick the output holds the level above,
// upsampled trilinearly (what a sample climbing the parent chain reads), and zero inside octants the parent's child mask marks
// empty (the runtime's missing-octant skirt).
#ifdef HSTR_DSTORAGE
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <dstorage.h>
#else
#include "GDeflate.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
constexpr uint64_t kLibraryMagic = 0x3442494C52545348ull;
constexpr uint32_t kCoreValues = 512;
constexpr uint32_t kBrickValues = 1000;
constexpr uint8_t kTransformHaar = 1;

struct BlobRef
{
    uint64_t offset = 0;
    uint32_t compressed = 0;
    uint32_t raw = 0;
};
struct PageBlobs
{
    BlobRef meta;
    BlobRef payload;
};
struct LibraryHeader
{
    uint64_t magic;
    uint32_t version;
    uint32_t assetCount;
    uint64_t assetTableOffset;
    uint64_t packageBytes;
    uint64_t sourceBytes;
    float lambda;
    float transmittanceError;
    uint64_t packKey;
};
struct AssetEntry
{
    char name[64];
    int32_t sourceMin[3];
    uint32_t dims[3];
    float voxelWorld;
    uint32_t topLevel;
    uint32_t proxyLevel;
    uint32_t proxyDims[3];
    uint32_t pad;
    uint64_t densityHash;
    BlobRef proxy;
    PageBlobs coarse;
    uint64_t chunkTableOffset;
    uint64_t level0Bricks;
    uint64_t storedBricks;
    uint64_t storedBytes;
};
struct BrickHeader
{
    uint32_t coord;
    uint8_t level;
    uint8_t childMask;
    uint8_t transform;
    uint8_t flags;
    float step;
    uint32_t payloadBytes;
    float valueMin;
    float valueRange;
    uint32_t payloadOffset;
};
struct PageEntry
{
    uint32_t coord;
    uint32_t pad;
    PageBlobs blobs;
};
struct PageHeader
{
    uint32_t brickCount;
    uint32_t pageCount;
};
static_assert(sizeof(BlobRef) == 16 && sizeof(PageBlobs) == 32 && sizeof(BrickHeader) == 28 && sizeof(PageEntry) == 40);

// --- CloudFormat.h, verbatim in substance ---------------------------------------------------------------------------------------
float atlasValue(uint8_t code, float valueMin, float valueRange)
{
    const float t = float(code) / 255.f;
    return valueMin + valueRange * t * t;
}
uint8_t atlasCode(float value, float valueMin, float valueRange)
{
    if (valueRange <= 0.f)
        return 0;
    const float t = std::sqrt(std::clamp((value - valueMin) / valueRange, 0.f, 1.f));
    return uint8_t(std::lround(t * 255.f));
}
void haarStepInverse(float* v, uint32_t offset, uint32_t stride, uint32_t n)
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
void haarInverse(float v[kCoreValues])
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
const std::array<uint16_t, kCoreValues>& coefficientScan()
{
    static const std::array<uint16_t, kCoreValues> scan = []
    {
        std::array<uint16_t, kCoreValues> s{};
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
void unpackCoefficients(const uint8_t* data, int32_t q[kCoreValues])
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
}
void reconstructBrick(const uint8_t* parentCodes, float parentMin, float parentRange, const uint32_t parity[3], const float* residual,
                      float values[kBrickValues])
{
    for (uint32_t z = 0; z < 10; ++z)
        for (uint32_t y = 0; y < 10; ++y)
            for (uint32_t x = 0; x < 10; ++x)
            {
                float prediction = 0.f;
                if (parentCodes)
                {
                    const float local[3] = {float(x) - 1.f, float(y) - 1.f, float(z) - 1.f};
                    float texel[3];
                    int base[3];
                    float f[3];
                    for (int a = 0; a < 3; ++a)
                    {
                        texel[a] = 4.f * float(parity[a]) + (local[a] + 0.5f) * 0.5f - 0.5f + 1.f;
                        base[a] = int(std::floor(texel[a]));
                        f[a] = texel[a] - float(base[a]);
                    }
                    for (uint32_t corner = 0; corner < 8; ++corner)
                    {
                        int q[3] = {base[0] + int(corner & 1), base[1] + int((corner >> 1) & 1), base[2] + int(corner >> 2)};
                        for (int a = 0; a < 3; ++a)
                            q[a] = std::clamp(q[a], 1, 8);
                        const float w = ((corner & 1) ? f[0] : 1.f - f[0]) * ((corner & 2) ? f[1] : 1.f - f[1]) *
                                        ((corner & 4) ? f[2] : 1.f - f[2]);
                        prediction += w * atlasValue(parentCodes[q[0] + 10 * (q[1] + 10 * q[2])], parentMin, parentRange);
                    }
                }
                const bool core = x >= 1 && x <= 8 && y >= 1 && y <= 8 && z >= 1 && z <= 8;
                const float r = core && residual ? residual[(x - 1) + 8 * ((y - 1) + 8 * (z - 1))] : 0.f;
                values[x + 10 * (y + 10 * z)] = std::max(0.f, prediction + r);
            }
}
// ---------------------------------------------------------------------------------------------------------------------------------

std::vector<uint8_t> readBlob(std::ifstream& file, const BlobRef& blob)
{
    std::vector<uint8_t> raw;
    if (blob.raw == 0)
        return raw;
    std::vector<uint8_t> compressed(blob.compressed);
    file.clear();
    file.seekg(std::streamoff(blob.offset));
    if (!file.read(reinterpret_cast<char*>(compressed.data()), std::streamsize(compressed.size())))
        throw std::runtime_error("short read");
    raw.resize(blob.raw);
#ifdef HSTR_DSTORAGE
    static IDStorageCompressionCodec* codec = []
    {
        IDStorageCompressionCodec* c = nullptr;
        if (FAILED(DStorageCreateCompressionCodec(DSTORAGE_COMPRESSION_FORMAT_GDEFLATE, 1, IID_PPV_ARGS(&c))))
            throw std::runtime_error("DStorageCreateCompressionCodec failed");
        return c;
    }();
    size_t written = 0;
    if (FAILED(codec->DecompressBuffer(compressed.data(), compressed.size(), raw.data(), raw.size(), &written)) || written != raw.size())
        throw std::runtime_error("GDeflate failed");
#else
    if (!GDeflate::Decompress(raw.data(), raw.size(), compressed.data(), compressed.size(), 1))
        throw std::runtime_error("GDeflate failed");
#endif
    return raw;
}

struct Page
{
    std::vector<BrickHeader> bricks;
    std::vector<PageEntry> pages;
    std::vector<uint8_t> payload;
};

Page readPage(std::ifstream& file, const BlobRef& meta, const BlobRef& payload)
{
    Page page;
    const std::vector<uint8_t> raw = readBlob(file, meta);
    if (raw.size() < sizeof(PageHeader))
        return page;
    PageHeader header;
    std::memcpy(&header, raw.data(), sizeof(header));
    page.bricks.resize(header.brickCount);
    page.pages.resize(header.pageCount);
    std::memcpy(page.bricks.data(), raw.data() + sizeof(PageHeader), page.bricks.size() * sizeof(BrickHeader));
    std::memcpy(page.pages.data(), raw.data() + sizeof(PageHeader) + page.bricks.size() * sizeof(BrickHeader),
                page.pages.size() * sizeof(PageEntry));
    page.payload = readBlob(file, payload);
    return page;
}

struct Stored
{
    uint8_t codes[kBrickValues];
    float valueMin;
    float valueRange;
    uint8_t childMask;
};

uint64_t key(uint32_t level, uint32_t coord) { return (uint64_t(level) << 32) | coord; }
void unpack(uint32_t c, uint32_t b[3])
{
    b[0] = c & 1023u;
    b[1] = (c >> 10) & 1023u;
    b[2] = c >> 20;
}

void writeNpy(const std::string& path, const std::vector<uint16_t>& data, const uint64_t shape[3])
{
    // Shape (z, y, x), C order: index x + nx (y + ny z).
    char dict[256];
    std::snprintf(dict, sizeof(dict), "{'descr': '<f2', 'fortran_order': False, 'shape': (%llu, %llu, %llu), }",
                  (unsigned long long)shape[2], (unsigned long long)shape[1], (unsigned long long)shape[0]);
    std::string header(dict);
    const size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY\x01\x00", 8);
    const uint16_t len = uint16_t(header.size());
    out.write(reinterpret_cast<const char*>(&len), 2);
    out.write(header.data(), std::streamsize(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * 2));
}

uint16_t toHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exponent = int32_t((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = x & 0x7FFFFFu;
    if (exponent <= 0)
    {
        if (exponent < -10)
            return uint16_t(sign);
        mantissa |= 0x800000u;
        const uint32_t shift = uint32_t(14 - exponent);
        uint32_t half = mantissa >> shift;
        if ((mantissa >> (shift - 1)) & 1u)
            ++half;
        return uint16_t(sign | half);
    }
    if (exponent >= 31)
        return uint16_t(sign | 0x7C00u);
    uint32_t half = (uint32_t(exponent) << 10) | (mantissa >> 13);
    if (mantissa & 0x1000u)
        ++half;
    return uint16_t(sign | half);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 10)
    {
        std::fprintf(stderr, "usage: hstrlib_dense <package.hstrlib> <level> <out.npy> [x0 y0 z0 x1 y1 z1]\n");
        return 2;
    }
    const uint32_t target = uint32_t(std::atoi(argv[2]));
    std::ifstream file(argv[1], std::ios::binary);
    LibraryHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file || header.magic != kLibraryMagic || header.version != 2)
    {
        std::fprintf(stderr, "not an HSTRLIB4 v2 package\n");
        return 1;
    }
    AssetEntry entry;
    file.seekg(std::streamoff(header.assetTableOffset));
    file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    const uint32_t chunkDims[3] = {entry.dims[0] / 128u, entry.dims[1] / 128u, entry.dims[2] / 128u};
    std::fprintf(stderr, "asset '%s' dims %u %u %u voxelWorld %g top %u proxyLevel %u proxy %u %u %u lambda %g p99 %g stored %llu\n",
                 entry.name, entry.dims[0], entry.dims[1], entry.dims[2], entry.voxelWorld, entry.topLevel, entry.proxyLevel,
                 entry.proxyDims[0], entry.proxyDims[1], entry.proxyDims[2], header.lambda, header.transmittanceError,
                 (unsigned long long)entry.storedBricks);

    std::unordered_map<uint64_t, Stored> stored;
    stored.reserve(size_t(entry.storedBricks) * 2);
    auto commit = [&](const Page& page)
    {
        // Pages list parents before their children (coarse page top-down by level; a level-2 page its brick, then each level-1
        // child followed by its level-0 children), so a parent is always reconstructed first.
        for (const BrickHeader& b : page.bricks)
        {
            Stored s;
            s.valueMin = b.valueMin;
            s.valueRange = b.valueRange;
            s.childMask = b.childMask;
            uint32_t c[3];
            unpack(b.coord, c);
            const uint32_t parity[3] = {c[0] & 1u, c[1] & 1u, c[2] & 1u};
            const auto parent = stored.find(key(b.level + 1u, (c[0] >> 1) | ((c[1] >> 1) << 10) | ((c[2] >> 1) << 20)));
            float residual[kCoreValues];
            const bool hasResidual = b.payloadBytes != 0;
            if (hasResidual)
            {
                int32_t q[kCoreValues];
                unpackCoefficients(page.payload.data() + b.payloadOffset, q);
                for (uint32_t i = 0; i < kCoreValues; ++i)
                    residual[i] = float(q[i]) * b.step;
                if (b.transform == kTransformHaar)
                    haarInverse(residual);
            }
            float values[kBrickValues];
            reconstructBrick(parent != stored.end() ? parent->second.codes : nullptr, parent != stored.end() ? parent->second.valueMin : 0.f,
                             parent != stored.end() ? parent->second.valueRange : 0.f, parity, hasResidual ? residual : nullptr, values);
            for (uint32_t i = 0; i < kBrickValues; ++i)
                s.codes[i] = atlasCode(values[i], b.valueMin, b.valueRange);
            stored[key(b.level, b.coord)] = s;
        }
    };

    commit(readPage(file, entry.coarse.meta, entry.coarse.payload));
    std::vector<PageBlobs> chunks(size_t(chunkDims[0]) * chunkDims[1] * chunkDims[2]);
    file.clear();
    file.seekg(std::streamoff(entry.chunkTableOffset));
    file.read(reinterpret_cast<char*>(chunks.data()), std::streamsize(chunks.size() * sizeof(PageBlobs)));
    size_t pageCount = 0;
    for (const PageBlobs& chunk : chunks)
    {
        if (chunk.meta.raw == 0)
            continue;
        const Page page = readPage(file, chunk.meta, chunk.payload);
        if (target <= 3)
            commit(page);
        if (target <= 2)
            for (const PageEntry& p : page.pages)
            {
                commit(readPage(file, p.blobs.meta, p.blobs.payload));
                ++pageCount;
            }
    }
    size_t perLevel[16] = {};
    for (const auto& [k, s] : stored)
        ++perLevel[std::min<uint64_t>(k >> 32, 15)];
    std::fprintf(stderr, "stored bricks by level:");
    for (uint32_t l = 0; l <= entry.topLevel; ++l)
        std::fprintf(stderr, " L%u %zu", l, perLevel[l]);
    std::fprintf(stderr, " (%zu level-2 pages)\n", pageCount);

    // Dense levels top-down: upsample, zero empty octants, paste stored cores.
    uint64_t lo[3] = {0, 0, 0};
    uint64_t hi[3] = {entry.dims[0] >> target, entry.dims[1] >> target, entry.dims[2] >> target};
    if (argc == 10)
        for (int a = 0; a < 3; ++a)
        {
            lo[a] = uint64_t(std::atoll(argv[4 + a]));
            hi[a] = uint64_t(std::atoll(argv[7 + a]));
        }
    // Work level by level over the box's footprint at that level (with a one-voxel margin for the trilinear upsample).
    std::vector<float> coarse;
    uint64_t cLo[3] = {}, cDim[3] = {};
    for (int32_t level = int32_t(entry.topLevel); level >= int32_t(target); --level)
    {
        const uint32_t shift = uint32_t(level) - target;
        uint64_t fLo[3], fDim[3];
        for (int a = 0; a < 3; ++a)
        {
            const uint64_t full = entry.dims[a] >> level;
            const uint64_t l0 = lo[a] >> shift;
            const uint64_t h0 = ((hi[a] + (1ull << shift) - 1) >> shift);
            fLo[a] = l0 > 0 ? l0 - 1 : 0;
            fDim[a] = std::min(full, h0 + 1) - fLo[a];
        }
        std::vector<float> fine(fDim[0] * fDim[1] * fDim[2], 0.f);
        // Upsample the coarser level: fine voxel i centre at (i + 0.5) / 2 - 0.5 in coarse voxels.
        if (!coarse.empty())
            for (uint64_t z = 0; z < fDim[2]; ++z)
                for (uint64_t y = 0; y < fDim[1]; ++y)
                    for (uint64_t x = 0; x < fDim[0]; ++x)
                    {
                        const uint64_t g[3] = {fLo[0] + x, fLo[1] + y, fLo[2] + z};
                        float p[3];
                        int64_t b[3];
                        float f[3];
                        for (int a = 0; a < 3; ++a)
                        {
                            p[a] = (float(g[a]) + 0.5f) * 0.5f - 0.5f - float(cLo[a]);
                            b[a] = int64_t(std::floor(p[a]));
                            f[a] = p[a] - float(b[a]);
                        }
                        // As reconstructBrick reads a parent: within the coarse brick holding this voxel only (its apron is its
                        // own edge texels), so no density bleeds across a brick face into an empty neighbour.
                        int64_t brickLo[3], brickHi[3];
                        for (int a = 0; a < 3; ++a)
                        {
                            const int64_t first = int64_t(((g[a] >> 1) >> 3) << 3);
                            brickLo[a] = std::max<int64_t>(first - int64_t(cLo[a]), 0);
                            brickHi[a] = std::min<int64_t>(first + 7 - int64_t(cLo[a]), int64_t(cDim[a]) - 1);
                        }
                        float v = 0.f;
                        for (uint32_t corner = 0; corner < 8; ++corner)
                        {
                            int64_t q[3] = {b[0] + (corner & 1), b[1] + ((corner >> 1) & 1), b[2] + (corner >> 2)};
                            for (int a = 0; a < 3; ++a)
                                q[a] = std::clamp<int64_t>(q[a], brickLo[a], brickHi[a]);
                            const float w = ((corner & 1) ? f[0] : 1.f - f[0]) * ((corner & 2) ? f[1] : 1.f - f[1]) *
                                            ((corner & 4) ? f[2] : 1.f - f[2]);
                            v += w * coarse[q[0] + cDim[0] * (q[1] + cDim[1] * q[2])];
                        }
                        // The parent's child mask: an octant without density is empty. Where the parent is not stored either, the
                        // codec dropped it (its own parent predicts it) and the runtime reads the nearest stored ancestor: the
                        // upsampled value stands, already zero where an ancestor's mask emptied it.
                        const uint32_t pc = uint32_t(g[0] >> 4) | (uint32_t(g[1] >> 4) << 10) | (uint32_t(g[2] >> 4) << 20);
                        const auto parent = stored.find(key(uint32_t(level) + 1u, pc));
                        if (parent != stored.end())
                        {
                            const uint32_t oct = uint32_t((g[0] >> 3) & 1) | (uint32_t((g[1] >> 3) & 1) << 1) | (uint32_t((g[2] >> 3) & 1) << 2);
                            if (!(parent->second.childMask & (1u << oct)))
                                v = 0.f;
                        }
                        fine[x + fDim[0] * (y + fDim[1] * z)] = v;
                    }
        // Paste stored bricks of this level.
        const uint64_t b0[3] = {fLo[0] / 8, fLo[1] / 8, fLo[2] / 8};
        const uint64_t b1[3] = {(fLo[0] + fDim[0] + 7) / 8, (fLo[1] + fDim[1] + 7) / 8, (fLo[2] + fDim[2] + 7) / 8};
        for (uint64_t bz = b0[2]; bz < b1[2]; ++bz)
            for (uint64_t by = b0[1]; by < b1[1]; ++by)
                for (uint64_t bx = b0[0]; bx < b1[0]; ++bx)
                {
                    const auto it = stored.find(key(uint32_t(level), uint32_t(bx) | (uint32_t(by) << 10) | (uint32_t(bz) << 20)));
                    if (it == stored.end())
                        continue;
                    const Stored& s = it->second;
                    for (uint32_t z = 0; z < 8; ++z)
                        for (uint32_t y = 0; y < 8; ++y)
                            for (uint32_t x = 0; x < 8; ++x)
                            {
                                const int64_t g[3] = {int64_t(bx * 8 + x) - int64_t(fLo[0]), int64_t(by * 8 + y) - int64_t(fLo[1]),
                                                      int64_t(bz * 8 + z) - int64_t(fLo[2])};
                                if (g[0] < 0 || g[1] < 0 || g[2] < 0 || g[0] >= int64_t(fDim[0]) || g[1] >= int64_t(fDim[1]) ||
                                    g[2] >= int64_t(fDim[2]))
                                    continue;
                                fine[g[0] + fDim[0] * (g[1] + fDim[1] * g[2])] =
                                    atlasValue(s.codes[(x + 1) + 10 * ((y + 1) + 10 * (z + 1))], s.valueMin, s.valueRange);
                            }
                }
        coarse.swap(fine);
        for (int a = 0; a < 3; ++a)
        {
            cLo[a] = fLo[a];
            cDim[a] = fDim[a];
        }
    }
    uint64_t shape[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
    std::vector<uint16_t> out(shape[0] * shape[1] * shape[2]);
    for (uint64_t z = 0; z < shape[2]; ++z)
        for (uint64_t y = 0; y < shape[1]; ++y)
            for (uint64_t x = 0; x < shape[0]; ++x)
                out[x + shape[0] * (y + shape[1] * z)] =
                    toHalf(coarse[(lo[0] + x - cLo[0]) + cDim[0] * ((lo[1] + y - cLo[1]) + cDim[1] * (lo[2] + z - cLo[2]))]);
    writeNpy(argv[3], out, shape);
    std::fprintf(stderr, "wrote %s (%llu x %llu x %llu, level %u)\n", argv[3], (unsigned long long)shape[0], (unsigned long long)shape[1],
                 (unsigned long long)shape[2], target);
    return 0;
}
