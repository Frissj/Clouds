/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
/** Offline compiler of an OpenVDB cloud library into HSTR cloud packages, one file per cloud (see CloudFormat.h), in two stages so
    that everything after reading the source is repeatable without it.

    HSTRCloudCompiler build <vdb directory> <cache directory>
        Once per source cloud (skipped while the VDB is unchanged): its canonical cache <cloud>.hstrcloud holds the source's true
        level-0 brick samples (lossless at the atlas's precision), the means of the coarse levels, the proxies and the density hash.

    HSTRCloudCompiler pack <cache directory> <output directory> [--quality E [--quality-tail T] | --lambda L | --budget-mb N] [--max-error E]
                           [--weight-floor F] [--density-scale S] [--transform haar|identity|cdf53|legacy] [--deadzone D]
                           [--sample R] [--clouds a,b,...] [--dry-run] [--force]
        Packaging from the canonical caches only, into <output directory>/<cloud>.hstrlib. Each brick's residual is transformed
        (3D Haar by default), quantised with the step that minimises weight * error + lambda * bytes, and GDeflate-compressed in
        pages. The error is the optical depth of the brick's reconstruction error; its weight is the brick's visibility from
        outside the cloud (over 26 directions), at least --weight-floor (default 0.01; 1: unweighted). The floor also protects
        cloud interiors for cameras flying through. --max-error caps every brick's unweighted error regardless of cost. The
        cloud's lambda is chosen by one of:
        --quality E (the default, kDefaultQuality): per cloud, the largest lambda whose close-up transmittance error meets E. The
           error is measured on the encoded density itself, without rendering: every voxel column of the chunks on each axis
           integrates the reconstruction a close view renders against the truth, and |e^-tau_true - e^-tau_coded| is attenuated
           by the proxy's optical depth in front of the chunk. Its 99th percentile over --sample of the chunks meets E, and its
           99.9th percentile meets --quality-tail (default kTailFactor * E).
           --density-scale is the extinction per unit of density per VDB world unit (the renderer's density scale times the sea's
           size scale); 1 by default.
        --lambda L: every cloud at L.
        --budget-mb N: one lambda for every cloud so the packages fit N MB in total (a soft limit: an overshoot warns), from each
           cloud's bytes at a ladder of lambdas measured on --sample of its chunks and cached in the cache directory. One lambda
           everywhere is the rate-distortion optimal split of a budget between clouds.
        A package whose key (source density, codec and lambda or quality settings) is unchanged is skipped unless --force; a cloud
        with the density of one already packed gets a copy of its package. --clouds packs a subset (by file name without
        extension) and --dry-run only reports, encoding --sample of the chunks (--transform cdf53 and legacy are dry-run only: the
        GPU decodes Haar and identity).
*/
#include "RenderPasses/HSTRCloud/CloudFormat.h"
#include "Utils/Math/Float16.h"
#include "Core/Platform/OS.h"
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

using namespace hstrcloud;

namespace
{
/// Bump whenever encodeBrick or the page layout changes the package bytes: cached analyses of older codecs are then remeasured.
constexpr uint32_t kCodecVersion = 3;

/// Default close-up transmittance error of pack (--quality): the 99th percentile of voxel-column errors a cloud's lambda meets.
/// 0.02 and 0.05 both rendered indistinguishably from near-lossless on cloud_cumulus_1_size_1 (45 and 28 MB against 193 MB).
constexpr float kDefaultQuality = 0.02f;
/// Default 99.9th-percentile limit as a multiple of the 99th-percentile target (--quality-tail overrides). Packages that rendered
/// like near-lossless had a tail of about 4.5 times their target: the limit only binds on clouds with unusually heavy tails.
constexpr float kTailFactor = 5.f;

/// The analysis measures package bytes at lambdas a factor 2.5 apart: 1e-8 to 9e-3 weighted optical depth per byte.
constexpr uint32_t kLadderSize = 16;
double ladderLambda(double position)
{
    return 1e-8 * std::pow(2.5, position);
}

template<typename F>
void parallelFor(size_t count, F&& f)
{
    const size_t threadCount = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), count);
    std::atomic<size_t> next{0};
    std::exception_ptr failure;
    std::mutex failureMutex;
    // The first exception stops the remaining work and is rethrown on the calling thread.
    auto worker = [&]()
    {
        for (size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1))
        {
            try
            {
                f(i);
            }
            catch (...)
            {
                std::lock_guard lock(failureMutex);
                if (!failure)
                    failure = std::current_exception();
                next = count;
            }
        }
    };
    std::vector<std::thread> threads;
    for (size_t t = 1; t < threadCount; ++t)
        threads.emplace_back(worker);
    worker();
    for (auto& thread : threads)
        thread.join();
    if (failure)
        std::rethrow_exception(failure);
}

double seconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

uint64_t fnv(uint64_t hash, uint64_t value)
{
    return (hash ^ value) * 1099511628211ull;
}

uint64_t sourceSignature(const std::filesystem::path& path)
{
    uint64_t hash = 1469598103934665603ull;
    hash = fnv(hash, std::filesystem::file_size(path));
    return fnv(hash, uint64_t(std::filesystem::last_write_time(path).time_since_epoch().count()));
}

using Core = std::array<float, kCoreValues>;

/// Mean cores of one pyramid level, keyed by packed brick coordinate.
struct Level
{
    std::unordered_map<uint32_t, uint32_t> index;
    std::vector<Core> cores;
    std::vector<uint32_t> coords;
    std::vector<uint8_t> childMask;

    uint32_t add(uint32_t coord)
    {
        auto [it, inserted] = index.try_emplace(coord, uint32_t(coords.size()));
        if (inserted)
        {
            coords.push_back(coord);
            childMask.push_back(0);
            cores.emplace_back();
            cores.back().fill(0.f);
        }
        return it->second;
    }
};

/// Adds a child core into its parent's means (8 children per parent voxel) and marks the child in the parent's mask.
void accumulate(Level& parents, uint32_t childCoord, const Core& child)
{
    const uint3 b = BrickHeader{childCoord}.brick();
    const uint32_t p = parents.add(BrickHeader::pack(b / 2u));
    const uint3 o = (b % 2u) * 8u;
    parents.childMask[p] |= uint8_t(1u << ((b.x & 1u) + 2u * (b.y & 1u) + 4u * (b.z & 1u)));
    Core& parent = parents.cores[p];
    for (uint32_t z = 0; z < 8; ++z)
        for (uint32_t y = 0; y < 8; ++y)
            for (uint32_t x = 0; x < 8; ++x)
            {
                const uint3 q = (o + uint3(x, y, z)) / 2u;
                parent[q.x + 8 * (q.y + 8 * q.z)] += 0.125f * child[x + 8 * (y + 8 * z)];
            }
}

/** Canonical cloud cache (<cache directory>/<cloud>.hstrcloud), written once per source by `build`:
 *
 *   header | chunk blobs | proxy blob | coarse blob | chunk table
 *
 * A chunk blob is the deflated level-0 cores of one chunk (packCores); levels 1-3 are their means, rebuilt on read. The coarse blob
 * holds the means of levels 4 to top; the proxy blob the runtime's float16 proxies. Every blob is deflated (BlobRef::raw set).
 */
constexpr uint64_t kCanonicalMagic = 0x314E414352545348ull; // HSTRCAN1
constexpr uint32_t kCanonicalVersion = 2;

/// A cloud's name and geometry, independent of the package format.
struct CanonicalGeometry
{
    char name[64] = {};
    int32_t sourceMin[3] = {};
    uint32_t dims[3] = {};
    float voxelWorld = 1.f;
    uint32_t topLevel = 0;
    uint32_t proxyLevel = 0;
    uint32_t proxyDims[3] = {};
    uint64_t level0Bricks = 0;
};

struct CanonicalHeader
{
    uint64_t magic = kCanonicalMagic;
    uint32_t version = kCanonicalVersion;
    uint32_t chunkCount = 0;
    uint64_t sourceSignature = 0;
    uint64_t sourceBytes = 0;
    uint64_t densityHash = 0;
    CanonicalGeometry geometry;
    BlobRef proxy;
    BlobRef coarse;
    uint64_t chunkTableOffset = 0;
};

struct CanonicalChunk
{
    uint32_t chunk = 0; ///< Chunk index in the cloud's chunk grid.
    uint32_t bricks = 0;
    BlobRef blob;
};

/// Level-0 cores of one chunk: brick count, coordinates, per brick its value minimum and range, then 512 uint16 codes per brick
/// (value = minimum + range * code / 65535, as fine as the smallest step of the atlas's sqrt coding), delta-coded for deflate.
std::vector<uint8_t> packCores(const Level& level)
{
    const uint32_t count = uint32_t(level.coords.size());
    std::vector<uint8_t> out(sizeof(uint32_t) + count * (sizeof(uint32_t) + 2 * sizeof(float) + kCoreValues * sizeof(uint16_t)));
    uint8_t* p = out.data();
    auto put = [&](const auto& value)
    {
        std::memcpy(p, &value, sizeof(value));
        p += sizeof(value);
    };
    put(count);
    for (uint32_t coord : level.coords)
        put(coord);
    std::vector<std::pair<float, float>> ranges(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        const auto [lo, hi] = std::minmax_element(level.cores[i].begin(), level.cores[i].end());
        ranges[i] = {*lo, *hi - *lo};
        put(ranges[i].first);
        put(ranges[i].second);
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        uint16_t previous = 0;
        for (float value : level.cores[i])
        {
            const uint16_t code =
                ranges[i].second > 0.f ? uint16_t(std::lround(std::clamp((value - ranges[i].first) / ranges[i].second, 0.f, 1.f) * 65535.f)) : 0;
            put(uint16_t(code - previous));
            previous = code;
        }
    }
    return out;
}

/// Inverse of packCores into an empty level.
void unpackCores(const std::vector<uint8_t>& raw, Level& level)
{
    const uint8_t* p = raw.data();
    const uint8_t* end = raw.data() + raw.size();
    auto get = [&](auto& value)
    {
        if (p + sizeof(value) > end)
            throw std::runtime_error("corrupt canonical chunk");
        std::memcpy(&value, p, sizeof(value));
        p += sizeof(value);
    };
    uint32_t count = 0;
    get(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        uint32_t coord = 0;
        get(coord);
        level.add(coord);
    }
    std::vector<std::pair<float, float>> ranges(count);
    for (auto& range : ranges)
    {
        get(range.first);
        get(range.second);
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        uint16_t code = 0;
        for (float& value : level.cores[i])
        {
            uint16_t delta = 0;
            get(delta);
            code = uint16_t(code + delta);
            value = ranges[i].first + ranges[i].second * float(code) / 65535.f;
        }
    }
}

/// Deflates bytes into a blob appended at the file's write position.
BlobRef appendBlob(std::ostream& file, const std::vector<uint8_t>& raw)
{
    std::vector<uint8_t> compressed;
    if (!deflateBytes(raw.data(), raw.size(), compressed))
        throw std::runtime_error("deflate failed");
    BlobRef ref;
    ref.offset = uint64_t(file.tellp());
    ref.compressed = uint32_t(compressed.size());
    ref.raw = uint32_t(raw.size());
    file.write(reinterpret_cast<const char*>(compressed.data()), std::streamsize(compressed.size()));
    return ref;
}

std::vector<uint8_t> readBlobFrom(std::istream& file, const BlobRef& ref)
{
    std::vector<uint8_t> compressed(ref.compressed);
    file.seekg(std::streamoff(ref.offset));
    if (!file.read(reinterpret_cast<char*>(compressed.data()), std::streamsize(compressed.size())))
        throw std::runtime_error("truncated canonical cache");
    std::vector<uint8_t> raw;
    if (!inflateBytes(compressed.data(), compressed.size(), raw, ref.raw))
        throw std::runtime_error("corrupt canonical blob");
    return raw;
}

bool readCanonicalHeader(const std::filesystem::path& path, CanonicalHeader& header)
{
    std::ifstream file(path, std::ios::binary);
    return file.read(reinterpret_cast<char*>(&header), sizeof(header)) && header.magic == kCanonicalMagic &&
           header.version == kCanonicalVersion;
}

/// A reconstructed brick as the GPU atlas holds it.
struct Reconstruction
{
    std::array<uint8_t, kBrickValues> codes;
    float valueMin = 0.f;
    float valueRange = 0.f;
};

struct EncodedBrick
{
    BrickHeader header;
    std::vector<uint8_t> payload;
    Reconstruction reconstruction;
    bool droppable = false; ///< The prediction alone is the best encoding.
    float error = 0.f;      ///< Optical-depth error over the brick span (mean plus a quarter of the maximum, times the span).
    float weight = 1.f;     ///< Visibility weight of the error in the rate-distortion cost.
};

/// Transform the encoder applies (the package holds only those the GPU decodes; CDF 5/3 is an experiment for dry runs).
enum class Transform
{
    Identity,
    Haar,
    Cdf53,
    Legacy, ///< The v3 codec (predicted, offset, 2, 4 or 8 bits within an error tolerance), for comparison in dry runs.
};

struct CodecSettings
{
    float lambda = 1e-4f;          ///< Weighted optical-depth error one byte is worth.
    float maxError = 0.f;          ///< When positive, no brick's (unweighted) error exceeds it.
    float deadzone = 0.2f;         ///< Quantiser rounding offset towards zero (0: round to nearest).
    float weightFloor = 0.01f;     ///< Least visibility weight of a brick's error (1: unweighted).
    float densityScale = 1.f;      ///< Extinction per unit of density per VDB world unit, for visibility and transmittance.
    Transform transform = Transform::Haar;
};

/// CDF 5/3 lifting step (symmetric extension), scaled to be near orthonormal: first half averages, second half details.
void cdf53Step(float* v, uint32_t offset, uint32_t stride, uint32_t n, bool inverse)
{
    float x[8];
    const uint32_t h = n / 2;
    if (!inverse)
    {
        for (uint32_t k = 0; k < n; ++k)
            x[k] = v[offset + k * stride];
        float s[4], d[4];
        for (uint32_t i = 0; i < h; ++i)
            d[i] = x[2 * i + 1] - 0.5f * (x[2 * i] + x[std::min(2 * i + 2, n - 2)]);
        for (uint32_t i = 0; i < h; ++i)
            s[i] = x[2 * i] + 0.25f * (d[i == 0 ? 0 : i - 1] + d[i]);
        for (uint32_t i = 0; i < h; ++i)
        {
            v[offset + i * stride] = s[i] * 1.41421356f;
            v[offset + (h + i) * stride] = d[i] * 0.70710678f;
        }
        return;
    }
    float s[4], d[4];
    for (uint32_t i = 0; i < h; ++i)
    {
        s[i] = v[offset + i * stride] * 0.70710678f;
        d[i] = v[offset + (h + i) * stride] * 1.41421356f;
    }
    for (uint32_t i = 0; i < h; ++i)
        x[2 * i] = s[i] - 0.25f * (d[i == 0 ? 0 : i - 1] + d[i]);
    for (uint32_t i = 0; i < h; ++i)
        x[2 * i + 1] = d[i] + 0.5f * (x[2 * i] + x[std::min(2 * i + 2, n - 2)]);
    for (uint32_t k = 0; k < n; ++k)
        v[offset + k * stride] = x[k];
}

void cdf53Transform(float v[kCoreValues], bool inverse)
{
    auto level = [&](uint32_t n)
    {
        auto axisX = [&] { for (uint32_t z = 0; z < n; ++z) for (uint32_t y = 0; y < n; ++y) cdf53Step(v, 8 * (y + 8 * z), 1, n, inverse); };
        auto axisY = [&] { for (uint32_t z = 0; z < n; ++z) for (uint32_t x = 0; x < n; ++x) cdf53Step(v, x + 64 * z, 8, n, inverse); };
        auto axisZ = [&] { for (uint32_t y = 0; y < n; ++y) for (uint32_t x = 0; x < n; ++x) cdf53Step(v, x + 8 * y, 64, n, inverse); };
        if (!inverse)
        {
            axisX();
            axisY();
            axisZ();
        }
        else
        {
            axisZ();
            axisY();
            axisX();
        }
    };
    if (!inverse)
        for (uint32_t n = 8; n >= 2; n /= 2)
            level(n);
    else
        for (uint32_t n = 2; n <= 8; n *= 2)
            level(n);
}

/// reconstructBrick without a residual, computed separably: a child value at local index i (apron included) interpolates parent
/// texels 4 parity + i / 2 and the next with weights 3/4 and 1/4 for even i, 1/4 and 3/4 for odd i, per axis.
void predictBrick(const Reconstruction* parent, uint3 parity, float values[kBrickValues])
{
    if (!parent)
    {
        std::fill_n(values, kBrickValues, 0.f);
        return;
    }
    float decoded[256];
    for (uint32_t c = 0; c < 256; ++c)
        decoded[c] = atlasValue(uint8_t(c), parent->valueMin, parent->valueRange);
    const uint3 o = parity * 4u;
    const float next[2] = {0.25f, 0.75f};
    // Along z over the parent's 6^3 texels the brick covers, then y, then x.
    float alongZ[6 * 6 * 10];
    for (uint32_t a = 0; a < 6; ++a)
        for (uint32_t b = 0; b < 6; ++b)
        {
            const uint32_t column = (o.x + a) + 10 * (o.y + b);
            for (uint32_t z = 0; z < 10; ++z)
            {
                const uint32_t pz = o.z + z / 2;
                const float f = next[z & 1];
                alongZ[a + 6 * (b + 6 * z)] =
                    (1.f - f) * decoded[parent->codes[column + 100 * pz]] + f * decoded[parent->codes[column + 100 * (pz + 1)]];
            }
        }
    float alongY[6 * 10 * 10];
    for (uint32_t z = 0; z < 10; ++z)
        for (uint32_t y = 0; y < 10; ++y)
        {
            const float f = next[y & 1];
            for (uint32_t a = 0; a < 6; ++a)
                alongY[a + 6 * (y + 10 * z)] = (1.f - f) * alongZ[a + 6 * (y / 2 + 6 * z)] + f * alongZ[a + 6 * (y / 2 + 1 + 6 * z)];
        }
    for (uint32_t z = 0; z < 10; ++z)
        for (uint32_t y = 0; y < 10; ++y)
        {
            const float* row = alongY + 6 * (y + 10 * z);
            for (uint32_t x = 0; x < 10; ++x)
            {
                const float f = next[x & 1];
                values[x + 10 * (y + 10 * z)] = std::max(0.f, (1.f - f) * row[x / 2] + f * row[x / 2 + 1]);
            }
        }
}

/// The v3 codec for comparisons (dry runs only; its payload is not decodable by the v4 runtime): the cheapest of the prediction,
/// a constant offset, and the residual's range quantised to 2, 4 or 8 bits whose unweighted error is within codec.maxError (the
/// tolerance). Payload bytes and errors are what v3 stored and reconstructed.
EncodedBrick encodeBrickLegacy(
    uint32_t coord,
    uint32_t level,
    uint8_t childMask,
    const Core& truth,
    const Reconstruction* parent,
    float span,
    float weight,
    const CodecSettings& codec
)
{
    const uint3 parity = BrickHeader{coord}.brick() % 2u;
    const uint8_t* parentCodes = parent ? parent->codes.data() : nullptr;
    float prediction[kBrickValues];
    reconstructBrick(parentCodes, parent ? parent->valueMin : 0.f, parent ? parent->valueRange : 0.f, parity, nullptr, prediction);
    Core residual;
    float residualMin = std::numeric_limits<float>::max();
    float residualMax = -std::numeric_limits<float>::max();
    double residualSum = 0.0;
    for (uint32_t z = 0; z < 8; ++z)
        for (uint32_t y = 0; y < 8; ++y)
            for (uint32_t x = 0; x < 8; ++x)
            {
                const uint32_t i = x + 8 * (y + 8 * z);
                residual[i] = truth[i] - prediction[(x + 1) + 10 * ((y + 1) + 10 * (z + 1))];
                residualMin = std::min(residualMin, residual[i]);
                residualMax = std::max(residualMax, residual[i]);
                residualSum += residual[i];
            }
    auto finish = [&](EncodedBrick& candidate, const float* spatial)
    {
        float values[kBrickValues];
        std::copy_n(prediction, kBrickValues, values);
        if (spatial)
            for (uint32_t z = 0; z < 8; ++z)
                for (uint32_t y = 0; y < 8; ++y)
                    for (uint32_t x = 0; x < 8; ++x)
                    {
                        float& v = values[(x + 1) + 10 * ((y + 1) + 10 * (z + 1))];
                        v = std::max(0.f, v + spatial[x + 8 * (y + 8 * z)]);
                    }
        const auto [lo, hi] = std::minmax_element(values, values + kBrickValues);
        Reconstruction& r = candidate.reconstruction;
        r.valueMin = *lo;
        r.valueRange = *hi - *lo;
        for (uint32_t i = 0; i < kBrickValues; ++i)
            r.codes[i] = atlasCode(values[i], r.valueMin, r.valueRange);
        candidate.header.valueMin = r.valueMin;
        candidate.header.valueRange = r.valueRange;
        double sum = 0.0;
        float maximum = 0.f;
        for (uint32_t z = 0; z < 8; ++z)
            for (uint32_t y = 0; y < 8; ++y)
                for (uint32_t x = 0; x < 8; ++x)
                {
                    const float error =
                        std::abs(atlasValue(r.codes[(x + 1) + 10 * ((y + 1) + 10 * (z + 1))], r.valueMin, r.valueRange) - truth[x + 8 * (y + 8 * z)]);
                    sum += error;
                    maximum = std::max(maximum, error);
                }
        candidate.error = (float(sum / kCoreValues) + 0.25f * maximum) * span;
        candidate.weight = weight;
    };
    EncodedBrick best;
    best.header.coord = coord;
    best.header.level = uint8_t(level);
    best.header.childMask = childMask;
    best.header.transform = 0xFF; // Not a v4 transform: its payload is v3's packed bits.
    best.header.flags = kBrickPredicted;
    best.droppable = true;
    finish(best, nullptr);
    if (best.error <= codec.maxError)
        return best;
    for (uint32_t bits : {0u, 2u, 4u, 8u})
    {
        EncodedBrick candidate;
        candidate.header = best.header;
        candidate.header.flags = 0;
        candidate.droppable = false;
        float spatial[kCoreValues];
        if (bits == 0)
        {
            std::fill_n(spatial, kCoreValues, float(residualSum / kCoreValues));
            candidate.payload.clear();
        }
        else
        {
            const uint32_t levels = (1u << bits) - 1u;
            const float step = (residualMax - residualMin) / float(levels);
            candidate.payload.assign((kCoreValues * bits + 7) / 8, 0);
            for (uint32_t i = 0; i < kCoreValues; ++i)
            {
                const uint32_t q = step > 0.f ? std::min(levels, uint32_t(std::lround((residual[i] - residualMin) / step))) : 0u;
                spatial[i] = residualMin + step * float(q);
                const uint32_t bit = i * bits;
                candidate.payload[bit / 8] |= uint8_t(q << (bit % 8));
                if (bit % 8 + bits > 8)
                    candidate.payload[bit / 8 + 1] |= uint8_t(q >> (8 - bit % 8));
            }
        }
        finish(candidate, spatial);
        if (candidate.error <= codec.maxError || bits == 8)
            return candidate;
    }
    return best;
}

/// Rate-distortion encoding of one brick: the prediction alone, or the residual's transform coefficients quantised with the step
/// that minimises weight * error + lambda * bytes (a coarse-to-fine search over steps a factor sqrt(2) apart). Errors are measured
/// on the true means after the atlas quantisation the GPU applies, exactly as the GPU will reconstruct them.
EncodedBrick encodeBrick(
    uint32_t coord,
    uint32_t level,
    uint8_t childMask,
    const Core& truth,
    const Reconstruction* parent,
    float span,
    float weight,
    const CodecSettings& codec
)
{
    if (codec.transform == Transform::Legacy)
        return encodeBrickLegacy(coord, level, childMask, truth, parent, span, weight, codec);
    BrickHeader base;
    base.coord = coord;
    base.level = uint8_t(level);
    base.childMask = childMask;
    base.transform = codec.transform == Transform::Identity ? kTransformIdentity : kTransformHaar;
    float prediction[kBrickValues];
    predictBrick(parent, BrickHeader{coord}.brick() % 2u, prediction);
    float coefficients[kCoreValues];
    for (uint32_t z = 0; z < 8; ++z)
        for (uint32_t y = 0; y < 8; ++y)
            for (uint32_t x = 0; x < 8; ++x)
                coefficients[x + 8 * (y + 8 * z)] = truth[x + 8 * (y + 8 * z)] - prediction[(x + 1) + 10 * ((y + 1) + 10 * (z + 1))];
    if (codec.transform == Transform::Haar)
        haarForward(coefficients);
    else if (codec.transform == Transform::Cdf53)
        cdf53Transform(coefficients, false);
    float largest = 0.f;
    for (float c : coefficients)
        largest = std::max(largest, std::abs(c));

    // Reconstruction of a residual (none: the prediction) into the atlas quantisation, as reconstructBrick and the GPU do: its
    // values' range, and optionally every atlas code. Returns the error against the truth.
    float values[kBrickValues];
    auto reconstruct = [&](const float* residual, float& valueMin, float& valueRange, uint8_t* codes)
    {
        std::copy_n(prediction, kBrickValues, values);
        if (residual)
            for (uint32_t z = 0; z < 8; ++z)
                for (uint32_t y = 0; y < 8; ++y)
                {
                    float* row = values + 1 + 10 * ((y + 1) + 10 * (z + 1));
                    const float* r = residual + 8 * (y + 8 * z);
                    for (uint32_t x = 0; x < 8; ++x)
                        row[x] = std::max(0.f, row[x] + r[x]);
                }
        float lo = values[0], hi = values[0];
        for (float v : values)
        {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        valueMin = lo;
        valueRange = hi - lo;
        const float inverseRange = valueRange > 0.f ? 1.f / valueRange : 0.f;
        auto code = [&](float v) { return valueRange > 0.f ? uint8_t(std::sqrt(std::clamp((v - lo) * inverseRange, 0.f, 1.f)) * 255.f + 0.5f) : uint8_t(0); };
        if (codes)
            for (uint32_t i = 0; i < kBrickValues; ++i)
                codes[i] = code(values[i]);
        float sum = 0.f;
        float maximum = 0.f;
        const float scale = valueRange / (255.f * 255.f);
        for (uint32_t z = 0; z < 8; ++z)
            for (uint32_t y = 0; y < 8; ++y)
            {
                const float* row = values + 1 + 10 * ((y + 1) + 10 * (z + 1));
                const float* t = truth.data() + 8 * (y + 8 * z);
                for (uint32_t x = 0; x < 8; ++x)
                {
                    const float c = float(code(row[x]));
                    const float error = std::abs(lo + scale * c * c - t[x]);
                    sum += error;
                    maximum = std::max(maximum, error);
                }
            }
        return (sum / float(kCoreValues) + 0.25f * maximum) * span;
    };
    auto cost = [&](float error, uint32_t bytes)
    {
        const bool allowed = codec.maxError <= 0.f || error <= codec.maxError;
        return (allowed ? 0.0 : 1e30) + double(weight * error) + double(codec.lambda) * double(bytes);
    };

    float valueMin = 0.f, valueRange = 0.f;
    float bestError = reconstruct(nullptr, valueMin, valueRange, nullptr);
    double bestCost = cost(bestError, 0);
    int bestK = 0; // 0: the prediction.

    // Step k: largest * 2^(-k / 2). A candidate's bytes are counted without packing; only the chosen one is packed.
    const float bias = 0.5f - codec.deadzone;
    int32_t q[kCoreValues];
    float residual[kCoreValues];
    auto quantise = [&](int k)
    {
        const float step = largest * std::pow(2.f, -0.5f * float(k));
        const float inverseStep = 1.f / step;
        uint32_t bytes = 64;
        for (uint32_t i = 0; i < kCoreValues; ++i)
        {
            const int32_t magnitude = std::min(int32_t(std::abs(coefficients[i]) * inverseStep + bias), 1 << 20);
            const int32_t value = coefficients[i] < 0.f ? -magnitude : magnitude;
            q[i] = value;
            residual[i] = float(value) * step;
            const uint32_t u = (uint32_t(value) << 1) ^ uint32_t(value >> 31);
            bytes += magnitude == 0 ? 0 : u < 0x80 ? 1 : u < 0x4000 ? 2 : 3;
        }
        if (codec.transform == Transform::Haar)
            haarInverse(residual);
        else if (codec.transform == Transform::Cdf53)
            cdf53Transform(residual, true);
        return (bytes + 3) & ~3u;
    };
    std::array<double, 48> tried;
    tried.fill(-1.0);
    auto evaluate = [&](int k)
    {
        k = std::clamp(k, 1, int(tried.size()) - 1);
        if (tried[k] >= 0.0)
            return tried[k];
        const uint32_t bytes = quantise(k);
        float candidateMin, candidateRange;
        const float error = reconstruct(residual, candidateMin, candidateRange, nullptr);
        const double c = cost(error, uint32_t(sizeof(BrickHeader)) + bytes);
        tried[k] = c;
        if (c < bestCost)
        {
            bestCost = c;
            bestK = k;
            bestError = error;
        }
        return c;
    };
    if (largest > 0.f)
    {
        int centreK = 1;
        double coarse = std::numeric_limits<double>::max();
        for (int k = 1; k < int(tried.size()); k += 6)
        {
            const double c = evaluate(k);
            if (c < coarse)
            {
                coarse = c;
                centreK = k;
            }
        }
        for (int delta : {3, 1})
        {
            const int centre = centreK;
            for (int k : {centre - delta, centre + delta})
                if (evaluate(k) <= tried[std::clamp(centreK, 1, int(tried.size()) - 1)])
                    centreK = std::clamp(k, 1, int(tried.size()) - 1);
        }
        // The error cap holds even when no step's cost fits it: the finest step reconstructs within the atlas quantisation.
        if (codec.maxError > 0.f && bestError > codec.maxError)
            evaluate(int(tried.size()) - 1);
    }

    EncodedBrick best;
    best.header = base;
    best.weight = weight;
    best.error = bestError;
    Reconstruction& r = best.reconstruction;
    if (bestK == 0)
    {
        best.header.flags = kBrickPredicted;
        best.droppable = true;
        reconstruct(nullptr, r.valueMin, r.valueRange, r.codes.data());
    }
    else
    {
        best.header.step = largest * std::pow(2.f, -0.5f * float(bestK));
        best.header.payloadBytes = quantise(bestK);
        packCoefficients(q, best.payload);
        if (best.payload.size() != best.header.payloadBytes)
            throw std::logic_error("encodeBrick counted packed coefficient bytes wrongly");
        reconstruct(residual, r.valueMin, r.valueRange, r.codes.data());
    }
    best.header.valueMin = r.valueMin;
    best.header.valueRange = r.valueRange;
    return best;
}

/// One source cloud during compilation.
class Asset
{
public:
    using ChunkLevels = std::array<Level, 4>;

    std::filesystem::path source;
    AssetEntry entry;
    uint64_t signature = 0;
    uint64_t densityHash = 0;
    uint64_t sourceBytes = 0;

    void load()
    {
        openvdb::initialize();
        openvdb::io::File vdb(source.string());
        vdb.open();
        if (vdb.hasGrid("density"))
            mGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(vdb.readGrid("density"));
        for (auto name = vdb.beginName(); !mGrid && name != vdb.endName(); ++name)
            mGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(vdb.readGrid(name.gridName()));
        if (!mGrid)
            throw std::runtime_error("no float grid in " + source.string());
        vdb.close();
        entry.voxelWorld = float(mGrid->voxelSize()[0]);
        const openvdb::CoordBBox bbox = mGrid->evalActiveVoxelBoundingBox();
        auto floorTo = [](int32_t v) { return v >= 0 ? v / 128 * 128 : -((-v + 127) / 128) * 128; };
        const int3 lo(floorTo(bbox.min().x()), floorTo(bbox.min().y()), floorTo(bbox.min().z()));
        const int3 hi(floorTo(bbox.max().x() + 128), floorTo(bbox.max().y() + 128), floorTo(bbox.max().z() + 128));
        mOrigin = lo;
        mDims = uint3(hi - lo);
        if (any(mDims > uint3(8192)))
            throw std::runtime_error("cloud larger than 8192 voxels per axis: " + source.string());
        mTopLevel = kChunkLevel;
        while (any(mDims > uint3(8u << mTopLevel)))
            ++mTopLevel;
        mProxyLevel = 0;
        while (any(mDims / (1u << mProxyLevel) > uint3(64)))
            ++mProxyLevel;
        mProxyDims = mDims / (1u << mProxyLevel);
        mChunkDims = mDims / kChunkVoxels;
        for (uint32_t i = 0; i < 3; ++i)
        {
            entry.sourceMin[i] = mOrigin[i];
            entry.dims[i] = mDims[i];
            entry.proxyDims[i] = mProxyDims[i];
        }
        entry.topLevel = mTopLevel;
        entry.proxyLevel = mProxyLevel;
        const std::string name = source.stem().string();
        std::copy_n(name.c_str(), std::min<size_t>(name.size(), sizeof(entry.name) - 1), entry.name);

        // Level-0 bricks: leaf nodes and the bricks active tiles cover, grouped by chunk.
        std::vector<uint32_t> bricks;
        const openvdb::Coord origin(lo.x, lo.y, lo.z);
        const auto& tree = mGrid->tree();
        for (auto leaf = tree.cbeginLeaf(); leaf; ++leaf)
        {
            const openvdb::Coord o = leaf->origin() - origin;
            bricks.push_back(BrickHeader::pack(uint3(o.x(), o.y(), o.z()) / 8u));
        }
        auto tile = tree.cbeginValueOn();
        tile.setMaxDepth(tile.getLeafDepth() - 1);
        for (; tile; ++tile)
        {
            if (!tile.isTileValue() || !(*tile > 0.f))
                continue;
            const openvdb::CoordBBox box = tile.getBoundingBox();
            const uint3 b0 = uint3(max(int3(box.min().x(), box.min().y(), box.min().z()) - lo, int3(0))) / 8u;
            const uint3 b1 = min(uint3(max(int3(box.max().x(), box.max().y(), box.max().z()) - lo, int3(0))) / 8u, mDims / 8u - 1u);
            for (uint32_t z = b0.z; z <= b1.z; ++z)
                for (uint32_t y = b0.y; y <= b1.y; ++y)
                    for (uint32_t x = b0.x; x <= b1.x; ++x)
                        bricks.push_back(BrickHeader::pack(uint3(x, y, z)));
        }
        std::sort(bricks.begin(), bricks.end(), [&](uint32_t a, uint32_t b) { return chunkOf(a) != chunkOf(b) ? chunkOf(a) < chunkOf(b) : a < b; });
        bricks.erase(std::unique(bricks.begin(), bricks.end()), bricks.end());
        mChunks.clear();
        for (size_t i = 0; i < bricks.size();)
        {
            size_t j = i;
            while (j < bricks.size() && chunkOf(bricks[j]) == chunkOf(bricks[i]))
                ++j;
            mChunks.push_back({chunkOf(bricks[i]), std::vector<uint32_t>(bricks.begin() + i, bricks.begin() + j)});
            i = j;
        }
        entry.level0Bricks = bricks.size();
    }

    void unload()
    {
        mGrid.reset();
        mCoarse.clear();
        mCached.clear();
        mIsCached.clear();
        mChunks.clear();
        mChunkBlobs.clear();
        mProxy.clear();
        mExposure.clear();
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            mDepthBefore[axis].clear();
            mDepthAfter[axis].clear();
        }
    }

    /// Stage 1: sweeps the loaded source into its canonical cache (written to a temporary file, renamed when complete).
    void writeCanonical(const std::filesystem::path& path)
    {
        const std::filesystem::path temporary = path.string() + ".partial";
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        CanonicalHeader header;
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        std::vector<CanonicalChunk> table(mChunks.size());
        std::mutex writeMutex;
        sweep(
            [&](size_t c, const ChunkLevels& levels)
            {
                const std::vector<uint8_t> raw = packCores(levels[0]);
                std::vector<uint8_t> compressed;
                if (!deflateBytes(raw.data(), raw.size(), compressed))
                    throw std::runtime_error("deflate failed");
                std::lock_guard lock(writeMutex);
                table[c].chunk = mChunks[c].chunk;
                table[c].bricks = uint32_t(levels[0].coords.size());
                table[c].blob.offset = uint64_t(file.tellp());
                table[c].blob.compressed = uint32_t(compressed.size());
                table[c].blob.raw = uint32_t(raw.size());
                file.write(reinterpret_cast<const char*>(compressed.data()), std::streamsize(compressed.size()));
            }
        );
        header.chunkCount = uint32_t(table.size());
        header.sourceSignature = signature;
        header.sourceBytes = sourceBytes;
        header.densityHash = densityHash;
        CanonicalGeometry& g = header.geometry;
        std::copy(std::begin(entry.name), std::end(entry.name), g.name);
        std::copy_n(entry.sourceMin, 3, g.sourceMin);
        std::copy_n(entry.dims, 3, g.dims);
        std::copy_n(entry.proxyDims, 3, g.proxyDims);
        g.voxelWorld = entry.voxelWorld;
        g.topLevel = entry.topLevel;
        g.proxyLevel = entry.proxyLevel;
        g.level0Bricks = entry.level0Bricks;
        header.proxy = appendBlob(file, mProxy);
        std::vector<uint8_t> coarse;
        auto put = [&](const uint8_t* data, size_t size) { coarse.insert(coarse.end(), data, data + size); };
        for (uint32_t level = kChunkLevel; level <= mTopLevel; ++level)
        {
            const Level& means = mCoarse[level];
            const uint32_t count = uint32_t(means.coords.size());
            put(reinterpret_cast<const uint8_t*>(&count), sizeof(count));
            put(reinterpret_cast<const uint8_t*>(means.coords.data()), count * sizeof(uint32_t));
            put(means.childMask.data(), count);
            put(reinterpret_cast<const uint8_t*>(means.cores.data()), count * sizeof(Core));
        }
        header.coarse = appendBlob(file, coarse);
        header.chunkTableOffset = uint64_t(file.tellp());
        file.write(reinterpret_cast<const char*>(table.data()), std::streamsize(table.size() * sizeof(CanonicalChunk)));
        file.seekp(0);
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        file.close();
        if (!file)
            throw std::runtime_error("failed to write " + temporary.string());
        std::filesystem::rename(temporary, path);
    }

    /// Stage 2: opens a canonical cache in place of the source; chunk means are read from it on demand.
    void open(const std::filesystem::path& path)
    {
        CanonicalHeader header;
        if (!readCanonicalHeader(path, header))
            throw std::runtime_error("not a canonical cloud cache of this version: " + path.string());
        mCachePath = path;
        const CanonicalGeometry& g = header.geometry;
        entry = AssetEntry();
        std::copy(std::begin(g.name), std::end(g.name), entry.name);
        std::copy_n(g.sourceMin, 3, entry.sourceMin);
        std::copy_n(g.dims, 3, entry.dims);
        std::copy_n(g.proxyDims, 3, entry.proxyDims);
        entry.voxelWorld = g.voxelWorld;
        entry.topLevel = g.topLevel;
        entry.proxyLevel = g.proxyLevel;
        entry.level0Bricks = g.level0Bricks;
        signature = header.sourceSignature;
        densityHash = header.densityHash;
        mOrigin = int3(entry.sourceMin[0], entry.sourceMin[1], entry.sourceMin[2]);
        mDims = uint3(entry.dims[0], entry.dims[1], entry.dims[2]);
        mChunkDims = mDims / kChunkVoxels;
        mProxyDims = uint3(entry.proxyDims[0], entry.proxyDims[1], entry.proxyDims[2]);
        mTopLevel = entry.topLevel;
        mProxyLevel = entry.proxyLevel;
        std::ifstream file(path, std::ios::binary);
        std::vector<CanonicalChunk> table(header.chunkCount);
        file.seekg(std::streamoff(header.chunkTableOffset));
        if (!file.read(reinterpret_cast<char*>(table.data()), std::streamsize(table.size() * sizeof(CanonicalChunk))))
            throw std::runtime_error("truncated canonical cache: " + path.string());
        mChunks.clear();
        mChunkBlobs.clear();
        for (const CanonicalChunk& chunk : table)
        {
            mChunks.push_back({chunk.chunk, {}});
            mChunkBlobs.push_back(chunk.blob);
        }
        mProxy = readBlobFrom(file, header.proxy);
        const std::vector<uint8_t> coarse = readBlobFrom(file, header.coarse);
        const uint8_t* p = coarse.data();
        const uint8_t* end = coarse.data() + coarse.size();
        auto take = [&](uint8_t* data, size_t size)
        {
            if (p + size > end)
                throw std::runtime_error("corrupt canonical coarse levels: " + path.string());
            std::memcpy(data, p, size);
            p += size;
        };
        mCoarse.assign(mTopLevel + 1, Level());
        for (uint32_t level = kChunkLevel; level <= mTopLevel; ++level)
        {
            uint32_t count = 0;
            take(reinterpret_cast<uint8_t*>(&count), sizeof(count));
            std::vector<uint32_t> coords(count);
            take(reinterpret_cast<uint8_t*>(coords.data()), count * sizeof(uint32_t));
            Level& means = mCoarse[level];
            for (uint32_t coord : coords)
                means.add(coord);
            take(means.childMask.data(), count);
            take(reinterpret_cast<uint8_t*>(means.cores.data()), count * sizeof(Core));
        }
        computeVisibility();
    }

    size_t chunkCount() const { return mChunks.size(); }

    /// Level-4 brick coordinate of chunk c.
    uint32_t chunkCoord(size_t c) const
    {
        const uint32_t chunk = mChunks[c].chunk;
        return BrickHeader::pack(uint3(chunk % mChunkDims.x, (chunk / mChunkDims.x) % mChunkDims.y, chunk / (mChunkDims.x * mChunkDims.y)));
    }

    /// Reads and caches the means of some chunks (the others stay uncached and empty).
    void cacheChunks(const std::vector<size_t>& chunks)
    {
        mCached.resize(mChunks.size());
        mIsCached.resize(mChunks.size(), 0);
        parallelFor(chunks.size(), [&](size_t i) { readChunk(chunks[i], mCached[chunks[i]], nullptr, nullptr); });
        for (size_t c : chunks)
            mIsCached[c] = 1;
    }

    /// The means of chunk c when they were cached, else null (they are read from the cache file).
    const ChunkLevels* cachedLevels(size_t c) const { return c < mIsCached.size() && mIsCached[c] ? &mCached[c] : nullptr; }

    /// Sweep over every chunk of the loaded source: level-4 means, the proxies and the density hash; each chunk's levels 0-3 go to
    /// the sink (called in parallel).
    void sweep(const std::function<void(size_t, const ChunkLevels&)>& sink)
    {
        const auto start = std::chrono::steady_clock::now();
        std::vector<float> proxyMean(size_t(mProxyDims.x) * mProxyDims.y * mProxyDims.z, 0.f);
        std::vector<float> proxyMax(proxyMean.size(), 0.f);
        std::vector<Core> level4(mChunks.size());
        std::vector<uint8_t> level4Mask(mChunks.size(), 0);
        std::vector<uint64_t> hashes(mChunks.size(), 0);
        std::mutex mutex;
        parallelFor(
            mChunks.size(),
            [&](size_t c)
            {
                ChunkLevels levels;
                std::vector<std::pair<uint32_t, float>> maxima;
                readChunk(c, levels, &maxima, &hashes[c]);
                sink(c, levels);
                level4[c].fill(0.f);
                for (size_t i = 0; i < levels[3].coords.size(); ++i)
                {
                    const uint3 b = BrickHeader{levels[3].coords[i]}.brick();
                    level4Mask[c] |= uint8_t(1u << ((b.x & 1u) + 2u * (b.y & 1u) + 4u * (b.z & 1u)));
                    const uint3 o = (b % 2u) * 8u;
                    for (uint32_t z = 0; z < 8; ++z)
                        for (uint32_t y = 0; y < 8; ++y)
                            for (uint32_t x = 0; x < 8; ++x)
                            {
                                const uint3 q = (o + uint3(x, y, z)) / 2u;
                                level4[c][q.x + 8 * (q.y + 8 * q.z)] += 0.125f * levels[3].cores[i][x + 8 * (y + 8 * z)];
                            }
                }
                std::lock_guard lock(mutex);
                if (mProxyLevel <= 3)
                    for (size_t i = 0; i < levels[mProxyLevel].coords.size(); ++i)
                        writeProxy(proxyMean, BrickHeader{levels[mProxyLevel].coords[i]}.brick(), levels[mProxyLevel].cores[i]);
                for (const auto& [coord, value] : maxima)
                {
                    // Proxy voxels overlapping the brick's values with the trilinear apron.
                    const uint3 b = BrickHeader{coord}.brick();
                    const uint3 plo = uint3(max(int3(b * 8u) - 1, int3(0))) / (1u << mProxyLevel);
                    const uint3 phi = min(uint3(int3(b * 8u) + 8) / (1u << mProxyLevel), mProxyDims - 1u);
                    for (uint32_t z = plo.z; z <= phi.z; ++z)
                        for (uint32_t y = plo.y; y <= phi.y; ++y)
                            for (uint32_t x = plo.x; x <= phi.x; ++x)
                            {
                                float& m = proxyMax[x + mProxyDims.x * (size_t(y) + mProxyDims.y * size_t(z))];
                                m = std::max(m, value);
                            }
                }
            }
        );
        densityHash = 1469598103934665603ull;
        for (uint64_t h : hashes)
            densityHash = fnv(densityHash, h);
        densityHash = fnv(densityHash, fnv(mDims.x, fnv(mDims.y, mDims.z)));

        // Levels 4 to top.
        mCoarse.assign(mTopLevel + 1, Level());
        for (size_t c = 0; c < mChunks.size(); ++c)
        {
            const uint3 chunk(mChunks[c].chunk % mChunkDims.x, (mChunks[c].chunk / mChunkDims.x) % mChunkDims.y, mChunks[c].chunk / (mChunkDims.x * mChunkDims.y));
            const uint32_t i = mCoarse[kChunkLevel].add(BrickHeader::pack(chunk));
            mCoarse[kChunkLevel].cores[i] = level4[c];
            mCoarse[kChunkLevel].childMask[i] = level4Mask[c];
        }
        for (uint32_t level = kChunkLevel; level < mTopLevel; ++level)
            for (size_t i = 0; i < mCoarse[level].coords.size(); ++i)
                accumulate(mCoarse[level + 1], mCoarse[level].coords[i], mCoarse[level].cores[i]);
        if (mProxyLevel > 3)
            for (size_t i = 0; i < mCoarse[mProxyLevel].coords.size(); ++i)
                writeProxy(proxyMean, BrickHeader{mCoarse[mProxyLevel].coords[i]}.brick(), mCoarse[mProxyLevel].cores[i]);

        mProxy.resize(proxyMean.size() * 2 * sizeof(float16_t));
        float16_t* out = reinterpret_cast<float16_t*>(mProxy.data());
        for (size_t i = 0; i < proxyMean.size(); ++i)
        {
            out[i] = float16_t(proxyMean[i]);
            float16_t rounded(proxyMax[i]);
            if (float(rounded) < proxyMax[i])
                rounded = float16_t(proxyMax[i] * (1.f + 1.f / 1024.f) + 1e-4f);
            out[proxyMean.size() + i] = rounded;
        }
        std::cout << "  sweep: " << mChunks.size() << " chunks, " << entry.level0Bricks << " level-0 bricks in " << seconds(start) << " s\n";
    }

    /// Statistics of stored bricks.
    struct EncodeStats
    {
        uint64_t bricks = 0;
        uint64_t predicted = 0;
        uint64_t nonzero = 0; ///< Non-zero coefficients.
        uint64_t payloadBytes = 0;
        double error = 0.0;
        double weightedError = 0.0;
        float maxError = 0.f;
        double encodeSeconds = 0.0;   ///< Thread seconds encoding bricks.
        double compressSeconds = 0.0; ///< Thread seconds in GDeflate.
        uint64_t metaCompressed = 0;    ///< GDeflate bytes of page metas (brick headers and directories).
        uint64_t payloadCompressed = 0; ///< GDeflate bytes of page payloads.
        /// Quantised coefficients per subband group of the scan (largest coordinate 0, 1, 2-3, 4-7): values -64 to 64, then escapes.
        std::array<std::array<uint64_t, 130>, 4> histogram = {};
        /// Close-up transmittance errors of voxel columns (measureTransmittance), log-spaced from 1e-6 to 1 with 64 bins per decade.
        static constexpr uint32_t kErrorBins = 384;
        std::array<uint64_t, kErrorBins> transmittance = {};
        uint64_t columns = 0;
        double transmittanceSum = 0.0;
        float transmittanceMax = 0.f;

        void addColumn(float value)
        {
            ++columns;
            transmittanceSum += value;
            transmittanceMax = std::max(transmittanceMax, value);
            const int bin = value > 1e-6f ? int((std::log10(value) + 6.f) * 64.f) : 0;
            ++transmittance[std::clamp(bin, 0, int(kErrorBins) - 1)];
        }

        /// The error below which a fraction p of the columns lie (the upper edge of its bin).
        float transmittancePercentile(double p) const
        {
            const uint64_t target = uint64_t(std::ceil(p * double(columns)));
            uint64_t count = 0;
            for (uint32_t b = 0; b < kErrorBins; ++b)
            {
                count += transmittance[b];
                if (count >= target && count > 0)
                    return std::pow(10.f, float(b + 1) / 64.f - 6.f);
            }
            return 0.f;
        }

        void add(const EncodedBrick& brick)
        {
            ++bricks;
            predicted += brick.droppable ? 1 : 0;
            payloadBytes += brick.payload.size();
            if (!brick.payload.empty() && brick.header.transform != 0xFF)
            {
                for (uint32_t i = 0; i < 64; ++i)
                    nonzero += std::bitset<8>(brick.payload[i]).count();
                int32_t q[kCoreValues];
                unpackCoefficients(brick.payload.data(), q);
                const auto& scan = coefficientScan();
                for (uint32_t s = 0; s < kCoreValues; ++s)
                {
                    const uint32_t group = s == 0 ? 0 : s < 8 ? 1 : s < 64 ? 2 : 3;
                    const int32_t value = q[scan[s]];
                    ++histogram[group][std::abs(value) <= 64 ? size_t(value + 64) : 129];
                }
            }
            error += brick.error;
            weightedError += brick.weight * brick.error;
            maxError = std::max(maxError, brick.error);
        }

        void merge(const EncodeStats& other)
        {
            bricks += other.bricks;
            predicted += other.predicted;
            nonzero += other.nonzero;
            payloadBytes += other.payloadBytes;
            error += other.error;
            weightedError += other.weightedError;
            maxError = std::max(maxError, other.maxError);
            encodeSeconds += other.encodeSeconds;
            compressSeconds += other.compressSeconds;
            metaCompressed += other.metaCompressed;
            payloadCompressed += other.payloadCompressed;
            for (size_t g = 0; g < histogram.size(); ++g)
                for (size_t v = 0; v < histogram[g].size(); ++v)
                    histogram[g][v] += other.histogram[g][v];
            for (size_t b = 0; b < kErrorBins; ++b)
                transmittance[b] += other.transmittance[b];
            columns += other.columns;
            transmittanceSum += other.transmittanceSum;
            transmittanceMax = std::max(transmittanceMax, other.transmittanceMax);
        }

        /// Bytes an ideal order-0 coder per subband group would spend on the coefficients (escapes counted as 2 bytes extra).
        double entropyBytes() const
        {
            double bits = 0.0;
            for (const auto& group : histogram)
            {
                double total = 0.0;
                for (uint64_t count : group)
                    total += double(count);
                for (size_t v = 0; v < group.size(); ++v)
                    if (group[v] > 0)
                        bits += double(group[v]) * (-std::log2(double(group[v]) / total) + (v == 129 ? 16.0 : 0.0));
            }
            return bits / 8.0;
        }
    };

    /// A GDeflate-compressed page: meta and payload with their raw sizes.
    struct CompressedPage
    {
        std::vector<uint8_t> meta;
        std::vector<uint8_t> payload;
        uint32_t metaRaw = 0;
        uint32_t payloadRaw = 0;

        uint64_t bytes() const { return meta.size() + payload.size(); }
    };

    static CompressedPage compressPage(const std::vector<uint8_t>& meta, const std::vector<uint8_t>& payload)
    {
        CompressedPage page;
        if (!compressGDeflate(meta.data(), meta.size(), page.meta) ||
            (!payload.empty() && !compressGDeflate(payload.data(), payload.size(), page.payload)))
        {
            char message[128];
            std::snprintf(message, sizeof(message), "GDeflate compression failed: HRESULT 0x%08X", uint32_t(lastGDeflateResult()));
            throw std::runtime_error(message);
        }
        page.metaRaw = uint32_t(meta.size());
        page.payloadRaw = uint32_t(payload.size());
        return page;
    }

    struct CoarseResult
    {
        CompressedPage page;
        std::unordered_map<uint32_t, Reconstruction> level4; ///< By chunk coordinate.
        EncodeStats stats;
    };

    /// Levels top to 4, top-down. Every coarse brick is kept (a predicted one is materialised): they are few.
    CoarseResult encodeCoarse(const CodecSettings& codec) const
    {
        CoarseResult result;
        std::vector<BrickHeader> headers;
        std::vector<uint8_t> payloads;
        std::unordered_map<uint32_t, Reconstruction> parents;
        for (uint32_t level = mTopLevel + 1; level-- > kChunkLevel;)
        {
            const Level& means = mCoarse[level];
            std::vector<EncodedBrick> encoded(means.coords.size());
            parallelFor(
                means.coords.size(),
                [&](size_t i)
                {
                    const uint3 b = BrickHeader{means.coords[i]}.brick();
                    const auto parent = level < mTopLevel ? parents.find(BrickHeader::pack(b / 2u)) : parents.end();
                    encoded[i] = encodeBrick(
                        means.coords[i], level, means.childMask[i], means.cores[i], parent != parents.end() ? &parent->second : nullptr,
                        span(level), brickWeight(level, means.coords[i], codec), codec
                    );
                }
            );
            std::unordered_map<uint32_t, Reconstruction> next;
            for (auto& brick : encoded)
            {
                brick.header.payloadOffset = uint32_t(payloads.size());
                payloads.insert(payloads.end(), brick.payload.begin(), brick.payload.end());
                headers.push_back(brick.header);
                result.stats.add(brick);
                next[brick.header.coord] = brick.reconstruction;
            }
            parents.swap(next);
        }
        result.level4 = std::move(parents);
        result.page = compressPage(makeMeta(headers, {}), payloads);
        return result;
    }

    struct ChunkResult
    {
        std::vector<uint8_t> chunkMeta; ///< Raw: its directory's blob offsets are filled in when the level-2 pages are written.
        CompressedPage chunkPayload;    ///< Only the payload half is used.
        std::vector<std::pair<uint32_t, CompressedPage>> pages; ///< Level-2 coordinate and page.
        EncodeStats stats;

        /// Compressed bytes (the chunk meta estimated with its directory offsets still zero).
        uint64_t bytes() const
        {
            uint64_t sum = chunkPayload.payload.size();
            if (!chunkMeta.empty())
            {
                std::vector<uint8_t> meta;
                compressGDeflate(chunkMeta.data(), chunkMeta.size(), meta);
                sum += meta.size();
            }
            for (const auto& page : pages)
                sum += page.second.bytes();
            return sum;
        }
    };

    /// Levels 3 to 0 of one chunk, top-down from its level-4 reconstruction; droppable bricks without stored descendants vanish.
    /// Without pages, only the chunk's transmittance error is measured (for the quality search).
    ChunkResult encodeChunk(size_t c, const CodecSettings& codec, const Reconstruction& level4, bool parallelLevels, bool pages = true) const
    {
        const ChunkLevels* cached = cachedLevels(c);
        ChunkLevels read;
        if (!cached)
            readChunk(c, read, nullptr, nullptr);
        const ChunkLevels& levels = cached ? *cached : read;
        std::array<std::vector<EncodedBrick>, 4> encoded;
        std::array<const std::unordered_map<uint32_t, uint32_t>*, 4> lookup;
        for (uint32_t level = 0; level < 4; ++level)
            lookup[level] = &levels[level].index;
        const auto encodeStart = std::chrono::steady_clock::now();
        for (uint32_t level = 4; level-- > 0;)
        {
            const Level& means = levels[level];
            encoded[level].resize(means.coords.size());
            // Siblings only read their parents: a level can encode in parallel.
            auto encode = [&](size_t i)
            {
                const uint3 b = BrickHeader{means.coords[i]}.brick();
                const Reconstruction* parent = &level4;
                if (level < 3)
                    parent = &encoded[level + 1][lookup[level + 1]->at(BrickHeader::pack(b / 2u))].reconstruction;
                encoded[level][i] = encodeBrick(
                    means.coords[i], level, means.childMask[i], means.cores[i], parent, span(level), brickWeight(level, means.coords[i], codec), codec
                );
            };
            if (parallelLevels)
                parallelFor(means.coords.size(), encode);
            else
                for (size_t i = 0; i < means.coords.size(); ++i)
                    encode(i);
        }
        const double encodeSeconds = seconds(encodeStart);
        ChunkResult result;
        result.stats.encodeSeconds = encodeSeconds;
        measureTransmittance(c, levels[0], encoded[0], codec, result.stats);
        if (!pages)
            return result;
        // Bottom-up: a brick is kept if it is needed itself or a descendant is kept.
        std::array<std::vector<uint8_t>, 4> kept;
        for (uint32_t level = 0; level < 4; ++level)
            kept[level].assign(encoded[level].size(), 0);
        for (uint32_t level = 0; level < 4; ++level)
        {
            for (size_t i = 0; i < encoded[level].size(); ++i)
                kept[level][i] |= encoded[level][i].droppable ? 0 : 1;
            if (level + 1 < 4)
                for (size_t i = 0; i < encoded[level].size(); ++i)
                    if (kept[level][i])
                        kept[level + 1][lookup[level + 1]->at(BrickHeader::pack(encoded[level][i].header.brick() / 2u))] = 1;
        }
        auto timedCompress = [&](const std::vector<uint8_t>& meta, const std::vector<uint8_t>& payload)
        {
            const auto start = std::chrono::steady_clock::now();
            CompressedPage page = compressPage(meta, payload);
            result.stats.compressSeconds += seconds(start);
            result.stats.metaCompressed += page.meta.size();
            result.stats.payloadCompressed += page.payload.size();
            return page;
        };
        auto emit =[&](EncodedBrick& brick, std::vector<BrickHeader>& headers, std::vector<uint8_t>& payloads)
        {
            brick.header.payloadOffset = uint32_t(payloads.size());
            payloads.insert(payloads.end(), brick.payload.begin(), brick.payload.end());
            headers.push_back(brick.header);
            result.stats.add(brick);
        };
        // Level-2 pages with their kept descendants.
        for (size_t i2 = 0; i2 < encoded[2].size(); ++i2)
        {
            if (!kept[2][i2])
                continue;
            std::vector<BrickHeader> headers;
            std::vector<uint8_t> payloads;
            const uint3 b2 = encoded[2][i2].header.brick();
            emit(encoded[2][i2], headers, payloads);
            for (uint32_t c1 = 0; c1 < 8; ++c1)
            {
                const uint3 b1 = b2 * 2u + uint3(c1 & 1u, (c1 >> 1) & 1u, c1 >> 2);
                const auto it1 = lookup[1]->find(BrickHeader::pack(b1));
                if (it1 == lookup[1]->end() || !kept[1][it1->second])
                    continue;
                emit(encoded[1][it1->second], headers, payloads);
                for (uint32_t c0 = 0; c0 < 8; ++c0)
                {
                    const uint3 b0 = b1 * 2u + uint3(c0 & 1u, (c0 >> 1) & 1u, c0 >> 2);
                    const auto it0 = lookup[0]->find(BrickHeader::pack(b0));
                    if (it0 != lookup[0]->end() && kept[0][it0->second])
                        emit(encoded[0][it0->second], headers, payloads);
                }
            }
            result.pages.emplace_back(BrickHeader::pack(b2), timedCompress(makeMeta(headers, {}), payloads));
        }
        // The chunk page: kept level-3 bricks and the page directory (blob offsets filled in when written).
        std::vector<BrickHeader> headers;
        std::vector<uint8_t> payloads;
        for (size_t i3 = 0; i3 < encoded[3].size(); ++i3)
            if (kept[3][i3])
                emit(encoded[3][i3], headers, payloads);
        std::vector<PageEntry> directory(result.pages.size());
        for (size_t p = 0; p < result.pages.size(); ++p)
        {
            const CompressedPage& page = result.pages[p].second;
            directory[p].coord = result.pages[p].first;
            directory[p].blobs.meta.compressed = uint32_t(page.meta.size());
            directory[p].blobs.meta.raw = page.metaRaw;
            directory[p].blobs.payload.compressed = uint32_t(page.payload.size());
            directory[p].blobs.payload.raw = page.payloadRaw;
        }
        if (!headers.empty())
        {
            result.chunkMeta = makeMeta(headers, directory);
            if (!payloads.empty())
                result.chunkPayload = timedCompress({}, payloads);
        }
        return result;
    }

    const std::vector<uint8_t>& proxy() const { return mProxy; }
    uint32_t chunkIndex(size_t c) const { return mChunks[c].chunk; }
    uint32_t chunkTableSize() const { return mChunkDims.x * mChunkDims.y * mChunkDims.z; }

    static std::vector<uint8_t> makeMeta(const std::vector<BrickHeader>& headers, const std::vector<PageEntry>& pages)
    {
        PageHeader header;
        header.brickCount = uint32_t(headers.size());
        header.pageCount = uint32_t(pages.size());
        std::vector<uint8_t> out(sizeof(PageHeader) + headers.size() * sizeof(BrickHeader) + pages.size() * sizeof(PageEntry));
        uint8_t* p = out.data();
        std::memcpy(p, &header, sizeof(header));
        p += sizeof(header);
        std::memcpy(p, headers.data(), headers.size() * sizeof(BrickHeader));
        p += headers.size() * sizeof(BrickHeader);
        std::memcpy(p, pages.data(), pages.size() * sizeof(PageEntry));
        return out;
    }

    /// Visibility weight of a brick's error: e^-tau of the least optical depth from the brick's most exposed proxy cell to outside
    /// the cloud along the six axes, at least codec.weightFloor. Silhouettes and skins keep full weight; thick cores little.
    float brickWeight(uint32_t level, uint32_t coord, const CodecSettings& codec) const
    {
        if (codec.weightFloor >= 1.f || mExposure.empty())
            return 1.f;
        const uint3 b = BrickHeader{coord}.brick();
        const uint3 lo = min((b * 8u << level) >> mProxyLevel, mProxyDims - 1u);
        const uint3 hi = min((((b + 1u) * 8u << level) - 1u) >> mProxyLevel, mProxyDims - 1u);
        float depth = std::numeric_limits<float>::max();
        for (uint32_t z = lo.z; z <= hi.z; ++z)
            for (uint32_t y = lo.y; y <= hi.y; ++y)
                for (uint32_t x = lo.x; x <= hi.x; ++x)
                    depth = std::min(depth, mExposure[x + mProxyDims.x * (size_t(y) + mProxyDims.y * size_t(z))]);
        return std::max(std::exp(-depth * codec.densityScale), codec.weightFloor);
    }

    /// From the proxy means (optical depths in density times world units, before the codec's density scale):
    /// - per proxy cell, the least depth from the cell's centre to outside the cloud over 26 directions (axes, face and body
    ///   diagonals), so a cell visible from any of them keeps its weight. (Dilating it by a neighbouring cell, 4 bricks at this
    ///   resolution, grew a package 45% for no measured gain: bricks already take the most exposed cell they overlap.)
    /// - along each axis, the depth of the cells before and after each cell (measureTransmittance).
    void computeVisibility()
    {
        const size_t count = size_t(mProxyDims.x) * mProxyDims.y * mProxyDims.z;
        const float16_t* means = reinterpret_cast<const float16_t*>(mProxy.data());
        const float cell = float(1u << mProxyLevel) * entry.voxelWorld;
        const int3 dims(mProxyDims);
        auto index = [&](int3 p) { return size_t(p.x) + size_t(dims.x) * (size_t(p.y) + size_t(dims.y) * size_t(p.z)); };
        auto density = [&](int3 p) { return std::max(0.f, float(means[index(p)])) * cell; };

        mExposure.assign(count, std::numeric_limits<float>::max());
        parallelFor(
            size_t(dims.z),
            [&](size_t z)
            {
                for (int y = 0; y < dims.y; ++y)
                    for (int x = 0; x < dims.x; ++x)
                    {
                        const int3 p(x, y, int(z));
                        float best = std::numeric_limits<float>::max();
                        for (int dz = -1; dz <= 1; ++dz)
                            for (int dy = -1; dy <= 1; ++dy)
                                for (int dx = -1; dx <= 1; ++dx)
                                {
                                    if (dx == 0 && dy == 0 && dz == 0)
                                        continue;
                                    const int3 d(dx, dy, dz);
                                    const float length = std::sqrt(float(dx * dx + dy * dy + dz * dz));
                                    float depth = 0.5f * density(p) * length;
                                    for (int3 q = p + d; all(q >= int3(0)) && all(q < dims) && depth < best; q += d)
                                        depth += density(q) * length;
                                    best = std::min(best, depth);
                                }
                        mExposure[index(p)] = best;
                    }
            }
        );

        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            const uint32_t u = (axis + 1) % 3;
            const uint32_t v = (axis + 2) % 3;
            mDepthBefore[axis].assign(count, 0.f);
            mDepthAfter[axis].assign(count, 0.f);
            for (int a = 0; a < dims[u]; ++a)
                for (int b = 0; b < dims[v]; ++b)
                    for (int direction : {1, -1})
                    {
                        float sum = 0.f;
                        for (int k = 0; k < dims[axis]; ++k)
                        {
                            int3 p;
                            p[axis] = direction > 0 ? k : dims[axis] - 1 - k;
                            p[u] = a;
                            p[v] = b;
                            const size_t i = index(p);
                            (direction > 0 ? mDepthBefore : mDepthAfter)[axis][i] = sum;
                            sum += density(p);
                        }
                    }
        }
    }

    /// Close-up transmittance error of one chunk: its level-0 reconstruction (stored and predicted bricks alike, as a close view
    /// renders them) against the truth, integrated along every voxel column of the chunk on each axis. A column's error
    /// |e^-tau_true - e^-tau_coded| is attenuated by the proxy's optical depth in front of the chunk from its more exposed side, so
    /// errors deep inside the cloud count as little as they show.
    void measureTransmittance(size_t c, const Level& truth, const std::vector<EncodedBrick>& coded, const CodecSettings& codec, EncodeStats& stats) const
    {
        constexpr uint32_t n = kChunkVoxels;
        const uint32_t chunk = mChunks[c].chunk;
        const uint3 chunkCoord(chunk % mChunkDims.x, (chunk / mChunkDims.x) % mChunkDims.y, chunk / (mChunkDims.x * mChunkDims.y));
        const float scale = entry.voxelWorld * codec.densityScale;
        // Per axis, the true and coded optical depth of each column (the other two axes in increasing order).
        std::vector<float> depth(6 * size_t(n) * n, 0.f);
        for (size_t i = 0; i < truth.coords.size(); ++i)
        {
            const uint3 base = (BrickHeader{truth.coords[i]}.brick() % 16u) * 8u;
            const Reconstruction& r = coded[i].reconstruction;
            const float codeScale = r.valueRange / (255.f * 255.f);
            for (uint32_t z = 0; z < 8; ++z)
                for (uint32_t y = 0; y < 8; ++y)
                    for (uint32_t x = 0; x < 8; ++x)
                    {
                        const float code = float(r.codes[(x + 1) + 10 * ((y + 1) + 10 * (z + 1))]);
                        const float t = truth.cores[i][x + 8 * (y + 8 * z)] * scale;
                        const float d = (r.valueMin + codeScale * code * code) * scale;
                        const uint32_t vx = base.x + x, vy = base.y + y, vz = base.z + z;
                        const size_t columns[3] = {size_t(vy) * n + vz, size_t(vx) * n + vz, size_t(vx) * n + vy};
                        for (uint32_t axis = 0; axis < 3; ++axis)
                        {
                            depth[(2 * axis) * n * n + columns[axis]] += t;
                            depth[(2 * axis + 1) * n * n + columns[axis]] += d;
                        }
                    }
        }
        const float proxyScale = codec.densityScale;
        const uint32_t other[3][2] = {{1, 2}, {0, 2}, {0, 1}};
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            const uint32_t first = (chunkCoord[axis] * n) >> mProxyLevel;
            const uint32_t last = ((chunkCoord[axis] + 1) * n - 1) >> mProxyLevel;
            for (uint32_t u = 0; u < n; ++u)
                for (uint32_t v = 0; v < n; ++v)
                {
                    const size_t column = size_t(u) * n + v;
                    const float tauTrue = depth[(2 * axis) * n * n + column];
                    const float tauCoded = depth[(2 * axis + 1) * n * n + column];
                    if (tauTrue <= 0.f && tauCoded <= 0.f)
                        continue;
                    uint3 cell;
                    cell[other[axis][0]] = (chunkCoord[other[axis][0]] * n + u) >> mProxyLevel;
                    cell[other[axis][1]] = (chunkCoord[other[axis][1]] * n + v) >> mProxyLevel;
                    cell[axis] = first;
                    const float before = mDepthBefore[axis][cell.x + mProxyDims.x * (size_t(cell.y) + mProxyDims.y * size_t(cell.z))];
                    cell[axis] = last;
                    const float after = mDepthAfter[axis][cell.x + mProxyDims.x * (size_t(cell.y) + mProxyDims.y * size_t(cell.z))];
                    const float exposure = std::exp(-std::min(before, after) * proxyScale);
                    stats.addColumn(exposure * std::abs(std::exp(-tauTrue) - std::exp(-tauCoded)));
                }
        }
    }

private:
    struct Chunk
    {
        uint32_t chunk;
        std::vector<uint32_t> bricks;
    };

public:
    uint32_t chunkOf(uint32_t coord) const
    {
        const uint3 c = BrickHeader{coord}.brick() / 16u;
        return c.x + mChunkDims.x * (c.y + mChunkDims.y * c.z);
    }

    float span(uint32_t level) const { return 8.f * float(1u << level) * entry.voxelWorld; }

    void writeProxy(std::vector<float>& proxy, uint3 b, const Core& core) const
    {
        for (uint32_t z = 0; z < 8; ++z)
            for (uint32_t y = 0; y < 8; ++y)
                for (uint32_t x = 0; x < 8; ++x)
                {
                    const uint3 p = b * 8u + uint3(x, y, z);
                    if (all(p < mProxyDims))
                        proxy[p.x + mProxyDims.x * (size_t(p.y) + mProxyDims.y * size_t(p.z))] = core[x + 8 * (y + 8 * z)];
                }
    }

    /// Levels 0-3 of chunk c: level-0 cores from the canonical cache when one is open, else from the source (bricks whose values
    /// including the apron are all zero are empty, and maxima and hash are measured); then means.
    void readChunk(size_t c, ChunkLevels& levels, std::vector<std::pair<uint32_t, float>>* maxima, uint64_t* hash) const
    {
        if (!mGrid)
        {
            std::ifstream file(mCachePath, std::ios::binary);
            unpackCores(readBlobFrom(file, mChunkBlobs[c]), levels[0]);
            for (uint32_t level = 0; level < 3; ++level)
                for (size_t i = 0; i < levels[level].coords.size(); ++i)
                    accumulate(levels[level + 1], levels[level].coords[i], levels[level].cores[i]);
            return;
        }
        auto accessor = mGrid->getConstUnsafeAccessor();
        const openvdb::Coord origin(mOrigin.x, mOrigin.y, mOrigin.z);
        uint64_t h = 1469598103934665603ull;
        for (uint32_t coord : mChunks[c].bricks)
        {
            const uint3 b = BrickHeader{coord}.brick();
            const openvdb::Coord base = origin + openvdb::Coord(int(b.x * 8) - 1, int(b.y * 8) - 1, int(b.z * 8) - 1);
            Core core;
            float maximum = 0.f;
            for (int z = 0; z < 10; ++z)
                for (int y = 0; y < 10; ++y)
                    for (int x = 0; x < 10; ++x)
                    {
                        const float v = accessor.getValue(base + openvdb::Coord(x, y, z));
                        const float value = std::isfinite(v) && v > 0.f ? v : 0.f;
                        maximum = std::max(maximum, value);
                        if (x >= 1 && x <= 8 && y >= 1 && y <= 8 && z >= 1 && z <= 8)
                            core[(x - 1) + 8 * ((y - 1) + 8 * (z - 1))] = value;
                    }
            if (maximum <= 0.f)
                continue;
            const uint32_t i = levels[0].add(coord);
            levels[0].cores[i] = core;
            if (maxima)
                maxima->emplace_back(coord, maximum);
            if (hash)
            {
                h = fnv(h, coord);
                for (float value : core)
                {
                    uint32_t bits;
                    std::memcpy(&bits, &value, sizeof(bits));
                    h = fnv(h, bits);
                }
            }
        }
        for (uint32_t level = 0; level < 3; ++level)
            for (size_t i = 0; i < levels[level].coords.size(); ++i)
                accumulate(levels[level + 1], levels[level].coords[i], levels[level].cores[i]);
        if (hash)
            *hash = h;
    }

private:
    openvdb::FloatGrid::Ptr mGrid;
    int3 mOrigin = int3(0);
    uint3 mDims = uint3(0);
    uint3 mChunkDims = uint3(0);
    uint3 mProxyDims = uint3(0);
    uint32_t mTopLevel = 0;
    uint32_t mProxyLevel = 0;
    std::vector<Chunk> mChunks;
    std::vector<Level> mCoarse;
    std::vector<ChunkLevels> mCached;
    std::vector<uint8_t> mIsCached;
    std::vector<uint8_t> mProxy;
    std::filesystem::path mCachePath;
    std::vector<BlobRef> mChunkBlobs;             ///< Per chunk, in the canonical cache.
    std::vector<float> mExposure;                 ///< computeVisibility.
    std::array<std::vector<float>, 3> mDepthBefore; ///< computeVisibility.
    std::array<std::vector<float>, 3> mDepthAfter;  ///< computeVisibility.
};

struct Analysis
{
    uint64_t signature = 0;
    uint64_t densityHash = 0;
    std::array<double, kLadderSize> bytes = {};
};

std::unordered_map<std::string, Analysis> readAnalysis(const std::filesystem::path& path)
{
    std::unordered_map<std::string, Analysis> result;
    std::ifstream file(path, std::ios::binary);
    uint32_t count = 0;
    if (!file.read(reinterpret_cast<char*>(&count), sizeof(count)))
        return result;
    for (uint32_t i = 0; i < count; ++i)
    {
        char name[64];
        Analysis analysis;
        file.read(name, sizeof(name));
        file.read(reinterpret_cast<char*>(&analysis), sizeof(analysis));
        if (file)
            result[std::string(name, strnlen(name, sizeof(name)))] = analysis;
    }
    return result;
}

void writeAnalysis(const std::filesystem::path& path, const std::unordered_map<std::string, Analysis>& analyses)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const uint32_t count = uint32_t(analyses.size());
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [key, analysis] : analyses)
    {
        char name[64] = {};
        std::copy_n(key.c_str(), std::min<size_t>(key.size(), 63), name);
        file.write(name, sizeof(name));
        file.write(reinterpret_cast<const char*>(&analysis), sizeof(analysis));
    }
}

/// Stage 1: the canonical cache of every VDB in the source directory whose cache is missing or from another source file.
void buildCaches(const std::filesystem::path& sourceDirectory, const std::filesystem::path& cacheDirectory)
{
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::filesystem::path> sources;
    for (const auto& file : std::filesystem::directory_iterator(sourceDirectory))
        if (file.is_regular_file() && file.path().extension() == ".vdb")
            sources.push_back(file.path());
    std::sort(sources.begin(), sources.end());
    if (sources.empty())
        throw std::runtime_error("no VDB files in " + sourceDirectory.string());
    std::filesystem::create_directories(cacheDirectory);
    uint64_t sourceBytes = 0;
    uint64_t cacheBytes = 0;
    for (const auto& source : sources)
    {
        const std::filesystem::path path = cacheDirectory / (source.stem().string() + ".hstrcloud");
        const uint64_t bytes = std::filesystem::file_size(source);
        sourceBytes += bytes;
        CanonicalHeader header;
        if (std::filesystem::exists(path) && readCanonicalHeader(path, header) && header.sourceSignature == sourceSignature(source))
        {
            std::cout << source.stem().string() << ": canonical cache up to date\n";
            cacheBytes += std::filesystem::file_size(path);
            continue;
        }
        const auto assetStart = std::chrono::steady_clock::now();
        std::cout << "building " << source.stem().string() << "\n";
        Asset asset;
        asset.source = source;
        asset.signature = sourceSignature(source);
        asset.sourceBytes = bytes;
        asset.load();
        asset.writeCanonical(path);
        asset.unload();
        cacheBytes += std::filesystem::file_size(path);
        std::cout << "  " << double(std::filesystem::file_size(path)) / 1048576.0 << " MB from " << double(bytes) / 1048576.0 << " MB of VDB in "
                  << seconds(assetStart) << " s\n";
    }
    std::cout << "canonical caches: " << double(cacheBytes) / 1048576.0 << " MB from " << double(sourceBytes) / 1048576.0 << " MB of VDB, "
              << seconds(start) << " s\n";
}

std::vector<std::string> splitList(const std::string& list)
{
    std::vector<std::string> items;
    std::stringstream stream(list);
    for (std::string item; std::getline(stream, item, ',');)
        if (!item.empty())
            items.push_back(item);
    return items;
}

uint64_t floatBits(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool readPackageHeader(const std::filesystem::path& path, LibraryHeader& header)
{
    std::ifstream file(path, std::ios::binary);
    return file.read(reinterpret_cast<char*>(&header), sizeof(header)) && header.magic == kLibraryMagic && header.version == kLibraryVersion;
}

/// A package under another cloud's name (for clouds of identical density).
void copyPackage(const std::filesystem::path& from, const std::filesystem::path& to, const std::string& name)
{
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing);
    std::fstream file(to, std::ios::binary | std::ios::in | std::ios::out);
    LibraryHeader header;
    AssetEntry entry;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    file.seekg(std::streamoff(header.assetTableOffset));
    file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
    std::fill(std::begin(entry.name), std::end(entry.name), 0);
    std::copy_n(name.c_str(), std::min<size_t>(name.size(), sizeof(entry.name) - 1), entry.name);
    file.seekp(std::streamoff(header.assetTableOffset));
    file.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    if (!file)
        throw std::runtime_error("failed to copy the package " + from.string() + " to " + to.string());
}

struct PackResult
{
    Asset::EncodeStats stats;
    uint64_t bytes = 0;
};

/// Encodes an opened cloud and writes its package. A dry run only counts the bytes; a sample rate below 1 encodes every k-th chunk
/// and scales the chunk bytes (errors are the sample's).
PackResult writePackage(Asset& asset, const CodecSettings& codec, const std::filesystem::path& path, bool dryRun, double sampleRate, uint64_t packKey)
{
    std::fstream file;
    const std::filesystem::path temporary = path.string() + ".partial";
    if (!dryRun)
    {
        file.open(temporary, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        if (!file)
            throw std::runtime_error("cannot write " + temporary.string());
    }
    LibraryHeader header;
    header.assetCount = 1;
    header.sourceBytes = asset.sourceBytes;
    header.lambda = codec.lambda;
    header.packKey = packKey;
    uint64_t written = sizeof(header);
    if (!dryRun)
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    // Blobs are appended at the running offset (counted, not written, in a dry run).
    auto append = [&](const std::vector<uint8_t>& bytes, uint32_t raw)
    {
        BlobRef ref;
        ref.offset = written;
        ref.compressed = uint32_t(bytes.size());
        ref.raw = raw;
        if (!dryRun)
            file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        written += bytes.size();
        return ref;
    };
    std::vector<uint8_t> compressed;
    compressGDeflate(asset.proxy().data(), asset.proxy().size(), compressed);
    asset.entry.densityHash = asset.densityHash;
    asset.entry.proxy = append(compressed, uint32_t(asset.proxy().size()));
    auto coarse = asset.encodeCoarse(codec);
    asset.entry.coarse.meta = append(coarse.page.meta, coarse.page.metaRaw);
    asset.entry.coarse.payload = append(coarse.page.payload, coarse.page.payloadRaw);
    Asset::EncodeStats stats = coarse.stats;

    // Chunks in parallel, written in order of completion (their table is written last).
    std::vector<PageBlobs> table(asset.chunkTableSize());
    std::mutex writeMutex;
    const size_t stride = std::max<size_t>(1, size_t(std::lround(1.0 / sampleRate)));
    std::vector<size_t> chunks;
    for (size_t c = 0; c < asset.chunkCount(); c += stride)
        chunks.push_back(c);
    const uint64_t chunkStart = written;
    parallelFor(
        chunks.size(),
        [&](size_t s)
        {
            const size_t c = chunks[s];
            auto result = asset.encodeChunk(c, codec, coarse.level4.at(asset.chunkCoord(c)), false);
            std::lock_guard lock(writeMutex);
            stats.merge(result.stats);
            if (result.chunkMeta.empty())
                return;
            // Level-2 pages first; then the chunk page with their final offsets.
            PageHeader pageHeader;
            std::memcpy(&pageHeader, result.chunkMeta.data(), sizeof(pageHeader));
            auto* directory = reinterpret_cast<PageEntry*>(result.chunkMeta.data() + sizeof(PageHeader) + pageHeader.brickCount * sizeof(BrickHeader));
            for (size_t p = 0; p < result.pages.size(); ++p)
            {
                const Asset::CompressedPage& page = result.pages[p].second;
                directory[p].blobs.meta = append(page.meta, page.metaRaw);
                if (page.payloadRaw > 0)
                    directory[p].blobs.payload = append(page.payload, page.payloadRaw);
            }
            std::vector<uint8_t> meta;
            compressGDeflate(result.chunkMeta.data(), result.chunkMeta.size(), meta);
            stats.metaCompressed += meta.size();
            const uint32_t chunk = asset.chunkIndex(c);
            table[chunk].meta = append(meta, uint32_t(result.chunkMeta.size()));
            if (result.chunkPayload.payloadRaw > 0)
                table[chunk].payload = append(result.chunkPayload.payload, result.chunkPayload.payloadRaw);
        }
    );
    if (stride > 1)
    {
        written = chunkStart + uint64_t(double(written - chunkStart) * double(asset.chunkCount()) / double(chunks.size()));
        std::cout << "  sampled " << chunks.size() << " of " << asset.chunkCount() << " chunks; sizes are scaled, errors are the sample's\n";
    }
    asset.entry.storedBricks = stats.bricks;
    asset.entry.chunkTableOffset = written;
    if (!dryRun)
        file.write(reinterpret_cast<const char*>(table.data()), std::streamsize(table.size() * sizeof(PageBlobs)));
    written += table.size() * sizeof(PageBlobs);
    header.assetTableOffset = written;
    written += sizeof(AssetEntry);
    asset.entry.storedBytes = written;
    header.packageBytes = written;
    header.transmittanceError = stats.transmittancePercentile(0.99);
    if (!dryRun)
    {
        file.write(reinterpret_cast<const char*>(&asset.entry), sizeof(AssetEntry));
        file.seekp(0);
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        file.close();
        if (!file)
            throw std::runtime_error("failed to write " + temporary.string());
        std::filesystem::rename(temporary, path);
    }
    PackResult result;
    result.stats = stats;
    result.bytes = written;
    return result;
}

/// The largest lambda (smallest package) whose close-up transmittance error over every k-th chunk (the sample) is within the
/// target at the 99th percentile and within the tail target at the 99.9th: decade steps from 3e-4 until the targets are
/// bracketed, then bisection in log lambda to a factor 1.15. Probes encode without building pages.
double searchLambda(Asset& asset, CodecSettings codec, float target, float tailTarget, double sampleRate)
{
    const auto start = std::chrono::steady_clock::now();
    const size_t stride = std::max<size_t>(1, size_t(std::lround(1.0 / sampleRate)));
    std::vector<size_t> sample;
    for (size_t c = 0; c < asset.chunkCount(); c += stride)
        sample.push_back(c);
    asset.cacheChunks(sample);
    int probes = 0;
    auto probe = [&](double logLambda)
    {
        codec.lambda = float(std::pow(10.0, logLambda));
        const auto coarse = asset.encodeCoarse(codec);
        Asset::EncodeStats stats;
        std::mutex mutex;
        parallelFor(
            sample.size(),
            [&](size_t s)
            {
                const auto result = asset.encodeChunk(sample[s], codec, coarse.level4.at(asset.chunkCoord(sample[s])), false, false);
                std::lock_guard lock(mutex);
                stats.merge(result.stats);
            }
        );
        ++probes;
        return stats.transmittancePercentile(0.99) <= target && stats.transmittancePercentile(0.999) <= tailTarget;
    };
    constexpr double kFinest = -8.0, kCoarsest = -1.0;
    double pass = std::numeric_limits<double>::quiet_NaN();
    double fail = std::numeric_limits<double>::quiet_NaN();
    double x = std::log10(3e-4);
    if (probe(x))
    {
        pass = x;
        while (std::isnan(fail) && pass < kCoarsest)
        {
            const double next = std::min(pass + 1.0, kCoarsest);
            (probe(next) ? pass : fail) = next;
        }
    }
    else
    {
        fail = x;
        while (std::isnan(pass) && fail > kFinest)
        {
            const double next = std::max(fail - 1.0, kFinest);
            (probe(next) ? pass : fail) = next;
        }
        if (std::isnan(pass))
        {
            std::cout << "  warning: no lambda meets the quality target; packing at the finest lambda\n";
            return std::pow(10.0, kFinest);
        }
    }
    if (!std::isnan(fail))
        for (int i = 0; i < 4; ++i)
        {
            const double mid = 0.5 * (pass + fail);
            (probe(mid) ? pass : fail) = mid;
        }
    std::cout << "  quality search: lambda " << std::pow(10.0, pass) << " after " << probes << " probes of " << sample.size() << " chunks, "
              << seconds(start) << " s\n";
    return std::pow(10.0, pass);
}
} // namespace

int main(int argc, char** argv)
{
    std::cout << std::unitbuf;
    SetUnhandledExceptionFilter(
        [](EXCEPTION_POINTERS* exception) -> LONG
        {
            std::cerr << "crash 0x" << std::hex << exception->ExceptionRecord->ExceptionCode << std::dec << "\n"
                      << Falcor::getStackTrace() << std::endl;
            return EXCEPTION_EXECUTE_HANDLER;
        }
    );
    const std::string mode = argc > 1 ? argv[1] : "";
    if (argc < 4 || (mode != "build" && mode != "pack"))
    {
        std::cout << "HSTRCloudCompiler build <vdb directory> <cache directory>\n"
                     "HSTRCloudCompiler pack <cache directory> <output directory> [--quality E [--quality-tail T] | --lambda L | --budget-mb N]\n"
                     "                       [--max-error E]\n"
                     "                       [--weight-floor F] [--density-scale S] [--transform haar|identity|cdf53|legacy] [--deadzone D]\n"
                     "                       [--sample R] [--clouds a,b,...] [--dry-run] [--force]\n";
        return 1;
    }
    if (mode == "build")
    {
        try
        {
            buildCaches(argv[2], argv[3]);
            return 0;
        }
        catch (const std::exception& e)
        {
            std::cerr << "error: " << e.what() << "\n";
            return 1;
        }
    }
    const std::filesystem::path cacheDirectory = argv[2];
    const std::filesystem::path outputDirectory = argv[3];
    double budgetMB = 0.0;    // --budget-mb: one lambda for every cloud so the packages fit the budget.
    double fixedLambda = 0.0; // --lambda: every cloud at this rate-distortion trade-off.
    float quality = kDefaultQuality; // Otherwise each cloud's lambda meets this close-up transmittance error,
    float tail = 0.f;                // and this one at the 99.9th percentile (0: kTailFactor * quality).
    double sampleRate = 0.2;
    bool dryRun = false;
    bool force = false;
    CodecSettings codec;
    std::unordered_set<std::string> only;
    for (int i = 4; i < argc; ++i)
    {
        const std::string option = argv[i];
        const bool hasValue = i + 1 < argc;
        if (option == "--dry-run")
            dryRun = true;
        else if (option == "--force")
            force = true;
        else if (!hasValue)
        {
            std::cerr << "option " << option << " needs a value\n";
            return 1;
        }
        else if (option == "--budget-mb")
            budgetMB = std::max(1.0, std::stod(argv[++i]));
        else if (option == "--lambda")
            fixedLambda = std::max(1e-12, std::stod(argv[++i]));
        else if (option == "--quality")
            quality = std::max(1e-6f, std::stof(argv[++i]));
        else if (option == "--quality-tail")
            tail = std::max(1e-6f, std::stof(argv[++i]));
        else if (option == "--max-error")
            codec.maxError = std::max(0.f, std::stof(argv[++i]));
        else if (option == "--weight-floor")
            codec.weightFloor = std::clamp(std::stof(argv[++i]), 0.f, 1.f);
        else if (option == "--density-scale")
            codec.densityScale = std::max(0.f, std::stof(argv[++i]));
        else if (option == "--deadzone")
            codec.deadzone = std::clamp(std::stof(argv[++i]), 0.f, 0.5f);
        else if (option == "--transform")
        {
            const std::string name = argv[++i];
            if (name == "haar")
                codec.transform = Transform::Haar;
            else if (name == "identity")
                codec.transform = Transform::Identity;
            else if (name == "cdf53")
                codec.transform = Transform::Cdf53;
            else if (name == "legacy")
                codec.transform = Transform::Legacy;
            else
            {
                std::cerr << "unknown transform " << name << " (haar, identity, cdf53)\n";
                return 1;
            }
        }
        else if (option == "--sample")
            sampleRate = std::clamp(std::stod(argv[++i]), 0.001, 1.0);
        else if (option == "--clouds")
            for (const std::string& name : splitList(argv[++i]))
                only.insert(name);
        else
        {
            std::cerr << "unknown option " << option << "\n";
            return 1;
        }
    }
    if ((codec.transform == Transform::Cdf53 || codec.transform == Transform::Legacy) && !dryRun)
    {
        std::cerr << "--transform cdf53 and legacy are comparisons the GPU does not decode: they need --dry-run\n";
        return 1;
    }
    if (codec.transform == Transform::Legacy && codec.maxError <= 0.f)
    {
        std::cerr << "--transform legacy needs --max-error (its v3 tolerance)\n";
        return 1;
    }
    if (budgetMB > 0.0 && fixedLambda > 0.0)
    {
        std::cerr << "--budget-mb and --lambda are exclusive\n";
        return 1;
    }
    try
    {
        const auto start = std::chrono::steady_clock::now();
        std::vector<Asset> assets;
        uint64_t sourceBytes = 0;
        for (const auto& file : std::filesystem::directory_iterator(cacheDirectory))
        {
            CanonicalHeader header;
            if (!file.is_regular_file() || file.path().extension() != ".hstrcloud" ||
                (!only.empty() && !only.count(file.path().stem().string())))
                continue;
            if (!readCanonicalHeader(file.path(), header))
                throw std::runtime_error("not a canonical cloud cache of this version: " + file.path().string() + " (rerun build)");
            Asset asset;
            asset.source = file.path();
            asset.signature = header.sourceSignature;
            asset.densityHash = header.densityHash;
            asset.sourceBytes = header.sourceBytes;
            sourceBytes += header.sourceBytes;
            assets.push_back(std::move(asset));
        }
        std::sort(assets.begin(), assets.end(), [](const Asset& a, const Asset& b) { return a.source < b.source; });
        if (assets.empty())
            throw std::runtime_error("no canonical cloud caches in " + cacheDirectory.string() + (only.empty() ? "" : " matching --clouds"));
        if (!only.empty() && assets.size() != only.size())
            throw std::runtime_error("--clouds names a cloud without a canonical cache");

        // 1. Analysis, cached per cloud by its density, the codec and its settings, and the sample rate.
        const std::filesystem::path analysisPath = cacheDirectory / "analysis.bin";
        auto analyses = readAnalysis(analysisPath);
        uint64_t codecKey = kCodecVersion;
        for (float setting : {codec.maxError, codec.weightFloor, codec.deadzone, codec.densityScale, float(codec.transform)})
            codecKey = fnv(codecKey, floatBits(setting));
        const uint64_t settingsKey = fnv(codecKey, uint64_t(std::lround(sampleRate * 1e6)));
        for (Asset& asset : assets)
        {
            if (budgetMB <= 0.0)
                break;
            const std::string name = asset.source.stem().string();
            const uint64_t analysisKey = fnv(asset.densityHash, settingsKey);
            auto cached = analyses.find(name);
            if (cached != analyses.end() && cached->second.signature == analysisKey)
                continue;
            const auto assetStart = std::chrono::steady_clock::now();
            std::cout << "analysing " << name << "\n";
            asset.open(asset.source);
            Analysis analysis;
            analysis.signature = analysisKey;
            analysis.densityHash = asset.densityHash;
            // Every k-th chunk, so the sample spans the whole cloud; its means are read once for every lambda.
            const size_t stride = std::max<size_t>(1, size_t(std::lround(1.0 / sampleRate)));
            std::vector<size_t> sample;
            for (size_t c = 0; c < asset.chunkCount(); c += stride)
                sample.push_back(c);
            const double scale = double(asset.chunkCount()) / double(sample.size());
            asset.cacheChunks(sample);
            std::vector<uint8_t> proxyCompressed;
            compressGDeflate(asset.proxy().data(), asset.proxy().size(), proxyCompressed);
            for (uint32_t t = 0; t < kLadderSize; ++t)
            {
                CodecSettings rung = codec;
                rung.lambda = float(ladderLambda(t));
                const auto coarse = asset.encodeCoarse(rung);
                std::atomic<uint64_t> sampled{0};
                parallelFor(
                    sample.size(),
                    [&](size_t s) { sampled += asset.encodeChunk(sample[s], rung, coarse.level4.at(asset.chunkCoord(sample[s])), false).bytes(); }
                );
                analysis.bytes[t] = double(sampled) * scale + double(coarse.page.bytes()) + double(proxyCompressed.size()) +
                                    double(asset.chunkTableSize()) * sizeof(PageBlobs) + sizeof(AssetEntry) + sizeof(LibraryHeader);
            }
            asset.unload();
            analyses[name] = analysis;
            writeAnalysis(analysisPath, analyses);
            std::cout << "  " << seconds(assetStart) << " s; MB at lambda " << ladderLambda(0) << ".." << ladderLambda(kLadderSize - 1) << ":";
            for (double b : analysis.bytes)
                std::cout << " " << int(b / 1048576.0);
            std::cout << "\n";
        }

        // 2. The lambda of every cloud: the quality search per cloud (step 3), --lambda, or with a budget the smallest lambda (finest
        // quality) whose estimate fits it: one lambda for every cloud, so every byte buys the same weighted error everywhere (a soft
        // limit: an overshoot warns).
        const uint64_t budgetBytes = uint64_t(budgetMB * 1048576.0);
        if (budgetMB > 0.0)
        {
            // Estimated bytes at a continuous ladder position (log-linear between measured lambdas).
            auto estimate = [&](double ladder)
            {
                double sum = 0.0;
                for (const Asset& asset : assets)
                {
                    const auto& b = analyses.at(asset.source.stem().string()).bytes;
                    const uint32_t lo = std::min<uint32_t>(uint32_t(std::max(ladder, 0.0)), kLadderSize - 2);
                    const double f = std::clamp(ladder - lo, 0.0, 1.0);
                    sum += std::exp(std::log(std::max(b[lo], 1.0)) * (1.0 - f) + std::log(std::max(b[lo + 1], 1.0)) * f);
                }
                return sum;
            };
            double ladder = 0.0;
            if (estimate(0.0) > double(budgetBytes))
            {
                double lo = 0.0, hi = kLadderSize - 1;
                for (int i = 0; i < 40; ++i)
                {
                    const double mid = 0.5 * (lo + hi);
                    (estimate(mid) > double(budgetBytes) ? lo : hi) = mid;
                }
                ladder = hi;
            }
            codec.lambda = float(ladderLambda(ladder));
            std::cout << "lambda " << codec.lambda << " for an estimated " << estimate(ladder) / 1048576.0 << " MB (budget " << budgetMB << " MB)\n";
        }
        else if (fixedLambda > 0.0)
        {
            codec.lambda = float(fixedLambda);
            std::cout << "lambda " << codec.lambda << " (fixed)\n";
        }
        else
            std::cout << "per-cloud lambda for a close-up transmittance error of " << quality << " at the 99th percentile and "
                      << (tail > 0.f ? tail : kTailFactor * quality) << " at the 99.9th (over " << sampleRate * 100.0 << "% of the chunks)\n";

        // 3. One package per cloud (a dry run only reports). A package whose key (source density and every setting) matches is up to
        // date; a cloud with the density of one packed in this run is a copy of its package.
        if (!dryRun)
            std::filesystem::create_directories(outputDirectory);
        std::unordered_map<uint64_t, std::filesystem::path> packedDensity;
        uint64_t totalBytes = 0;
        size_t upToDate = 0, copies = 0;
        for (Asset& asset : assets)
        {
            const std::string name = asset.source.stem().string();
            const std::filesystem::path path = outputDirectory / (name + ".hstrlib");
            const uint64_t modeKey = budgetMB <= 0.0 && fixedLambda <= 0.0
                                         ? fnv(fnv(fnv(1, floatBits(quality)), floatBits(tail)), uint64_t(std::lround(sampleRate * 1e6)))
                                         : fnv(2, floatBits(codec.lambda));
            const uint64_t packKey = fnv(fnv(codecKey, modeKey), asset.densityHash);
            LibraryHeader existing;
            if (!dryRun && !force && readPackageHeader(path, existing) && existing.packKey == packKey)
            {
                std::cout << name << ": up to date (" << double(existing.packageBytes) / 1048576.0 << " MB)\n";
                totalBytes += existing.packageBytes;
                packedDensity.try_emplace(asset.densityHash, path);
                ++upToDate;
                continue;
            }
            if (!dryRun)
                if (auto original = packedDensity.find(asset.densityHash); original != packedDensity.end())
                {
                    copyPackage(original->second, path, name);
                    std::cout << name << ": identical density to " << original->second.stem().string() << ", copied its package\n";
                    totalBytes += std::filesystem::file_size(path);
                    ++copies;
                    continue;
                }

            const auto assetStart = std::chrono::steady_clock::now();
            std::cout << "packing " << name << "\n";
            asset.open(asset.source);
            CodecSettings cloudCodec = codec;
            if (budgetMB <= 0.0 && fixedLambda <= 0.0)
                cloudCodec.lambda = float(searchLambda(asset, codec, quality, tail > 0.f ? tail : kTailFactor * quality, sampleRate));
            const PackResult result = writePackage(asset, cloudCodec, path, dryRun, dryRun ? sampleRate : 1.0, packKey);
            const Asset::EncodeStats& stats = result.stats;
            totalBytes += result.bytes;
            if (!dryRun)
                packedDensity.try_emplace(asset.densityHash, path);
            const double coded = double(std::max<uint64_t>(stats.bricks - stats.predicted, 1));
            std::cout << "  lambda " << cloudCodec.lambda << ": " << stats.bricks << " bricks stored of " << asset.entry.level0Bricks
                      << " source level-0 bricks, " << double(result.bytes) / 1048576.0 << " MB, " << seconds(assetStart) << " s\n"
                      << "  close-up transmittance error over " << stats.columns << " voxel columns: mean "
                      << stats.transmittanceSum / double(std::max<uint64_t>(stats.columns, 1)) << ", 99th percentile "
                      << stats.transmittancePercentile(0.99) << ", 99.9th percentile " << stats.transmittancePercentile(0.999) << ", max "
                      << stats.transmittanceMax << "\n"
                      << "  " << stats.predicted << " predicted; coded bricks average " << double(stats.nonzero) / coded << " non-zero coefficients, "
                      << double(stats.payloadBytes) / coded << " payload bytes before GDeflate\n"
                      << "  optical-depth error per brick: mean " << stats.error / double(std::max<uint64_t>(stats.bricks, 1)) << ", weighted mean "
                      << stats.weightedError / double(std::max<uint64_t>(stats.bricks, 1)) << ", max " << stats.maxError << "\n"
                      << "  thread seconds: encoding " << stats.encodeSeconds << ", GDeflate " << stats.compressSeconds << "\n"
                      << "  levels 0-3 GDeflate: metas " << double(stats.metaCompressed) / 1048576.0 << " MB, payloads "
                      << double(stats.payloadCompressed) / 1048576.0 << " MB (raw " << double(stats.payloadBytes) / 1048576.0
                      << " MB; order-0 entropy of the coefficients " << stats.entropyBytes() / 1048576.0 << " MB)\n";
            asset.unload();
        }
        std::cout << (dryRun ? "dry run, would write " : "wrote ") << assets.size() << " packages to " << outputDirectory.string() << " ("
                  << upToDate << " up to date, " << copies << " copies): " << double(totalBytes) / 1048576.0 << " MB from "
                  << double(sourceBytes) / 1048576.0 << " MB of VDB, " << seconds(start) << " s\n";
        if (budgetMB > 0.0 && totalBytes > budgetBytes)
            std::cout << "warning: the packages exceed the " << budgetMB << " MB budget (a soft limit)\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
