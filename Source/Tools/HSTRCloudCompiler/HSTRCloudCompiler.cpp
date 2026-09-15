/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
/** Offline compiler of an OpenVDB cloud library into one budgeted HSTR cloud package (see CloudFormat.h), in two stages so that
    everything after reading the source is repeatable without it.

    HSTRCloudCompiler build <vdb directory> <cache directory>
        Once per source cloud (skipped while the VDB is unchanged): its canonical cache <cloud>.hstrcloud holds the source's true
        level-0 brick samples (lossless at the atlas's precision), the means of the coarse levels, the proxies and the density hash.

    HSTRCloudCompiler pack <cache directory> <output.hstrlib> [--budget-mb N | --tolerance T] [--sample R] [--clouds a,b,...]
        Packaging from the canonical caches only; tolerance, codec, pages, deduplication and compression all live here:
        1. Analysis: for every cloud, the package bytes at a ladder of optical-depth tolerances, measured on a sample of its chunks.
           Cached in the cache directory per cloud and codec version, so new budgets cost only step 3.
        2. One tolerance for the whole library is interpolated so the package fits the budget (a soft limit: every cloud gets the
           same quality, an overshoot warns), or --tolerance fixes it.
        3. Every cloud is encoded at that tolerance and written, with a report of its brick encodings. Clouds with identical density
           are stored once. --clouds packs a subset (by file name without extension) for quick codec experiments.
*/
#include "RenderPasses/HSTRCloud/CloudFormat.h"
#include "Utils/Math/Float16.h"
#include "Core/Platform/OS.h"
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <array>
#include <atomic>
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
constexpr uint32_t kCodecVersion = 1;

constexpr uint32_t kLadderSize = 16;
float ladderTolerance(uint32_t i)
{
    return 2e-4f * std::pow(2.f, float(i)); // 2e-4 to 6.6 (optical depth over a brick span, in VDB units)
}

template<typename F>
void parallelFor(size_t count, F&& f)
{
    const size_t threadCount = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), count);
    std::atomic<size_t> next{0};
    auto worker = [&]()
    {
        for (size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1))
            f(i);
    };
    std::vector<std::thread> threads;
    for (size_t t = 1; t < threadCount; ++t)
        threads.emplace_back(worker);
    worker();
    for (auto& thread : threads)
        thread.join();
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
constexpr uint32_t kCanonicalVersion = 1;

struct CanonicalHeader
{
    uint64_t magic = kCanonicalMagic;
    uint32_t version = kCanonicalVersion;
    uint32_t chunkCount = 0;
    uint64_t sourceSignature = 0;
    uint64_t sourceBytes = 0;
    uint64_t densityHash = 0;
    AssetEntry entry; ///< Name, geometry and level0Bricks; the package fields are unused.
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
    bool droppable = false; ///< The prediction alone is within tolerance.
};

/// Chooses the cheapest residual that reconstructs the brick's true means within tolerance (mean plus a quarter of the maximum
/// absolute error, times the brick's span), measured after the atlas quantisation the GPU applies.
EncodedBrick encodeBrick(uint32_t coord, uint32_t level, uint8_t childMask, const Core& truth, const Reconstruction* parent, float span, float tolerance)
{
    EncodedBrick best;
    best.header.coord = coord;
    best.header.level = uint8_t(level);
    best.header.childMask = childMask;
    const uint3 parity = BrickHeader{coord}.brick() % 2u;
    // Prediction (residual zero) for all 1000 values.
    BrickHeader predictedHeader = best.header;
    predictedHeader.flags = kBrickPredicted;
    float prediction[kBrickValues];
    reconstructBrick(
        parent ? parent->codes.data() : nullptr,
        parent ? parent->valueMin : 0.f,
        parent ? parent->valueRange : 0.f,
        parity,
        predictedHeader,
        nullptr,
        prediction
    );
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

    auto evaluate = [&](EncodedBrick& candidate) -> float
    {
        float values[kBrickValues];
        reconstructBrick(
            parent ? parent->codes.data() : nullptr,
            parent ? parent->valueMin : 0.f,
            parent ? parent->valueRange : 0.f,
            parity,
            candidate.header,
            candidate.payload.data(),
            values
        );
        float lo = values[0];
        float hi = values[0];
        for (float v : values)
        {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        Reconstruction& r = candidate.reconstruction;
        r.valueMin = lo;
        r.valueRange = hi - lo;
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
                    const float error = std::abs(
                        atlasValue(r.codes[(x + 1) + 10 * ((y + 1) + 10 * (z + 1))], r.valueMin, r.valueRange) - truth[x + 8 * (y + 8 * z)]
                    );
                    sum += error;
                    maximum = std::max(maximum, error);
                }
        return (float(sum / kCoreValues) + 0.25f * maximum) * span;
    };

    // Predicted only.
    best.header.flags = kBrickPredicted;
    if (evaluate(best) <= tolerance)
    {
        best.droppable = true;
        return best;
    }
    // A constant offset, then 2, 4 and 8 bits.
    for (uint32_t bits : {0u, 2u, 4u, 8u})
    {
        EncodedBrick candidate;
        candidate.header = best.header;
        candidate.header.flags = 0;
        candidate.header.bits = uint8_t(bits);
        if (bits == 0)
        {
            candidate.header.residualMin = float(residualSum / kCoreValues);
            candidate.header.residualStep = 0.f;
        }
        else
        {
            const uint32_t levels = (1u << bits) - 1u;
            candidate.header.residualMin = residualMin;
            candidate.header.residualStep = (residualMax - residualMin) / float(levels);
            candidate.payload.assign((kCoreValues * bits + 7) / 8, 0);
            for (uint32_t i = 0; i < kCoreValues; ++i)
            {
                const uint32_t q = candidate.header.residualStep > 0.f
                                       ? std::min(levels, uint32_t(std::lround((residual[i] - residualMin) / candidate.header.residualStep)))
                                       : 0u;
                const uint32_t bit = i * bits;
                candidate.payload[bit / 8] |= uint8_t(q << (bit % 8));
                if (bit % 8 + bits > 8)
                    candidate.payload[bit / 8 + 1] |= uint8_t(q >> (8 - bit % 8));
            }
        }
        if (evaluate(candidate) <= tolerance || bits == 8)
            return candidate;
    }
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
        mChunks.clear();
        mChunkBlobs.clear();
        mProxy.clear();
    }

    /// Stage 1: sweeps the loaded source into its canonical cache (written to a temporary file, renamed when complete).
    void writeCanonical(const std::filesystem::path& path, uint64_t sourceBytes)
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
        header.entry = entry;
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
        entry = header.entry;
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
    }

    size_t chunkCount() const { return mChunks.size(); }

    /// Reads and caches the means of some chunks (the others stay uncached and empty).
    void cacheChunks(const std::vector<size_t>& chunks)
    {
        mCached.resize(mChunks.size());
        parallelFor(chunks.size(), [&](size_t i) { readChunk(chunks[i], mCached[chunks[i]], nullptr, nullptr); });
    }

    /// The sweep's means of chunk c when they were cached, else null (they are read from the source).
    const ChunkLevels* cachedLevels(size_t c) const { return c < mCached.size() ? &mCached[c] : nullptr; }

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

    struct CoarseResult
    {
        std::vector<uint8_t> page; ///< Undeflated.
        std::unordered_map<uint32_t, Reconstruction> level4; ///< By chunk coordinate.
        uint64_t bricks = 0;
    };

    /// Levels top to 4, top-down.
    CoarseResult encodeCoarse(float tolerance) const
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
                        span(level), tolerance
                    );
                    // The coarse levels are few and always kept: a predicted brick is materialised.
                    if (encoded[i].droppable)
                        encoded[i].header.flags = kBrickPredicted;
                }
            );
            std::unordered_map<uint32_t, Reconstruction> next;
            for (auto& brick : encoded)
            {
                brick.header.payloadOffset = uint32_t(payloads.size());
                payloads.insert(payloads.end(), brick.payload.begin(), brick.payload.end());
                headers.push_back(brick.header);
                next[brick.header.coord] = brick.reconstruction;
            }
            parents.swap(next);
        }
        result.level4 = std::move(parents);
        result.bricks = headers.size();
        result.page = makePage(headers, {}, payloads);
        return result;
    }

    struct ChunkResult
    {
        std::vector<uint8_t> chunkPage;                       ///< Undeflated, page offsets relative to the level-2 page list.
        std::vector<std::pair<uint32_t, std::vector<uint8_t>>> pages; ///< Level-2 coordinate and deflated page.
        std::vector<uint32_t> pageRaw;
        uint64_t bricks = 0;
        std::array<uint64_t, 5> modes = {}; ///< Stored bricks by encoding: predicted, constant offset, 2, 4 and 8 bits.
        uint64_t payloadBytes = 0;          ///< Undeflated residual bytes.
    };

    /// Levels 3 to 0 of one chunk, top-down from its level-4 reconstruction; droppable bricks without stored descendants vanish.
    ChunkResult encodeChunk(size_t c, float tolerance, const Reconstruction& level4, bool parallelLevels) const
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
                encoded[level][i] = encodeBrick(means.coords[i], level, means.childMask[i], means.cores[i], parent, span(level), tolerance);
            };
            if (parallelLevels)
                parallelFor(means.coords.size(), encode);
            else
                for (size_t i = 0; i < means.coords.size(); ++i)
                    encode(i);
        }
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
        ChunkResult result;
        auto emit = [&](EncodedBrick& brick, std::vector<BrickHeader>& headers, std::vector<uint8_t>& payloads)
        {
            if (brick.droppable)
            {
                brick.header.flags = kBrickPredicted;
                brick.header.bits = 0;
                brick.payload.clear();
            }
            brick.header.payloadOffset = uint32_t(payloads.size());
            payloads.insert(payloads.end(), brick.payload.begin(), brick.payload.end());
            headers.push_back(brick.header);
            ++result.bricks;
            ++result.modes[brick.droppable ? 0 : 1 + (brick.header.bits == 0 ? 0 : brick.header.bits == 2 ? 1 : brick.header.bits == 4 ? 2 : 3)];
            result.payloadBytes += brick.payload.size();
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
            const std::vector<uint8_t> raw = makePage(headers, {}, payloads);
            std::vector<uint8_t> compressed;
            deflateBytes(raw.data(), raw.size(), compressed);
            result.pages.emplace_back(BrickHeader::pack(b2), std::move(compressed));
            result.pageRaw.push_back(uint32_t(raw.size()));
        }
        // The chunk page: kept level-3 bricks and the page directory (offsets filled in when written).
        std::vector<BrickHeader> headers;
        std::vector<uint8_t> payloads;
        for (size_t i3 = 0; i3 < encoded[3].size(); ++i3)
            if (kept[3][i3])
                emit(encoded[3][i3], headers, payloads);
        std::vector<PageEntry> directory(result.pages.size());
        for (size_t p = 0; p < result.pages.size(); ++p)
        {
            directory[p].coord = result.pages[p].first;
            directory[p].blob.compressed = uint32_t(result.pages[p].second.size());
            directory[p].blob.raw = result.pageRaw[p];
        }
        if (!headers.empty())
            result.chunkPage = makePage(headers, directory, payloads);
        return result;
    }

    const std::vector<uint8_t>& proxy() const { return mProxy; }
    uint32_t chunkIndex(size_t c) const { return mChunks[c].chunk; }
    uint32_t chunkTableSize() const { return mChunkDims.x * mChunkDims.y * mChunkDims.z; }

    static std::vector<uint8_t> makePage(const std::vector<BrickHeader>& headers, const std::vector<PageEntry>& pages, const std::vector<uint8_t>& payloads)
    {
        PageHeader header;
        header.brickCount = uint32_t(headers.size());
        header.pageCount = uint32_t(pages.size());
        std::vector<uint8_t> out(sizeof(PageHeader) + headers.size() * sizeof(BrickHeader) + pages.size() * sizeof(PageEntry) + payloads.size());
        uint8_t* p = out.data();
        std::memcpy(p, &header, sizeof(header));
        p += sizeof(header);
        std::memcpy(p, headers.data(), headers.size() * sizeof(BrickHeader));
        p += headers.size() * sizeof(BrickHeader);
        std::memcpy(p, pages.data(), pages.size() * sizeof(PageEntry));
        p += pages.size() * sizeof(PageEntry);
        std::memcpy(p, payloads.data(), payloads.size());
        return out;
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
    std::vector<uint8_t> mProxy;
    std::filesystem::path mCachePath;
    std::vector<BlobRef> mChunkBlobs; ///< Per chunk, in the canonical cache.
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
        asset.load();
        asset.writeCanonical(path, bytes);
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
                     "HSTRCloudCompiler pack <cache directory> <output.hstrlib> [--budget-mb N | --tolerance T] [--sample R] [--clouds a,b,...]\n";
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
    const std::filesystem::path output = argv[3];
    double budgetMB = 1000.0;
    double sampleRate = 0.2;
    double fixedTolerance = 0.0; // --tolerance: every cloud at this optical-depth tolerance, without analysis or budget.
    std::unordered_set<std::string> only;
    for (int i = 4; i + 1 < argc; i += 2)
    {
        const std::string option = argv[i];
        if (option == "--budget-mb")
            budgetMB = std::stod(argv[i + 1]);
        else if (option == "--tolerance")
            fixedTolerance = std::max(1e-6, std::stod(argv[i + 1]));
        else if (option == "--sample")
            sampleRate = std::clamp(std::stod(argv[i + 1]), 0.001, 1.0);
        else if (option == "--clouds")
            for (const std::string& name : splitList(argv[i + 1]))
                only.insert(name);
        else
        {
            std::cerr << "unknown option " << option << "\n";
            return 1;
        }
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
            sourceBytes += header.sourceBytes;
            assets.push_back(std::move(asset));
        }
        std::sort(assets.begin(), assets.end(), [](const Asset& a, const Asset& b) { return a.source < b.source; });
        if (assets.empty())
            throw std::runtime_error("no canonical cloud caches in " + cacheDirectory.string() + (only.empty() ? "" : " matching --clouds"));
        if (!only.empty() && assets.size() != only.size())
            throw std::runtime_error("--clouds names a cloud without a canonical cache");

        // 1. Analysis, cached per cloud by its density, the codec version and the sample rate.
        const std::filesystem::path analysisPath = cacheDirectory / "analysis.bin";
        auto analyses = readAnalysis(analysisPath);
        for (Asset& asset : assets)
        {
            if (fixedTolerance > 0.0)
                break;
            const std::string name = asset.source.stem().string();
            const uint64_t analysisKey = fnv(fnv(asset.densityHash, kCodecVersion), uint64_t(std::lround(sampleRate * 1e6)));
            auto cached = analyses.find(name);
            if (cached != analyses.end() && cached->second.signature == analysisKey)
                continue;
            const auto assetStart = std::chrono::steady_clock::now();
            std::cout << "analysing " << name << "\n";
            asset.open(asset.source);
            Analysis analysis;
            analysis.signature = analysisKey;
            analysis.densityHash = asset.densityHash;
            // Every k-th chunk, so the sample spans the whole cloud; its means are read once for every tolerance.
            const size_t stride = std::max<size_t>(1, size_t(std::lround(1.0 / sampleRate)));
            std::vector<size_t> sample;
            for (size_t c = 0; c < asset.chunkCount(); c += stride)
                sample.push_back(c);
            const double scale = double(asset.chunkCount()) / double(sample.size());
            asset.cacheChunks(sample);
            const uint32_t chunkDimsX = asset.entry.dims[0] / kChunkVoxels;
            const uint32_t chunkDimsY = asset.entry.dims[1] / kChunkVoxels;
            for (uint32_t t = 0; t < kLadderSize; ++t)
            {
                const float tolerance = ladderTolerance(t);
                const auto coarse = asset.encodeCoarse(tolerance);
                std::vector<uint8_t> coarseCompressed;
                deflateBytes(coarse.page.data(), coarse.page.size(), coarseCompressed);
                std::atomic<uint64_t> sampled{0};
                parallelFor(
                    sample.size(),
                    [&](size_t s)
                    {
                        const uint32_t chunk = asset.chunkIndex(sample[s]);
                        const uint3 coord(chunk % chunkDimsX, (chunk / chunkDimsX) % chunkDimsY, chunk / (chunkDimsX * chunkDimsY));
                        const auto result = asset.encodeChunk(sample[s], tolerance, coarse.level4.at(BrickHeader::pack(coord)), false);
                        uint64_t bytes = 0;
                        if (!result.chunkPage.empty())
                        {
                            std::vector<uint8_t> compressed;
                            deflateBytes(result.chunkPage.data(), result.chunkPage.size(), compressed);
                            bytes += compressed.size();
                        }
                        for (const auto& page : result.pages)
                            bytes += page.second.size();
                        sampled += bytes;
                    }
                );
                std::vector<uint8_t> proxyCompressed;
                deflateBytes(asset.proxy().data(), asset.proxy().size(), proxyCompressed);
                analysis.bytes[t] = double(sampled) * scale + double(coarseCompressed.size()) + double(proxyCompressed.size()) +
                                    double(asset.chunkTableSize()) * sizeof(BlobRef) + sizeof(AssetEntry);
            }
            asset.unload();
            analyses[name] = analysis;
            writeAnalysis(analysisPath, analyses);
            std::cout << "  " << seconds(assetStart) << " s; MB at tolerance " << ladderTolerance(0) << ".." << ladderTolerance(kLadderSize - 1) << ":";
            for (double b : analysis.bytes)
                std::cout << " " << int(b / 1048576.0);
            std::cout << "\n";
        }

        // 2. One tolerance for the library: interpolate log(bytes) over the ladder. Identical clouds count once.
        std::unordered_map<uint64_t, size_t> firstWithHash;
        std::vector<size_t> aliasOf(assets.size(), kNoAlias);
        for (size_t i = 0; i < assets.size(); ++i)
        {
            auto [it, inserted] = firstWithHash.try_emplace(assets[i].densityHash, i);
            if (!inserted)
                aliasOf[i] = it->second;
        }
        // Estimated bytes of asset i at a continuous ladder position (log-linear between measured tolerances).
        auto assetBytes = [&](size_t i, double ladder)
        {
            if (aliasOf[i] != kNoAlias || fixedTolerance > 0.0)
                return 0.0;
            const auto& b = analyses.at(assets[i].source.stem().string()).bytes;
            const uint32_t lo = std::min<uint32_t>(uint32_t(std::max(ladder, 0.0)), kLadderSize - 2);
            const double f = std::clamp(ladder - lo, 0.0, 1.0);
            return std::exp(std::log(std::max(b[lo], 1.0)) * (1.0 - f) + std::log(std::max(b[lo + 1], 1.0)) * f);
        };
        auto estimate = [&](double ladder)
        {
            double sum = double(sizeof(LibraryHeader) + assets.size() * sizeof(AssetEntry));
            for (size_t i = 0; i < assets.size(); ++i)
                sum += assetBytes(i, ladder);
            return sum;
        };
        // 2. The finest ladder position whose estimate fits the budget (a soft limit: the one tolerance is kept for every cloud).
        const uint64_t budgetBytes = uint64_t(budgetMB * 1048576.0);
        double ladder = 0.0;
        if (fixedTolerance > 0.0)
            ladder = std::log2(fixedTolerance / ladderTolerance(0));
        else if (estimate(0.0) > double(budgetBytes))
        {
            double lo = 0.0, hi = kLadderSize - 1;
            for (int i = 0; i < 40; ++i)
            {
                const double mid = 0.5 * (lo + hi);
                (estimate(mid) > double(budgetBytes) ? lo : hi) = mid;
            }
            ladder = hi;
        }
        const float tolerance = float(ladderTolerance(0) * std::pow(2.0, ladder));
        if (fixedTolerance > 0.0)
            std::cout << "tolerance " << tolerance << " (fixed)\n";
        else
            std::cout << "tolerance " << tolerance << " for an estimated " << estimate(ladder) / 1048576.0 << " MB (budget " << budgetMB << " MB)\n";

        // 3. Compile.
        std::filesystem::create_directories(output.parent_path());
        const std::filesystem::path temporary = output.string() + ".partial";
        std::fstream file(temporary, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
        LibraryHeader header;
        header.assetCount = uint32_t(assets.size());
        header.sourceBytes = sourceBytes;
        header.tolerance = tolerance;
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        auto append = [&](const std::vector<uint8_t>& bytes)
        {
            BlobRef ref;
            ref.offset = uint64_t(file.tellp());
            ref.compressed = uint32_t(bytes.size());
            file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
            return ref;
        };
        for (size_t i = 0; i < assets.size(); ++i)
        {
            Asset& asset = assets[i];
            const auto assetStart = std::chrono::steady_clock::now();
            if (aliasOf[i] != kNoAlias)
            {
                asset.entry = assets[aliasOf[i]].entry;
                const std::string name = asset.source.stem().string();
                std::fill(std::begin(asset.entry.name), std::end(asset.entry.name), 0);
                std::copy_n(name.c_str(), std::min<size_t>(name.size(), 63), asset.entry.name);
                asset.entry.aliasOf = uint32_t(aliasOf[i]);
                std::cout << name << ": identical density to " << assets[aliasOf[i]].entry.name << ", shared\n";
                continue;
            }
            const uint64_t assetOffset = uint64_t(file.tellp());
            std::cout << "packing " << asset.source.stem().string() << "\n";
            asset.open(asset.source);
            std::vector<uint8_t> compressed;
            deflateBytes(asset.proxy().data(), asset.proxy().size(), compressed);
            asset.entry.proxy = append(compressed);
            asset.entry.proxy.raw = uint32_t(asset.proxy().size());
            auto coarse = asset.encodeCoarse(tolerance);
            deflateBytes(coarse.page.data(), coarse.page.size(), compressed);
            asset.entry.coarse = append(compressed);
            asset.entry.coarse.raw = uint32_t(coarse.page.size());
            asset.entry.storedBricks = coarse.bricks;

            // Chunks in parallel, written in order of completion (their table is written last).
            std::vector<BlobRef> table(asset.chunkTableSize());
            std::mutex writeMutex;
            const uint32_t chunkDimsX = asset.entry.dims[0] / kChunkVoxels;
            const uint32_t chunkDimsY = asset.entry.dims[1] / kChunkVoxels;
            uint64_t bricks = 0;
            uint64_t payloadBytes = 0;
            std::array<uint64_t, 5> modes = {};
            parallelFor(
                asset.chunkCount(),
                [&](size_t c)
                {
                    const uint32_t chunk = asset.chunkIndex(c);
                    const uint3 coord(chunk % chunkDimsX, (chunk / chunkDimsX) % chunkDimsY, chunk / (chunkDimsX * chunkDimsY));
                    auto result = asset.encodeChunk(c, tolerance, coarse.level4.at(BrickHeader::pack(coord)), false);
                    std::lock_guard lock(writeMutex);
                    bricks += result.bricks;
                    payloadBytes += result.payloadBytes;
                    for (size_t m = 0; m < modes.size(); ++m)
                        modes[m] += result.modes[m];
                    if (result.chunkPage.empty())
                        return;
                    // Level-2 pages first; then the chunk page with their final offsets.
                    PageHeader pageHeader;
                    std::memcpy(&pageHeader, result.chunkPage.data(), sizeof(pageHeader));
                    auto* directory =
                        reinterpret_cast<PageEntry*>(result.chunkPage.data() + sizeof(PageHeader) + pageHeader.brickCount * sizeof(BrickHeader));
                    for (size_t p = 0; p < result.pages.size(); ++p)
                        directory[p].blob.offset = append(result.pages[p].second).offset;
                    std::vector<uint8_t> page;
                    deflateBytes(result.chunkPage.data(), result.chunkPage.size(), page);
                    table[chunk] = append(page);
                    table[chunk].raw = uint32_t(result.chunkPage.size());
                }
            );
            asset.entry.storedBricks += bricks;
            asset.entry.chunkTableOffset = uint64_t(file.tellp());
            file.write(reinterpret_cast<const char*>(table.data()), std::streamsize(table.size() * sizeof(BlobRef)));
            asset.entry.storedBytes = uint64_t(file.tellp()) - assetOffset;
            std::cout << "  " << asset.entry.storedBricks << " bricks stored of " << asset.entry.level0Bricks << " source level-0 bricks, "
                      << double(asset.entry.storedBytes) / 1048576.0 << " MB, " << seconds(assetStart) << " s\n"
                      << "  levels 0-3: predicted " << modes[0] << ", offset " << modes[1] << ", 2-bit " << modes[2] << ", 4-bit " << modes[3]
                      << ", 8-bit " << modes[4] << "; headers " << double(bricks * sizeof(BrickHeader)) / 1048576.0 << " MB, residuals "
                      << double(payloadBytes) / 1048576.0 << " MB before deflate\n";
            asset.unload();
        }
        header.assetTableOffset = uint64_t(file.tellp());
        for (const Asset& asset : assets)
            file.write(reinterpret_cast<const char*>(&asset.entry), sizeof(AssetEntry));
        header.packageBytes = uint64_t(file.tellp());
        file.seekp(0);
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        file.close();
        std::filesystem::rename(temporary, output);
        std::cout << "wrote " << output.string() << ": " << double(header.packageBytes) / 1048576.0 << " MB from "
                  << double(sourceBytes) / 1048576.0 << " MB of VDB at tolerance " << tolerance << ", " << seconds(start) << " s\n";
        if (fixedTolerance <= 0.0 && header.packageBytes > budgetBytes)
            std::cout << "warning: the package exceeds the " << budgetMB << " MB budget (a soft limit)\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
