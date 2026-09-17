/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "CloudSea.h"

#include <algorithm>

namespace hstrcloud
{
namespace
{
uint32_t hashUint(uint32_t value)
{
    const uint32_t state = value * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct TileRng
{
    uint32_t state;
    float next()
    {
        state = hashUint(state);
        return float(state >> 8) / 16777216.f;
    }
};
} // namespace

CloudSea::CloudSea(std::vector<CloudAsset> assets, const CloudSeaDesc& desc) : mAssets(std::move(assets)), mDesc(desc)
{
    FALCOR_CHECK(!mAssets.empty(), "HSTRCloud: the cloud sea needs at least one asset.");
    mDesc.tileVoxels = std::max(16u, (mDesc.tileVoxels + 15u) / 16u * 16u);
    mDesc.tiles = std::max(2u, mDesc.tiles);
    mVoxelWorld = mDesc.tileWorld / float(mDesc.tileVoxels);
    const uint32_t layerVoxels = std::max(16u, (uint32_t(std::ceil(mDesc.layerHeight / mVoxelWorld)) + 15u) / 16u * 16u);
    mDims = uint3(mDesc.tiles * mDesc.tileVoxels, layerVoxels, mDesc.tiles * mDesc.tileVoxels);

    // Clouds keep their relative physical sizes: one scale fits the largest into a tile.
    mFitScale = std::numeric_limits<float>::max();
    for (const CloudAsset& asset : mAssets)
    {
        uint3 lo(std::numeric_limits<uint32_t>::max());
        uint3 hi(0);
        for (uint32_t z = 0; z < asset.proxyDims.z; ++z)
            for (uint32_t y = 0; y < asset.proxyDims.y; ++y)
                for (uint32_t x = 0; x < asset.proxyDims.x; ++x)
                    if (asset.proxyMax[x + asset.proxyDims.x * (size_t(y) + asset.proxyDims.y * size_t(z))] > 0.f)
                    {
                        lo = min(lo, uint3(x, y, z));
                        hi = max(hi, uint3(x, y, z));
                    }
        if (any(lo > hi))
            lo = hi = uint3(0);
        mContentMin.push_back(lo * (1u << asset.proxyLevel));
        mContentMax.push_back(min((hi + 1u) * (1u << asset.proxyLevel) - 1u, asset.dims - 1u));
        const float3 extent = float3(mContentMax.back() - mContentMin.back() + 1u) * asset.voxelWorld;
        mFitScale = std::min(mFitScale, 0.98f * std::min(mDesc.tileWorld / std::max(extent.x, extent.z), mDesc.layerHeight / extent.y));
    }
    mMean.assign(size_t(mDims.x) * mDims.y * mDims.z, 0.f);
    mMax.assign(mMean.size(), 0.f);
    mTiles.resize(mDesc.tiles * mDesc.tiles);
    mRequested.assign(mTiles.size(), int2(std::numeric_limits<int32_t>::min()));
    const uint32_t workerCount = std::max(1u, std::thread::hardware_concurrency() / 4);
    for (uint32_t i = 0; i < workerCount; ++i)
        mWorkers.emplace_back([this] { worker(); });
    logInfo(
        "HSTRCloud: cloud sea of {} assets, {}^2 tiles of {} world units, domain {} voxels of {:.2f}, cloud scale {:.3g}.",
        mAssets.size(),
        mDesc.tiles,
        mDesc.tileWorld,
        mDims,
        mVoxelWorld,
        mFitScale
    );
}

CloudSea::~CloudSea()
{
    {
        std::lock_guard lock(mMutex);
        mStop = true;
    }
    mWake.notify_all();
    for (auto& thread : mWorkers)
        thread.join();
}

uint32_t CloudSea::slotIndex(int2 worldTile) const
{
    const int32_t n = int32_t(mDesc.tiles);
    const int2 slot((worldTile.x % n + n) % n, (worldTile.y % n + n) % n);
    return uint32_t(slot.x + n * slot.y);
}

uint32_t CloudSea::pendingTiles() const
{
    std::lock_guard lock(mMutex);
    return uint32_t(mJobs.size() + mResults.size()) + mBusy;
}

CloudSea::Tile CloudSea::makeTile(int2 world) const
{
    Tile tile;
    tile.world = world;
    TileRng rng{hashUint(uint32_t(world.x) * 73856093u ^ hashUint(uint32_t(world.y) * 19349663u ^ hashUint(mDesc.seed)))};
    tile.occupied = rng.next() < mDesc.coverage;
    const uint32_t assetID = std::min(uint32_t(rng.next() * float(mAssets.size())), uint32_t(mAssets.size()) - 1);
    const CloudAsset& asset = mAssets[assetID];
    const uint32_t turns = std::min(uint32_t(rng.next() * 4.f), 3u);
    const bool mirror = rng.next() < 0.5f;
    const float scale = kCloudSeaMinScale + (1.f - kCloudSeaMinScale) * rng.next();
    const float sourceVoxelWorld = asset.voxelWorld * mFitScale * scale;
    const float3 contentVoxels = float3(mContentMax[assetID] - mContentMin[assetID] + 1u);
    const float3 extent = contentVoxels * sourceVoxelWorld;
    const float3 footprint = (turns & 1u) ? float3(extent.z, extent.y, extent.x) : extent;
    const float3 offset(
        rng.next() * std::max(0.f, mDesc.tileWorld - footprint.x),
        0.3f * rng.next() * std::max(0.f, mDesc.layerHeight - footprint.y),
        rng.next() * std::max(0.f, mDesc.tileWorld - footprint.z)
    );

    // Content voxel c (relative to the content minimum) to tile-local world q: q = M c + t, with M a scaled, mirrored quarter turn.
    float3x3 linear = float3x3::identity();
    if (mirror)
        linear[0][0] = -1.f;
    for (uint32_t i = 0; i < turns; ++i)
        linear = mul(float3x3{0.f, 0.f, 1.f, 0.f, 1.f, 0.f, -1.f, 0.f, 0.f}, linear);
    const float3x3 forward = linear * sourceVoxelWorld;
    const float3 centre = offset + 0.5f * footprint;
    const float3 translation = centre - mul(forward, 0.5f * (contentVoxels - 1.f));
    // The inverse of a scaled signed permutation is its transpose over the squared scale.
    const float3x3 inverse = transpose(linear) * (1.f / sourceVoxelWorld);
    // Domain tile-local voxel p (voxel centres at integers) to asset source voxel x: q = (p + 0.5) size.
    const float3x3 a = inverse * mVoxelWorld;
    const float3 b = mul(inverse, float3(0.5f * mVoxelWorld) - translation) + float3(mContentMin[assetID]);
    tile.instance.row0 = float4(a[0][0], a[0][1], a[0][2], b.x);
    tile.instance.row1 = float4(a[1][0], a[1][1], a[1][2], b.y);
    tile.instance.row2 = float4(a[2][0], a[2][1], a[2][2], b.z);
    tile.instance.asset = assetID;
    tile.instance.sourceVoxelWorld = sourceVoxelWorld;
    tile.instance.proxyLevel = std::log2(mVoxelWorld / sourceVoxelWorld);
    tile.instance.occupied = tile.occupied ? 1u : 0u;
    const float3 tileCorner = mDesc.origin + float3(float(world.x), 0.f, float(world.y)) * mDesc.tileWorld;
    tile.worldMin = tileCorner + offset;
    tile.worldMax = tileCorner + offset + footprint;
    return tile;
}

CloudSea::Result CloudSea::rasterize(const Job& job) const
{
    Result result;
    result.slot = job.slot;
    result.tile = makeTile(job.world);
    const uint32_t r = mDesc.tileVoxels;
    result.mean.assign(size_t(r) * mDims.y * r, 0.f);
    result.max.assign(result.mean.size(), 0.f);
    if (!result.tile.occupied)
        return result;
    const HSTRCloudInstance& instance = result.tile.instance;
    const CloudAsset& asset = mAssets[instance.asset];
    const float3x3 a{
        instance.row0.x, instance.row0.y, instance.row0.z, instance.row1.x, instance.row1.y, instance.row1.z, instance.row2.x,
        instance.row2.y, instance.row2.z};
    const float3 b(instance.row0.w, instance.row1.w, instance.row2.w);
    const float proxyScale = float(1u << asset.proxyLevel);
    const float3 halfBox =
        0.5f * float3(
                   std::abs(a[0][0]) + std::abs(a[0][1]) + std::abs(a[0][2]),
                   std::abs(a[1][0]) + std::abs(a[1][1]) + std::abs(a[1][2]),
                   std::abs(a[2][0]) + std::abs(a[2][1]) + std::abs(a[2][2])
               );
    const uint32_t subsamples = std::clamp(uint32_t(std::ceil(mVoxelWorld / instance.sourceVoxelWorld / proxyScale)), 1u, 4u);
    auto proxyIndex = [&](uint3 p) { return p.x + asset.proxyDims.x * (size_t(p.y) + asset.proxyDims.y * size_t(p.z)); };
    auto proxyMean = [&](float3 x)
    {
        const float3 l = (x + 0.5f) / proxyScale - 0.5f;
        const int3 base = int3(floor(l));
        const float3 f = l - float3(base);
        float value = 0.f;
        for (uint32_t corner = 0; corner < 8; ++corner)
        {
            const int3 q = base + int3(corner & 1, (corner >> 1) & 1, corner >> 2);
            if (any(q < 0) || any(q >= int3(asset.proxyDims)))
                continue;
            const float3 w((corner & 1) ? f.x : 1.f - f.x, (corner & 2) ? f.y : 1.f - f.y, (corner & 4) ? f.z : 1.f - f.z);
            value += w.x * w.y * w.z * asset.proxyMean[proxyIndex(uint3(q))];
        }
        return value;
    };

    const float3 tileCorner = mDesc.origin + float3(float(job.world.x), 0.f, float(job.world.y)) * mDesc.tileWorld;
    const int3 lo = max(int3(floor((result.tile.worldMin - tileCorner) / mVoxelWorld)) - 2, int3(0));
    const int3 hi = min(int3(ceil((result.tile.worldMax - tileCorner) / mVoxelWorld)) + 2, int3(r, mDims.y, r) - 1);
    for (int32_t z = lo.z; z <= hi.z; ++z)
        for (int32_t y = lo.y; y <= hi.y; ++y)
            for (int32_t x = lo.x; x <= hi.x; ++x)
            {
                const float3 p{float(x), float(y), float(z)};
                float mean = 0.f;
                for (uint32_t sz = 0; sz < subsamples; ++sz)
                    for (uint32_t sy = 0; sy < subsamples; ++sy)
                        for (uint32_t sx = 0; sx < subsamples; ++sx)
                        {
                            const float3 u = (float3(float(sx), float(sy), float(sz)) + 0.5f) / float(subsamples) - 0.5f;
                            mean += proxyMean(mul(a, p + u) + b);
                        }
                mean /= float(subsamples * subsamples * subsamples);
                // Conservative maximum over the voxel's footprint plus one source voxel of trilinear support.
                const float3 centre = mul(a, p) + b;
                const float3 boxLo = (centre - halfBox - 1.f) / proxyScale;
                const float3 boxHi = (centre + halfBox + 1.f) / proxyScale;
                if (any(boxHi < 0.f) || any(boxLo >= float3(asset.proxyDims)))
                    continue;
                const uint3 plo = uint3(max(floor(boxLo), float3(0.f)));
                const uint3 phi = min(uint3(max(floor(boxHi), float3(0.f))), asset.proxyDims - 1u);
                float maximum = 0.f;
                for (uint32_t qz = plo.z; qz <= phi.z; ++qz)
                    for (uint32_t qy = plo.y; qy <= phi.y; ++qy)
                        for (uint32_t qx = plo.x; qx <= phi.x; ++qx)
                            maximum = std::max(maximum, asset.proxyMax[proxyIndex(uint3(qx, qy, qz))]);
                const size_t index = size_t(x) + r * (size_t(y) + size_t(mDims.y) * size_t(z));
                result.mean[index] = std::min(mean, maximum);
                result.max[index] = maximum;
            }
    return result;
}

void CloudSea::apply(Result& result)
{
    const uint32_t r = mDesc.tileVoxels;
    const uint3 corner(result.slot % mDesc.tiles * r, 0, result.slot / mDesc.tiles * r);
    for (uint32_t z = 0; z < r; ++z)
        for (uint32_t y = 0; y < mDims.y; ++y)
        {
            const size_t source = size_t(r) * (size_t(y) + size_t(mDims.y) * z);
            const size_t target = corner.x + size_t(mDims.x) * (size_t(y) + size_t(mDims.y) * (corner.z + z));
            std::copy_n(result.mean.begin() + source, r, mMean.begin() + target);
            std::copy_n(result.max.begin() + source, r, mMax.begin() + target);
        }
    mTiles[result.slot] = result.tile;
}

void CloudSea::worker()
{
    for (;;)
    {
        Job job;
        {
            std::unique_lock lock(mMutex);
            mWake.wait(lock, [&] { return mStop || !mJobs.empty(); });
            if (mStop)
                return;
            job = mJobs.front();
            mJobs.pop_front();
            ++mBusy;
        }
        Result result = rasterize(job);
        {
            std::lock_guard lock(mMutex);
            mResults.push_back(std::move(result));
            --mBusy;
        }
        mWake.notify_all();
    }
}

std::vector<uint32_t> CloudSea::update(float3 cameraPosition, bool applyResults)
{
    const int32_t n = int32_t(mDesc.tiles);
    const int2 cameraTile(
        int32_t(std::floor((cameraPosition.x - mDesc.origin.x) / mDesc.tileWorld)),
        int32_t(std::floor((cameraPosition.z - mDesc.origin.z) / mDesc.tileWorld))
    );
    std::vector<uint32_t> changed;
    std::vector<Result> results;
    {
        std::lock_guard lock(mMutex);
        if (any(cameraTile != mCenter))
        {
            mCenter = cameraTile;
            std::vector<Job> jobs;
            for (int32_t dz = 0; dz < n; ++dz)
                for (int32_t dx = 0; dx < n; ++dx)
                {
                    const int2 world = cameraTile - n / 2 + int2(dx, dz);
                    const uint32_t slot = slotIndex(world);
                    if (all(mRequested[slot] == world))
                        continue;
                    mRequested[slot] = world;
                    jobs.push_back({slot, world});
                }
            // Superseded requests for the same slots are dropped; the nearest tiles come first.
            mJobs.erase(
                std::remove_if(mJobs.begin(), mJobs.end(), [&](const Job& job) { return any(mRequested[job.slot] != job.world); }),
                mJobs.end()
            );
            std::sort(
                jobs.begin(),
                jobs.end(),
                [&](const Job& p, const Job& q)
                {
                    const int2 dp = abs(p.world - cameraTile);
                    const int2 dq = abs(q.world - cameraTile);
                    return std::max(dp.x, dp.y) < std::max(dq.x, dq.y);
                }
            );
            mJobs.insert(mJobs.end(), jobs.begin(), jobs.end());
        }
        if (applyResults)
            results.swap(mResults);
    }
    mWake.notify_all();
    for (Result& result : results)
    {
        if (any(mRequested[result.slot] != result.tile.world))
            continue; // A later request replaced it.
        apply(result);
        changed.push_back(result.slot);
    }
    return changed;
}

void CloudSea::fill(float3 cameraPosition)
{
    update(cameraPosition);
    {
        std::unique_lock lock(mMutex);
        mWake.wait(lock, [&] { return mJobs.empty() && mBusy == 0; });
    }
    update(cameraPosition);
}
} // namespace hstrcloud
