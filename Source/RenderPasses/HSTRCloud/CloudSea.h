/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "CloudLibrary.h"
#include "HSTRCloudTypes.slang"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace hstrcloud
{
struct CloudSeaDesc
{
    float3 origin = float3(0.f); ///< World corner of tile (0, 0) at the base of the cloud layer.
    float tileWorld = 320.f;     ///< Tile edge in world units.
    float layerHeight = 160.f;   ///< Height of the cloud layer.
    uint32_t tileVoxels = 64;    ///< Domain voxels per tile edge (a multiple of 16).
    uint32_t tiles = 8;          ///< Tiles per window axis (the domain is a torus of tiles x tiles).
    uint32_t seed = 1;
    float coverage = 0.85f; ///< Probability that a tile holds a cloud.
};

/// An endless procedural cloud sea. World tile (tx, tz) holds a hashed library asset with a quarter-turn rotation, mirror, scale
/// and offset. A camera-centred window of tiles x tiles is resident in a toroidal domain (slot = tile mod tiles): tiles entering the
/// window are rasterized into the domain proxy on a worker thread, so flight never runs out of clouds and memory stays constant.
class CloudSea
{
public:
    struct Tile
    {
        int2 world = int2(std::numeric_limits<int32_t>::min());
        bool occupied = false;
        HSTRCloudInstance instance = {};
        float3 worldMin = float3(0.f); ///< World bounds of the instance.
        float3 worldMax = float3(0.f);
    };

    CloudSea(std::vector<CloudAsset> assets, const CloudSeaDesc& desc);
    ~CloudSea();

    /// Recentres the window on the camera and applies finished rasterizations. Returns the slots whose content changed.
    std::vector<uint32_t> update(float3 cameraPosition);
    /// Rasterizes every tile of the window around the camera before returning.
    void fill(float3 cameraPosition);

    const std::vector<CloudAsset>& getAssets() const { return mAssets; }
    const CloudSeaDesc& getDesc() const { return mDesc; }
    uint3 getDims() const { return mDims; }
    float getVoxelWorld() const { return mVoxelWorld; }
    const std::vector<float>& getMean() const { return mMean; }
    const std::vector<float>& getMax() const { return mMax; }
    const std::vector<Tile>& getTiles() const { return mTiles; }
    uint32_t slotIndex(int2 worldTile) const;
    uint32_t pendingTiles() const;

private:
    struct Job
    {
        uint32_t slot = 0;
        int2 world = int2(0);
    };
    struct Result
    {
        Tile tile;
        uint32_t slot = 0;
        std::vector<float> mean; ///< Tile box of tileVoxels x dims.y x tileVoxels.
        std::vector<float> max;
    };

    Tile makeTile(int2 world) const;
    Result rasterize(const Job& job) const;
    void apply(Result& result);
    void worker();

    std::vector<CloudAsset> mAssets;
    CloudSeaDesc mDesc;
    uint3 mDims = uint3(0);
    float mVoxelWorld = 1.f;
    float mFitScale = 1.f;                ///< World size of a VDB unit so the largest cloud fits a tile.
    std::vector<uint3> mContentMin;       ///< Per asset: bounds of non-zero source voxels.
    std::vector<uint3> mContentMax;
    std::vector<float> mMean;
    std::vector<float> mMax;
    std::vector<Tile> mTiles;             ///< Per slot.
    std::vector<int2> mRequested;         ///< Per slot: world tile queued or applied.
    int2 mCenter = int2(std::numeric_limits<int32_t>::min());

    mutable std::mutex mMutex;
    std::condition_variable mWake;
    std::deque<Job> mJobs;
    std::vector<Result> mResults;
    uint32_t mBusy = 0;
    bool mStop = false;
    std::vector<std::thread> mWorkers;
};
} // namespace hstrcloud
