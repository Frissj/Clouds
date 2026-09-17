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

/// Where the sea's workers write a tile's GPU upload: host-visible staging buffers of one tile's packed volume each (HSTRCloud).
class TileStaging
{
public:
    virtual ~TileStaging() = default;
    /// A free buffer and its mapped data, or -1 when none is free (the tile then uploads from host memory). Thread safe.
    virtual int32_t acquire(uint32_t*& data) = 0;
    /// Returns a buffer whose tile was dropped before its upload. Thread safe.
    virtual void discard(int32_t handle) = 0;
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
        int32_t contentLow = -1;  ///< Lowest domain voxel layer with a non-zero maximum (-1: none).
        int32_t contentHigh = -1; ///< Highest.
    };
    /// A slot's domain proxy: tileVoxels x dims.y x tileVoxels voxels (x + r (y + dims.y z)).
    struct TileVolume
    {
        /// What the GPU domain still has to take on for the slot.
        enum class Upload : uint8_t
        {
            Done,   ///< Nothing: the GPU holds the volume.
            Zero,   ///< An empty tile.
            Staged, ///< The packed volume in staging buffer `staged`.
            Packed, ///< The packed volume in `packed` (no staging buffer was free).
        };
        std::vector<float> mean; ///< Mean density (the residency's visibility reads it).
        /// The GPU upload, converted on the worker, per voxel: float16 mean in the low bits, and above it the float16 conservative
        /// maximum over the voxel's footprint, rounded up (the RG16 texel layout).
        std::vector<uint32_t> packed;
        Upload upload = Upload::Zero;
        int32_t staged = -1;
    };

    CloudSea(std::vector<CloudAsset> assets, const CloudSeaDesc& desc);
    ~CloudSea();

    /// Recentres the window on the camera and applies finished rasterizations (unless applyResults is false: they wait for a later
    /// update, while something is reading the tiles). Returns the slots whose content changed.
    std::vector<uint32_t> update(float3 cameraPosition, bool applyResults = true);
    /// Rasterizes every tile of the window around the camera before returning.
    void fill(float3 cameraPosition);

    const std::vector<CloudAsset>& getAssets() const { return mAssets; }
    const CloudSeaDesc& getDesc() const { return mDesc; }
    uint3 getDims() const { return mDims; }
    float getVoxelWorld() const { return mVoxelWorld; }
    /// World size of a VDB unit at instance scale 1 (an asset's source voxel is asset.voxelWorld times this, times the scale).
    float getFitScale() const { return mFitScale; }
    /// Mean density of a domain voxel (inside the domain).
    float meanAt(uint3 voxel) const
    {
        const uint32_t r = mDesc.tileVoxels;
        const uint32_t slot = voxel.x / r + mDesc.tiles * (voxel.z / r);
        return mVolumes[slot].mean[voxel.x % r + size_t(r) * (voxel.y + size_t(mDims.y) * (voxel.z % r))];
    }
    const TileVolume& getVolume(uint32_t slot) const { return mVolumes[slot]; }
    /// The volume of a slot the GPU upload takes on (it sets Upload::Done and frees the packed copy).
    TileVolume& getUploadVolume(uint32_t slot) { return mVolumes[slot]; }
    /// Workers write GPU uploads into the staging buffers from here on (it must outlive the sea).
    void setStaging(TileStaging* pStaging)
    {
        std::lock_guard lock(mMutex);
        mpStaging = pStaging;
    }
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
        TileVolume volume;
    };

    Tile makeTile(int2 world) const;
    Result rasterize(const Job& job, TileStaging* pStaging) const;
    void apply(Result& result);
    void discard(TileVolume& volume);
    void worker();

    std::vector<CloudAsset> mAssets;
    CloudSeaDesc mDesc;
    uint3 mDims = uint3(0);
    float mVoxelWorld = 1.f;
    float mFitScale = 1.f;                ///< World size of a VDB unit so the largest cloud fits a tile.
    std::vector<uint3> mContentMin;       ///< Per asset: bounds of non-zero source voxels.
    std::vector<uint3> mContentMax;
    std::vector<TileVolume> mVolumes;     ///< Per slot (a finished tile's volume is swapped in).
    std::vector<Tile> mTiles;             ///< Per slot.
    std::vector<int2> mRequested;         ///< Per slot: world tile queued or applied.
    int2 mCenter = int2(std::numeric_limits<int32_t>::min());

    mutable std::mutex mMutex;
    std::condition_variable mWake;
    std::deque<Job> mJobs;
    std::vector<Result> mResults;
    uint32_t mBusy = 0;
    bool mStop = false;
    TileStaging* mpStaging = nullptr;
    std::vector<std::thread> mWorkers;
};
} // namespace hstrcloud
