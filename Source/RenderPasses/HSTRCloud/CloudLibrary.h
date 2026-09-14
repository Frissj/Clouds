/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "CloudFormat.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace hstrcloud
{
/// Always-resident description of one cloud of a compiled library package. Fine bricks stream from its chunk and level-2 pages.
struct CloudAsset
{
    std::string name;
    int3 sourceMin = int3(0);
    uint3 dims = uint3(0); ///< Source voxels, multiples of 128.
    uint3 chunkDims = uint3(0);
    float voxelWorld = 1.f; ///< World size of one source voxel in the VDB.
    uint32_t topLevel = 4;
    uint32_t proxyLevel = 0;
    uint3 proxyDims = uint3(0);
    std::vector<float> proxyMean; ///< Level-proxyLevel mean density.
    std::vector<float> proxyMax;  ///< Maximum source density (with the trilinear apron) per proxy voxel.
    std::vector<uint8_t> coarsePage; ///< Inflated page of the bricks of levels >= 4.
    std::vector<BlobRef> chunks;     ///< Chunk pages.

    uint32_t chunkIndex(uint3 c) const { return c.x + chunkDims.x * (c.y + chunkDims.y * c.z); }
};

struct CloudLibrary
{
    std::filesystem::path path;
    LibraryHeader header;
    std::vector<CloudAsset> assets; ///< Clouds with distinct density (aliases of identical clouds are dropped).
};

/// A page as the runtime uses it: brick headers, the level-2 page directory (chunk pages) and the packed residuals.
struct DecodedPage
{
    std::vector<BrickHeader> bricks;
    std::vector<PageEntry> pages;
    std::vector<uint8_t> payload;
};

CloudLibrary loadCloudLibrary(const std::filesystem::path& path);

/// Splits an inflated page.
bool decodePage(const std::vector<uint8_t>& raw, DecodedPage& page);

/// Reads and inflates one blob.
bool readBlob(std::ifstream& file, const BlobRef& blob, std::vector<uint8_t>& raw);

} // namespace hstrcloud
