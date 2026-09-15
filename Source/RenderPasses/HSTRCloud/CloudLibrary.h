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
    uint32_t file = 0; ///< Package file (CloudLibrary::files) its blobs are in.
    int3 sourceMin = int3(0);
    uint3 dims = uint3(0); ///< Source voxels, multiples of 128.
    uint3 chunkDims = uint3(0);
    float voxelWorld = 1.f; ///< World size of one source voxel in the VDB.
    uint32_t topLevel = 4;
    uint32_t proxyLevel = 0;
    uint3 proxyDims = uint3(0);
    std::vector<float> proxyMean;       ///< Level-proxyLevel mean density.
    std::vector<float> proxyMax;        ///< Maximum source density (with the trilinear apron) per proxy voxel.
    std::vector<uint8_t> coarseMeta;    ///< Decompressed meta of the page of the bricks of levels >= 4.
    std::vector<uint8_t> coarsePayload; ///< Its decompressed payload.
    std::vector<PageBlobs> chunks;      ///< Chunk pages.

    uint32_t chunkIndex(uint3 c) const { return c.x + chunkDims.x * (c.y + chunkDims.y * c.z); }
};

struct CloudLibrary
{
    std::vector<std::filesystem::path> files;
    std::vector<CloudAsset> assets; ///< Clouds with distinct density (later copies of an identical cloud are dropped).
};

/// A page's meta as the runtime uses it: brick headers and the level-2 page directory (chunk pages).
struct DecodedPage
{
    std::vector<BrickHeader> bricks;
    std::vector<PageEntry> pages;
};

/// Loads a package file, or every package (*.hstrlib) in a directory.
CloudLibrary loadCloudLibrary(const std::filesystem::path& path);

/// Splits a decompressed page meta; payloadBytes (the page's decompressed payload size) bounds its bricks' payloads.
bool decodeMeta(const std::vector<uint8_t>& raw, uint32_t payloadBytes, DecodedPage& page);

/// Reads and GDeflate-decompresses one blob on the CPU.
bool readBlob(std::ifstream& file, const BlobRef& blob, std::vector<uint8_t>& raw);

} // namespace hstrcloud
