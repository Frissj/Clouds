/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "CloudLibrary.h"
#include "Utils/Math/Float16.h"

namespace hstrcloud
{
bool readBlob(std::ifstream& file, const BlobRef& blob, std::vector<uint8_t>& raw)
{
    if (blob.compressed == 0)
        return false;
    std::vector<uint8_t> compressed(blob.compressed);
    file.clear();
    file.seekg(std::streamoff(blob.offset));
    if (!file.read(reinterpret_cast<char*>(compressed.data()), std::streamsize(compressed.size())))
        return false;
    return inflateBytes(compressed.data(), compressed.size(), raw, blob.raw);
}

bool decodePage(const std::vector<uint8_t>& raw, DecodedPage& page)
{
    if (raw.size() < sizeof(PageHeader))
        return false;
    PageHeader header;
    std::memcpy(&header, raw.data(), sizeof(header));
    const size_t bricksEnd = sizeof(PageHeader) + size_t(header.brickCount) * sizeof(BrickHeader);
    const size_t pagesEnd = bricksEnd + size_t(header.pageCount) * sizeof(PageEntry);
    if (pagesEnd > raw.size())
        return false;
    page.bricks.resize(header.brickCount);
    page.pages.resize(header.pageCount);
    std::memcpy(page.bricks.data(), raw.data() + sizeof(PageHeader), page.bricks.size() * sizeof(BrickHeader));
    std::memcpy(page.pages.data(), raw.data() + bricksEnd, page.pages.size() * sizeof(PageEntry));
    page.payload.assign(raw.begin() + pagesEnd, raw.end());
    for (const BrickHeader& brick : page.bricks)
        if (brick.payloadOffset + brick.payloadBytes() > page.payload.size())
            return false;
    return true;
}

CloudLibrary loadCloudLibrary(const std::filesystem::path& path)
{
    CloudLibrary library;
    library.path = path;
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(&library.header), sizeof(library.header)) || library.header.magic != kLibraryMagic ||
        library.header.version != kLibraryVersion)
        FALCOR_THROW("HSTRCloud: '{}' is not an HSTR cloud library package (compile it with HSTRCloudCompiler).", path);
    std::vector<AssetEntry> entries(library.header.assetCount);
    file.seekg(std::streamoff(library.header.assetTableOffset));
    file.read(reinterpret_cast<char*>(entries.data()), std::streamsize(entries.size() * sizeof(AssetEntry)));
    if (!file)
        FALCOR_THROW("HSTRCloud: cannot read the asset table of '{}'.", path);
    for (const AssetEntry& entry : entries)
    {
        if (entry.aliasOf != kNoAlias)
            continue; // Identical density: the original already represents it.
        CloudAsset asset;
        asset.name = std::string(entry.name, strnlen(entry.name, sizeof(entry.name)));
        asset.sourceMin = int3(entry.sourceMin[0], entry.sourceMin[1], entry.sourceMin[2]);
        asset.dims = uint3(entry.dims[0], entry.dims[1], entry.dims[2]);
        asset.chunkDims = asset.dims / kChunkVoxels;
        asset.voxelWorld = entry.voxelWorld;
        asset.topLevel = entry.topLevel;
        asset.proxyLevel = entry.proxyLevel;
        asset.proxyDims = uint3(entry.proxyDims[0], entry.proxyDims[1], entry.proxyDims[2]);
        std::vector<uint8_t> proxy;
        const size_t proxyCount = size_t(asset.proxyDims.x) * asset.proxyDims.y * asset.proxyDims.z;
        if (!readBlob(file, entry.proxy, proxy) || proxy.size() != proxyCount * 2 * sizeof(float16_t))
            FALCOR_THROW("HSTRCloud: corrupt proxy of '{}' in '{}'.", asset.name, path);
        const auto* packed = reinterpret_cast<const float16_t*>(proxy.data());
        asset.proxyMean.resize(proxyCount);
        asset.proxyMax.resize(proxyCount);
        for (size_t i = 0; i < proxyCount; ++i)
        {
            asset.proxyMean[i] = float(packed[i]);
            asset.proxyMax[i] = float(packed[proxyCount + i]);
        }
        if (!readBlob(file, entry.coarse, asset.coarsePage))
            FALCOR_THROW("HSTRCloud: corrupt coarse page of '{}' in '{}'.", asset.name, path);
        asset.chunks.resize(size_t(asset.chunkDims.x) * asset.chunkDims.y * asset.chunkDims.z);
        file.seekg(std::streamoff(entry.chunkTableOffset));
        file.read(reinterpret_cast<char*>(asset.chunks.data()), std::streamsize(asset.chunks.size() * sizeof(BlobRef)));
        if (!file)
            FALCOR_THROW("HSTRCloud: cannot read the chunk table of '{}' in '{}'.", asset.name, path);
        library.assets.push_back(std::move(asset));
    }
    logInfo(
        "HSTRCloud: cloud library '{}': {} clouds, {:.1f} MB compiled from {:.1f} MB of VDB at optical-depth tolerance {:.3g}.",
        path.filename(),
        library.assets.size(),
        double(library.header.packageBytes) / 1048576.0,
        double(library.header.sourceBytes) / 1048576.0,
        library.header.tolerance
    );
    return library;
}
} // namespace hstrcloud
