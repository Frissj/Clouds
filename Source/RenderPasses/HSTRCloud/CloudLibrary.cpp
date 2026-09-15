/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "CloudLibrary.h"
#include "Utils/Math/Float16.h"

#include <unordered_set>

namespace hstrcloud
{
bool readBlob(std::ifstream& file, const BlobRef& blob, std::vector<uint8_t>& raw)
{
    raw.clear();
    if (blob.raw == 0)
        return true;
    if (blob.compressed == 0)
        return false;
    std::vector<uint8_t> compressed(blob.compressed);
    file.clear();
    file.seekg(std::streamoff(blob.offset));
    if (!file.read(reinterpret_cast<char*>(compressed.data()), std::streamsize(compressed.size())))
        return false;
    return decompressGDeflate(compressed.data(), compressed.size(), raw, blob.raw);
}

bool decodeMeta(const std::vector<uint8_t>& raw, uint32_t payloadBytes, DecodedPage& page)
{
    if (raw.size() < sizeof(PageHeader))
        return false;
    PageHeader header;
    std::memcpy(&header, raw.data(), sizeof(header));
    const size_t bricksEnd = sizeof(PageHeader) + size_t(header.brickCount) * sizeof(BrickHeader);
    const size_t pagesEnd = bricksEnd + size_t(header.pageCount) * sizeof(PageEntry);
    if (pagesEnd != raw.size())
        return false;
    page.bricks.resize(header.brickCount);
    page.pages.resize(header.pageCount);
    std::memcpy(page.bricks.data(), raw.data() + sizeof(PageHeader), page.bricks.size() * sizeof(BrickHeader));
    std::memcpy(page.pages.data(), raw.data() + bricksEnd, page.pages.size() * sizeof(PageEntry));
    for (const BrickHeader& brick : page.bricks)
        if (uint64_t(brick.payloadOffset) + brick.payloadBytes > payloadBytes || brick.payloadOffset % 4 != 0)
            return false;
    return true;
}

CloudLibrary loadCloudLibrary(const std::filesystem::path& path)
{
    CloudLibrary library;
    if (std::filesystem::is_directory(path))
    {
        for (const auto& file : std::filesystem::directory_iterator(path))
            if (file.is_regular_file() && file.path().extension() == ".hstrlib")
                library.files.push_back(file.path());
        std::sort(library.files.begin(), library.files.end());
        if (library.files.empty())
            FALCOR_THROW("HSTRCloud: no cloud packages (*.hstrlib) in '{}' (pack them with HSTRCloudCompiler).", path);
    }
    else
        library.files.push_back(path);

    std::unordered_set<uint64_t> densities;
    uint64_t packageBytes = 0;
    uint64_t sourceBytes = 0;
    size_t duplicates = 0;
    for (uint32_t f = 0; f < library.files.size(); ++f)
    {
        const std::filesystem::path& filePath = library.files[f];
        std::ifstream file(filePath, std::ios::binary);
        LibraryHeader header;
        if (!file.read(reinterpret_cast<char*>(&header), sizeof(header)) || header.magic != kLibraryMagic || header.version != kLibraryVersion)
            FALCOR_THROW("HSTRCloud: '{}' is not an HSTR cloud package of this version (pack it with HSTRCloudCompiler).", filePath);
        packageBytes += header.packageBytes;
        sourceBytes += header.sourceBytes;
        std::vector<AssetEntry> entries(header.assetCount);
        file.seekg(std::streamoff(header.assetTableOffset));
        file.read(reinterpret_cast<char*>(entries.data()), std::streamsize(entries.size() * sizeof(AssetEntry)));
        if (!file)
            FALCOR_THROW("HSTRCloud: cannot read the asset table of '{}'.", filePath);
        for (const AssetEntry& entry : entries)
        {
            CloudAsset asset;
            asset.name = std::string(entry.name, strnlen(entry.name, sizeof(entry.name)));
            if (!densities.insert(entry.densityHash).second)
            {
                // Identical density to a cloud already loaded: it adds nothing to the sea.
                ++duplicates;
                continue;
            }
            asset.file = f;
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
                FALCOR_THROW(
                    "HSTRCloud: corrupt proxy of '{}' in '{}' (GDeflate result 0x{:08X}).", asset.name, filePath, uint32_t(lastGDeflateResult())
                );
            const auto* packed = reinterpret_cast<const float16_t*>(proxy.data());
            asset.proxyMean.resize(proxyCount);
            asset.proxyMax.resize(proxyCount);
            for (size_t i = 0; i < proxyCount; ++i)
            {
                asset.proxyMean[i] = float(packed[i]);
                asset.proxyMax[i] = float(packed[proxyCount + i]);
            }
            if (!readBlob(file, entry.coarse.meta, asset.coarseMeta) || !readBlob(file, entry.coarse.payload, asset.coarsePayload))
                FALCOR_THROW("HSTRCloud: corrupt coarse page of '{}' in '{}'.", asset.name, filePath);
            asset.chunks.resize(size_t(asset.chunkDims.x) * asset.chunkDims.y * asset.chunkDims.z);
            file.clear();
            file.seekg(std::streamoff(entry.chunkTableOffset));
            file.read(reinterpret_cast<char*>(asset.chunks.data()), std::streamsize(asset.chunks.size() * sizeof(PageBlobs)));
            if (!file)
                FALCOR_THROW("HSTRCloud: cannot read the chunk table of '{}' in '{}'.", asset.name, filePath);
            logInfo(
                "HSTRCloud: cloud '{}': {:.1f} MB at lambda {:.3g}, close-up transmittance error {:.4f} (99th percentile).",
                asset.name,
                double(header.packageBytes) / 1048576.0,
                header.lambda,
                header.transmittanceError
            );
            library.assets.push_back(std::move(asset));
        }
    }
    logInfo(
        "HSTRCloud: cloud library '{}': {} clouds from {} packages ({} identical skipped), {:.1f} MB packed from {:.1f} MB of VDB.",
        path,
        library.assets.size(),
        library.files.size(),
        duplicates,
        double(packageBytes) / 1048576.0,
        double(sourceBytes) / 1048576.0
    );
    return library;
}
} // namespace hstrcloud
