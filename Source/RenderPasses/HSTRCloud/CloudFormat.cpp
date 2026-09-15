/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "CloudFormat.h"

#include <zlib.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <dstorage.h>

namespace hstrcloud
{
namespace
{
thread_local int32_t gLastGDeflateResult = 0;

/// One single-threaded codec per calling thread (callers parallelise across blobs).
IDStorageCompressionCodec* gdeflateCodec()
{
    struct Holder
    {
        IDStorageCompressionCodec* codec = nullptr;
        HRESULT result = S_OK;
        Holder() { result = DStorageCreateCompressionCodec(DSTORAGE_COMPRESSION_FORMAT_GDEFLATE, 1, IID_PPV_ARGS(&codec)); }
        ~Holder()
        {
            if (codec)
                codec->Release();
        }
    };
    thread_local Holder holder;
    gLastGDeflateResult = holder.result;
    return holder.codec;
}
} // namespace

bool compressGDeflate(const uint8_t* data, size_t size, std::vector<uint8_t>& out)
{
    out.clear();
    if (size == 0)
        return true; // An absent blob.
    IDStorageCompressionCodec* codec = gdeflateCodec();
    if (!codec)
        return false;
    out.resize(codec->CompressBufferBound(size));
    size_t written = 0;
    gLastGDeflateResult = codec->CompressBuffer(data, size, DSTORAGE_COMPRESSION_BEST_RATIO, out.data(), out.size(), &written);
    if (FAILED(gLastGDeflateResult))
        return false;
    out.resize(written);
    return true;
}

bool decompressGDeflate(const uint8_t* data, size_t size, std::vector<uint8_t>& out, size_t rawSize)
{
    IDStorageCompressionCodec* codec = gdeflateCodec();
    if (!codec)
        return false;
    out.resize(rawSize);
    size_t written = 0;
    gLastGDeflateResult = codec->DecompressBuffer(data, size, out.data(), rawSize, &written);
    return SUCCEEDED(gLastGDeflateResult) && written == rawSize;
}

int32_t lastGDeflateResult()
{
    return gLastGDeflateResult;
}

bool deflateBytes(const uint8_t* data, size_t size, std::vector<uint8_t>& out)
{
    uLongf bound = compressBound(uLong(size));
    out.resize(bound);
    if (compress2(out.data(), &bound, data, uLong(size), Z_BEST_COMPRESSION) != Z_OK)
        return false;
    out.resize(bound);
    return true;
}

bool inflateBytes(const uint8_t* data, size_t size, std::vector<uint8_t>& out, size_t rawSize)
{
    out.resize(rawSize);
    uLongf length = uLongf(rawSize);
    return uncompress(out.data(), &length, data, uLong(size)) == Z_OK && length == rawSize;
}
} // namespace hstrcloud
