/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "CloudFormat.h"

#include <zlib.h>

namespace hstrcloud
{
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
