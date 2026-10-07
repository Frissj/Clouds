#!/bin/sh
# Builds hstrlib_dense (Linux / WSL / MinGW) against the reference GDeflate decoder:
#   git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/DirectStorage.git && (cd DirectStorage && git sparse-checkout set GDeflate)
#   git clone --depth 1 https://github.com/NVIDIA/libdeflate.git nvlibdeflate
#   DS=DirectStorage/GDeflate/GDeflate LD=nvlibdeflate sh build_hstrlib_dense.sh
set -e
HERE=$(dirname "$0")
: "${DS:?path to DirectStorage/GDeflate/GDeflate}"
: "${LD:?path to NVIDIA libdeflate}"
OUT=${OUT:-$HERE/hstrlib_dense}
TMP=$(mktemp -d)
gcc -O2 -c -I"$LD" "$LD/lib/gdeflate_decompress.c" "$LD/lib/utils.c" "$LD/lib/x86/cpu_features.c" -o "$TMP/a.o" 2>/dev/null || {
  for f in gdeflate_decompress utils x86/cpu_features; do gcc -O2 -c -I"$LD" "$LD/lib/$f.c" -o "$TMP/$(basename $f).o"; done; }
g++ -O2 -std=c++17 -c -I"$DS" -I"$LD" "$DS/GDeflateDecompress.cpp" -o "$TMP/gd.o"
g++ -O2 -std=c++17 -c -I"$DS" "$HERE/hstrlib_dense.cpp" -o "$TMP/main.o"
g++ -o "$OUT" "$TMP"/*.o -lpthread
rm -rf "$TMP"
echo "built $OUT"
