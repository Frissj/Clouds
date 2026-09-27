# Intel® Volumetric Clouds Library — HSTR packages

Download the compressed full- or half-resolution set from the
[clouds-v1 release](https://github.com/Frissj/Clouds/releases/tag/clouds-v1).
Each set contains the five different dense cloud shapes numbered 0–4.

## Install and launch

1. Check out the `clouds-v1` tag of this repository, or a compatible newer revision, and build Mogwai and HSTRCloud.
2. Extract either ZIP into the repository root. The archive supplies `IntelSea/full/` or `IntelSea/half/` automatically.
3. Double-click `RunIntelCloudSeaFull.bat` or `RunIntelCloudSeaHalf.bat`. These launch Mogwai with its window visible.

Install both sets to switch between them. Each script loads exclusively its matching folder,
uses density scale 1.0, and occupies every tile in a 16 × 16 resident window.
Tile occupancy does not guarantee a seamless cloud blanket: placement still uses the existing one-cloud-per-tile renderer.
First-render initialization can be lengthy. Scene loading and script syntax were checked;
the first-render check was stopped before completion, so visual quality and performance are unverified.

## Contents and modifications

- Source: [Intel® Volumetric Clouds Library](https://dpel.aswf.io/intel-cloud-library/), `intelCloudLib_dense.0.L.vdb` through `intelCloudLib_dense.4.L.vdb`.
- Full: all five L sources compiled into HSTR packages (293.902 MiB total).
- Half: the same sources averaged over 2 × 2 × 2 voxel blocks, with voxel size doubled to maintain physical scale (149.264 MiB total).
- Both use lossy Haar/GDeflate encoding with fixed lambda 0.0001. Automatic quality search is disabled by default; explicit `--quality` enables it.
- No density multiplier was applied. The `.L` in filenames identifies the original source, including in the half-resolution folder.
- `.hstrlib` requires this project's compatible HSTR renderer; it is not a VDB file. Package format version is 2.
- Original VDBs, intermediate `.hstrcloud` caches, and partial files are excluded from the downloads and Git history.

`manifest.json` records package sizes, SHA-256 hashes, voxel sizes, padded dimensions and compiler-reported compression errors.
Those errors describe compression against each resolution's cache; they do not measure downsampling or rendered-scene quality.
SHA256SUMS.txt in the release verifies the ZIP downloads.

## License and attribution

Intel® Volumetric Clouds Library Copyright 2022 Intel Corp. All rights reserved.

These modified assets retain the [ASWF Digital Assets License v1.1](LICENSE.txt),
which permits the uses listed in that license, including research, development and benchmarking.
Publications showing images derived from the assets must include the copyright notice above.
This packaging does not imply endorsement by Intel.

## Repackage local compiled files

Run `python scripts/HSTR/PackageIntelClouds.py` from the repository root.
It validates both five-file sets, writes the manifest, and creates the ZIPs and checksums
in the ignored `IntelSea/releases/` directory. Use `--output PATH` to select another directory.
