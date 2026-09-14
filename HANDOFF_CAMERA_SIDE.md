# HSTRCloud camera-side work: handoff

Companion to `HANDOFF.md` (unchanged). Covers the session of 2026-09-13/14 that
rebuilt the HSTRCloud camera path for 4K output. Read the "Deviation" section first.

## State of the tree

- Committed by the user: `54e05fc2 preprocessing` (parallel CPU preprocessing).
- Uncommitted: `Source/RenderPasses/HSTRCloud/*`, one small change in
  `Source/Falcor/Rendering/Volumes/HSTRHierarchy.cpp` (nested `parallelFor` runs
  serially on worker threads), and two new scripts in `scripts/HSTR/`.
- Release build of `HSTRCloud` and `FalcorTest` succeeds; `FalcorTest -f 'HSTR.*'`
  passes 24/24.

## Deviation from HANDOFF.md (must be resolved)

`HANDOFF.md` forbids replacing transport with a conventional shadow march. The
current sun lighting does exactly that:

- `computeFineSun` precomputes sun optical depth for every voxel (a volumetric
  shadow map).
- `residualAtVoxel` adds `E * sum_n a^n p(theta; c^n g) exp(-b^n tau)`, a
  multi-octave multiple-scattering approximation. Octaves 1-3 read a
  density-weighted, Gaussian-spread version (`gatherSunOctaves`, `blurSunOctaves`).
- Its parameters were fitted to the path-traced reference (below). The HST
  boundary solve then carries only the sky (`hstSunFraction = 0`).

It is the only reason the image now resembles the reference, but it is not
transport-driven. The transport-driven fix is to inject the sun as a volume source
into the HST and propagate it through the Schur hierarchy. That needs
internal-source operators per node; the existing `localSourceRadiance` path only
maps one leaf to the root. It also needs finer angular traces: six-face P0/order-2
cannot hold a g = 0.85 sun.

To get back the pre-deviation lighting without deleting code, set
`residualStrength = 0` and `hstSunFraction = 1`.

## What is implemented and worth keeping

Camera path (no transport re-solve on camera motion):

- **Tile camera basis.** One per 8x8 tile: bilinear from shared tile-corner queries
  plus a centre bubble term. The hierarchical surplus flags tiles for exact
  per-pixel integration, with hysteresis (`basisTolerance`, `basisHysteresis`).
  Queries are deterministic and only rebuilt when the camera, cut or lighting
  changes.
- **Full-resolution transmittance** per pixel through the tile's cut, using
  majorant-bounded steps. The majorant grid is 4-voxel blocks dilated by one block,
  built on the CPU in `uploadExtinction`. Extinction is sampled with a zero border.
- **Cut front-to-back order** comes from the HST BSP itself: a per-node key built
  from the near/far child bits along the ancestor path (`bspOrderKey`), sorted per
  tile with a groupshared bitonic sort. This fixed the thin vertical radiance lines
  that the old centre-ray sort caused.
- **Cut split/merge hysteresis** (`cutHysteresis`), using a double-buffered
  per-node acceptance state.
- **Camera lighting lattice at 2x per leaf** (resolves the 2x2 face modes).

Tooling:

- **`setProperties`** implemented: scripts can change parameters at runtime without
  rebuilding the hierarchy.
- **Unbiased reference path tracer inside the pass** (`debugView 6`). It does delta
  tracking over the same extinction texture and majorant grid, with sun NEE via
  ratio tracking, HG phase, albedo 0.99, the same sky gradient, and Russian
  roulette. It accumulates into an internal running average; 1080p costs about
  0.2 s per spp.
- **`compareReference` property.** Computes linear L1 and log(1+x) L1 of the HST
  frame against that average and exposes them as the read-only properties
  `referenceError` and `referenceLogError`.
- `scripts/HSTR/CompareToReference.py`: accumulates the reference per view, then
  sweeps a parameter grid (`HSTR_GRID` JSON env var) and writes sorted results.
  `scripts/HSTR/BenchHSTRCloud.py`: headless 4K profiler timings plus captures.
  Both write to `%TEMP%/hstr_bench` and are driven by `HSTR_*` env vars.
- Falcor's `ErrorMeasurePass` crashes (access violation) loading the captured EXRs
  through `Texture::createFromFile`. That is why the comparison lives inside
  HSTRCloud.

Debug views (`debugView`):

| Value | View |
|---|---|
| 1 | Refined tiles |
| 2 | March cost |
| 3 | Opacity |
| 4 | Exact everywhere |
| 5 | Sun field at first surface |
| 6 | Reference path tracer |
| 7 | Rasterized sun residual only |

## Measurements (RTX 4080 Laptop)

HSTRCloud GPU time at 3840x2160:

| Version | Static camera | Orbiting camera |
|---|---|---|
| Before this session (`4aa1a7b`) | 2.97 ms | 4.29 ms |
| Camera path only, HST lighting | 2.1 ms | 3.5 ms |
| Current (with the deviating sun field) | 3.5 ms | 4.8 ms |

With the orbiting camera, the cut (project + bin + sort) costs 1.2 ms and the camera
basis 0.27 ms. Resolve cost is dominated by per-pixel sun evaluation in resident
leaves.

Error against the 256-spp reference (sum of log errors over the static, backlit and
oblique views, 1080p):

| Version | Summed log error |
|---|---|
| Original HST-only lighting | 0.733 |
| Current fitted sun field | 0.444 |

The static view still has log error 0.165.

## Known problems

- **Shadowed areas and the cloud underside** are flat sky-blue. The reference shows
  dark grey-blue with structure. The missing pieces are puff-scale sky occlusion and
  multiply scattered light under the cloud; the HST sky field is too coarse and too
  bright there.
- **Voxel-scale blockiness.** `wdas_cloud_eighth.vdb` voxels are about 18 px at 4K,
  and the sun field is stored per voxel, so light and shadow edges follow voxel
  faces. Finer data exists in `Downloads\wdas_cloud\`: quarter (68 MB), half
  (492 MB) and full (2.9 GB). Quarter fits the current dense texture path. Half and
  full need sparse or brick storage (or direct NanoVDB sampling) and larger HST
  leaves to keep the leaf count near 16k.
- **Stepped silhouettes** follow the voxel grid. They are inherent to trilinear
  filtering of this data at 4K; the reference shows the same.
- **Camera motion is not free**: the cut rebuild costs 1.2 ms. Binning large node
  projections into 8x8 tiles dominates; 16x16 tiles, or reprojecting only nodes whose
  footprint changed, are the obvious levers.
- The inside-cloud path is still the old NanoVDB march (`integrateVolume`).

## Experiments tried and removed (do not repeat)

- **Local Jacobi diffusion** of sun deposition on 2-voxel cells as a multiplicative
  residual. Dense lit cells are thicker than a transport mean free path and trap
  fluence, which renders as bright spots. More sweeps do not help.
- **Leaf-local 8^3 Schur solves as residual pages**, normalised per leaf. Each leaf
  is lit as an isolated box, which produces leaf-sized horizontal bands, and it
  costs about 44 s of CPU per sun change.

## Build, run, compare

```powershell
python tools/run_clang_format.py -i --clang-format-executable 'C:\packman-repo\chk\clang-format\15.0.6-windows-x86_64\clang-format.exe' Source/RenderPasses/HSTRCloud/HSTRCloud.cpp Source/RenderPasses/HSTRCloud/HSTRCloud.h
cmd.exe /d /s /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && "C:\packman-repo\chk\cmake\3.24.1+nv3-windows-x86_64\bin\cmake.exe" --build build\windows-ninja-msvc --config Release --target HSTRCloud'
$env:HSTR_TAG='run'; & .\build\windows-ninja-msvc\bin\Release\Mogwai.exe --headless --script=scripts/HSTR/BenchHSTRCloud.py
$env:HSTR_TAG='fit'; $env:HSTR_GRID='{"residualStrength":[0.0,1.0]}'; & .\build\windows-ninja-msvc\bin\Release\Mogwai.exe --headless --script=scripts/HSTR/CompareToReference.py
```

Scene load takes 20-50 s (the CPU hierarchy build).
