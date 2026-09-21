import os
from pathlib import Path
from falcor import *

# Interactive cloud sea: an endless procedural layer of a compiled cloud library, rendered through the beam view with fine density
# paged in around the camera. The clouds are compiled offline from the VDBs, once into canonical caches, then packed one file per
# cloud into a library directory (--quality picks each cloud's compression; --budget-mb N fits a total instead):
#   HSTRCloudCompiler.exe build <clouds_hr/vdb> <clouds_hr/cache>
#   HSTRCloudCompiler.exe pack <clouds_hr/cache> <clouds_hr/library>
# Environment overrides: HSTR_CLOUD_LIBRARY (a library directory or one .hstrlib), HSTR_CLOUD_POOL_MB, HSTR_CLOUD_LOADS, HSTR_SEA_TILES.
library = Path(os.environ.get("HSTR_CLOUD_LIBRARY", str(Path.home() / "Downloads" / "clouds_hr" / "library")))
tiles = int(os.environ.get("HSTR_SEA_TILES", "8"))

g = RenderGraph("CloudSea")
g.addPass(createPass("HSTRCloud", {
    "debugView": 9,
    "cloudLibrary": str(library),
    "cloudProxyResolution": int(os.environ.get("HSTR_SEA_VOXELS", "64")),
    "cloudSeaTiles": tiles,
    "cloudSeaCoverage": float(os.environ.get("HSTR_SEA_COVERAGE", "0.85")),
    "cloudBrickPoolMB": int(os.environ.get("HSTR_CLOUD_POOL_MB", "256")),
    "cloudBrickLoadsPerFrame": int(os.environ.get("HSTR_CLOUD_LOADS", "1024")),
    # Clamped to the resident window of tiles around the camera.
    "seaViewDistance": 20000.0,
    "worldCacheCellVoxels": int(os.environ.get("HSTR_SEA_CELL", "4")),
    "worldCachePhotons": 65536,
    "worldCacheUpdates": 1,
    "worldCacheBakeInterval": 8,
    "sunNearVoxels": 2.0,
    # Two sampled voxels per camera step, never more than one sea voxel. A 2-voxel cap let the first step into a fine brick cross
    # ~5 of its voxels with one sample: against the 4K path tracer (farside, 8x8 blocks) 0.1265 log error, 0.0998 capped, 0.0996 at
    # 1-voxel steps, which cost 60-78% more; the cap costs 3-8%.
    "minStepVoxels": 2.0,
    "maxStepVoxels": 1.0,
    # Sun bakes scheduled on the GPU (read when the residency is created; HSTR_GPU_SUN=0 keeps the CPU scheduler).
    "cloudGpuSun": os.environ.get("HSTR_GPU_SUN", "1") != "0",
    # Fine sea detail slips between the corners and centre of larger tiles, whose centre test then accepts them as blocks. At 4K
    # against the per-pixel march (tolerance 0.05; mean 8-bit display error): near 16x3 0.52, 8x2 0.33, 4x1 0.21; far side 0.29,
    # 0.24, 0.20; sea overview 0.22, 0.14, 0.08. GPU ms with the queries rebuilt every frame (moving camera), near view:
    # 16x3 183, 8x2 203, 4x1 285, the per-pixel march 333. A brute-force stopgap until the tile test sees sub-tile detail.
    "beamTileSize": 4,
    "beamLevels": 1,
    # Whole query rays: split segments each start at unit transmittance, so those behind opaque cloud march on to the view
    # distance. Near view at 4K, 4x1: queries 201 -> 85 ms, total 293 -> 175 ms, display error 0.21 -> 0.16.
    "beamSegments": 1,
    # The persistent octahedral beam image: the beam basis and its residual live on a world-fixed sphere of directions, so turning
    # re-marches nothing already seen, and a per-block parallax guard over a 4 x 4 hierarchy lists only what translation broke.
    # 4K GPU ms park / look / flick / walk / sprint 0.49 / 0.51 / 0.72 / 0.92 / 0.56, 0.137% of pixels over 0.02 against the
    # per-pixel march (a3d4d427). It replaces the screen-space temporal beam at tolerance 0.05.
    "beamTolerance": 0.01,
    "beamTemporal": False,
    "beamRefFrame": True,
    "beamOct": True,
    "beamOctFull": False,
    "beamOctScale": 1.0,
    "beamGuard": True,
    "beamRefresh": 256,
    "beamRefreshBlock": 4,
    "beamDepthTolerance": 0.05,
    "beamCarryTolerance": 0.0,
    "beamScreenResidual": False,
}), "HSTRCloud")
g.addPass(createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0}), "ToneMapper")
g.addEdge("HSTRCloud.color", "ToneMapper.src")
g.markOutput("ToneMapper.dst")
m.addGraph(g)
m.loadScene("data/HSTR/cloud_sea.pyscene")
