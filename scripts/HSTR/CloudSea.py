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
    # Doubled (2026-09-25, 4K, one settle, against the exact march; the gate is under 1% of pixels over 0.02): walk 5.73 -> 4.14
    # ms at 0.137 -> 0.425%, sprint 7.15 -> 5.36 at 0.028 -> 0.127% (budgetstep1). 3x fails walk (2.94%). A deliberate
    # approximation: it spends the gate's headroom on the step, with beamOctScale 0.5 and the warp below.
    "minStepVoxels": 4.0,
    "maxStepVoxels": 2.0,
    # The shipping switch mask (11773) with transmittance-scaled steps (bit 32768: behind transmittance T a step grows by up to
    # 1 / sqrt(T), 4x at most): walk 4.14 -> 3.91 ms at 0.425 -> 0.513% (budgetstep1).
    "beamShipMask": 11773 | 32768,
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
    # Half the octahedral image's angular resolution (a texel ~2 pixels at the view centre): walk 3.91 -> 2.46 ms at 0.513 ->
    # 0.760%, sprint 5.06 -> 2.90 at 0.155 -> 0.283% (budgetoct2 / budgetjitter2).
    "beamOctScale": 0.5,
    "beamGuard": True,
    # A guard block holds through 8 texels of parallax, and the resolve warps what it holds to the current camera by the query
    # depths (beamWarp, a field per lattice point). Walk 2.52 -> 2.19 ms at 0.761 -> 0.938% (budgetwarpfield1); unwarped at 8 it
    # fails (1.986%). Sprint is unchanged: every block expires at 20 units a frame, so there is nothing held to warp.
    "beamGuardParallax": 8.0,
    "beamWarp": True,
    # ...and only while at least a quarter of the classified on-screen blocks are held, decided on the GPU each build
    # (writeBeamWarpArgs): walk holds 62%, sprint 0-0.9%, so sprint skips the field and the warped resolve. Sprint 2.95-2.97 ->
    # 2.85 ms, walk unchanged, errors bit-identical in both (warpauto2).
    "beamWarpAuto": 0.25,
    # Held few blocks, the build loosens its tile test instead: tolerance 0.01 -> 0.05 as the held share falls 0.25 -> 0.05,
    # decided on the GPU (decideBeamPolicy). 4K sprint 2.93 -> 2.52 ms at 0.28 -> 0.48% over 0.02, jog 2.79 -> 2.66 at 0.58 ->
    # 0.85%, walk unchanged (policy4). Longer steps as well failed on jog's content (policy3).
    "beamPolicy": True,
    "beamPolicyStep": 1.0,
    "beamPolicyTolerance": 0.05,
    "beamPolicyHeldLow": 0.05,
    "beamPolicyHeldHigh": 0.25,
    # The colour output at RGBA16Float: the tone mapper, which reads it, 0.39 -> 0.07 ms at 4K, errors identical (colorfmt1).
    "colorFormat": 1,
    # The anchoring build covers the whole sphere (about 97 ms once, at 4K), so a turn lands on directions already built.
    "beamPrebuild": True,
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
