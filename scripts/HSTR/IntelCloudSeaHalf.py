import os
from pathlib import Path
from falcor import *

# Intel cloud sea: five dense shapes, exclusively half resolution, at original density.
# Every tile is occupied; a 16 x 16 resident window expands the original 8 x 8 sea.
# Pool, loads, tiles and voxel resolution remain configurable through HSTR_* below.
library = Path(__file__).resolve().parents[2] / "IntelSea" / "half"
expected = {f"intelCloudLib_dense.{i}.L.hstrlib" for i in range(5)}
if {p.name for p in library.glob("*.hstrlib")} != expected:
    raise RuntimeError(f"Expected exactly the five half-resolution Intel clouds in {library}")
tiles = int(os.environ.get("HSTR_SEA_TILES", "16"))

g = RenderGraph("CloudSea")
g.addPass(createPass("HSTRCloud", {
    "debugView": 9,
    "cloudLibrary": str(library),
    "cloudProxyResolution": int(os.environ.get("HSTR_SEA_VOXELS", "64")),
    "cloudSeaTiles": tiles,
    "cloudSeaCoverage": 1.0,
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
    # Centreless root tiles: no centre queries (half the dirty query's rays), the tile test the corners' curvature against
    # beamTolerance, which is that test's own scale (HSTRCloud.cs.slang beamTileRefines has the numbers). Against centres at
    # 0.01 (cache and sun with the rest) -> 0.02 (cache 4, sun 1.5 apart) -> centreless 0.4, 4K sunset walk: units 4.88 -> 3.92 ->
    # ~3.99 ms, query 3.3 -> 3.3 -> 2.46 ms; sprint units 5.6 -> 4.9 -> 4.7, query 3.9 -> 3.9 -> 2.8. Squared log error against
    # the path trace (noise subtracted): 0.020288 -> 0.020363 -> 0.020376 (sunset 960), and lower on the 696-spp crop.
    # In motion, against a fresh rebuild at each scored camera (HSTR_MOTION_SCORE 2, tiletest_motionq, old test vs this one with
    # edge 0.95 below): walk 2.516% -> 2.379% of pixels over 0.02 (worst 3.48 -> 2.93%, p99.9 0.49 -> 0.21), sprint 5.600% ->
    # 3.192% (worst 10.5 -> 4.35%, p99.9 1.16 -> 0.49).
    "beamCentreless": True,
    "beamTolerance": 0.4,
    # The silhouette test only past a transmittance range of 0.95 (was the 0.2 default; the hill's BEAM view had 0.5). Squared
    # error against the path trace at curvature 0.4, edge 0.5 / 0.75 / 0.8 / 0.9 / 0.95 / 1.0 (off): 0.020376 / 0.020378 /
    # 0.020382 / 0.020379 / 0.020378 / 0.020553 (sunset 960), crop 0.082245 / 0.95 0.082235 - the error is all in the ranges past
    # 0.95. 4K walk units 6.66M -> 4.93M, march/units 3.63 -> 3.13 ms; sprint 7.83M -> 4.97M, 3.99 -> 3.33 ms (edge_time, against
    # 0.5, twice each).
    "beamEdgeContrast": 0.95,
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
    # Its loose end 0.05 -> 1.0 with the centreless tolerance: 2.5x the tight end 0.4 (timed at sprint, centreless_sprint).
    "beamPolicy": True,
    "beamPolicyTolerance": 1.0,
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
m.loadScene(str(library.parents[1] / "data/HSTR/cloud_sea.pyscene"))
