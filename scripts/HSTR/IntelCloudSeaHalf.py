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
    # MEASURED (2026-10-06; quality: squared log excess against the 326-spp path trace of sunset_hill 960x540, noise subtracted,
    # anchor 0.020228; time: 4K sunset walk units + query, interleaved anchors 3.81 / 3.62 / 3.71 / 4.07 ms, qualtime3):
    # minStepVoxels 5 + beamTolerance 0.6: +1.5% excess, 2.97 / 3.05 ms (-0.79 ms), motion 2.61 / 2.49% over 0.02 against
    # anchors ~2.04%. minStepVoxels 6 + tolerance 0.6: +3.6%, 2.61 ms (-1.2 ms), motion 3.35%. Alone (qualfront1/2): step 5
    # +0.5%, step 6 +2.4%, tolerance 0.6 +0.6%, 0.8 +3.2%; beamOctScale 0.42 +4.1%, 0.35 +7.4%, 0.25 +18.7%. The 8 x 8 block
    # gate scores every one of these 0.000% (blind); not shipped - a quality trade for the owner to take.
    # Images (qualfront2/3 compare_zoom*.png): tolerance 0.6 cuts silhouettes into straight tile-diagonal edges (a coarse tile
    # accepted across the edge); the step alone does not. Step alone, interleaved (qualtime4, anchors 3.96 / 3.74 / 4.19 ms):
    # step 5 3.58 / 3.48 ms (-0.43 ms), motion 3.19 (one 4.2% chunk) / 2.45% against ~2.1%; step 6 3.07 ms (-0.9 ms), motion
    # 4.02% - the longer step also costs motion stability.
    # Tolerance 0.2 (no visible tile blocks; tol02 / tol02time, anchors 4.20 / 3.95 / 3.93 / 4.11 ms, motion ~2.05%): alone
    # 4.52 ms (+0.47), excess -0.8%, motion 2.19%; + step 5 3.81 ms (-0.24), excess +0.0%, motion 3.83%; + step 6 3.77 ms
    # (-0.28), +1.5%, motion 3.34%; tolerance 0.3 + step 6 +1.6%. The step pays for the tolerance but costs motion stability.
    "minStepVoxels": 4.0,
    "maxStepVoxels": 2.0,
    # The cache read as one camera-baked texel a lit sample (bakeWorldCacheView) instead of five SH texels and two evaluations:
    # 4K walk units + query -0.47 ms net of its 0.10 ms bake, squared excess against the path trace +0.4% (cacheview1 /
    # cacheviewpt2; numbers beside worldCacheView). An explicit approximation (the cell centre's view direction), owner-approved.
    "worldCacheView": True,
    # The shipping switch mask (11773) with transmittance-scaled steps (bit 32768: behind transmittance T a step grows by up to
    # 1 / sqrt(T), 4x at most): walk 4.14 -> 3.91 ms at 0.425 -> 0.513% (budgetstep1).
    "beamShipMask": 11773 | 32768,
    # Per-layer occupancy at majorant-block scale (hstrDomainLayers, from the domain build's conservative maxima widened by the
    # coarsest brick level's reach): a sample in a block with no layer skips to the end of the run of such blocks (up to 8), and a
    # layer without its bit is skipped without its lookup. 4K sunset walk, units + query ms, two pairs in one launch (tight16): off
    # 5.97, per-layer skip 4.84, + run skip 4.59 (-23%); over 0.02 in motion 2.33% -> 2.16%. Squared excess against the path trace
    # 0.020293 -> 0.020163 (sunset_hill_tight15). An explicit approximation: 0.085% of density samples fall in blocks whose layer
    # bit is clear (tight13; lossy-compressed bricks hold density outside the source maxima) and are dropped. Without the reach
    # margin it saved ~1.0 ms but dropped 0.41% (tight5 / tight12); jog and sprint were measured on that version: 7.68 -> 6.50
    # and 6.94 -> 5.67 ms (tight6).
    # Mode 4 also tests the nearest domain voxel's layer bits before a layer's lookup: walk 4.49 -> 4.15 ms (tight18, two pairs),
    # squared excess 0.020163 -> 0.020175 (sunset_hill_tight17), motion over 0.02 within the arms' spread. It catches 72% of the
    # brick walks that end empty, and drops 0.27% of density samples (probe VoxelDense, against 0.08% at block scale).
    "cloudLayerTightSkip": 4,
    # Mode 4's voxel-bits load wraps its voxel by selects, not signed integer modulo (exact: seaVoxel already wrapped it). Found in
    # the packed mask's Nsight source trace, where the same modulo took 17.3% of the units kernel's samples. wrap2 (4K sunset walk,
    # three interleaved triples against the shipped anchor at each position): units + query 3.91 / 3.79 / 4.18 vs ~4.06 / ~4.0 /
    # ~4.45 ms, about -0.2 ms; the packed mask with the same fix (cloudLayerPacked 15) ~-0.09 ms, within the noise.
    "cloudLayerWrapSelect": True,
    # The layer-free run skip jumps by a Chebyshev distance field over the majorant blocks (exact). rundist1 (4K sunset walk,
    # off / on / off / on): units 2.294 / 2.208 / 2.310 / 2.135, query 1.787 / 1.756 / 1.798 / 1.721 ms (~-0.18 ms together);
    # over 0.02 2.36 / 2.34% (second pair). Run loads 5.66M -> 3.82M a frame, capped runs 36k -> 5.6k.
    "cloudLayerRunDistance": True,
    # Dirty units start at their tile's least lattice first-density distance less 8 sea voxels (an approximation). 4K sunset walk:
    # ustart6 (motion score 2, hint in every pass) units off 2.45 / 2.65 / 2.54 -> m8 2.39 / 2.30 ms, query 1.81 / 1.79 / 1.82 ->
    # 1.74 / 1.70, over 0.02 2.26 / 2.31 / 2.35 -> 2.54 (one score 3.35, the rest 2.24-2.32) / 2.30%; m4 2.40 / 2.45%. Path trace
    # (sunset_hill_ustart3, squared excess): off 0.020187, m8 0.020178, m2 0.020201, sanity -8 0.037139. Misses (ustart4): 55 of
    # 117k hinted units a frame start past their first density; 25% of hinted rays differ from a march from zero by > 2% only
    # because the step positions move (the march's own discretisation, neutral against the path trace).
    "beamUnitStart": 8.0,
    # Far-sea density steps at least two far texels' footprint (past ~6.5k units a texel spans a sea voxel). farfoot2 (4K walk):
    # farSea 0.650 / 0.636 -> 0.578 (1) / 0.499 (2) / 0.470 ms (8). Path trace, horizon band of sunset_hill_farfoot1 (6.9k px):
    # rms log 0.1476 / 0.1477 -> 0.1470 (2), p99 0.495 -> 0.502; sanity cap 32 0.1593 / 0.589.
    "seaFarFootprint": 2.0,
    # Sun bakes a frame while the camera moves (HSTRCloud.cpp's default 256 was tuned on frame time alone, when a bake cost less).
    # 4K sunset walk, bakeCloudSun ms / over 0.02 (motion score 2): bakes1 256 0.73 / 2.52%, 128 0.35 / 2.28%, 64 0.19 / 2.26%,
    # 256 again 0.63 / 2.73%; bakeexit1 256 1.01 / 2.41%, 128 0.50 / 2.25%, 256 again 0.80 / 2.26%. Sprint scores swung 3-11% on
    # the 256 anchor alone (one bad chunk decides them), with 128 at 2.9%: no sign of bake lag there either. Parked keeps the
    # full rate, so the backlog still drains when the camera stops.
    "cloudSunBakesMoving": 128,
    # Sun bake step, level voxels (default 0.5). An explicit approximation. The bake's cost is its density steps: ~92 a texel, 98.5%
    # of texels marching to the full reach (bake probe, HSTR_SUN_BAKE_PROBE). bakestep4 (4K sunset walk, bakes kept across arms):
    # bakeCloudSun 0.5 0.557 / 0.412 ms, 1 0.229, 2 0.123. Quality (sunset_hill_bakestep5, every arm fully rebaked, 960 x 540 exact
    # view against the 0.5 anchor repeated last): step 2 0.063% of pixels over 2/255, p99 1/255, p99.9 2/255, max 11/255; step 4
    # 0.72%, p99.9 9/255, max 23/255; sanity (constant brick density in the bake) 2.6%, p99.9 40/255, max 79/255. The squared-excess
    # gate barely sees bakes (0.020159 anchor, 0.020152 step 2, 0.019845 sanity), so the image comparison decides.
    "cloudSunBakeStep": 2.0,
    # Sun bakes scheduled on the GPU (read when the residency is created; HSTR_GPU_SUN=0 keeps the CPU scheduler).
    "cloudGpuSun": os.environ.get("HSTR_GPU_SUN", "1") != "0",
    # Fine sea detail slips between the corners and centre of larger tiles, whose centre test then accepts them as blocks. At 4K
    # against the per-pixel march (tolerance 0.05; mean 8-bit display error): near 16x3 0.52, 8x2 0.33, 4x1 0.21; far side 0.29,
    # 0.24, 0.20; sea overview 0.22, 0.14, 0.08. GPU ms with the queries rebuilt every frame (moving camera), near view:
    # 16x3 183, 8x2 203, 4x1 285, the per-pixel march 333. A brute-force stopgap until the tile test sees sub-tile detail.
    # MEASURED and REJECTED (centreless 0.4, edge 0.95): 8. Its first numbers were a bug - listBeamFailedTileUnits kept a failed
    # tile's units in one 32-bit mask, so at 8 (64 units) units 32-63 were never listed: fewer units, cheaper, and stale stripes
    # in motion (7.7% of pixels over 0.02). Fixed, motion is fine (walk 2.54% / 2.69%, sprint 10.8% / 6.3% at 4 / 8,
    # tile8_fixq), but it costs more: 4K walk query 2.15 -> 1.33 ms and units 2.82 -> 3.83, sprint query 2.81 -> 1.48 and units
    # 3.24 -> 5.20 (tile8_fixtime, twice each) - a failing tile marches 64 units.
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
    # MEASURED (guardpar / inval_share / dirty_reasons, 4K sunset walk, after the centreless tile test): 12 / 16 list ~5% fewer
    # blocks and units - the multiscale pyramid's coarser levels (2^(L-1) blocks of travel) bind, not this level-0 budget. The
    # walk lists 17.4k of ~30k on-screen blocks a frame (12.6k held, 1.9k unverified, the rest failing the bound); invalidation off
    # or residency frozen leaves the listing unchanged (17.2k), so content changes are not where the dirty work comes from.
    # Spending motion error on the forward walk (an explicit approximation, motion only: a parked frame lists nothing, so the
    # path-trace gate cannot see it). Level 0 holds a block on beamGuardMotion 1's shift (|dP x w| + reach |dP|) at 4 texels OR
    # the isotropic |dP| at 8 (beamGuardIsoParallax): forward motion gets mode 1's tighter bound, sideways motion keeps today's.
    # 4K sunset, HSTR_MOTION_SCORE 2, units + query ms / over 0.02 mean / dirty units a frame, anchors iso p8 first and last:
    # walk (guardor1) iso 5.26 / 2.33% / 116k, m1 p4 3.30 / 3.17% / 81k, OR 2.95 / 3.27% / 79k, sanity iso p16 3.47 / 2.53% /
    # 109k, iso 3.67 / 2.76% / 108k - about -1.0 ms for +0.7 points against the drifting anchors; strafe +x (guardor2) iso 4.21 /
    # 4.46% / 126k, m1 p4 4.87 / 2.31% / 170k, OR 3.90 / 4.61% / 116k, iso 3.97 / 4.38% / 120k - mode 1 p4 alone pays ~+0.8 ms
    # sideways (as motion_strafe1), the OR does not. Sprint lists every block whatever the budget (motion_sprint1).
    # REJECTED for what it looks like: dark chips along every silhouette in motion (guardor_img: 1178 pixels a frame under 0.6x the
    # rebuild's luminance, iso p8 14). Mode 1 p4 alone has them too (guardor_img2); beamWarp off removes them and misaligns all
    # else (guardor_img3, 3.36% -> 4.58%): held blocks really drift under mode 1, and the warp field, interpolated between lattice
    # points on both sides of a near edge over far cloud, folds. Bounding the near-to-far parallax a held block carries
    # (beamGuardWarpParallax 1) removes the chips and the gain together (guardwarp2, interleaved, 10 scores each: iso p8 3.68 /
    # 3.75 ms, 2.03 / 2.16%, 69 / 7 chip pixels; OR + warp 1 3.55 / 3.77 ms, 2.22 / 2.01%, 46 / 23; units 111k -> 100k). The
    # walk's -1 ms was the blocks the warp cannot follow. All three settings stay available, off.
    "beamGuardMotion": 0,
    "beamGuardParallax": 8.0,
    "beamGuardIsoParallax": 0.0,
    "beamGuardWarpParallax": 0.0,
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
