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
    # HSTR_SEA_COVERAGE: the share of tiles holding a cloud (a sparse sky for the budget-scaled refill; CloudSea.py's override).
    "cloudSeaCoverage": float(os.environ.get("HSTR_SEA_COVERAGE", "1.0")),
    "cloudBrickPoolMB": int(os.environ.get("HSTR_CLOUD_POOL_MB", "256")),
    # 4096: 1024 was saturated on 112-115 of 120 walk frames (loadcap3); see HSTRCloud.h mCloudBrickLoadsPerFrame.
    "cloudBrickLoadsPerFrame": int(os.environ.get("HSTR_CLOUD_LOADS", "4096")),
    # Clamped to the resident window of tiles around the camera.
    "seaViewDistance": 20000.0,
    "worldCacheCellVoxels": int(os.environ.get("HSTR_SEA_CELL", "4")),
    # 262144 photons a batch, 10 batches a tile (worldCacheTarget below): the photon total of 40 x 65536. A focused update is
    # bound by its latency, not its photons, so bigger batches are cheaper per photon. MEASURED (photonsize1, 4K sunset sprint
    # 20, 3 x 240 frames, chunk GPU / ms an update over norefill 2.31): 65536 x 40 3.75 / 4.14 (anchor), 2.76-3.13 ms an update;
    # 131072 x 20 3.43, 4.95; 262144 x 10 3.37, 8.2 - per 65536 photons 2.76-3.13 / 2.48 / 2.05. Path trace (photonpt1, hill,
    # whole cache from a restart): excess 0.023239 (anchor identical) / 0.023433 / 0.023072, p99 0.4871 / p99.9 0.6889 in all;
    # sanity 65536 x 10 0.023832 - quality is the photon total's. Cost: an update frame pays ~8 ms of photons, not ~3.
    # Smaller batches do not buy the spikes back (spikefront1, stratified, 2 x 240 frames, HSTRCloud GPU per frame - mean, frames
    # over 4 / 6 ms, longest): sprint 20 262144 x 10 3.04, 34 / 29, 16.2 ms (again 3.04 / 3.02, 25 / 22-24, 17.1); 131072 x 20
    # 3.38, 60 / 31, 15.1; 65536 x 40 3.65 / 3.60, 101 / 17-18, 12.9-13.1. Walk 2 alike (262144 x 10 2.22-2.78, 65536 x 40 3.2-3.3).
    # The longest frame without any refill is ~10 ms (spikes1), so the floor of the spikes is not the photons'.
    "worldCachePhotons": 262144,
    "worldCacheUpdates": 1,
    "worldCacheBakeInterval": 8,
    # Refill: 40 batches of 65536 photons a tile (10 of 262144, worldCachePhotons above), sync, photons focused on the starving
    # tiles (1/16 survive elsewhere, weighted 16). The numbers below are of 65536-photon batches. A deliberate
    # approximation of the parked image: path trace (t40pt1, hill 960 wide, whole cache from a restart) excess 0.023239 against
    # 0.022971 at 64 (+1.2%), p99 0.4871 / p99.9 0.6889 unchanged, max 0.824 / 0.827 (sanity t10 0.023832); focus 1/16 0.022927
    # (focuspt2). For it, honest sprint (focushonest2 / continue1, 4K sunset sprint 20, tiles applied after each walk, 3 x 240
    # frames, chunk GPU, near-camera batches mean): 3.69 / 3.69 / 3.57 ms, 32.8 / 32.6 / 31.2, against the urgent 26 + span 4
    # policy (worldCacheUrgentTarget, kept in C++) 4.10 / 3.79 ms, 28.5 / 23.6 and t64 focused 4.73, 37.6; norefill 2.42.
    # The photon queue and slices do not shrink the refill, they spread it (slicecost1, beside worldCacheAsyncSpan).
    # current1 (these launcher defaults, 2 x 240-frame chunks, desired 243k): walk 2 2.70 / 2.90 ms (160 focused updates, 48
    # tiles applied in 4 events, backlog 55-77k), sprint 20 3.76 / 3.31 (204 updates, 447 tiles); parked after either, draining
    # at the 512 bake cap, 1.46-1.81. The walk refills now: under the old apply gate it applied no tile in 240 frames.
    "worldCacheTarget": 10,
    # Sea tiles apply beside a running residency walk (seaApplyDuringWalk, C++ default on; the walk reads its own copy of the
    # slots), in batches of 32 ready tiles (2 rows) or whatever is left once nothing more is being rasterized. Held for the walk
    # (seaApplyAfterWalk alone) the window trailed the camera by a walk (~200 ms of CPU, 50-60 sprint frames) and a slow walk
    # emptied the view (seahold1; chunks of 3.6k-39k desired). MEASURED (walkbeside1 / applybatch1, 4K sunset, 2 x 240 frames,
    # chunk GPU, photon frames, chunk mean desired): sprint 20 at once (batch 0) 6.0-6.4 ms, 183-200 photon frames - every small
    # apply restarts a few tiles, so some tile always starves and the refill runs every frame; batch 32 3.40 / 3.73, 63 / 81,
    # 243k / 243k; batch 64 2.98 / 3.49, 33 / 61, 243k / 205k (fewest 161k); the old hold 2.93 / 2.71, 21 / 20, 243k / 159k
    # (fewest 62k). Walk 2: batch 32 2.14 / 2.33, hold 2.28 / 2.28. The hold's cheaper sprint was partly a lagging view.
    "seaApplyBatch": 32,
    "worldCacheFocusFar": 0.0625,
    # Stratified photon spawn: consecutive photons side by side along a Morton curve of the face, sources in contiguous index
    # ranges, every photon still uniform. MEASURED (strattime1, 4K sunset parked, 10 rows restarted, 4 profiled rounds a arm):
    # photon update 7.56-7.67 / 7.52-7.68 ms (ship, anchor) -> 6.48-6.66 (-13%). Path trace (stratpt2, hill, whole cache from a
    # restart, seeds 0 / 1 / 2): excess 0.023072 / 0.022793 / 0.023038 (mean 0.022968) -> 0.023335 / 0.022900 / 0.022974 (mean
    # 0.023070, +0.4% within the 0.0003-0.0004 seed spread); p99 0.4871 / p99.9 0.6889 in all; anchor identical. At 65536 x 10
    # it beats iid spawns (0.023488 against 0.023832, stratpt1 / patchpt1). worldCachePatch (random patches) failed the gate.
    "worldCacheStratify": True,
    # Flight cull: under focus, a photon flight whose cells (box + 1 cell) lie only in tiles taking no deposits skips its deposit
    # walk. Exact by construction. MEASURED (cullpark1, 4K sunset parked, 2 rows restarted, 4 profiled rounds): photon update
    # 6.59-6.85 ms (ship, anchor) -> 4.73-4.98 (-28%). Path trace (cullpt2, hill, 4 rows restarted, refilled under focus, seeds
    # aligned across arms): ship / cull / ship again excess 0.023210 identical to the last digit; misaligned sanity 0.023412,
    # flightCull 2 (cull everything) 0.077495 (cullpt1) - the gate sees the refilled rows.
    # Moving (cullslice1, 4K sunset, 2 x 240 frames, HSTRCloud GPU, chunk mean / frames over 6 ms): walk 2 cull 2.25 / 2.41 (5 /
    # 8), off 2.78 / 2.44 (19 / 9), cull again 2.53 / 2.58 (13 / 8); sprint 20 cull 2.82 / 2.86 (17 / 22), off 3.02 (20; its 2nd
    # chunk's desired collapsed to 39k), again 3.07 / 2.89 (26 / 18). 70-85% of sprint photon frames still go over 6 ms.
    "worldCacheFlightCull": True,
    # Focus roulette: under focus, a photon scattering more than one tile (wrapped, Chebyshev) from every starving tile survives at
    # 1/8, weighted up (weight window, never rolled twice). Unbiased. MEASURED (roulettepark1, 4K sunset parked, 2 rows restarted,
    # 4 profiled rounds): photon update 4.20-4.57 / 4.24-4.47 ms (ship, anchor) -> 3.25-3.47 at 1/8 (-24%); 1/4 3.51-3.59; 1/8 at
    # reach 2 3.51-3.76; ceiling (1e-4) 3.01-3.16. Frames over 6 ms in the 120-frame chunk 7 -> 0. Path trace (roulettept3, hill
    # 960 wide, 4 rows restarted, refilled under focus, launcher beam, seeds aligned): ship / ship again 0.024521 identical, 1/8
    # 0.024568 (+0.2%), 1/4 0.024512, sanity 1e-4 0.024796; seed spread (ship misaligned) 0.024731; p99 0.4871 / p99.9 0.6889 in all.
    "worldCacheFocusRoulette": 0.125,
    # Refill scaled by the frame's headroom (HSTRCloud.h worldCacheBudgetMs): every tile is held at 6 batches (worldCacheUrgentAll)
    # and the camera's 7 x 7 tiles too, baked at once (worldCacheUrgentTarget / Near); from 6 to the target 10 the refill spends only
    # what the frame leaves under 2 ms, so a sparse sky (or a parked camera) refills at full rate and a dense sprint does not.
    # MEASURED: targetpt1 (hill path trace, each arm converged to its own target, 262144-photon batches): 10 0.024564, 8 0.024444,
    # 6 0.024489, sanity 2 0.027800 (p99 0.4871 -> 0.5793), 10 again 0.025184 - 6 within the anchor's drift. floor6 (4K sunset,
    # 240-frame chunks, chunk GPU): sprint 20 ship 3.545 / 3.423, budget 3.217 ms (updates 89 / 98 -> 55, the camera's tiles 9.3
    # batches on average, never under 6); walk 2 2.153 / 2.274, budget 2.296 (all its updates are new tiles under 6). sparse1
    # (HSTR_SEA_COVERAGE 0.25): walk 1.779 / 1.583, budget 1.792; sprint 2.322 / 2.468, budget 2.355 - the camera's tiles at 10
    # throughout, unchanged where there is headroom. budgetpt1 (path trace, parked): ship 0.024564, budget 0.024564 identical
    # (converged in 69 frames against 13), sanity t2 0.025031. A floor near the camera only (6 within 3) bought the sprint 2.87 ms
    # (budget3) but left the far tiles unrefilled: lagbudget2 14.0% over 0.02 against the converged cache, ship 7.8% (single
    # sprint scores, noisy - lagbudget4's ship anchors 11.8 / 15.4%), the middle distance visibly darker.
    # budgetfinal1 (these settings against the old refill, one launch): walk 2 2.111 / 2.210 against 2.363 ms (6 / 6 updates
    # against 21), sprint 20 3.415 / 3.209 against 3.462 (66 / 61 against 104; the camera's tiles 8.4 / 8.0 batches on average).
    "worldCacheBudgetMs": 2.0,
    "worldCacheUrgentAll": 6,
    "worldCacheUrgentTarget": 6,
    "worldCacheUrgentNear": 3,
    "worldCacheAsync": False,
    "worldCacheAsyncSpan": False,
    # 2 slices with the flight cull (cullslice1, as above): walk 2.38 / 2.97 ms (3 / 6 frames over 6 ms), sprint 3.24 / 2.84 (29
    # / 18) - fewer walk spikes, but the mean +0.1-0.4 in 3 of 4 chunks and more frames over 3 / 4 ms. Before the cull: slicespike1.
    # Again with focus roulette and tiles applied in batches of 32 (slices2, sprint 20 chunk GPU / frames over 6 ms): 1 slice 3.68
    # / 3.74 (25 / 30), again 3.80 / 3.60 (28 / 36); 2 slices 4.08 / 4.21 (11 / 13); 4 slices 4.25 / 4.14 (12 / 13). Walk 2: 1
    # slice 2.17 / 2.37 (again 2.46 / 2.45), 2 slices 2.51 / 2.60. Slices halve the sprint spikes for +0.4-0.5 ms of mean.
    "worldCacheSlices": 1,
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
    # 1 / sqrt(T), 4x at most): walk 4.14 -> 3.91 ms at 0.425 -> 0.513% (budgetstep1). Bit 65536: the per-lane layer loop
    # (leanExtinctionLayered), exact (lanept1); query 0.716 / 0.709 -> 0.681, units 0.781 / 0.781 -> 0.768 ms (layerloop3).
    "beamShipMask": 11773 | 32768 | 65536,
    # The dirty query's two threads a ray march it together, two steps an iteration (marchBeamLeanPair; same steps as the serial
    # march, paircheck1): query 0.706 / 0.697 -> 0.626 ms, units unchanged (pair3); motion over 0.02 3.53 / 5.55 -> 3.60% (pairq1).
    "beamDirtySegments": 2,
    "beamQueryPair": True,
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
    # The far layer stored over the smooth sky: the resolve takes sky and far sea behind in one fetch. halfres1 (4K walk):
    # resolve/pixels 0.398 / 0.376 -> 0.356 / 0.291 ms. Path trace (sunset_hill_farsky2): excess 0.024075 -> 0.023937, p99 0.4871
    # both, max 0.861 -> 0.828.
    "seaFarSky": True,
    # Checkerboard resolve: one parity resolved, the other the mean of its four neighbours where all came from a basis and agree
    # within beamHalfContrast, the rest resolved exactly. An explicit approximation. GPU trace, 4K walk, same frames: HSTRCloud
    # 2.98 / 3.34 (warplines1) -> 2.86 / 3.21 ms (halftrace3), resolve range 0.66 / 0.67 -> 0.52 / 0.56. Path trace
    # (sunset_hill_farsky2, 960 x 540 - fewer pixels a beam unit than 4K, so the mean errs more there): excess 0.024075 ->
    # 0.024076. Moving (halfscore2, with seaFarSky, alternated at matched positions, % over 0.02): off 6.48 / 5.41 / 6.07 / 6.14,
    # on 6.52 / 2.17 / 6.14 / 5.62.
    "beamHalfResolve": True,
    # Sun bakes a frame while the camera moves (HSTRCloud.cpp's default 256 was tuned on frame time alone, when a bake cost less).
    # 4K sunset walk, bakeCloudSun ms / over 0.02 (motion score 2): bakes1 256 0.73 / 2.52%, 128 0.35 / 2.28%, 64 0.19 / 2.26%,
    # 256 again 0.63 / 2.73%; bakeexit1 256 1.01 / 2.41%, 128 0.50 / 2.25%, 256 again 0.80 / 2.26%. Sprint scores swung 3-11% on
    # the 256 anchor alone (one bad chunk decides them), with 128 at 2.9%: no sign of bake lag there either. Parked keeps the
    # full rate, so the backlog still drains when the camera stops.
    # 128 -> 64 (sunbake2-4, 4K sunset walk, arms 128 / 64 / 32 / sanity 8 / 128): bakeCloudSun 0.136 / 0.101 -> 0.069 ms (32 0.032,
    # 8 0.010). The motion score cannot see bake rate (sanity 8 scored 5.53% against 5.49 / 5.77%: its fresh rebuild reads the same
    # bakes), so sunbake4 counted the march's lit samples resolved from an ancestor's bake (kBeamSunAncestor), 240-frame chunks:
    # ship 81% (1.47 levels up), 64 78% (1.59), sanity 8 85% (1.63), ship again 62% (1.75) - 64 inside the anchors' drift (the
    # backlog drained 378k -> 275k over the run), the sanity arm above it. Unbaked / coarse fallbacks 0 in every arm; units and
    # query unchanged (probe on: 2.98 / 3.10 vs 3.16, query 1.12 / 1.12 vs 1.10). Known limit, not caused by this: while walking,
    # 60-85% of lit samples light from an ancestor's bake ~1.5 levels coarser (300-400k bakes waiting); the path-trace gate, run
    # parked and fully baked, never sees it.
    "cloudSunBakesMoving": 64,
    # Parked at most 512 bakes a frame (under cloudSunBakesPerFrame's 1024; moving, cloudSunBakesMoving's 64 still rules). Parked
    # after a sprint ~250-330k outdated bakes drain, outdated ones answering meanwhile (cloudSunKeepStale), and the main queue
    # waits at the frame's end for the residency queue's bakes. MEASURED (bakecap1, 4K, parked after a 240-frame sprint 20,
    # four sprint + park passes in one launch, the last three 60-frame windows of each): cap 1024 2.36 / 2.76 / 2.55 ms and
    # again 1.65 / 2.05 / 2.38; 512 1.62 / 1.62 / 1.66; 256 1.61 / 1.63 / 1.69; settled ~0.8-0.95. Drain ~24k bakes a window
    # at 512 against ~60k at 1024: twice as long, under 2 ms. Benchmarks that score a settled sun settle uncapped (sunset_hill).
    "cloudSunBakesCap": 512,
    # Sun stamp, levels 0-1 by changed-cell prefix sums (stampSunFine) instead of every change's sweep box (stampSunChange). The full
    # stamp stored 9-34M times a run, 88-93% into entries already holding the run's epoch (stampskip2: overlapping sweeps); reading
    # first (cloudSunStampSkipSame) was no faster. stampfine4 (4K sunset walk, 96 profiled frames, scheduleSun every 4th): stamp
    # ship 1.379 / ship again 1.586 ms a run (0.316 / 0.380 a frame) -> coarse 0.049 + fine 0.643 (0.086 a frame; with the listed
    # beam marks 0.044 + 0.723, 0.160 a frame, over 1.3x the changes). Exact: cloudSunStampFineCheck 1 (full stamp after, counting
    # entries the fine stamp missed) and 2 (full stamp first, counting entries it stamped beyond) 0 over 101 checked runs, no
    # list fallbacks (capacity 262144, peak 174911 candidates). One stamp a cell per class (stampfine3) cost 1.76 ms a run: 97%
    # of its threads stored nothing; the wave per (class, cell) tests the union box first. stampfine5: a lane per (cell, class) for
    # that test, the wave then over the passing pairs' entries - stampFine 1.7-2.1 -> 0.20-0.23 ms a run, coarse 0.07; full stamp
    # 1.23 / 1.72 ms a run in the same launch; per frame ~0.065 vs 0.17-0.41 ms. Checks 1 / 2: 0 over 103 runs.
    "cloudSunStampFine": True,
    # markBeamChanges: one thread per (change, instance of its asset) instead of per sea slot. stampfine2 / 4: 0.029 / 0.038 ms a
    # run against ship 0.052 / 0.054 and 0.047 / 0.049.
    "beamChangeAssetSlots": True,
    # GPU sun bakes mark the beam blocks that read them (markBeamChanges). Never happened before: the bake count was read after
    # gpuSunDispatched cleared it. markbakes1 (4K sunset walk, motion score 2): ship 8.00% / on 5.24% / ship again 5.13% over 0.02,
    # markBeamChanges 0.031 / 0.027 / 0.033 ms, units 0.73 / 0.82 / 0.95 - no measurable cost or quality change; on as correct.
    # Parked after a sprint the ~250-330k outdated bakes drain at 1024 a frame and each frame's marks re-march ~982 beam blocks
    # (sundirty2: the change list's bakes are the only invalidator; march 0.63-0.69 ms profiled, 0.007-0.010 with marks off -
    # markdrain2). But it costs nothing measurable (markdrain3, clean, on / off / on: drain windows 2.36-2.86 / 1.79-2.76 /
    # 1.80-2.30 ms): the main queue waits at the frame's end for the residency queue's bakes anyway.
    "beamMarkBakes": True,
    # worldCacheView rebaked every 4th frame while the camera moves (never while parked and unchanged). cacheview1 (4K sunset walk,
    # motion score 2, share over 0.02 / p99.9): every 1 0.065 ms a frame 4.84% / 0.97, every 2 0.028 5.66% / 0.49, every 4 0.015
    # 5.12% / 0.41, sanity view off 7.25% / 0.58, every 1 again 0.055 6.40% / 0.41 - 2 and 4 inside the anchors' spread, the
    # sanity arm above it. The first anchor ran under-mapped (213880 of 243302), the rest mapped in full.
    "worldCacheViewEvery": 4,
    # Sun bake step, level voxels (default 0.5). An explicit approximation. The bake's cost is its density steps: ~92 a texel, 98.5%
    # of texels marching to the full reach (bake probe, HSTR_SUN_BAKE_PROBE). bakestep4 (4K sunset walk, bakes kept across arms):
    # bakeCloudSun 0.5 0.557 / 0.412 ms, 1 0.229, 2 0.123. Quality (sunset_hill_bakestep5, every arm fully rebaked, 960 x 540 exact
    # view against the 0.5 anchor repeated last): step 2 0.063% of pixels over 2/255, p99 1/255, p99.9 2/255, max 11/255; step 4
    # 0.72%, p99.9 9/255, max 23/255; sanity (constant brick density in the bake) 2.6%, p99.9 40/255, max 79/255. The squared-excess
    # gate barely sees bakes (0.020159 anchor, 0.020152 step 2, 0.019845 sanity), so the image comparison decides.
    "cloudSunBakeStep": 2.0,
    # Camera marches skip brick cells whose density bound gives at most 0.05 optical depth across, and end a ray at transmittance
    # 0.02 (was 1e-3). The second is an explicit approximation. Path trace (hillthin1, sunset_hill 960x540, 326 spp, squared excess
    # with the noise subtracted): ship 0.020607, thin 0.05 0.020607 (identical), + min transmittance 0.02 0.021053 (+2.2%; p99 /
    # p99.9 unchanged 0.487 / 0.689, max 0.720 -> 0.722), sanity thin 1.0 0.024740 (+20%, max 0.948), ship again 0.020607.
    # Time (thintime2, 4K sunset walk, query + units ms): ship 1.631 / again 1.686, thin 1.585, thin + 0.02 1.484 (-0.17 against
    # the anchors' mean); motion score 6.4 / 6.1% anchors, 5.8% / 2.4%.
    # Higher, MEASURED, no gain (2026-10-13): path trace (thinpt1) excess 0.024574 -> 0.1 0.024558 / 0.2 0.024809 (+1.0%) / 0.4
    # 0.026017 (+5.9%, past the sanity arm's +4.6%); walk (thintime3, arm means) ship 2.95 / 3.11, 0.1 3.04, 0.2 3.03, 0.4 3.47 ms.
    "cloudThinDepth": 0.05,
    "cloudMinTransmittance": 0.02,
    # Sun bakes scheduled on the GPU (read when the residency is created; HSTR_GPU_SUN=0 keeps the CPU scheduler).
    "cloudGpuSun": os.environ.get("HSTR_GPU_SUN", "1") != "0",
    # Residency GPU work (uploads, commits, pages, sun scheduling and bakes, the sun resolve) on a COMPUTE queue beside the frame,
    # published a frame late (HSTRCloud::mAsyncResidency). asyncab3 (4K sunset walk, profiler off, 6 cycles, chunk GPU over both
    # queues): off 3.86 / off2 4.25, async 3.78 / async2 3.81 ms (-0.26, 5 of 6 cycles), sanity (serial +1.7 ms probe) 5.97; churn
    # chunks at the load cap in 13-19 of 20 frames as off. asyncab1 motion score 2: off 5.75 / 5.57%, async 5.84 / 5.76% over 0.02.
    "asyncResidency": True,
    # The residency's CPU update on its own thread, from the end of one frame's residency recording (moved to the frame's end) to
    # the next's (HSTRCloud::mCloudResidencyWorker). worker4 (4K sunset walk at 2 units a frame, arms in one launch, 80 frames
    # each): update 6.22 / 7.88 ms a frame on the frame's thread (off / off again); on, 8.36 ms on the worker of which the frame
    # waited 5.33 - 3.03 ms a frame off the frame's thread; walk mean 9.39 / 8.93 / 11.87 ms, p50 5.62 / 4.60 / 8.60. Motion score 2
    # (worker3): over 0.02 9.12 / 8.87 / 8.75%, p99.9 0.97 / 0.58 / 1.16; parked mapped = desired.
    "cloudResidencyWorker": True,
    # A frame whose residency update is still running does not wait for it (HSTRCloud::mCloudResidencySkipBusy; the residency runs
    # on a frame clock). Once the update fitted a frame (parallel fade ends, maps, merge, upload, sun): skipbusy5 (4K sunset walk at 2
    # units a frame, arms alternating, 80 frames each) walk mean 3.07 ms against 4.54 / 4.85 waiting, p50 2.95 against 3.80 / 3.79,
    # p95 4.54 against 9.89 / 9.01, max 5.84 against 13.1 / 31.4; 32 updates in 80 frames, chunk-end backlogs 0 / 11k / 24k against
    # 0 / 14k / 27k (one chunk 95k: a cut landing at its end). Motion score 2 (skipbusy6, arms on, skip, skip2, on2): over 0.02
    # 5.48 / 4.33% against 5.23 / 5.41%, p99.9 0.344 or less in every arm.
    "cloudResidencySkipBusy": True,
    # The async cut walk's lists built in parallel ranges, the same lists (CloudView::cutTailParallel): cutrays1 (4K sunset walk at
    # 2 units a frame, walks bucketed by the configuration they ran under) tail 73.7 -> 14.0 ms a walk.
    "cloudCutTailParallel": True,
    # Visibility rays stepping tile coordinates instead of dividing, corner rays stopped once they cannot raise the brick's best
    # visibility (CloudView::cutRays 5): the same visibility, checked - 0 of 4.7M rays and refreshes differing (cutrays1, blocked1,
    # cutoff1, waves2). Coarse phase 0.80 -> 0.52-0.55 ms per 1k rays against the reference loop's 0.78-1.09 (cutoff1).
    "cloudCutRays": 5,
    # The walk's visibility refreshes traced on the GPU and applied before the next walk (CloudView::cutGpuVisibility): gpuvis2 (4K
    # sunset walk at 2 units a frame, arms cpu / gpu / gpu2 / cpu2, 80 frames each) walk 156 -> 91 ms, coarse phase 76.7 -> 10.8 ms and
    # 0.869 -> 0.136 G thread cycles a walk; frame mean 4.07 / 4.09 against 4.29 / 3.90 (neutral). Motion score 2 (gpuvis1): over 0.02
    # 5.40 / 5.49% against 5.53 / 5.45%, p99.9 0.344-0.41 in every arm; 0 results dropped.
    "cloudCutGpuVisibility": True,
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
