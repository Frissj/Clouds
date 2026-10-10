from pathlib import Path
from falcor import *

# Sunset over the Intel cloud sea: the half-resolution Intel sea exactly as IntelCloudSeaHalf.py sets it up (its clouds, residency and
# measured renderer settings), lit and seen through pl-sky's atmosphere (skyModel 2; Atmosphere.h) with the sun low ahead.
launcher = Path(__file__).resolve().parent / "IntelCloudSeaHalf.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
m.activeGraph.getPass("HSTRCloud").set_properties({
    # Two staggered layers of the Intel clouds (CloudSeaDesc::layers): each layer-1 cloud sits across four layer-0 tiles, so the
    # clouds overlap and merge, and their bases spread over the whole layer height.
    "cloudSeaLayers": 2,
    # The near march stops at 2000 world units (the 16-tile window allows 7 x 330 = 2310); the far sea draws the rest. The horizon
    # rays it cuts are the march's longest: 4K walk (viewtime2, frozen) units + query 5.49 -> 3.20 ms, live (viewlive1) 5.80-5.95 ->
    # 3.60-3.65 ms, far sea 0.35 -> 0.69 ms and resolve pixels 0.12 -> 0.41 ms back. Against the 326-spp path trace
    # (sunset_hill_0_0_3840x2160_960x540, hillview5) excess squared log error 0.02048 -> 0.02057 (+0.4%); 1600 is +3.1% (the far
    # sea's proxy draws mid-distance clouds as blobs). Score a changed distance on its second arm: the first after a change is
    # deterministically worse (0.0274 vs 0.0211 at 1600, cause not found).
    "seaViewDistance": 2000.0,
    # A quality trade the owner took (2026-10-08): camera step 4 -> 5 sea voxels. Measured (IntelCloudSeaHalf.py's notes, qualfront /
    # qualtime4): +0.5% excess squared log error against the 326-spp path trace, units + query -0.43 ms on the 4K walk, motion
    # slightly less stable. Step 6 (+2.3% in hillcent1, -0.9 ms) was offered and not taken.
    # 2026-10-09, toward 2 ms (owner: take the trades): step 7, beamOctScale 0.42, tile tolerance 0.6 (policy ceiling 1.4).
    # MEASURED (combo1, 4K walk, frozen after a baking settle, worldCacheView, overlaps off): units + query 2.50 / 2.38 -> 1.62 ms
    # (-0.82); moving frames against a fresh rebuild 2.32% over 0.02 vs ship 2.34 / 2.63% (scale 0.42 + tolerance 0.6 alone scored
    # 4.51% in the same run - motion scores are position-dependent, see 24429297). Against the 326-spp path trace (combopt1): excess
    # 0.020766 -> 0.022117 (+6.5%), p99 0.4871 / p99.9 0.6889 unchanged, max 0.721 -> 0.790; anchors repeat to the digit, sanity
    # 0.025365. Alone (tradetime1 / tradept1): scale 0.42 -0.59 ms +3.6%, tolerance 0.6 -0.29 ms +0.8%, step 7 -0.28 ms +2.5%,
    # step 6 -0.09 ms (not worth it).
    # Then (tradetime2 / tradept2, same conditions): beamOctScale 0.42 -> 0.35, units + query 1.62 / 1.57 -> 1.40 ms, excess +2.9%;
    # far sea at scale 4, farSea 0.65 / 0.74 -> 0.415 ms, +4.4%; both 0.022117 -> 0.023703 (+7.2%), p99 / p99.9 unchanged, max
    # 0.790 -> 0.727. No gain (same run): far sea to 10000, step 8, tolerance 0.8 (-0.04 ms), refresh 16 (-0.09 ms).
    # Then (tradetime3 / tradept3, same conditions, anchors ship 0.659 query / 0.701 units / 0.409 farSea ms): far sea scale 8
    # query 0.544, units 0.603, farSea 0.324 (-0.30 ms), excess 0.023703 -> 0.025976 (+9.6%), p99 / p99.9 unchanged, max 0.893.
    # Not taken: beamOctScale 0.30 (-0.10 ms, +2.8%), 0.25 (-0.17 ms, +6.9%, residual +0.03); 0.25 with far scale 8 breaks p99
    # (0.4871 -> 0.5793, excess +15.6%). Sanity tol 10: 0.028843, anchors repeat to the digit.
    # Then beamOctScale 0.30 with far scale 8 (tradept4): excess 0.025976 -> 0.026557 (+2.2%), p99 / p99.9 / max unchanged (0.4871
    # / 0.6889 / 0.893), sanity 0.031137, anchors repeat to the digit; time from tradetime3 (scale 0.30 alone, -0.10 ms).
    # Then beamOctScale 0.25 with far scale 6 (tradept5 / tradetime4): excess 0.026567 -> 0.026733 (+0.6%), p99 / p99.9 unchanged,
    # max 0.893 -> 0.820; query + units + dirtyTiles + resolve + farSea 2.283 / 2.345 -> 2.241 / 2.273 ms (-0.057; farSea +0.05,
    # the beam passes -0.11). Same run, each breaking p99 (0.4871 -> 0.5793): scale 0.27 (+2.4%), tolerance 0.7, step 7.5.
    # A lower internal resolution for an upscaler (DLSS) - MEASURED and closed (resprice1, 4K walk, sunset_motion per-arm "_res",
    # the beam's angle held at 4K's 0.25 by scaling beamOctScale): resolve 0.466 / 0.477 at 4K (anchors) -> 0.239 at 2560x1440
    # (0.375), 0.161 at 1920x1080 (0.5); march 0.836 / 0.930 -> 0.988 / 0.775 (inside the anchors' spread). The beam already marches
    # at a quarter of 4K's angle, so only the pixel passes scale: ~0.31 ms saved at 1080p before the upscaler's own cost (not
    # measured here) and its ghosting on volumes. Stop rule was 0.5 ms.
    "minStepVoxels": 7.0,
    "beamOctScale": 0.25,
    "seaFarScale": 6,
    "beamTolerance": 0.6,
    "beamPolicyTolerance": 1.4,
    # Two layers ask for about twice the sun bakes of one; until a brick's bake lands its samples take the live sun march. 1024 a frame
    # (bakeCloudSun ~0.2-2 ms while baking). NOT MEASURED against 256: the A/B that seemed to show it (1871ae53) changed the rate at
    # runtime, which did not rebuild the residency then, so its second arm was the first one settled longer.
    "cloudSunBakesPerFrame": 1024,
    # The GPU sun scheduling every 4th frame with 4x the moving cap (the frames between release only). MEASURED (sunevery1, 4K walk
    # 2, live residency, one launch, every 1 / 2 / 4 / 1 / 2): cloudSea 0.446 / 0.286 / 0.144 / 0.417 / 0.230 ms (scheduleSun 0.295
    # / 0.213 / 0.114 / 0.293 / 0.168, resolveCloudSun 0.143 / 0.066 / 0.024 / 0.118 / 0.055); query + units within the anchors'
    # spread (1.59 / 1.63 / 1.53 / 1.38 / 1.38); bakes waiting at the arms' ends 180k / 189k / 187k / 211k / 209k, stale 116k /
    # 108k / 107k / 107k / 105k - the backlog drains as fast.
    # Re-measured with 4096 loads a frame (sunevery2, 4K walk, 240-frame chunks, every 4 / 8 / 16 / sanity 1 / 4): stamp 0.126 /
    # 0.089 / 0.103 / 0.160 / 0.112 ms a frame; changes stamped per 240 frames 83.7k / 71.1k / 95.2k / 143.6k - dedupe over a longer
    # interval barely shrinks the stamp's work (~350 cell changes a frame), so a longer interval buys ~0.03 ms. Left at 4.
    "cloudSunEvery": 4,
    # The low sun's horizontal banding: past the fine near reach the sun depth comes from the domain's coarse sheared field (5.16-unit
    # layers), which at 4 degrees of elevation draws a bright/dark band per layer on the lit faces (~30 px apart at 4K). A reach of 8
    # sea voxels (~41 world units) takes the handoff deep enough that the bands are gone (4 leaves faint ones, 32 is no better than 8).
    # It asks for many more bricks and bakes (desired at the 243k atlas cap, ~600k bakes), so the sun atlas is doubled (~1056 MB at the
    # 256 MB pool). 4K, same process, 1800-frame settle then a 60-degree turn, reach 8: pool x1 parked 1.63 ms, turn 4.62, after the
    # turn 4.45 (p95 12.6) with 309k bakes waiting; x2 parked 1.74, turn 2.80, after 4.32 (p95 16.0) with 41k waiting when settled.
    # Reach 2 (banded), earlier run: parked 3.11, after the turn 4.93.
    "sunNearVoxels": 8.0,
    "cloudSunPoolScale": 2,
    # 4 degrees above the horizon, ahead and to the right of the start view.
    "sunDirection": float3(0.30, 0.07, 0.95),
    # The sky, the aerial perspective and the clouds' sun and sky light come from the atmosphere; sunRadiance and skyRadiance are
    # derived from it. A world unit is 10 m and the ground 1 km below y = 0, which puts the sea at ~1-2.6 km.
    "skyModel": 2,
    "atmosphereSunIntensity": 20.0,
    "atmosphereWorldToKm": 0.01,
    "atmosphereGroundY": -100.0,
    "atmosphereAerialDistance": 150.0,
})
