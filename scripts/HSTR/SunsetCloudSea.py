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
    # Two layers ask for about twice the sun bakes of one; until a brick's bake lands its samples take the live sun march. 1024 a frame
    # (bakeCloudSun ~0.2-2 ms while baking). NOT MEASURED against 256: the A/B that seemed to show it (1871ae53) changed the rate at
    # runtime, which did not rebuild the residency then, so its second arm was the first one settled longer.
    "cloudSunBakesPerFrame": 1024,
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
