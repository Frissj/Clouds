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
    # Two layers ask for ~170k sun bakes (one layer ~82k); until a brick's bake lands its samples take the live sun march and every
    # bake that lands re-marches the beam through it. 4K, same process, settled over 1800 frames then a 60-degree turn: 256 a frame
    # never settled (130k still waiting, parked 7.29 ms, after the turn 8.13); 1024 drained in ~900 frames, parked 2.26 ms (profiled
    # GPU 2.89 ms, bakeCloudSun 0.18 ms while baking), after the turn 3.45 ms with the new 21k backlog gone within 90 frames.
    "cloudSunBakesPerFrame": 1024,
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
