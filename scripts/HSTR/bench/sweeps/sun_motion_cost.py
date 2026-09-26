from sea_config import mk

# What a moving sun still costs after sunColumns / sunInvalidateAngle, and the throttles for it:
# - sunFieldAngle: the sun field and octaves rebuilt once the sun has turned this far (radians) from their build, and when it stops.
# - cloudSunBakeAngle: degrees between per-brick sun bake generations (each re-resolves the sun slot table, ~0.27 ms).
# - the world cache as the shipped scene runs it (CloudSea.py: worldCacheUpdates 1, a photon update every frame; the harness
#   freezes it after its 64-sample settle), before and after worldCacheTarget (updates only while a tile is under 64 batches).
# Live-cache arms last: they leave the cache updating for whatever follows them in the process.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/sun_motion_cost.py --steps 1
#          --motions "park 0 0 0" "park 0 0 0.01"
TESTS = [
    ("ship", mk()),
    ("field 1 deg", mk(sunFieldAngle=0.0175)),
    ("field 2 deg", mk(sunFieldAngle=0.035)),
    ("bake 1 deg", mk(cloudSunBakeAngle=1.0)),
    ("bake 2 deg", mk(cloudSunBakeAngle=2.0)),
    ("live cache", mk(worldCacheUpdates=1, worldCacheTarget=0)),
    ("live cache target 64", mk(worldCacheUpdates=1, worldCacheTarget=64)),
]
