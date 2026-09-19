from sea_config import mk

# Exact-query count candidates after virtual paging/locality work. All arms keep the shipped density integration and are first
# scored against the exact per-pixel march; only candidates that survive this gate are compared with the saved path-traced image.
TESTS = [
    ["4x1", mk(beamTileSize=4)],
    ["8x1", mk(beamTileSize=8)],
    ["16x1", mk(beamTileSize=16)],
    ["8x2 adaptive", mk(beamTileSize=8, beamLevels=2, beamAdaptiveRoot=True)],
    ["16x3 adaptive", mk(beamTileSize=16, beamLevels=3, beamAdaptiveRoot=True)],
]
