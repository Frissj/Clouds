from sea_config import mk

TESTS = [
    ["4x1", mk(beamTileSize=4)],
    ["8x1", mk(beamTileSize=8)],
    ["8x2 adaptive", mk(beamTileSize=8, beamLevels=2, beamAdaptiveRoot=True)],
]
