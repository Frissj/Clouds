from sea_config import mk

# motion: beamRefresh after the single-reconstruction carry and the onto-itself fast path.
S = dict(beamShip=True, beamCarryTolerance=0.02)
TESTS = [
    ["ship", mk(beamShip=True)],
    ["r2", mk(beamRefresh=2, **S)],
    ["r3", mk(beamRefresh=3, **S)],
    ["r4", mk(beamRefresh=4, **S)],
    ["r4 steps", mk(beamRefresh=4, beamRefreshDebug=7, **S)],
    ["r4 b4", mk(beamRefresh=4, beamRefreshBlock=4, **S)],
]
