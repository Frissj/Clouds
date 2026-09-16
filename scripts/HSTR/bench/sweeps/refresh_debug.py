from sea_config import mk

# motion: which beamRefresh check rejects the carried values (beamRefreshDebug modes).
R = dict(beamShip=True, beamRefresh=2, beamCarryTolerance=0.02)
TESTS = [
    ["r2", mk(**R)],
    ["r2 all", mk(beamRefreshDebug=2, **R)],
    ["r2 no depth", mk(beamRefreshDebug=4, **R)],
    ["r2 no tile", mk(beamRefreshDebug=5, **R)],
    ["r2 no carry", mk(beamRefreshDebug=6, **R)],
]
