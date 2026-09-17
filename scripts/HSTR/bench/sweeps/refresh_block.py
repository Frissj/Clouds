from sea_config import mk

# motion: beamRefresh with block-coherent cells (beamRefreshBlock) and a slower pattern for tile centres (beamRefreshCentres).
S = dict(beamShip=True, beamCarryTolerance=0.02)
TESTS = [
    ["ship", mk(beamShip=True)],
    ["r4", mk(beamRefresh=4, **S)],
    ["r4 b2", mk(beamRefresh=4, beamRefreshBlock=2, **S)],
    ["r4 b4", mk(beamRefresh=4, beamRefreshBlock=4, **S)],
    ["r4 b8", mk(beamRefresh=4, beamRefreshBlock=8, **S)],
    ["r2 b8", mk(beamRefresh=2, beamRefreshBlock=8, **S)],
    ["r2 c4", mk(beamRefresh=2, beamRefreshCentres=4, **S)],
    ["r4 c8", mk(beamRefresh=4, beamRefreshCentres=8, **S)],
]
