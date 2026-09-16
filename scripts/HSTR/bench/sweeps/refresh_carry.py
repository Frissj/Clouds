from sea_config import mk

# motion: beamRefresh with the carry tolerance, 2x2 against 3x3.
S = dict(beamShip=True, beamCarryTolerance=0.02)
TESTS = [
    ["ship", mk(beamShip=True)],
    ["r2 ct.02", mk(beamRefresh=2, **S)],
    ["r3 ct.02", mk(beamRefresh=3, **S)],
    ["r3 ct.01", mk(beamRefresh=3, beamShip=True, beamCarryTolerance=0.01)],
    ["r4 ct.02", mk(beamRefresh=4, **S)],
]
