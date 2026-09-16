from sea_config import mk

# motion: beamRefresh against the shipped beam view, with the specialization on.
S = dict(beamShip=True)
TESTS = [
    ["ship", mk(**S)],
    ["r2", mk(beamRefresh=2, **S)],
    ["r3", mk(beamRefresh=3, **S)],
    ["r2 ct.02", mk(beamRefresh=2, beamCarryTolerance=0.02, **S)],
    ["r2 ct.01", mk(beamRefresh=2, beamCarryTolerance=0.01, **S)],
    ["r2 ct.02 px1", mk(beamRefresh=2, beamCarryTolerance=0.02, beamParallax=1.0, **S)],
    ["r2 ct.01 px.5", mk(beamRefresh=2, beamCarryTolerance=0.01, beamParallax=0.5, **S)],
]
