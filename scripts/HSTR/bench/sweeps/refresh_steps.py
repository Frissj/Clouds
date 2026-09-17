from sea_config import mk

# motion: where beamRefresh's query time goes - the steps marched and carried (debug 7), and the carry path's own cost (debug 2
# accepts every reprojection after a single lattice read).
S = dict(beamShip=True, beamCarryTolerance=0.02, beamRefresh=4)
TESTS = [
    ["ship", mk(beamShip=True)],
    ["r4", mk(**S)],
    ["r4 steps", mk(beamRefreshDebug=7, **S)],
    ["r4 unchecked", mk(beamRefreshDebug=2, **S)],
]
