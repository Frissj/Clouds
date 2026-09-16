from sea_config import mk

# ab: the tile oracle (beamOracle) against the shipped tests, per tiling, with and without tile centres.
O = dict(beamOracle=1)
CL = dict(beamCentreless=True)
TESTS = [
    ["anchor", mk()],
    ["or t4", mk(**O)],
    ["or t4 cl", mk(**O, **CL)],
    ["or cl n2", mk(beamOracle=2, **CL)],
    ["or cl n4", mk(beamOracle=4, **CL)],
    ["or cl b.03", mk(beamOracle=1, beamOracleBar=0.03, **CL)],
    ["or t8", mk(beamTileSize=8, **O)],
    ["or t8 cl", mk(beamTileSize=8, **O, **CL)],
    ["or t8 l2", mk(beamTileSize=8, beamLevels=2, **O)],
    ["or t16 l3", mk(beamTileSize=16, beamLevels=3, **O)],
    ["or t4l2", mk(beamLevels=2, **O)],
    ["cl curv.05", mk(beamTolerance=0.05, beamEdgeContrast=0.0, **CL)],
    ["cl curv.1", mk(beamTolerance=0.1, beamEdgeContrast=0.0, **CL)],
    ["cl e.5", mk(**CL)],
    ["anchor end", mk()],
]
