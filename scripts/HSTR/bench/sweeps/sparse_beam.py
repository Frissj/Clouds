from sea_config import mk

# Keep temporal reuse disabled so parked and moving measurements pay the same
# camera-space rebuild. The two sparse arms isolate direct packed-result resolve
# from the query compiler itself.
COMMON = dict(
    beamTileSize=32,
    beamLevels=4,
    beamSegments=1,
    beamTemporal=False,
    beamRefresh=0,
    beamQueue=False,
    beamGridDispatch=False,
)

TESTS = [
    ("legacy lattice", mk(**COMMON, beamSparse=False)),
    ("sparse start 0", mk(**COMMON, beamSparse=True, beamSparseDirect=True, beamSparseMinLevel=0)),
    ("sparse start 1", mk(**COMMON, beamSparse=True, beamSparseDirect=True, beamSparseMinLevel=1)),
    ("sparse start 2", mk(**COMMON, beamSparse=True, beamSparseDirect=True, beamSparseMinLevel=2)),
]
