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
    ("shipping 4x1", mk(beamSparse=False)),
    ("sparse volume", mk(**COMMON, beamSparse=True, beamSparseCut=False, beamSparseMinLevel=2)),
    ("sparse cut start 2", mk(**COMMON, beamSparse=True, beamSparseCut=True, beamSparseMinLevel=2)),
    ("sparse cut start 1", mk(**COMMON, beamSparse=True, beamSparseCut=True, beamSparseMinLevel=1)),
    ("sparse cut start 0", mk(**COMMON, beamSparse=True, beamSparseCut=True, beamSparseMinLevel=0)),
]
