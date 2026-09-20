from sea_config import mk

# Diagnostic for isolating register/occupancy cost from actual live-sun work.
# The disabled arms are not shipping candidates: moving/live-residency validation
# must be served by a separate fallback queue before the rare path can be removed.
SPARSE = dict(
    beamTileSize=32,
    beamLevels=4,
    beamSegments=1,
    beamTemporal=False,
    beamRefresh=0,
    beamQueue=False,
    beamGridDispatch=False,
    beamSparse=True,
    beamSparseCut=False,
    beamSparseMinLevel=2,
)

TESTS = [
    ("shipping live sun", mk(beamSparse=False, cloudSunLiveMarch=True)),
    ("shipping no live sun", mk(beamSparse=False, cloudSunLiveMarch=False)),
    ("sparse live sun", mk(**SPARSE, cloudSunLiveMarch=True)),
    ("sparse no live sun", mk(**SPARSE, cloudSunLiveMarch=False)),
]
