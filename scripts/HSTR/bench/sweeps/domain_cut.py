from sea_config import mk

# The projected domain transport cut against the two evaluators it replaces. Every arm shares one sparse hierarchy so the only
# controlled difference is what a basis position costs: a full camera march (sparse volume) or the tile's front-to-back cut, whose
# cells are either taken from their trilinear page or marched over their own segment alone.
#
# cutTransmittanceTolerance is the whole question: it is the transmittance error a page may stand in for, so it decides how much of
# a ray survives as unresolved residual. 0 marches every cell the cut visits (the cut is then only an ordering and an empty-space
# reject), and the arms above it trade image error for the cells they hand to a page.
COMMON = dict(
    beamTileSize=32,
    beamLevels=4,
    beamSegments=1,
    beamTemporal=False,
    beamRefresh=0,
    beamQueue=False,
    beamGridDispatch=False,
    beamSparse=True,
    beamSparseMinLevel=2,
)

# Nothing is put back between tests, so every arm sets every key any other arm sets (ab_test.py).
TESTS = [
    ("shipping 4x1", mk(beamSparse=False, beamSparseCut=False, cutTransmittanceTolerance=0.01)),
    ("sparse volume", mk(**COMMON, beamSparseCut=False, cutTransmittanceTolerance=0.01)),
    ("cut tol 0", mk(**COMMON, beamSparseCut=True, cutTransmittanceTolerance=0.0)),
    ("cut tol 0.002", mk(**COMMON, beamSparseCut=True, cutTransmittanceTolerance=0.002)),
    ("cut tol 0.01", mk(**COMMON, beamSparseCut=True, cutTransmittanceTolerance=0.01)),
    ("cut tol 0.05", mk(**COMMON, beamSparseCut=True, cutTransmittanceTolerance=0.05)),
]
