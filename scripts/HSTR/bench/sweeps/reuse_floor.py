from sea_config import mk

# How far the SHIPPED feature set already is from the 2 ms target, parked and flying.
#
# Every sparse/cut experiment on this view has run with temporal reuse switched off on purpose (sparse_beam.py, domain_cut.py:
# "keep temporal reuse disabled so parked and moving measurements pay the same camera-space rebuild"), which is right for isolating
# an evaluator and wrong for reading the frame budget. beamRefresh carries most basis points from the previous build and beamTemporal
# reprojects the frame, so the two of them move the parked number by several milliseconds and are exactly what a moving camera puts
# at risk. This measures the floor the current evaluator can reach before any architectural change, so the remaining gap is known.
#
# Arms are the shipped dense 4x1 lattice; the sparse compiler is measured separately (sparse_coherence.py) and is slower here.
COMMON = dict(
    beamSparse=False,
    beamSparseCut=False,
    beamSparseMinLevel=2,
    beamQueue=False,
    beamGridDispatch=False,
    beamSegments=1,
)

TESTS = [
    ("no reuse", mk(**COMMON, beamRefresh=0, beamRefreshBlock=1, beamTemporal=False)),
    ("refresh 4", mk(**COMMON, beamRefresh=4, beamRefreshBlock=4, beamTemporal=False)),
    ("temporal", mk(**COMMON, beamRefresh=0, beamRefreshBlock=1, beamTemporal=True)),
    ("refresh 4 + temporal", mk(**COMMON, beamRefresh=4, beamRefreshBlock=4, beamTemporal=True)),
]
