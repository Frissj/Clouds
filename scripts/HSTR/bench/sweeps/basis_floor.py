from sea_config import mk

# How much of the basis pass is WORK, and how much is threads that exist only to early-out?
#
# The query pass launches one thread per lattice point in the beam image - 1.69x the screen at beamRefMargin 0.15 - and with the
# persistent lattice most of those threads now read one validity word and return. Before designing a work generator that emits the
# entering strip, the refresh subset and the parallax failures directly, this prices the ceiling on that redesign: if the launch
# itself is most of the 1.12 ms, generating work is the whole remaining budget; if it is a small part, the basis cost is real
# marching and the redesign wins little.
#
# beamRefresh = n refreshes one of n*n phases per build, so the marched share is 1/n^2: 1/256, 1/4096, 1/65536 below. The last is
# effectively "carry everything", which is not a shippable image - points are never re-marched - but it is exactly the arm that
# isolates the fixed cost. Quality is expected to degrade in the high-n arms and is not the point; read the times.
#
# The margin arm is meaningful PARKED ONLY. At beamRefMargin 0 the frame re-anchors as soon as the camera turns at all, and every
# re-anchor is a full rebuild, so its turning numbers measure re-anchoring rather than the surplus.
COMMON = dict(
    beamTileSize=4,
    beamLevels=1,
    beamSegments=1,
    beamTemporal=False,
    beamSparse=False,
    beamSparseCut=False,
    beamSparseMinLevel=2,
    beamQueue=False,
    beamGridDispatch=False,
    beamRefreshBlock=4,
    beamCarryTolerance=0.0,
    beamDepthTolerance=0.05,
    beamRefreshDebug=0,
    beamRefFrame=True,
)

TESTS = [
    ("ref refresh 16", mk(**COMMON, beamRefresh=16, beamRefMargin=0.15)),
    ("ref refresh 64", mk(**COMMON, beamRefresh=64, beamRefMargin=0.15)),
    ("ref refresh 256", mk(**COMMON, beamRefresh=256, beamRefMargin=0.15)),
    ("ref refresh 256 m0", mk(**COMMON, beamRefresh=256, beamRefMargin=0.0)),
]
