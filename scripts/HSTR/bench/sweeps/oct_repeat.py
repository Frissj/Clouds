from sea_config import mk

# Is an octahedral arm's cost REPRODUCIBLE within one run, or does it depend on what ran before it?
#
# The same configuration - oct x1.0, beamTolerance 0.05 - measured 0.36 ms under fast yaw in one run and 2.41 ms in the next, with
# the rectangle control reproducing to 0.07 ms across both. Until that is explained no octahedral timing means anything, so this
# repeats the identical arm three times, interleaved with the rectangle, inside a single run.
#
# The suspicion is that the harness's assumption of independent arms is false here. The octahedral image is fixed to the WORLD and
# persists until its dimensions change, so an arm inherits every direction the previous arm marched: an oct arm that follows another
# oct arm of the same scale starts warm, while one that follows the rectangle starts cold and pays for every direction the motion
# exposes. If that is it, the repeats will fall monotonically and the rectangle will not move.
COMMON = dict(
    beamTileSize=4,
    beamLevels=1,
    beamSegments=1,
    beamTemporal=False,
    beamSparse=False,
    beamSparseCut=False,
    beamQueue=False,
    beamGridDispatch=False,
    beamRefreshBlock=4,
    beamCarryTolerance=0.0,
    beamDepthTolerance=0.05,
    beamRefreshDebug=0,
    beamRefFrame=True,
    beamRefresh=256,
    beamTolerance=0.05,
)

OCT = dict(COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0)
TESTS = [
    ("oct A", mk(**OCT)),
    ("oct B", mk(**OCT)),
    ("rect 1", mk(**COMMON, beamOct=False, beamRefMargin=0.15)),
    ("oct C", mk(**OCT)),
    ("rect 2", mk(**COMMON, beamOct=False, beamRefMargin=0.15)),
    ("oct D", mk(**OCT)),
]
