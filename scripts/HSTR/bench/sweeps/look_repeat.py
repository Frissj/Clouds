from sea_config import mk

# At 0.01 rad/frame the octahedral image dispatches 32 query threads and 256 residual threads - exactly what a parked camera
# dispatches - yet the profiler attributes 0.81 ms to the residual march. 256 rays cannot cost that, so something one-off is being
# averaged into the 48-frame window. The obvious candidate is the fresh build every arm now starts with: beamReset clears the
# lattice, the residual and the level map, and for the octahedral sphere at 720p that is roughly 300 MB of texture.
#
# Repeating the identical arm separates the two. If the cost is a one-off, the arms will not agree with each other in a way that
# tracks nothing but their position in the run; if it is real per-frame work, they will all report it.
COMMON = dict(
    beamTileSize=4, beamLevels=1, beamSegments=1, beamTemporal=False, beamSparse=False, beamSparseCut=False,
    beamQueue=False, beamGridDispatch=False, beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05,
    beamRefreshDebug=0, beamRefFrame=True, beamRefresh=256, beamTolerance=0.05, beamGuard=True,
)
OCT = dict(COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0)
TESTS = [
    ("oct 1", mk(**OCT)),
    ("oct 2", mk(**OCT)),
    ("oct 3", mk(**OCT)),
    ("rect", mk(**COMMON, beamOct=False, beamRefMargin=0.15)),
    ("oct 4", mk(**OCT)),
]
