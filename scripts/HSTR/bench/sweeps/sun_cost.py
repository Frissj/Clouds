from sea_config import mk

# With the beam's rotation and translation costs removed, the SUN is what is left.
#
# Profiled at fly 20 with the block guard on, residualPages/fineSun is 0.35 ms in both layouts - the largest single item in the
# rectangle's 0.89 ms frame, and identical in the octahedral one, because it has nothing to do with either. It is the per-tile sun
# page rebuild, throttled by cloudSunTilesPerFrame, plus sunOctaves, an unconditional full-volume gather and three blurs that runs
# whether or not anything changed.
#
# cloudSunTilesPerFrame was measured in an earlier session as cheaper AND slightly better at 1 than at the default 8, and none of
# the sweeps since have set it. So this checks both halves at once: whether the throttle still pays here, and what is left
# underneath it when it does. Quality is read as carefully as time - a sun page rebuilt more slowly is a sun page that is more
# out of date, and that shows up against the world-cache march or it does not.
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
    beamGuard=True,
)

RECT = dict(COMMON, beamOct=False, beamRefMargin=0.15)
OCT = dict(COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0)
TESTS = [
    ("rect sun 8", mk(**RECT, cloudSunTilesPerFrame=8)),
    ("rect sun 2", mk(**RECT, cloudSunTilesPerFrame=2)),
    ("rect sun 1", mk(**RECT, cloudSunTilesPerFrame=1)),
    ("oct sun 8", mk(**OCT, cloudSunTilesPerFrame=8)),
    ("oct sun 2", mk(**OCT, cloudSunTilesPerFrame=2)),
    ("oct sun 1", mk(**OCT, cloudSunTilesPerFrame=1)),
]
