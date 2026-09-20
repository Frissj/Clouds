from sea_config import mk

# Is yaw's cloudSea/scheduleSun cost caused by rotation producing new residency CUTS?
#
# CloudResidency redoes its cut once the camera turns by cutTurn degrees (default 3), and scheduleSunBakes puts mCutId straight
# into the inputs it compares to decide whether it can idle - so a rotation-induced cut defeats the idle skip and runs the whole
# GPU sun scheduler (scan, select, emit, evict, assign). A yaw of 0.004 rad/frame is 0.229 deg/frame, so that is a new cut about
# every 13 frames, yet scheduleSun reads 0.50 ms as a per-frame MEAN under yaw and does not appear at all parked.
#
# Raising cutTurn to 360 makes rotation alone never trigger a cut. If scheduleSun then disappears under yaw, the diagnosis holds
# and the fix is to stop a cut that only reprioritised bricks from forcing a full rescan - a sun page baked for a static sun stays
# radiometrically valid however the camera turns. If it does not disappear, the cost is somewhere else entirely.
COMMON = dict(
    beamTileSize=4, beamLevels=1, beamSegments=1, beamTemporal=False, beamSparse=False, beamSparseCut=False,
    beamQueue=False, beamGridDispatch=False, beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05,
    beamRefreshDebug=0, beamRefFrame=True, beamRefMargin=0.15, beamRefresh=256,
)

TESTS = [
    ("cutTurn 3 (default)", mk(**COMMON, cloudCutTurn=3.0)),
    ("cutTurn 360", mk(**COMMON, cloudCutTurn=360.0)),
]
