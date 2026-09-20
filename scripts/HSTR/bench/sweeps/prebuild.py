from sea_config import mk

# THE GATE on a persistent world-direction cache.
#
# The beam basis and the beam residual are already functions of world direction and already sit at fixed indices under rotation -
# but both are gated on beamScreenBounds, so only the part of the reference image the screen currently covers is ever built. A
# turn therefore exposes directions that have never been marched, and that is the entering-strip cost. beamRefPrebuild builds and
# keeps the WHOLE image instead, so a turn reveals nothing new and only changes which built directions get resolved.
#
# If yaw then costs what parked costs, the premise behind fixed world-direction sectors is demonstrated and worth building. If it
# does not, the remaining yaw cost is something else and sectors would not have removed it. Measured breakdown says to expect
# about 0.95 ms of the 1.15 ms yaw penalty to go (entering basis 0.56 + entering residual ~0.4) and roughly 0.53 ms to remain in
# cloudSea/scheduleSun, which turning triggers and no beam cache can help.
COMMON = dict(
    beamTileSize=4, beamLevels=1, beamSegments=1, beamTemporal=False, beamSparse=False, beamSparseCut=False,
    beamQueue=False, beamGridDispatch=False, beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05,
    beamRefreshDebug=0, beamRefFrame=True,
)

TESTS = [
    ("r256 normal", mk(**COMMON, beamRefresh=256, beamRefMargin=0.15, beamRefPrebuild=False)),
    ("r256 prebuilt", mk(**COMMON, beamRefresh=256, beamRefMargin=0.15, beamRefPrebuild=True)),
    ("r256 prebuilt m50", mk(**COMMON, beamRefresh=256, beamRefPrebuild=True, beamRefMargin=0.5)),
]
