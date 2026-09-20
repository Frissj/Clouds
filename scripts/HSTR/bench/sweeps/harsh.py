from sea_config import mk

# Fast camera motion, which every other sweep avoids. Three things here are not exercised at 0.004 rad/frame:
#
#   - RE-ANCHORING. The reference frame is a finite image; once the screen would leave it the frame re-anchors and that build
#     starts from nothing (about 6 ms parked, 14 under yaw). At 0.004 rad/frame that is one build in 59 and hides in the mean.
#     At 0.05 it is roughly one build in 5, so the amortised cost of re-anchoring IS the cost of turning.
#   - The entering strips. A fast turn exposes a wide strip rather than a few tiles, and if the rectangle subtraction
#     under-covers, the error shows here first and as a band down one edge rather than as a change in the mean.
#   - Fast translation, which takes the sweep path deliberately: a camera that moves can fail parallax anywhere.
#
# Read the WORST STEP and the max, not the mean. A generator that misses work produces a few very wrong pixels in the frames
# right after the motion, which a 12-step mean will bury.
COMMON = dict(
    beamTileSize=4, beamLevels=1, beamSegments=1, beamTemporal=False, beamSparse=False, beamSparseCut=False,
    beamQueue=False, beamGridDispatch=False, beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05,
    beamRefreshDebug=0, beamRefFrame=True, beamRefMargin=0.15,
)

TESTS = [
    ("r16 (insurance)", mk(**COMMON, beamRefresh=16)),
    ("r256", mk(**COMMON, beamRefresh=256)),
    ("r256 screen", mk(**COMMON, beamRefresh=256, beamRefFrame=False)),
]
