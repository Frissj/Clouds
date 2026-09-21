from sea_config import mk

# Where the three changes leave a camera that is actually being played with.
#
# The anchored rectangle is the baseline. The octahedral image removed re-anchoring, so rotation stopped costing; the block guard
# removed the translation discovery sweep, so moving stopped costing what it was costing; and beamScreenResidual marches a failed
# tile's pixels in screen space instead of resampling them out of an anisotropic map, which is where the octahedral image's quality
# deficit came from. Each was measured on its own. This measures them together, on motions a player produces rather than on the
# extremes that isolated the mechanisms:
#
#   look        a slow pan, the camera turning and nothing else
#   flick       0.05 rad per frame, about as fast as a mouse flick gets
#   walk        forward at 2 units a frame with a little turn, the common case
#   fly         forward at 20 with a turn, the aggressive case
#   sprint      forward at 20, no turn - the one posture where the rectangle still wins
#
# Quality is against the world-cache march at one-voxel steps, and it is the point of the third arm: the octahedral image is only
# worth having if it holds the bar, and at beamTolerance 0.02 with the screen residual it beat the rectangle at 720p.
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
    beamGuard=True,
)

RECT = dict(COMMON, beamOct=False, beamRefMargin=0.15)
OCT = dict(COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0)
TESTS = [
    ("rect", mk(**RECT, beamTolerance=0.05)),
    ("oct", mk(**OCT, beamTolerance=0.05)),
    ("oct screen", mk(**OCT, beamTolerance=0.02, beamScreenResidual=True)),
]
