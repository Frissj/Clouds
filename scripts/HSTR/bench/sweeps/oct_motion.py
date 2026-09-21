from sea_config import mk

# Does removing the anchor remove the re-anchor cost? This is what the whole octahedral image is for.
#
# The rectangle re-anchors when the screen would leave it: 16 times in 48 frames at 0.05 rad/frame, each one re-marching 100% of the
# image to cover the 4.4% of directions the turn actually exposed, which is 4.16 ms of a 6.84 ms fast turn at 4K. The octahedral
# image has no anchor at all - a direction's texel is fixed for the life of the renderer - so a turn should cost the entering
# directions and nothing else, and the level-0 query pass should stop caring about angular velocity.
#
# Run this at 720p, NOT 4K: the sphere at parity resolution is many times the screen and the residual does not fit at 4K until it is
# paged. So read the SHAPE - how each arm's cost changes from slow yaw to fast yaw - rather than the absolute milliseconds. If the
# rectangle climbs steeply and the octahedral image does not, that is the result; if both climb, the anchor was not the cause.
#
# beamOctScale 1.4 is angular parity with the screen (scale 1 undersamples: it matched the sphere's average density against the
# screen's centre density, and a perspective frame is denser than its own average). beamOctFull off so the screen's footprint is
# bounded rather than the whole sphere being built, which is what a real frame would do.
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
)

# beamScreenResidual marches a failed tile's pixels in screen space instead of resampling the beam-image residual, which removed
# the octahedral quality deficit outright at x1.0 resolution: with beamTolerance 0.005 it scores 0.063% of pixels over 0.02 against
# the rectangle's 0.527%, with max error BELOW the rectangle's and no pixels at all over 0.1. It costs the residual's carry, so
# every failed pixel marches every build - and tightening the tolerance fails more of them. This prices that.
TESTS = [
    ("rect m15", mk(**COMMON, beamOct=False, beamRefMargin=0.15, beamTolerance=0.05)),
    ("oct x1.0", mk(**COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0, beamTolerance=0.05)),
    ("oct scr t.05", mk(**COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0, beamScreenResidual=True, beamTolerance=0.05)),
    ("oct scr t.02", mk(**COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0, beamScreenResidual=True, beamTolerance=0.02)),
    ("oct scr t.005", mk(**COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0, beamScreenResidual=True, beamTolerance=0.005)),
]
