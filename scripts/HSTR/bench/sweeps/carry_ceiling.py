from sea_config import mk

# What would a per-block parallax certificate be worth?
#
# At 20 units per frame 98.93% of the 1.067M swept points survive translation, but every one of them runs beamReprojectQuery to
# find that out, and that reads a thirteen-sample neighbourhood. A certificate - one conservative depth bound per block of points -
# would let a surviving block skip the read entirely. beamAssumeCarry prices the ceiling on that before it is built: it lets a
# translated point take the same early-out a parked camera takes, without testing anything.
#
# The image it produces is WRONG. Nothing is ever invalidated by moving, so quality in that arm is meaningless and only its time is
# being read. The dispatch is left sweeping in both arms, so the difference is the per-thread test alone and not the work
# generation that would come after it.
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

RECT = dict(COMMON, beamOct=False, beamRefMargin=0.15)
OCT = dict(COMMON, beamOct=True, beamOctFull=False, beamOctScale=1.0)
TESTS = [
    ("rect", mk(**RECT, beamAssumeCarry=False)),
    ("rect ceiling", mk(**RECT, beamAssumeCarry=True)),
    ("oct", mk(**OCT, beamAssumeCarry=False)),
    ("oct ceiling", mk(**OCT, beamAssumeCarry=True)),
]
