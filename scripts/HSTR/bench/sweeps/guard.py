from sea_config import mk

# Does a per-block parallax certificate reach the ceiling the assume-carry probe measured?
#
# The renderer asks a million points whether translation invalidated them to find the eleven thousand it did, and each question
# reads a thirteen-sample neighbourhood. beamGuard keeps, per block of beamRefreshBlock lattice points, the smallest distance any
# of its points holds and the camera it was last verified from; while |dP| * beamGuardAngle stays under beamDepthTolerance times
# that distance, no point in the block can have failed and none of them are asked.
#
# The ceiling, measured by skipping the test outright with beamAssumeCarry (a wrong image, a real floor on cost):
#
#              rect   rect ceiling    oct   oct ceiling
#   fly 20     1.71       0.87        2.09      1.15      (level-0 query 0.82 -> 0.05, 0.86 -> 0.07)
#   fly turn   1.37       1.33        2.06      1.12
#
# So read two things. The TIME against that ceiling: a guard that certifies most blocks should land near it, and one that certifies
# none should land on the untouched arm. And the QUALITY against the world-cache march, which must not move at all - the guard is a
# claim that the per-point test would have passed, so if it is wrong the error shows up immediately.
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
    ("rect", mk(**RECT, beamGuard=False, beamAssumeCarry=False)),
    ("rect guard", mk(**RECT, beamGuard=True, beamAssumeCarry=False)),
    ("rect ceiling", mk(**RECT, beamGuard=False, beamAssumeCarry=True)),
    ("oct", mk(**OCT, beamGuard=False, beamAssumeCarry=False)),
    ("oct guard", mk(**OCT, beamGuard=True, beamAssumeCarry=False)),
    ("oct ceiling", mk(**OCT, beamGuard=False, beamAssumeCarry=True)),
]
