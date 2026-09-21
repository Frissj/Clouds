from sea_config import mk

# The screen residual against a persistent one, at 4K, on the gameplay motions.
#
# Plain oct resolves its failed tiles out of the beam image with a bilinear over four marched units, which is a resampling the
# anchored rectangle never paid: 0.43% of pixels over 0.02 at 4K against the rectangle's 0.146%. beamScreenResidual removes that
# by marching every failed pixel's own ray - 0.155% - but it keeps nothing, so it pays 3.7 ms parked and 4.4-6.8 ms moving.
#
# No rect arm here. In gameplay.py each oct arm followed a rect arm, so it started on a freshly reset sphere, and look / flick /
# sprint read 4.2-4.6 ms of "resolve" that an all-oct run does not show (flick oct 0.66 ms). The first arm of every motion is a
# repeat of the last, so arm order can be read off the drift.
#
# Sprint's ABSOLUTE error depends on how many frames ran before it, so compare arms within a run only. The same binary scored the
# screen arm 0.046% over 0.02 with this list's first four arms and 0.153% with the "oct 0.01" arm added - and every arm of the
# run moved with it (oct 0.02: 0.047% against 0.173%), so the ordering between arms held while the level did not.
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
    beamPrebuild=True,
    beamOct=True,
    beamOctFull=False,
    beamOctScale=1.0,
)

OCT = mk(**COMMON, beamTolerance=0.05, beamScreenResidual=False)
SCREEN = mk(**COMMON, beamTolerance=0.02, beamScreenResidual=True)
# The shipped oct arm and the screen arm differ in beamTolerance as well (0.05 against 0.02), so "oct 0.02" separates what the
# tighter basis test buys from what the screen residual buys.
OCT_TIGHT = mk(**COMMON, beamTolerance=0.02, beamScreenResidual=False)
# An oct arm straight after a screen arm used to read +0.17 ms parked in "march" at any tolerance (4K: 0.172 / 0.174 ms against
# 0.007 after an oct arm). That was the profiler, not the renderer: GpuTimer's resolve had no barrier before its readback copy, so
# every GPU timing was two frames old, and each arm's first two records were the previous arm's last two - 2 x 3.9 ms of screen
# march over 47 frames. Fixed in GpuTimer::resolve. It is what a3d4d427 misread as "t0.05 marches 0.17 ms parked where t0.01
# marches none": those t0.05 arms followed screen arms. The screen arms still run last, which costs nothing.
OCT_T001 = mk(**COMMON, beamTolerance=0.01, beamScreenResidual=False)
TESTS = [
    ("oct warm", OCT_T001),
    ("oct", OCT),
    ("oct 0.02", OCT_TIGHT),
    ("oct 0.01", OCT_T001),
    ("oct again", OCT),
    # The matched comparison: the screen residual at the same tile acceptance as "oct 0.01".
    ("screen 0.01", mk(**COMMON, beamTolerance=0.01, beamScreenResidual=True)),
    ("screen", SCREEN),
]
