from sea_config import mk

# beamPolicy: spend the error a build's held blocks do not carry on longer steps and a looser tile test, continuously in the held
# share h this build classified (decideBeamPolicy, on the GPU, so this frame's): s = saturate((high - h) / (high - low)), steps
# x lerp(1, step, s), tolerance lerp(0.01, tolerance, s). Walk holds 62% (s = 0: unchanged), sprint 0-1% (s = 1).
# sprintfront2 at fixed settings: sprint k2 tol .01 2.84 ms 0.282% -> k2.5 tol .05 2.25 ms 0.716% | k2.75 tol .03 2.24 0.822%.
# Jog (8 units a frame) is the case between, where the held share is partial and both errors add: the one to watch.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/policy.py --steps 1
#          --motions "walk 2 0.004" "jog 8 0.002" "sprint 20 0"
POLICY = dict(beamPolicy=True, beamPolicyStep=1.25, beamPolicyTolerance=0.05, beamPolicyHeldLow=0.1, beamPolicyHeldHigh=0.4)

FIRST = [
    ("default", mk(beamPolicy=False)),
    ("policy", mk(**POLICY)),
    ("policy k2.75", mk(**dict(POLICY, beamPolicyStep=1.375))),
    ("default again", mk(beamPolicy=False)),
]
# MEASURED (policy1; ms, over 0.02, p99.9; held share h; the decided step scale / tolerance):
#   walk   default 2.14 0.937% | policy 2.10 0.937% (h 0.61: s 0, unchanged) | k2.75 2.09 0.937% | again 2.11
#   jog    default 2.78 0.575% (h 0.14, so no warp at beamWarpAuto 0.25) | policy 2.44 1.173% (x1.21, tol .044) FAILS |
#          k2.75 2.41 1.528% FAILS | again 2.79 0.575%
#   sprint default 2.90 0.282% | policy 2.33 0.676% 0.043 | k2.75 2.22 0.846% 0.051 | again 2.93
# The held share does not measure how much of the budget is spent: jog's 14% held blocks go unwarped, which already doubles its
# error over sprint's, and the policy's step error adds to it. Next: a ramp that only spends when almost nothing is held, and
# a lower warp threshold so jog's held blocks are warped.
RAMP = dict(POLICY, beamPolicyHeldLow=0.02, beamPolicyHeldHigh=0.15)
SECOND = [
    ("default", mk(beamPolicy=False)),
    ("warp .05", mk(beamPolicy=False, beamWarpAuto=0.05)),
    ("ramp", mk(**RAMP)),
    ("ramp warp .05", mk(**RAMP, beamWarpAuto=0.05)),
    ("default again", mk(beamPolicy=False)),
]
# MEASURED (policy2; ms, over 0.02; h, decided k / tol, warp):
#   walk   default 2.16 0.938% | warp .05 2.24 | ramp 2.15 0.938% | ramp warp .05 2.12 | again 2.10       (h 0.62, s 0)
#   trot   default 2.86 0.723% | warp .05 2.86 | ramp 2.84 0.724% | ramp warp .05 2.85 | again 2.86       (h 0.25, s 0)
#   jog    default 2.79 0.576% | warp .05 2.80 0.575% | ramp 2.69 0.617% (k 1.01, tol .012) | again 2.78  (h 0.14)
#   sprint default 2.90 0.282% | warp .05 2.91 | ramp 2.30 0.676% (k 1.25, tol .05) | again 2.89       (h 0)
# Warping jog's held blocks changes nothing (0.576 -> 0.575%): its error is not parallax. Its scored frame is harder content:
# the settings that give sprint's frame 0.676% gave jog's 1.173% (policy1). So sprint's headroom was one frame's, not fast
# motion's, and a policy spending the gate has to be judged over several frames. Next: the settings forced on (s = 1 whatever
# is held) over three scored frames each of sprint and jog.
FORCE = dict(beamPolicy=True, beamPolicyHeldLow=0.98, beamPolicyHeldHigh=0.99)
THIRD = [
    ("default", mk(beamPolicy=False)),
    ("tol .03", mk(**FORCE, beamPolicyStep=1.0, beamPolicyTolerance=0.03)),
    ("tol .05", mk(**FORCE, beamPolicyStep=1.0, beamPolicyTolerance=0.05)),
    ("k2.25 tol .03", mk(**FORCE, beamPolicyStep=1.125, beamPolicyTolerance=0.03)),
    ("k2.5 tol .05", mk(**FORCE, beamPolicyStep=1.25, beamPolicyTolerance=0.05)),
    ("default again", mk(beamPolicy=False)),
]
# MEASURED (policy3, forced on, three scored frames each; ms, over 0.02 per frame, worst p99.9):
#   jog    default 2.81 0.575/0.560/0.554 | tol .03 2.44 0.786/0.759/0.738 | tol .05 2.37 0.885/0.849/0.838 |
#          k2.25 tol .03 2.32 0.929/0.922/0.889 | k2.5 tol .05 2.17 1.245/1.237/1.166 FAILS 0.072 | again 2.77
#   sprint default 2.90 0.282/0.284/0.276 | tol .03 2.59 0.408 | tol .05 2.52 0.476 | k2.25 tol .03 2.48 0.500 |
#          k2.5 tol .05 2.33 0.688 0.043 | again 2.89
# Longer steps fail on jog's harder content even with nothing else spent; tolerance alone holds with 0.1% to spare when forced
# onto it. So the policy spends tolerance only (step 1), up to 0.05, and only where little is held.
SHIP = dict(beamPolicy=True, beamPolicyStep=1.0, beamPolicyTolerance=0.05, beamPolicyHeldLow=0.05, beamPolicyHeldHigh=0.25)
TESTS = [
    ("default", mk(beamPolicy=False)),
    ("policy", mk(**SHIP)),
    ("default again", mk(beamPolicy=False)),
]
# MEASURED (policy4, two scored frames each; ms, over 0.02 per frame, decided tolerance):
#   walk   default 2.13 0.938/0.858 | policy 2.24* 0.938/0.858 (tol .010) | again 2.14   * every scope +7%, tone mapper too: drift
#   trot   default 2.83 0.724/0.704 | policy 2.82 0.724/0.768 (tol .015) | again 2.85
#   jog    default 2.80 0.576/0.559 | policy 2.66 0.801/0.850 (tol .050) | again 2.77
#   sprint default 2.94 0.282/0.284 | policy 2.52 0.476/0.467 (tol .050) | again 2.92
# SHIPPED: SHIP (sea_config.BEAM, CloudSea.py, HSTRCloud.h).
