import runpy
from pathlib import Path

# The guard's certificate under the warp (beamWarpCertify, beamCellRadius): a block holds while camera motion times the spread of
# inverse CLOUD depth around it (1 / near - 1 / far, sky excluded) stays under beamGuardParallax texels, at every level of the
# pyramid - the parallax the warp (beamWarp) cannot take out - instead of while motion over its nearest depth does. A listed block
# restarts its depth range. The old certificate expires every one of sprint's 30,027 blocks every frame at any budget
# (budgetparallax2); walk parallax 8 + warp 2.30 ms at 0.960% (budgetwarp3). Every arm sets every key.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_certify.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
FAST = runpy.run_path(str(Path(__file__).with_name("budget_jitter.py")))["arm"](2)


def arm(parallax, warp, certify):
    return dict(FAST, beamGuardParallax=parallax, beamWarp=warp, beamWarpCertify=certify)


TESTS = [
    ("par 1", arm(1.0, False, False)),
    ("par 8 warp", arm(8.0, True, False)),
    ("diff .25", arm(0.25, True, True)),
    ("diff .5", arm(0.5, True, True)),
    ("diff 1", arm(1.0, True, True)),
    ("diff 2", arm(2.0, True, True)),
    ("par 1 again", arm(1.0, False, False)),
]
# MEASURED and REMOVED (budgetcertify1, 4K; ms, % of pixels over 0.02, dirty blocks, units marched):
#   walk   par 1 2.50 0.759% 15,313 146k | par 8 warp 2.42 0.960% 11,462 71k | diff .25 2.51 1.578% 8,275 102k |
#          diff .5 2.32 2.225% 7,348 79k | diff 1 2.16 2.999% 6,765 55k | diff 2 2.02 3.874% 5,999 37k | again 2.50 0.759%
#   sprint par 1 2.96 0.283% 30,027 118k | par 8 warp 3.31 0.283% 30,027 | diff .25 3.21 0.362% 21,623 117k |
#          diff .5 3.21 0.594% 21,480 | diff 1 3.19 0.840% 21,282 | diff 2 3.20 1.066% 21,082 102k | again 2.94 0.283%
# Walk: every differential budget fails the gate, even .25 texel, where the old certificate + warp holds at parallax 8. It
# certifies blocks the warp cannot correct - the depths are one opacity-weighted distance per ray, blind to a cloud's thickness
# along it. Sprint: 28% of blocks held for the first time, and no time saved (query 0.82 -> 0.79, units unchanged): the blocks
# it can hold are cheap, next to sky; near dense cloud expires at 20 units a frame under any certificate. beamWarpCertify, the
# far-depth store and pyramid and the per-block restart are gone; the "diff" arms no longer vary.
