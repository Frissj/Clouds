import runpy
from pathlib import Path

# The parallax warp in the resolve (beamWarp, beamWarpPosition): a held block read where its content was seen from its capture
# camera, by the query depths, instead of along the current camera's direction. Whether reuse with a warp holds the gate where
# reuse without one does not: budgetparallax2 (in-flight scoring), walk beamGuardParallax 8 1.99 ms at 1.987% over 0.02 against
# 2.43 ms at 0.762% at 1. Sprint expires every block at any budget, so it only checks the warp changes nothing there yet.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_warp.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
FAST = runpy.run_path(str(Path(__file__).with_name("budget_jitter.py")))["arm"](2)

TESTS = [
    ("par 1", FAST),
    ("par 1 warp", dict(FAST, beamWarp=1)),
    ("par 8", dict(FAST, beamGuardParallax=8.0)),
    ("par 8 warp", dict(FAST, beamGuardParallax=8.0, beamWarp=1)),
    ("par 1 again", FAST),
]
# MEASURED (budgetwarp1): the warp arms bit-identical to the unwarped ones (walk par 8 1.439% both), and the resolve 0.45 ->
# 0.54 ms in every arm with the warp compiled in. The diagnostic arms find which early-out held it off.
DIAG = [
    ("par 8 moved", dict(FAST, beamGuardParallax=8.0, beamWarp=2)),
    ("par 8 unverified", dict(FAST, beamGuardParallax=8.0, beamWarp=3)),
    ("par 8 no depth", dict(FAST, beamGuardParallax=8.0, beamWarp=4)),
]
# MEASURED (budgetwarpdiag1, walk par 8, share of pixels painted): moved 4.648% (3.2% over the 1.439% baseline), unverified
# 1.439% (none), no depth 91.850% - sky pixels never warped, so a cloud sliding over sky was never followed. Warp now falls back
# to the nearest depth of the 3 x 3 guard blocks around.
# MEASURED and REMOVED (budgetwarp2, walk): par 1 0.760% | par 1 warp 0.760% | par 8 0.960% | par 8 warp 0.960%, every statistic
# identical, and the warp compiled into the resolve cost it 0.45 -> 0.74 ms in every arm, warp on or off. The warp does move
# lookups (3.2% of pixels by over half a texel) but no pixel across the gate: the extra error of a held block is not its
# parallax. par 8's error also varies with the arm before it (1.987 / 1.439 / 0.960% in budgetparallax2 / budgetwarp1 /
# budgetwarp2) while repeated arms reproduce - state carried between arms (beamReset clears mpBeamPixels[0] only, and not the
# tile level map). beamWarp and beamWarpPosition are gone; these arms no longer vary.
