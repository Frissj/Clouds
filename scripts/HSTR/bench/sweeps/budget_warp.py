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
    ("par 1", dict(FAST, beamWarp=False)),
    ("par 1 warp", dict(FAST, beamWarp=True)),
    ("par 4 warp", dict(FAST, beamGuardParallax=4.0, beamWarp=True)),
    ("par 8", dict(FAST, beamGuardParallax=8.0, beamWarp=False)),
    ("par 8 warp", dict(FAST, beamGuardParallax=8.0, beamWarp=True)),
    ("par 1 again", dict(FAST, beamWarp=False)),
]
# Every arm sets beamWarp explicitly (and sea_motion.py resets keys an arm leaves out): see the misreading below. The warp is a
# define on the two resolve passes (HSTR_BEAM_WARP), so the unwarped arms pay nothing for it.
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
# MEASURED (budgetwarp2, walk): par 1 0.760% | par 1 warp 0.760% | par 8 0.960% | par 8 warp 0.960% | par 1 again 0.760%.
# MISREAD at the time as "the warp changes nothing": the harness did not reset a key an arm leaves out, so beamWarp = 1 from
# "par 1 warp" stayed on through "par 8" and "par 1 again" - every arm after the first was warped. Against the unwarped par 8
# (1.985%, budgetreset1; 1.987%, budgetparallax2) the warp took walk par 8 to 1.439% (own depth only, budgetwarp1) and 0.960%
# (with the sky fallback, budgetwarp2): under the gate, at 2.31 ms against par 1's 2.49 - with the resolve 0.45 -> 0.74 ms from
# the warp compiled in, which is the cost to remove. The warp was taken out on that misreading (44ff6bcb's message says so too);
# sea_motion.py now resets every key an arm leaves out (PRIOR).
