import runpy
from pathlib import Path

# The guard's parallax budget (beamGuardParallax, texels; capped at one guard block, 8 texels) at the best in-gate frame (2x steps
# + tstep + beamOctScale 0.5). budgetaccept1: every one of the 30,027 on-screen guard blocks is dirty every frame at walk AND
# sprint - at 1 texel a 2-unit move invalidates every block nearer than ~4,500 units - so the persistent image gives no reuse
# while moving and the query pass (0.7-0.8 ms) re-queries the whole screen. Only ever measured at 1. A larger budget holds a
# block across several frames of motion at up to that many texels of parallax error.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_parallax.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
FAST = runpy.run_path(str(Path(__file__).with_name("budget_jitter.py")))["arm"](2)

TESTS = [
    ("par 1", FAST),
    ("par 2", dict(FAST, beamGuardParallax=2.0)),
    ("par 4", dict(FAST, beamGuardParallax=4.0)),
    ("par 8", dict(FAST, beamGuardParallax=8.0)),
    ("par 1 again", FAST),
]
