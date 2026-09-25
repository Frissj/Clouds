import runpy
from pathlib import Path

# The step length against the p99 gate (share of pixels over 0.02 below 1%; the default frame sits at 0.137% walk / 0.028% sprint):
# how much of the dirty march the headroom buys with the deterministic midpoint march alone, before a stochastic or reconstructed
# residual is built. minStepVoxels / maxStepVoxels raised together (fine bricks step minStepVoxels * fineVoxels, clamped to
# maxStepVoxels, so both must move or the step stays put), alone and with the transmittance-scaled steps (HSTR_SHIP bit 32768).
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_step.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
BASE = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["BASE"]
MASK = BASE["beamShipMask"]


def step(k, tstep=False):
    return dict(BASE, minStepVoxels=2.0 * k, maxStepVoxels=float(k), beamShipMask=MASK | (32768 if tstep else 0))


TESTS = [
    ("ship", BASE),
    ("step 1.5x", step(1.5)),
    ("step 2x", step(2)),
    ("step 3x", step(3)),
    ("step 2x tstep", step(2, True)),
    ("ship again", BASE),
]
