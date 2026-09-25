import runpy
from pathlib import Path

# Tile acceptance (beamTolerance, default 0.01) and early termination (cloudMinTransmittance, default 0.02) at the best in-gate
# frame (2x steps + tstep + beamOctScale 0.5): the units pass (1.00 ms sprint, budgetnsys1) marches the texels of tiles whose
# basis failed; a looser tolerance fails fewer. Termination cuts the deep tail of both dirty passes. Both spend the gate's
# headroom (walk 0.760%, sprint 0.283% of pixels over 0.02; the gate is 1%), so walk is scored too. All at one image size.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_accept.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
FAST = runpy.run_path(str(Path(__file__).with_name("budget_jitter.py")))["arm"](2)

TESTS = [
    ("base", FAST),
    ("tol .02", dict(FAST, beamTolerance=0.02)),
    ("tol .04", dict(FAST, beamTolerance=0.04)),
    ("minT .05", dict(FAST, cloudMinTransmittance=0.05)),
    ("tol .02 minT .05", dict(FAST, beamTolerance=0.02, cloudMinTransmittance=0.05)),
    ("base again", FAST),
]
