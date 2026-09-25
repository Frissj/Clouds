import runpy
from pathlib import Path

# Nsight capture of the best in-gate frame (budgetoct2 / budgetjitter2: 2x steps + transmittance-scaled steps + beamOctScale 0.5,
# sprint 2.90 ms at 0.283% over 0.02): where the remaining time goes. One arm, so the octahedral image is allocated once.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_nsight.py --steps 0 --motions "sprint 20 0" --nsys
TESTS = [("s2t oct .5", runpy.run_path(str(Path(__file__).with_name("budget_jitter.py")))["arm"](2))]
