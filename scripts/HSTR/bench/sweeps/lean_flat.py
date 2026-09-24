import runpy
from pathlib import Path

# Resolved level pages against the lean march (lean_march.FLAT). Run: python scripts/HSTR/bench/run_sea.py motion TAG
#     scripts/HSTR/bench/sweeps/lean_flat.py --steps 1 --motions "walk 2 0.004" "sprint 20 0"
TESTS = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["FLAT"]
