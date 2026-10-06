import runpy
from pathlib import Path

# Batched-load span evaluator against the plain one (lean_march.GATHER). Run:
#     python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_gather.py --steps 1 --motions "walk 2 0.004"
TESTS = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["GATHER"]
