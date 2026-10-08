import runpy
from pathlib import Path

# An apron-free, 8-aligned brick layout under the span evaluator (lean_march.COMPACT). Run:
#     HSTR_RES=1920x1080 python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_compact.py --steps 1 --motions "walk 2 0.004"
TESTS = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["COMPACT"]
