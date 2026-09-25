import runpy
from pathlib import Path

# The direct-span ceiling test (lean_march.SPAN). Run:
#     python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_span.py --steps 1
#     --motions "walk 2 0.004" "sprint 20 0"
TESTS = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["SPAN"]
