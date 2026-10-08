import runpy
from pathlib import Path

# The working set of a real brick sample (lean_march.WORKSET). Run:
#     HSTR_RES=1920x1080 python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_workset.py --steps 1 --motions "walk 2 0.004"
TESTS = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["WORKSET"]
