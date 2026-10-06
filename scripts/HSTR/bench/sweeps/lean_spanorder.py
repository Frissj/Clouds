import runpy
from pathlib import Path

# Span evaluation order, the resolved loop, the density / sun split and shared transfers (lean_march.SPANORDER). Run:
#     HSTR_RES=1920x1080 python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_spanorder.py --steps 1 --motions "walk 2 0.004"
TESTS = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["SPANORDER"]
