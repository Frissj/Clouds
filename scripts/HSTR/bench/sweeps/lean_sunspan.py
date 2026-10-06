import os
import runpy
from pathlib import Path

# Span-level sun: interpolation quality (lean_march.SUNSPAN) or, with HSTR_SPAN_SWEEP=CROSS, the brick crossings' lit samples. Run:
#     HSTR_RES=1920x1080 python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_sunspan.py --steps 1 --motions "walk 2 0.004"
TESTS = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))[os.environ.get("HSTR_SPAN_SWEEP", "SUNSPAN")]
