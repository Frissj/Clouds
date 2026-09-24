import runpy
from pathlib import Path

# Transmittance-scaled steps on the lean march (lean_march.TSTEP, opt-in HSTR_SHIP bits 16384 / 32768). Run:
#     python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_tstep.py --steps 1
#     --motions "walk 2 0.004" "sprint 20 0"
TESTS = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["TSTEP"]
