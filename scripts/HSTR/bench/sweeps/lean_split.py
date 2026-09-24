import runpy
from pathlib import Path

# Lighting split on the current default march (lean_march.SPLIT_SHIP), for Nsight DRAM attribution. Run:
#     python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_split.py --steps 0
#     --motions "walk 2 0.004" --nsys --nsys-set ad10x-gfxt
TESTS = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["SPLIT_SHIP"]
