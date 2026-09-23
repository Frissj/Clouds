import runpy
from pathlib import Path

# Experiment 6, fine-brick reuse: every dirty chord of the shipping frame re-walked at the march's step rule, its samples recorded
# per atlas brick visit and aggregated per (listing tile, brick) - the work a tile workgroup staging each fine brick once in shared
# memory would amortise the stage over. Answers samples per stage (the staging bytes per useful step), tiles per brick (the L2
# reuse between workgroups) and the distribution. Counters are the last scored step's frame; probe time is not a cost.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/brick_share.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
SHARE = runpy.run_path(str(Path(__file__).with_name("push_share.py")))["SHARE"]
TESTS = [("bricks", dict(SHARE, pushDilate=0, pushShareMode=8))]
