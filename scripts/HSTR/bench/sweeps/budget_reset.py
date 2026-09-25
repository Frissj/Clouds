import runpy
from pathlib import Path

# Does an arm carry state into the next one (beamReset incomplete), or does the result vary between processes? walk
# beamGuardParallax 8 scored 1.987 / 1.439 / 0.960% in three runs (budgetparallax2 / budgetwarp1 / budgetwarp2), two of them after
# the same preceding configuration, while repeated arms within a run reproduced. Here par 8 runs first, after par 1, and twice
# more in a row: if every par 8 matches, the reset is complete and the variation is between processes.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_reset.py --steps 1 --motions "walk 2 0.004"
FAST = runpy.run_path(str(Path(__file__).with_name("budget_jitter.py")))["arm"](2)
PAR8 = dict(FAST, beamGuardParallax=8.0)

TESTS = [
    ("par 8", PAR8),
    ("par 1", FAST),
    ("par 8 after 1", PAR8),
    ("par 8 after 8", PAR8),
]
# MEASURED (budgetreset1, walk): par 8 1.985% (11,462 dirty blocks, 71,088 units), par 1 0.760%, par 8 after 1 and after 8
# 1.985% with the same counts - beamReset is complete. The variation was the harness keeping beamWarp on from an earlier arm
# (see budget_warp.py), not state in the renderer, and not timing: the renderer schedules by frame count and the benchmark
# freezes residency and fills sea tiles synchronously.
