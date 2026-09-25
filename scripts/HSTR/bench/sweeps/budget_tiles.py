import runpy
from pathlib import Path

# The dirty passes' shape at the best in-gate frame (2x steps + tstep + beamOctScale 0.5). budgetnsys1 (sprint, Nsight): query
# 0.80 ms at 23 of 48 warps in flight and 22% SM issue, units 1.05 ms at 35 warps - both under half occupancy. The tile size and
# the dirty-ray slicing were tuned at beamOctScale 1.0 with ~4x the rays; at 0.5 the query / unit balance and the parallelism
# are different. beamTileSize rebalances query against unit work; beamDirtySegments splits a dirty ray across threads.
# beamTileSize changes the image's rounding, so it may reallocate: 4 first, the other sizes after, the anchor last.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_tiles.py --steps 1 --motions "sprint 20 0"
FAST = runpy.run_path(str(Path(__file__).with_name("budget_jitter.py")))["arm"](2)

TESTS = [
    ("t4", FAST),
    ("t2", dict(FAST, beamTileSize=2)),
    ("t4 again", FAST),
]
# Resizes on purpose (tile 2 and scale 1 reallocate the beam resources): with the free-and-wait on a beam image size change
# (setupBeam), the anchors must hold.

# MEASURED (budgettiles1, sprint): t4 s1 2.93 ms (query 0.79, units 1.00) 0.283%; beamDirtySegments 2 / 4: query 11.47 / 9.82 ms
# - the sliced dirty march is ~13x slower than one thread per ray at this ray count, not an occupancy win; t8 2.39 ms at 3.409%
# over 0.02 (fails the gate). The tile 2 arms and the anchor (3.61 ms) ran after reallocations and were slowed by them.
