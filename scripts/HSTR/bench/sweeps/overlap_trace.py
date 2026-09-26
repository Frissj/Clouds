from sea_config import mk

# beamFusedOverlap alone, for a GPU Trace: overlap1 (4K) measured it slower than the separate passes (sprint 3.11 against 2.36 ms)
# with identical answers. Whether the query and tile dispatches overlap at all, and what the tile waves do while they wait.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/overlap_trace.py --steps 0
#          --motions "sprint 20 0" --ngfx 40 41
TESTS = [("overlap", mk(beamDirtyFused=True, beamFusedOverlap=True))]
