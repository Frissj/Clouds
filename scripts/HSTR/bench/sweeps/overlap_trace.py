from sea_config import mk

# beamOverlapResolve alone, for a GPU Trace (run_sea.py --ngfx): overlap1 measured no change at all, so this shows what still
# orders the pixel pass behind the unit march (the API calls between the two dispatches, and whether their warps ever coexist).
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/overlap_trace.py --steps 0
#          --motions "sprint 20 0" --ngfx 40 41
TESTS = [("overlap", mk(beamOverlapResolve=True))]
