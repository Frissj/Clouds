from sea_config import mk

# beamDirtyFused alone, for a GPU Trace (run_sea.py --ngfx). fused15 (4K): without its own unit march the fused chain is still
# 0.1-0.3 ms behind the separate passes (sprint 2.64 against 2.33); the trace gives its registers, occupancy and hot spots.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_trace.py --steps 0
#          --motions "sprint 20 0" --ngfx 40 41
TESTS = [("fused", mk(beamDirtyFused=True, beamFusedUnits=False))]
