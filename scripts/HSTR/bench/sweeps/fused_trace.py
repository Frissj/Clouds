from sea_config import mk

# beamDirtyFused alone, for a GPU Trace (run_sea.py --ngfx): fused4 had the fused kernel 4-15x slower than the passes it replaces
# with identical answers. The trace gives its registers, occupancy and warp timeline.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_trace.py --steps 0
#          --motions "sprint 20 0" --ngfx 40 41
TESTS = [("fused", mk(beamDirtyFused=True))]
