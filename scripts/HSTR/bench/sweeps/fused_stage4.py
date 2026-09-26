from sea_config import mk

# DIAGNOSTIC: stage 4 (everything but the kernel's own units) matches the separate path but some frames' idle loops give up with
# every published block's items done and blocks still unpublished. The widened diagnostics name the counter that is stuck.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_stage4.py --steps 1 --motions "walk 2 0.004"
TESTS = [("stage4 tiles", mk(beamDirtyFused=True, beamFusedStage=4))]
