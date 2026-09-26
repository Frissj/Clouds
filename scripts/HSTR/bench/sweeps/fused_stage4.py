from sea_config import mk

# DIAGNOSTIC: stages 3 and 4 give up on some frames with every block reserved and tile items claimed but never finished
# (fstage3: 928 and 44). The item-wait diagnostics (kFusedDiagCount 17-22) say what the stuck claims wait on.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_stage4.py --steps 1 --motions "walk 2 0.004"
TESTS = [
    ("stage3 items", mk(beamDirtyFused=True, beamFusedStage=3)),
    ("stage4 tiles", mk(beamDirtyFused=True, beamFusedStage=4)),
]
