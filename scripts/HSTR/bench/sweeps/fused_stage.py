from sea_config import mk

# DIAGNOSTIC: each arm stops runBeamDirtyFused at a stage (beamFusedStage); the first arm that crashes names the stage at fault.
# Arms run in one process. fstage: stages 1-4 ran (4 matched the separate path's errors), 0 crashed; fstage2 then crashed at
# 5, which leaves claimed units unmarched by design. Now against the written-slot block queue (fused11/12 removed the device):
# 2 rays only, 3 tile items without their tests, 4 everything but the kernel's own units, 0 all of it.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_stage.py --steps 1 --motions "walk 2 0.004"
TESTS = [
    ("stage2 rays", mk(beamDirtyFused=True, beamFusedStage=2)),
    ("stage3 items", mk(beamDirtyFused=True, beamFusedStage=3)),
    ("stage4 tiles", mk(beamDirtyFused=True, beamFusedStage=4)),
    ("stage0 all", mk(beamDirtyFused=True, beamFusedStage=0)),
]
