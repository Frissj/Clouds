from sea_config import mk

# DIAGNOSTIC: fused5-9 removed the device on the first fused frame, with every wait bounded and aborting the kernel. Each arm
# stops runBeamDirtyFused at a stage (beamFusedStage); the first arm that crashes names the stage at fault. Arms run in one process.
# fstage: stages 1-4 ran (4 matched the separate path's errors), 0 crashed. Here 5 claims units without marching them (the units
# pass then starts past them) and 6 marches them in the kernel with the units pass marching every unit again.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_stage.py --steps 1 --motions "walk 2 0.004"
TESTS = [
    ("stage4 tiles", mk(beamDirtyFused=True, beamFusedStage=4)),
    ("stage5 claims", mk(beamDirtyFused=True, beamFusedStage=5)),
    ("stage6 march", mk(beamDirtyFused=True, beamFusedStage=6)),
    ("stage0 all", mk(beamDirtyFused=True, beamFusedStage=0)),
]
