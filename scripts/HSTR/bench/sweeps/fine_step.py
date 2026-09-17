from sea_config import mk

# Farside marches 5 sampled voxels per step: its bricks are finer than the cloudFineMinVoxels floor (0.05) on the step. Either
# sample the level the step can resolve (cloudStepFootprint) or lower the floor.
TESTS = [
    ["step 2", mk(cloudStepFootprint=0.0, cloudFineMinVoxels=0.05)],
    ["footprint 0.5", mk(cloudStepFootprint=0.5, cloudFineMinVoxels=0.05)],
    ["footprint 1", mk(cloudStepFootprint=1.0, cloudFineMinVoxels=0.05)],
    ["floor .025", mk(cloudStepFootprint=0.0, cloudFineMinVoxels=0.025)],
    ["floor .01", mk(cloudStepFootprint=0.0, cloudFineMinVoxels=0.01)],
    ["step 1", mk(cloudStepFootprint=0.0, cloudFineMinVoxels=0.05, minStepVoxels=1, maxStepVoxels=1)],
]
