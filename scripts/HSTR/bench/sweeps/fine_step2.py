from sea_config import mk

# Farside: the majorant rule lets thin fine-brick regions step the full maxStepVoxels. Bound the step's optical depth harder, or cap
# the step at one voxel while keeping the 2-voxel minimum's scale for fine bricks.
TESTS = [
    ["step 2", mk(stepOpticalDepth=0.5)],
    ["od .25", mk(stepOpticalDepth=0.25)],
    ["od .125", mk(stepOpticalDepth=0.125)],
    ["max 1", mk(stepOpticalDepth=0.5, maxStepVoxels=1)],
    ["min 1 max 2", mk(stepOpticalDepth=0.5, minStepVoxels=1)],
    ["step 1", mk(stepOpticalDepth=0.5, minStepVoxels=1, maxStepVoxels=1)],
]
