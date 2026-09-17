from sea_config import REFERENCE, mk

# truth --views farside: the shipped 2-voxel beam, the 1-voxel beam and the exact per-pixel march against the path tracer.
TESTS = [
    ["exact march", dict(REFERENCE)],
    ["beam step 2", mk()],
    ["beam step 1", mk(minStepVoxels=1, maxStepVoxels=1)],
]
