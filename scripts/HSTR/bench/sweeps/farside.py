from sea_config import REFERENCE, mk

# ab --views farside: why every beam configuration reads ~51% over 0.02 there. The reference itself checks the harness; the others
# move one beam setting at a time to the reference's value.
TESTS = [
    ["beam", mk()],
    ["reference", dict(REFERENCE)],
    ["minT .001", mk(cloudMinTransmittance=0.001)],
    ["thin 0", mk(cloudThinDepth=0.0)],
    ["step 1", mk(minStepVoxels=1, maxStepVoxels=1)],
    ["all three", mk(cloudMinTransmittance=0.001, cloudThinDepth=0.0, minStepVoxels=1, maxStepVoxels=1)],
]
