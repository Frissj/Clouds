from sea_config import mk

# ab: the walk of a camera density sample, split by where cloudCostProbe stops. 3 stops with the instance resolved, 5 after the
# directory and node descent, 6 after the parent climb and the octant test, 2 before the atlas fetch. So 5 - 3 is the descent and
# 6 - 5 the climb: what a flat page table keyed on (position, desired level) would replace. Steps are pinned so the arms cannot move
# the ray or sample count, and the first arm is the drift anchor.
TESTS = [
    ["base", mk(minStepVoxels=2, maxStepVoxels=2)],
    ["instance", mk(minStepVoxels=2, maxStepVoxels=2, cloudCostProbe=3)],
    ["descent", mk(minStepVoxels=2, maxStepVoxels=2, cloudCostProbe=5)],
    ["climb", mk(minStepVoxels=2, maxStepVoxels=2, cloudCostProbe=6)],
    ["no atlas", mk(minStepVoxels=2, maxStepVoxels=2, cloudCostProbe=2)],
]
