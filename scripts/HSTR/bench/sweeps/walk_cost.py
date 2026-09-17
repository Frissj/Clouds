from sea_config import mk

# ab: what a camera density sample costs now, split by cloudCostProbe (1 stops at the proxy, 2 stops before the atlas fetch, 3
# stops after cloudInstanceAt). The step schedule is pinned so the probes cannot move the ray or sample count, and the first arm is
# the drift anchor. The shipped split was ~6.8 ms of walk in a 9.7 ms sea frame, with the atlas at nothing; this re-measures it
# against the GPU sun scheduling and the streaming since.
TESTS = [
    ["base", mk(minStepVoxels=2, maxStepVoxels=2)],
    ["proxy only", mk(minStepVoxels=2, maxStepVoxels=2, cloudCostProbe=1)],
    ["no atlas", mk(minStepVoxels=2, maxStepVoxels=2, cloudCostProbe=2)],
    ["instance only", mk(minStepVoxels=2, maxStepVoxels=2, cloudCostProbe=3)],
]
