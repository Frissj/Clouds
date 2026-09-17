from sea_config import mk

# motion --live, run twice (HSTR_GPU_SUN=1 and 0): the residency's CPU time with sun bakes scheduled on the GPU or the CPU.
TESTS = [
    ["async", mk(cloudCutAsync=True)],
]
