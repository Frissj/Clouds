from sea_config import mk

# ab --views farside: which change to the 2-voxel march removes the farside bias (51% over 0.02 at 2 voxels, 0.33% at 1).
OFF = dict(cloudSourceLinear=False, cloudQuadrature=1, cloudLongitudinalOracle=0, cloudOracleCentroid=False, cloudTrapezoid=False)


def t(**overrides):
    return mk(**dict(OFF, **overrides))


TESTS = [
    ["beam", t()],
    ["step 1-2", t(minStepVoxels=1, maxStepVoxels=2)],
    ["step 1.5", t(minStepVoxels=1.5, maxStepVoxels=1.5)],
    ["sourceLinear", t(cloudSourceLinear=True)],
    ["trapezoid", t(cloudTrapezoid=True)],
    ["quadrature 2", t(cloudQuadrature=2)],
    ["oracle 4", t(cloudLongitudinalOracle=4)],
    ["centroid 4", t(cloudLongitudinalOracle=4, cloudOracleCentroid=True)],
    ["step 1", t(minStepVoxels=1, maxStepVoxels=1)],
]
