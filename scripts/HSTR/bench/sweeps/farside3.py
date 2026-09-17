from sea_config import mk

# ab --views farside --components 1,2,4,8: which light component carries the 2-voxel farside bias. The tests leave hstComponents to
# the configuration.


def t(**overrides):
    props = mk(**overrides)
    props.pop("hstComponents")
    return props


TESTS = [
    ["step 2", t()],
    ["step 1", t(minStepVoxels=1, maxStepVoxels=1)],
]
