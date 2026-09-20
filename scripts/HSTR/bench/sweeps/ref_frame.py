from sea_config import mk

# The rotation-invariant beam basis frame against the screen-space build it replaces.
#
# The basis is a function of world direction, so turning the camera invalidates nothing - it only moves where each basis ray lands.
# Keying the basis to the screen hides that: the direction a pixel needs after a turn falls between the previous lattice points, so
# a carried value is resampled through the previous basis. carry_ceiling.py priced that exactly (sea, 4K, yaw): carrying 15 of every
# 16 points reaches 4.48 ms but takes the error from 0.144% to 3.907% over 0.02 and p99.9 from 2.56e-2 to 1.45e-1, and the shipped
# checked carry is both SLOWER than no reuse (11.31 against 9.93 ms) and six times worse.
#
# In the reference frame the previous build marched the very ray a basis point needs, and its value sits at the very same index, so
# a carry is a read - no reprojection, no resampling, no interpolation. What a turn must still march is only what turns ON screen,
# which the build stamp in the lattice identifies. The per-pixel fallback keeps marching either way, so these arms isolate the basis.
COMMON = dict(
    beamTileSize=4,
    beamLevels=1,
    beamSegments=1,
    beamTemporal=False,
    beamSparse=False,
    beamSparseCut=False,
    beamSparseMinLevel=2,
    beamQueue=False,
    beamGridDispatch=False,
    beamRefreshBlock=4,
    beamCarryTolerance=0.0,
    beamDepthTolerance=0.05,
    beamRefreshDebug=0,
    beamRefMargin=0.15,
)

TESTS = [
    ("screen no reuse", mk(**COMMON, beamRefresh=0, beamRefFrame=False)),
    ("screen refresh 4", mk(**COMMON, beamRefresh=4, beamRefFrame=False)),
    ("ref refresh 4", mk(**COMMON, beamRefresh=4, beamRefFrame=True)),
    ("ref refresh 16", mk(**COMMON, beamRefresh=16, beamRefFrame=True)),
]
