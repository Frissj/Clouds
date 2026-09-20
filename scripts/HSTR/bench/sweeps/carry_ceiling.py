from sea_config import mk

# What a carried basis could be worth under rotation, before building a rotation-invariant basis frame.
#
# beamReprojectQuery already reprojects properly: it takes the point's own stored distance, projects it through beamPrevViewProj and
# reconstructs the previous lattice there. Under pure rotation that reprojection is EXACT - no world-space ray becomes invalid, only
# its screen address moves. What costs quality is that the needed direction does not land on the previous lattice, so the value is
# resampled through the previous basis, and the source comment records the consequence: "a turning camera, which reprojects exactly,
# still tripled the error - resampled through failing tiles, every frame".
#
# These arms price that. "checked" is the shipped carry. "unchecked" disables the three rejections (the failing-tile test via
# beamRefreshDebug 5, the depth agreement via a tolerance nothing can exceed, the carry spread via beamCarryTolerance 0), so every
# point the pattern does not refresh is carried whatever the resampling did. That is the CEILING of any carry scheme on this basis:
# its time is what a rotation-invariant basis would cost, and its error is what the resampling costs. refresh 16 pushes the same
# question further - one point in sixteen marched - so the two together separate "how cheap can carrying get" from "how wrong".
COMMON = dict(
    beamSparse=False,
    beamSparseCut=False,
    beamSparseMinLevel=2,
    beamQueue=False,
    beamGridDispatch=False,
    beamSegments=1,
    beamTemporal=False,
    beamRefreshBlock=4,
    beamCarryTolerance=0.0,
)
UNCHECKED = dict(beamRefreshDebug=5, beamDepthTolerance=1.0e9)

TESTS = [
    ("no reuse", mk(**COMMON, beamRefresh=0, beamRefreshDebug=0, beamDepthTolerance=0.05)),
    ("refresh4 checked", mk(**COMMON, beamRefresh=4, beamRefreshDebug=0, beamDepthTolerance=0.05)),
    ("refresh4 unchecked", mk(**COMMON, beamRefresh=4, **UNCHECKED)),
    ("refresh16 unchecked", mk(**COMMON, beamRefresh=16, **UNCHECKED)),
]
