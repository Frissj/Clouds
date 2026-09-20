from sea_config import mk

# What does it COST to address the lattice through a page table?
#
# The design this is for: stop anchoring the beam image to a camera pose at all, and store marched directions in a sparse, paged,
# world-fixed octahedral atlas with an LRU budget. That removes re-anchoring - which is what makes a fast turn 12.5 ms, because a
# re-anchor re-marches 100% of the image to cover the 4.4% of directions a 0.05 rad/frame turn actually exposes - without
# allocating the sphere: memory becomes a pool size rather than a solid angle. Octahedral because its direction transform has no
# transcendentals at all (abs, divide, select) and no poles, so looking up is an ordinary case rather than a fallback.
#
# Every lattice read in that design resolves through a page table first, and the resolve walks that path thirteen times per pixel.
# Adding a BRANCH there once cost 1.81 -> 2.81 ms, so the indirection is the thing most likely to kill the design, and it can be
# priced before any of it is built: beamPageIndirect routes the existing reads through a table filled with the IDENTITY. Same
# texture, same layout, same texel, same results - only the dependent buffer read and the page arithmetic are added.
#
# So read the times, and check that quality is IDENTICAL: if it is not, the identity mapping is wrong and the timings mean nothing.
# beamPageShift is the page edge in lattice points as a shift: 4 is 16 x 16 points (64 x 64 pixels at beamTileSize 4), 6 is 64 x 64.
#
# beamPageIndirect 1 resolves a page on EVERY lattice read; 2 resolves one per resolve pixel and then reads directly, which is what
# a page holding its own one-texel apron allows - a tile's 2 x 2 neighbourhood is then guaranteed in-page, so thirteen reads share
# one lookup. Measured, 1 costs a flat +0.16 ms parked and +0.17 under yaw, identical at both page sizes, so it is the dependent
# read itself rather than locality. The pair brackets what the real atlas would pay.
COMMON = dict(
    beamTileSize=4,
    beamLevels=1,
    beamSegments=1,
    beamTemporal=False,
    beamSparse=False,
    beamSparseCut=False,
    beamQueue=False,
    beamGridDispatch=False,
    beamRefreshBlock=4,
    beamCarryTolerance=0.0,
    beamDepthTolerance=0.05,
    beamRefreshDebug=0,
    beamRefFrame=True,
    beamRefMargin=0.15,
    beamRefresh=256,
)

TESTS = [
    ("direct", mk(**COMMON, beamPageIndirect=0)),
    ("per read", mk(**COMMON, beamPageIndirect=1, beamPageShift=4)),
    ("hoisted", mk(**COMMON, beamPageIndirect=2, beamPageShift=4)),
    ("direct again", mk(**COMMON, beamPageIndirect=0)),
    ("hoisted again", mk(**COMMON, beamPageIndirect=2, beamPageShift=4)),
]
