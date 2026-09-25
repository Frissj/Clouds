from sea_config import BEAM, mk

# Phase K1: the lit sample's sun chain started by reloading its instance (leanExtinction had it and let it go) and then the
# instance's asset, before the resolved bake and the far field could be addressed. HSTR_SHIP bit 524288 carries the instance's sun
# class, scale and near scale (a new instance field, sunNearScale) in LeanHit instead: two floats more, live within one step.
# strip2 priced the sun chain at +0.16 / +0.25 ms (walk query / units) and +0.25 / +0.23 (sprint) over density alone.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/sun_hit.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
CARRY = BEAM["beamShipMask"] | 524288

FIRST = [
    ("reload", mk()),
    ("carry", mk(beamShipMask=CARRY)),
    ("reload again", mk()),
    ("carry again", mk(beamShipMask=CARRY)),
]
# MEASURED (sunhit1, the instance record grown to 84 bytes): walk reload 2.10 / 2.04, carry 2.13 / 2.07; sprint 2.48 / 2.48,
#   2.50 / 2.51 - the carry slower everywhere. But the 84-byte record broke 16-byte alignment for every density sample's
#   instance load; the carry was not what cost.
# MEASURED (sunhit2, record kept at 80 bytes - scale replaced the occupied flag; ms, query, units):
#   walk   reload 2.118 0.571 0.715 | carry 2.114 0.566 0.715 | reload again 2.064 0.559 0.697 | carry again 2.058 0.558 0.695
#   sprint reload 2.495 0.846 0.733 | carry 2.474 0.840 0.722 | reload again 2.489 0.847 0.732 | carry again 2.487 0.841 0.731
#   Errors identical. The carry is neutral to slightly ahead (every pair), so it needs registers freed to pay properly.
# Next, cumulative on the carry, one change an arm: the policy's step scale out of the march (bit 1048576: it ships at 1, and
# min / max step return to constant-buffer values), then the view direction recomputed per lit sample (bit 2097152).
NOSCALE = CARRY | 1048576
SECOND = [
    ("reload", mk()),
    ("carry", mk(beamShipMask=CARRY)),
    ("carry noscale", mk(beamShipMask=NOSCALE)),
    ("carry noscale view", mk(beamShipMask=NOSCALE | 2097152)),
    ("reload again", mk()),
    ("carry noscale view again", mk(beamShipMask=NOSCALE | 2097152)),
]
# MEASURED (sunhit3; ms, query, units; errors identical in every arm):
#   walk   reload 2.105 0.573 0.699 | carry 2.097 | carry noscale 2.074 0.559 0.678 | + view 2.071 | reload again 2.050 |
#          carry noscale view again 2.012 0.541 0.666
#   sprint reload 2.472 0.838 0.717 | carry 2.474 | carry noscale 2.400 0.800 0.689 | + view 2.409 | reload again 2.465 |
#          carry noscale view again 2.401
# The policy's step scale cost sprint 0.07 ms and walk 0.02-0.03 at scale 1: removed from the march for good (the policy is
# tolerance-only). The view direction recomputed per lit sample: nothing - removed. The carry: still within noise.
# Now on that frame (bit 1048576 is free again, and reused): the carry again, and one per-ray register less - the reciprocal of
# voxelsPerUnit in place of voxelsPerUnit and worldPerVoxel (bit 1048576).
RCP = 1048576
TESTS = [
    ("reload", mk()),
    ("carry", mk(beamShipMask=CARRY)),
    ("reload rcp", mk(beamShipMask=BEAM["beamShipMask"] | RCP)),
    ("carry rcp", mk(beamShipMask=CARRY | RCP)),
    ("reload again", mk()),
    ("carry rcp again", mk(beamShipMask=CARRY | RCP)),
]
