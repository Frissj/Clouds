import runpy
from pathlib import Path

# The lean beam march (HSTR_SHIP bit 1024, marchBeamLean) against the old shipping march (mask 509): the same samples and values
# with less state live across a step, because the Nsight capture of the shipping walk frame showed the dirty passes
# occupancy-bound (22% / 32% warps in flight, 68% / 63% of warp slots unallocated on active SMs, DRAM read 31% / 22%). One
# controlled difference; the old march is repeated last as the drift anchor. The error columns must match.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_march.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
# MEASURED (lean1, 4K): walk 7.45 / 7.42 -> 6.63 ms, sprint 9.18 / 9.18 -> 8.18 ms, errors identical. Now the default (1533).
BASE = runpy.run_path(str(Path(__file__).with_name("march_cost_split.py")))["BASE"]
LEAN = dict(BASE, beamShipMask=509 | 1024)
OLD = dict(BASE, beamShipMask=509)
TESTS = [
    ("ship", OLD),
    ("lean", LEAN),
    ("ship again", OLD),
]

# The lean march's light terms switched off one at a time (march_cost_split's arms on the lean march). Images wrong on purpose
# except "lean", and so is the dirty set: the arms march 1.9-3.5% of pixels, so read cost per marched ray, not raw ms.
# MEASURED (leansplit1, walk, dirty query + units ms / marched): lean 5.50 / 3.5%, no sun 3.88 / 3.2%, no cache 4.42 / 2.8%,
# density only 2.56 / 1.9% - per marched ray the density chain is most of it, the sun about a quarter, the cache little.
SPLIT = [
    ("lean", LEAN),
    ("lean no sun", dict(LEAN, hstComponents=1 | 4 | 8)),
    ("lean no cache", dict(LEAN, hstComponents=1 | 2)),
    ("lean density only", dict(LEAN, hstComponents=1)),
    ("lean again", LEAN),
]

# The same split on the current default march (BASE carries sea_config's mask), for DRAM traffic per marched ray under
# run_sea.py --nsys --nsys-set ad10x-gfxt: which lookups the frame's VRAM reads come from.
SPLIT_SHIP = [
    ("all", BASE),
    ("no sun", dict(BASE, hstComponents=1 | 4 | 8)),
    ("no cache", dict(BASE, hstComponents=1 | 2)),
    ("density only", dict(BASE, hstComponents=1)),
]

# The resolved level pages (HSTR_SHIP bit 2048, cloudAssetBrickAtFlat): one word per sample instead of the page, the octant test
# and the parent climb; with it, the per-frame resolved sun slots (resolveCloudSunSlots). Exact: the error columns must match.
# MEASURED (leanflat1, pages alone): walk 6.56 / 6.52 -> 6.63 ms, no gain - the climb was not what the march waits on.
# MEASURED (leanflat2, pages + sun slots): walk 6.61 / 6.62 -> 6.06 ms (query 2.60 -> 2.24, units 2.86 -> 2.45), sprint 8.11 /
# 8.22 -> 7.55 ms, errors identical. The sun's key search (16 words a level, up to 15 ancestors) was. Now the default (3581).
FLAT_MASK = 509 | 1024 | 2048
FLAT = [
    ("lean", LEAN),
    ("lean flat", dict(LEAN, beamShipMask=FLAT_MASK)),
    ("lean again", LEAN),
]

# Zero-majorant steps (40% of the lean march's steps, leancount2) crossing whole zero 4-voxel majorant blocks (HSTR_SHIP bit
# 8192, majorantZeroExitFine) instead of one voxel each where the 16-voxel block is not all zero. Exact in what it skips; the later
# samples' phase moves, as with the 16-voxel skip, so the errors are compared, not required identical.
# MEASURED (leanzero1): walk 6.04 / 6.06 -> 5.97 ms, errors identical (sprint the same). Air steps are cheap; kept (11773).
ZERO = [
    ("flat", dict(BASE, beamShipMask=FLAT_MASK)),
    ("flat zero4", dict(BASE, beamShipMask=FLAT_MASK | 8192)),
    ("flat again", dict(BASE, beamShipMask=FLAT_MASK)),
]

# Transmittance-scaled steps (HSTR_SHIP bit 16384: up to 2x, 32768: up to 4x, at 1 / sqrt(T)): fewer lit samples where the pixel
# can no longer see much. An approximation - judged against the path-traced reference.
SHIP_MASK = FLAT_MASK | 8192
TSTEP = [
    ("ship", dict(BASE, beamShipMask=SHIP_MASK)),
    ("tstep 2x", dict(BASE, beamShipMask=SHIP_MASK | 16384)),
    ("tstep 4x", dict(BASE, beamShipMask=SHIP_MASK | 32768)),
    ("ship again", dict(BASE, beamShipMask=SHIP_MASK)),
]

# MEASURED (leantstep1, 4K): walk 5.99 / 5.99 -> 5.44 (2x) / 5.38 (4x) ms for 0.137% -> 0.146% / 0.143% over 0.02 (p99.9 the same,
# max 0.171 -> 0.135); sprint 7.44 / 7.48 -> 6.83 / 6.78 ms for 0.028% -> 0.028% / 0.029%. Opt-in: it trades quality.

# MEASURED and REMOVED (leanloads1, errors identical): exact load cuts that cost registers - the proxy only where it answers
# (bits 65536) 6.04 -> 6.16 ms walk, the sea tile instance cached along the ray (131072) 6.04 -> 6.53, both 6.62 (anchor 6.07);
# sprint 7.52 -> 7.68 / 8.12 / 8.22 (anchor 7.58). See leanExtinction.

# The cache held over lightingStride contributing samples (and never across a cache cell): an approximation of the cache, which
# is smooth at cell scale while the samples are a fraction of a cell apart. Judged against the path-traced reference.
# MEASURED and REMOVED (leanstride1, walk): 6.39 / 6.37 / 6.38 / 6.64 ms at stride 1 / 2 / 4 / 8 for 0.137% / 0.153% / 1.048% /
# 4.985% over 0.02. No time to buy; marchBeamLean evaluates the cache at every contributing sample again (the arms no longer vary).
STRIDE = [
    ("stride 1", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=1)),
    ("stride 2", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=2)),
    ("stride 4", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=4)),
    ("stride 8", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=8)),
    ("stride 1 again", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=1)),
]
