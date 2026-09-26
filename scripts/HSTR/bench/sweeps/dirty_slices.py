from sea_config import mk

# beamDirtySegments re-judged in today's regime. ngfx8 (Nsight shader profiler, 4K walk frame 40): the dirty query (0.49 ms, 96
# registers, cap 20 warps) and units (0.49 ms, 80 registers, cap 24) fill their SMs at launch and then only drain - one wave, no
# refill - so the last third of each pass holds a few long rays on an almost idle GPU. Slicing was rejected when the query was
# 6.4 ms of saturated work (HSTRCloudTypes.slang, beamDirtySegments); here it may shorten the tail. It restarts each slice's march,
# so the gate decides. Jog is the tightest motion. Anchor repeated for drift.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/dirty_slices.py --steps 3
#          --motions "walk 2 0.004" "jog 8 0.002"
# MEASURED and REJECTED (slices1; ms, q0 ms, >0.02, worst scored frame):
#   walk default 2.06 0.67 0.886% 0.937% | slices 2 2.12 0.71 0.912% 0.956% | slices 4 2.25 0.84 0.963% 1.009% | again 2.01 0.64
#   jog  default 2.52 0.84 0.829% 0.849% | slices 2 2.64 1.01 0.935% 0.955% | slices 4 2.86 1.26 1.100% 1.146% | again 2.52 0.84
# Slower and worse on both. A slice marches its whole segment even behind an opaque earlier slice (early termination is per
# slice), so the split duplicates exactly the long rays' work instead of spreading it; the restarts cost the gate.
TESTS = [
    ("default", mk()),
    ("slices 2", mk(beamDirtySegments=2)),
    ("slices 4", mk(beamDirtySegments=4)),
    ("default again", mk()),
]
