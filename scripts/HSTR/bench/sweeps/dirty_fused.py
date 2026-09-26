from sea_config import mk

# beamDirtyFused: the dirty chain (query, tile test, unit list, coarse rebuild, warp field, and the units there is time for) in one
# persistent dispatch, each piece claimed as soon as what it reads is final. ngfx9 (4K sprint): the query's last 0.29 of 0.92 ms
# was a few warps finishing the longest rays, and 0.38 ms more went to the rebuild / tile / argument passes before the unit march,
# with the GPU mostly idle through both. Same work and same answers, so the errors must not move. Anchor repeated for drift.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/dirty_fused.py --steps 1
#          --motions "walk 2 0.004" "jog 8 0.002" "sprint 20 0"
TESTS = [
    ("separate", mk()),
    ("fused", mk(beamDirtyFused=True)),
    ("separate again", mk()),
]
