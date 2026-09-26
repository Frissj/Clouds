from sea_config import mk

# beamFusedOverlap: the separate dirty query (its own 96 registers) counts finished blocks into the fused queue, and
# runBeamDirtyFused compiled without its ray loop tests their tiles from a dispatch right behind it, no barrier between - the tile
# test fills the query's tail without the query paying fusion's residency (ngfx18: one program, 128 registers). Units stay in the
# units pass. Same answers expected: errors, tiles tested and units listed must match the separate arms.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_overlap.py --steps 1
#          --motions "walk 2 0.004" "jog 8 0.002" "sprint 20 0"
TESTS = [
    ("separate", mk()),
    ("fused no units", mk(beamDirtyFused=True, beamFusedUnits=False)),
    ("overlap", mk(beamDirtyFused=True, beamFusedOverlap=True)),
    ("separate again", mk()),
]
