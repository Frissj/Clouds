from sea_config import mk

# beamFusedUnits: ngfx12 (4K sprint) had the fused kernel at 168 registers, 12 warps an SM, 60% of warp slots unallocated - the
# separate query takes 96. Without the in-kernel unit march (HSTR_FUSED_UNITS 0) the units pass marches every unit, still
# overlapped by the pixel resolve, and the kernel holds only the query and tile test.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_units.py --steps 1
#          --motions "walk 2 0.004" "jog 8 0.002" "sprint 20 0"
TESTS = [
    ("separate", mk()),
    ("fused units", mk(beamDirtyFused=True)),
    ("fused no units", mk(beamDirtyFused=True, beamFusedUnits=False)),
    ("separate again", mk()),
]
