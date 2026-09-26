from sea_config import mk

# DIAGNOSTIC for a GPU Trace's register count only (images are wrong: no tile test, no units): runBeamDirtyFused compiled as its
# ray loop alone (HSTR_FUSED_RAYS_ONLY). ngfx17: the whole fused kernel allocates 128 registers (121 live at the peak, inside
# marchBeam's brick lookup) against the separate query's 96. If the ray loop alone still takes ~120-128, the persistent query
# costs that itself and fusion is settled; if it drops a residency tier, the tile/unit half was contaminating it.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_rays_trace.py --steps 0
#          --motions "sprint 20 0" --ngfx 40 41
TESTS = [("fused rays only", mk(beamDirtyFused=True, beamFusedUnits=False, beamFusedRaysOnly=True))]
