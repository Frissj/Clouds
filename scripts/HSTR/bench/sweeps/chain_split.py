import runpy
from pathlib import Path

# Where the dirty march's time goes, for a direct resolved-brick path: (1) the fine-brick walk's steps split by what answered them
# (pushShareMode 8: air, empty fine brick, proxy with density, proxy at zero, fine brick - pushWalk*), and (2) the dirty passes'
# ms as a sample's lookup chain is cut short (cloudCostProbe: 3 stops once the instance is resolved, 2 after the hierarchy walk
# before the atlas fetch, 1 at the proxy) with the lighting off (hstComponents 1). The probes need the generic kernel, so the
# anchor for them is the same frame with the shipping defines off (beamShip False). The probe images are wrong on purpose: read the
# query and units timings with the marched steps, not the error.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/chain_split.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
# MEASURED (chainsplit1, 4K, one settle; query + units ms, walk / sprint): ship 6.94 / 8.49 (anchor 6.94 / 8.56 at the end),
# generic 9.46 / 11.74, no light 4.86 / 6.48, no atlas 3.97 / 5.40, instance 2.04 / 2.84, proxy ~1.9 / 2.68. On the generic
# kernel: lighting (sun + cache) ~4.6 ms (49%), the atlas fetch ~0.9, the sparse hierarchy walk ~1.9, the instance ~0.1-0.3, the
# step loop with the proxy ~1.9. The cut-short probes change the density and so the early outs: attribution, not exact splits.
# The walk's steps (pushWalk*, walk / sprint): 74.1 / 75.8% an EMPTY fine brick, 24.6 / 22.7% a fine sample, 0.2% proxy with
# density, 1.1-1.3% proxy at zero, no air - the non-fine steps are almost all exactly skippable by a brick-level DDA.
# MEASURED (costsplit1, 2026-10-08, sunset_motion 4K walk 2, frozen, beamDirtyStats, the same switches as HSTR_MOTION_ARMS -
# arms in HSTR_results/costsplit1.arms; units / query ms, units listed over 20 frames, ns a unit):
#   ship 1.840 / 1.406, 2.27M (16.2) | again 1.767 / 1.368, 2.44M (14.5)
#   no sky scatter 1.816 / 1.381, 2.34M (15.5) | no sun multiple 1.722 / 1.377, 2.43M (14.2)
#   no sun single 1.121 / 1.182, 1.39M (16.1) | no light 0.669 / 0.973, 0.88M (15.1)
#   generic 3.797 / 3.228, 2.20M (34.5) | generic no light 1.211 / 2.230, 0.89M (27.1) | cut at instance 1.214 / 2.120, 0.93M (26.0)
#   cut before atlas 1.303 / 2.213, 0.98M (26.6) | cut at proxy 0.190 / 0.374, 0.39M (9.6)
# A unit costs ~15 ns whatever the lighting: units time is the units' COUNT, and the single sun is what makes them - without it
# the tile test lists 39% fewer, without any light 62% fewer. In the query (blocks fixed, ~17k a frame) lighting is a per-sample
# cost, ~0.4 of 1.4 ms (the single sun ~0.2). Sky scatter and multiple sun are ~0.03-0.1 ms each. Far sea (~0.65) and resolve
# pixels (~0.55) do not move. The ship kernel runs at half the generic kernel's cost, so the cut-short probes (generic only)
# attribute a generic unit's ~27 ns mostly to everything past the proxy loop (9.6 ns), not to one link.
# MEASURED (livegap1, same conditions, residency frozen / live alternated in one launch): units 1.738 / 1.764 / 1.747 / 1.785,
# query 1.279 / 1.338 / 1.300 / 1.374 ms (frozen, live, frozen, live); units listed 2.27M / 2.31M / 2.33M / 2.30M. Live adds ~0.1 to
# the march and ~0.5 of residency passes: resolveCloudSun 0.11, sun scan 0.10, bakes 0.105, skirt masks 0.10-0.19, upload 0.065,
# stamp 0.03-0.05. The live flight's earlier ~1.3 ms gap over the frozen arms was mostly clock drift (it runs first). sunWaiting
# moves (265k -> 284k -> 205k, parked 231k at 128 bakes a frame): a real backlog the 1024-a-frame bake rate never clears, not churn.
SWEEPS = Path(__file__).parent
SHARE = runpy.run_path(str(SWEEPS / "push_share.py"))["SHARE"]
BASE = runpy.run_path(str(SWEEPS / "march_cost_split.py"))["BASE"]
GENERIC = dict(BASE, beamShip=False)
TESTS = [
    ("bricks", dict(SHARE, pushDilate=0, pushShareMode=8)),
    ("ship", BASE),
    ("generic", GENERIC),
    ("no light", dict(GENERIC, hstComponents=1)),
    ("instance", dict(GENERIC, hstComponents=1, cloudCostProbe=3)),
    ("no atlas", dict(GENERIC, hstComponents=1, cloudCostProbe=2)),
    ("proxy", dict(GENERIC, hstComponents=1, cloudCostProbe=1)),
    ("ship again", BASE),
]
