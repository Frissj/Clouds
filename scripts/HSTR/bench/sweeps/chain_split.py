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
