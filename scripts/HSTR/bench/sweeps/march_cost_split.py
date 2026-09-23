import runpy
from pathlib import Path

# Where the dirty march's time goes (handoff experiment 0): the shipping frame with the march's light terms switched off one at a
# time through hstComponents (runtime branches: the cost of the evaluations, not of the registers they hold). kComponentSunSingle
# (2) is every sunDepthAt + residualAtDepth; SunMultiple | SkyScattered (4 | 8) is every worldCacheRadiance; 1 leaves the density
# and the transmittance. These images are wrong on purpose: read the query and units timings, not the error.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/march_cost_split.py
#          --steps 1 --motions "walk 2 0.004" "sprint 20 0"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1, cloudSunAncestors=15, beamRepairProbe=False, beamShadowCarry=0,
            cellViews=False, beamPushProbe=False, hstComponents=15)
TESTS = [
    ("all", BASE),
    ("no sun", dict(BASE, hstComponents=1 | 4 | 8)),
    ("no cache", dict(BASE, hstComponents=1 | 2)),
    ("density only", dict(BASE, hstComponents=1)),
    ("all again", BASE),
]
