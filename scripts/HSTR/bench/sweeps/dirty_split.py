import runpy
from pathlib import Path

# Timing split of the dirty passes, OCT_T001 at the guard's defaults with the sun fallback climbing to the root.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/dirty_split.py --motions "walk 2 0.004" "sprint 20 0"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
TESTS = [("current", dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1, cloudSunAncestors=15))]
