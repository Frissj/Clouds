import runpy
from pathlib import Path

# Occupancy check on the dirty passes: the same frame with sunDepthAt's live near march compiled out (HSTR_SUN_LIVE 0). In a settled
# frozen sea every sun query is answered by a bake, so the frames should match; any time difference is the fallback's registers.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/dirty_sunlive.py --motions "walk 2 0.004" "sprint 20 0"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1)
TESTS = [("live", dict(BASE, cloudSunLiveMarch=True)), ("nolive", dict(BASE, cloudSunLiveMarch=False))]
