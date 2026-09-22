import runpy
from pathlib import Path

# Content changes under the persistent beam image (beamInvalidate). The sea replaces tiles as the camera crosses them - with
# residency frozen it now takes them on synchronously, so they land on fixed frames - and rebuilds the sun pages downwind. Off, a
# direction marched before a tile changed keeps the old cloud until camera motion happens to re-march it; on, the blocks whose
# directions cross the changed column (and its lighting reach) are unverified and re-marched.
#
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/content_invalidation.py
#        --motions "walk 2 0.004" "sprint 20 0"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1)

TESTS = [("stale", dict(BASE, beamInvalidate=False)), ("invalidate", dict(BASE, beamInvalidate=True))]
