import runpy
from pathlib import Path

# The sun slot pool holds one bake per density slot, but the sea's bricks need about three (orientation classes): it settles full,
# with ~126k bakes waiting for good, and a sample of an unbaked brick whose two nearest ancestors are unbaked too takes sunDepthAt's
# live march at every step - ~4.5 ms of the 4K walk. The fallback compared: climb two levels (as shipped), climb to the root, and
# the voxel sun field (the proxy, cloudSunLiveMarch off).
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/sun_fallback.py --motions "walk 2 0.004" "sprint 20 0"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1)
TESTS = [("climb2", dict(BASE, cloudSunAncestors=2)), ("climb15", dict(BASE, cloudSunAncestors=15)),
         ("proxy", dict(BASE, cloudSunAncestors=2, cloudSunLiveMarch=False))]
