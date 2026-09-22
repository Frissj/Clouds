import runpy
from pathlib import Path

# What a beam march step costs by part: every component (15), without the cache (sun multiple and sky: 3), without the single
# scatter's sun depth (13), and extinction only (1, background). Error numbers are against each arm's own exact frame, so only the
# times compare. Answers whether a geometry-only march is cheap enough to validate a reprojected sample.
# Run: HSTR_STEPS=0 python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/march_components.py --motions "walk 2 0.004"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1)
TESTS = [(f"c{c}", dict(BASE, hstComponents=c)) for c in (15, 13, 3, 1)]
