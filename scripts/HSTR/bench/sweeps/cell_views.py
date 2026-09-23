import runpy
from pathlib import Path

# Cell views (cellViews): the dirty query and the dirty unit march composed each ray from cached per-cell views of its transfer
# instead of marching it (composeBeam). cellViewDrift is the lateral misregistration (domain voxels) a view may accumulate.
# HISTORICAL: composeBeam is removed (per-ray traversal; 10.05 ms vs 7.57 walk, results cellviews1/cvdebug2-5), so cellViews=True
# no longer changes the frame. Kept as the record of those runs' configuration.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/cell_views.py --motions "walk 2 0.004" "sprint 20 0"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1, cloudSunAncestors=15, beamRepairProbe=False, beamShadowCarry=0,
            cellViews=False)
CELL = dict(BASE, cellViews=True, cellViewEdge=16, cellViewCapacity=16384, cellViewBuilds=4096, cellViewDrift=0.04,
            cellViewRefine=True)
TESTS = [
    ("base", BASE),
    ("cell .04", CELL),
    ("cell .02", dict(CELL, cellViewDrift=0.02)),
    ("cell .04 e8", dict(CELL, cellViewEdge=8)),
    ("base again", BASE),
]
