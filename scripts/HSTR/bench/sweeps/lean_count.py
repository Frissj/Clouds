import runpy
from pathlib import Path

# What the lean march's steps are (HSTR_SHIP bit 4096, marchBeamLean's counters into pushShare's walk slots): per frame, every
# dirty ray's steps split into majorant-zero steps (pushWalkAir), empty-box samples (pushWalkEmpty), proxy samples
# (pushWalkProxy), contributing samples (pushWalkProxyZero, alpha > 1e-6) and the rest - brick samples too faint to count; rays in
# pushBrickWalkSamples. The push probe only provides the counter buffer; its own time is not a cost here.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_count.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
SHARE = runpy.run_path(str(Path(__file__).with_name("push_share.py")))["SHARE"]
TESTS = [("count", dict(SHARE, pushDilate=0, pushShareMode=0, beamShipMask=509 | 1024 | 2048 | 4096))]
