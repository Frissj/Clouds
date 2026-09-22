import runpy
from pathlib import Path

# Slices per ray in the dirty query at 4K (beamDirtySegments), the evidence for its default of 1.
#
# 64 was measured at 720p on ~560 dirty rays, where the query is latency-bound and slices are free. Run with
#   python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/dirty_segment_budget.py
#          --motions "park 0 0" "look 0 0.01" "walk 2 0.004" "sprint 20 0"
# it read (HSTRCloud ms / query ms / >0.02 against the exact march):
#   walk   seg 1 28.2 / 6.4 / 0.136%   seg 64 62.0 / 38.2 / 0.554%
#   sprint seg 1 31.4 / 6.1 / 0.025%   seg 64 81.7 / 53.1 / 0.156%
#   park and look: no query at all (prebuilt sphere), 0.55 / 0.68 ms in every arm.
# A per-build adaptive count (beamDirtySegmentThreads: the most slices keeping rays x slices within 32k-256k threads) matched seg 1
# exactly in every motion - it always chose 1 - and was removed. "seg 1" runs first (warm-up) and last (drift).
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0)

TESTS = [
    ("seg 1", dict(BASE, beamDirtySegments=1)),
    ("seg 64", dict(BASE, beamDirtySegments=64)),
    ("seg 1 again", dict(BASE, beamDirtySegments=1)),
]
