from sea_config import mk

# Phase G: how much of the dirty query and unit passes is per-ray cost that stays however few samples a ray takes. beamStripProbe
# runs a copy of each pass over the same list before the real one (queryStrip, unitsStrip), compiled to stop early, writing
# nothing: 1 after the ray setup (the list read, beamQueryNeedsMarch, makeCutRay), 2 after one step, 3 the full march - the
# calibration, against the real pass in the same frame, of what running second (warm caches) or first does to a copy.
# Timings only (--steps 0): the image is the real passes' and does not change.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/strip.py --steps 0
#          --motions "walk 2 0.004" "sprint 20 0"
FIRST = [
    ("full copy", mk(beamStripProbe=3)),
    ("setup", mk(beamStripProbe=1)),
    ("one step", mk(beamStripProbe=2)),
    ("full copy again", mk(beamStripProbe=3)),
]
# MEASURED (strip1; ms, the copy's scope; the real pass ran beside it at 0.49-0.57 walk query, 0.69-0.71 units):
#   walk   full copy query 0.558 units 0.666 | setup 0.026 0.007 | one step 0.038 0.016 | full again 0.550 0.663
#   sprint full copy query 0.823 units 0.681 | setup 0.044 0.007 | one step 0.061 0.015 | full again 0.819 0.683
# ARCHITECTURAL LIMIT for per-ray work: the list read, beamQueryNeedsMarch and makeCutRay are 2-5% of the query and 1% of the
# units. Both passes are their samples; precomputed ray data or per-ray specialisation cannot pay.
# Phase J, per sample: 4 density alone (no sun chain, no cache), 5 density + sun, 6 density + cache.
TESTS = [
    ("full copy", mk(beamStripProbe=3)),
    ("density", mk(beamStripProbe=4)),
    ("density + sun", mk(beamStripProbe=5)),
    ("density + cache", mk(beamStripProbe=6)),
    ("full copy again", mk(beamStripProbe=3)),
]
