import runpy
from pathlib import Path

# pushEval (handoff Step 5, as a measurement): sample directions (one per 8 x 8 beam texels) consume the sorted shell lists front
# to back, compositing cells from their views and marching only the chords of cells without one (the residual), then the exact
# march of the same direction scores them. Answers: cells consumed per sample before opacity, view hit rate, residual share and
# steps, view builds, error against the exact march. The frame itself is unchanged.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/push_eval.py
#          --steps 2 --motions "walk 2 0.004"
PROBE = runpy.run_path(str(Path(__file__).with_name("push_probe.py")))
EVAL = dict(PROBE["PUSH"], pushShells=10, pushEval=True, pushSampleStep=8, cellViews=True, cellViewEdge=8, cellViewCapacity=131072,
            cellViewBuilds=16384, cellViewDrift=0.04, cellViewRefine=True)
# MEASURED (eval1): views - walk 3.0% of samples over 0.02 and 2.05 ms against 1.09 exact; sprint 59% residual, 4.9 ms; drift .04
# and .1 alike. The arms now split the error by source, every one scored against the exact march of the same samples (arm A):
#   B views (the known loser), C the cell operator's extinction with the chord's exact source (oracle), D + constant source,
#   E + linear source, F + trilinear emission. See pushOperatorChord.
OPERATOR = dict(EVAL, cellViews=False)
TESTS = [
    ("B views", dict(EVAL, pushEvalMode=0)),
    ("C tau+oracle", dict(OPERATOR, pushEvalMode=1)),
    ("D constant", dict(OPERATOR, pushEvalMode=2)),
    ("E linear", dict(OPERATOR, pushEvalMode=3)),
    ("F trilinear", dict(OPERATOR, pushEvalMode=4)),
]
