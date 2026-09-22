import runpy
from pathlib import Path

# Cell views at steady state: a ring big enough that views outlive the walk's churn (262k views of 8 x 8), so the composition's own
# cost shows once nearly every occupied cell has one. 16 x 16 at 65k beside it for the resolution's effect on quality.
CV = runpy.run_path(str(Path(__file__).with_name("cell_views.py")))
BIG = dict(CV["CELL"], cellViewCapacity=65536, cellViewBuilds=16384, cellViewDrift=0.1)
TESTS = [
    ("base", CV["BASE"]),
    ("e16 65k", BIG),
    ("e8 262k", dict(BIG, cellViewEdge=8, cellViewCapacity=262144)),
]
