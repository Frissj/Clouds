import runpy
from pathlib import Path

# Certification (experiment 5) and the source oracle's converse, on the shipping frame's real dirty rays, one settle.
# - "share": pushShareMode 2 | 4 - share7/8 reproduced (the anchor) and, per crossing, a central-difference Jacobian about the
#   snapped 1-frame texel with three guards: its predicted change accepting the stale value, the one-scalar L1 sensitivity bound
#   accepting it, and the jet's curvature estimate accepting its own prediction (pushShareCertify*, see countPushShare).
# - "tau+const" / "tau+tri": pushEval's cell operator with each chord's EXACT transmittance and the operator's constant /
#   trilinear source (pushEvalExactDepth, see pushOperatorChord), scored against the exact march of the same samples.
# Counters are the last scored step's frame; probe time is not a cost.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/certify.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
SWEEPS = Path(__file__).parent
SHARE = runpy.run_path(str(SWEEPS / "push_share.py"))["SHARE"]
OPERATOR = runpy.run_path(str(SWEEPS / "push_eval.py"))["OPERATOR"]
TESTS = [
    ("share", dict(SHARE, pushDilate=0, pushShareMode=6)),
    ("tau+const", dict(OPERATOR, pushEvalMode=2, pushEvalExactDepth=True)),
    ("tau+tri", dict(OPERATOR, pushEvalMode=4, pushEvalExactDepth=True)),
]
