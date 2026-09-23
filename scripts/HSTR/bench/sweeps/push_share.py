import runpy
from pathlib import Path

# Experiment 1, pushShare: how much work one evaluation per cell x BeamTile interaction could replace. Every dirty ray the shipping
# frame marched (query points and units) walks the push lists (shells) front to back, marching each crossed cell's chord; per
# interaction it counts the rays and their steps inside it (pushShareShell, pushShareHist*), and the same rays' exact steps
# (pushShareExactSteps) check that the crossings account for the march. Experiment 2 rides along: payloads per interaction
# (pushSharePayload*, see countPushShare). Counters are the last scored step's frame; the probe's own time is not a cost.
# MEASURED (share4): "dilate1" (every cell within one cell of an occupied one listed) added 3.2M crossings and no density at all -
# the same 8,604 rays over 0.02 in transmittance - so it is not an arm any more.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/push_share.py --steps 2 --motions "walk 2 0.004"
#      (~15 minutes after a shader edit, much of it recompiling ~108 programs).
PUSH = runpy.run_path(str(Path(__file__).with_name("push_probe.py")))["PUSH"]
SHARE = dict(PUSH, pushShells=10, pushShare=True)
# pushShareMode 2: experiment 4, the temporal oracle (see countPushShare); 1 re-runs the payload experiments 2 and 3.
TESTS = [("share", dict(SHARE, pushDilate=0, pushShareMode=2))]
