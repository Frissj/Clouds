import runpy
from pathlib import Path

# The octahedral beam image's angular resolution (beamOctScale; 1.0 already undersamples the screen, oct_motion.py) against the p99
# gate: the dirty march's cost is ~1.16 ms per-ray floor + 3.45 ms / step multiplier at walk (budgetstep1), so longer steps alone
# floor near 2.3 ms a frame and only fewer rays reach the rest. Scale s marches ~s^2 of the texels; the resolve filters the image.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_oct.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
STEP = runpy.run_path(str(Path(__file__).with_name("budget_step.py")))
BASE, step = STEP["BASE"], STEP["step"]
FAST = step(2, True)

TESTS = [
    ("step 2x tstep", FAST),
    ("s2t oct .7", dict(FAST, beamOctScale=0.7)),
    ("s2t oct .5", dict(FAST, beamOctScale=0.5)),
    ("step 2x tstep again", FAST),
]

# MEASURED (budgetoct1, 4K, errors only - the GPU slowed ~10x from the fourth walk arm on, same tile counts, nothing residency):
#   walk   s2t 3.94 ms 0.513% | oct .85 3.54 0.541% | oct .7 3.10 0.593% | oct .5 0.760% | ship oct .7 0.245%
#   sprint s2t 0.155% | oct .85 0.170% | oct .7 0.196% | oct .5 2.87 ms 0.283% | ship oct .7 0.060%
# MEASURED (budgetoct2, clean until the walk anchor): walk s2t 4.01 ms 0.513% | oct .7 3.11 0.593% | oct .5 2.46 0.760% (march
# 1.63, tile tests 0.19, classify 0.11, resolve 0.47); sprint oct .7 3.84 0.196%. The slowdown again followed the scale changes.

# MEASURED (budgetstep1, 4K, errors against the exact march; the gate is under 1% of pixels over 0.02):
#   walk   ship 5.73 ms 0.137% | 1.5x 4.71 0.189% | 2x 4.14 0.425% | 3x 3.47 2.941% | 2x tstep 3.91 0.513% | again 5.69 0.137%
#   sprint ship 7.15 ms 0.028% | 1.5x 6.03 0.046% | 2x 5.36 0.127% | 3x 4.56 0.803% | 2x tstep 5.06 0.155% | again 7.22 0.028%
# Walk query + units 4.61 -> 3.60 / 3.01 / 2.31 ms at 1 / 1.5 / 2 / 3x: ~1.16 + 3.45 / k. Outside the march ~1.15 ms (tile tests
# 0.44, classify + rebuild 0.2, resolve 0.46).
