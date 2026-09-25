import runpy
from pathlib import Path

# Step phase jitter (HSTR_SHIP bit 65536, marchBeamLean): the first step shortened by a per-direction hash, so a long step's error
# is noise across neighbouring texels rather than structure. Whether that buys back enough of the gate to take 2.5-3x steps. Every
# arm at beamOctScale 0.5: changing the scale within one process reallocates the octahedral image and, after two or three changes,
# the GPU ran ~10x slower on the same work (budgetoct1 / budgetoct2), so one scale per process.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/budget_jitter.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
STEP = runpy.run_path(str(Path(__file__).with_name("budget_step.py")))
step = STEP["step"]


def arm(k, jitter=False):
    a = dict(step(k, True), beamOctScale=0.5)
    if jitter:
        a["beamShipMask"] |= 65536
    return a


TESTS = [
    ("s2t", arm(2)),
    ("s2t jitter", arm(2, True)),
    ("s2.5t jitter", arm(2.5, True)),
    ("s3t jitter", arm(3, True)),
    ("s2t again", arm(2)),
]

# MEASURED and REMOVED (budgetjitter2, phase re-armed after every skip): walk s2t 2.46 ms 0.760% -> jitter 2.71 0.639% | s2.5t
# jitter 2.49 1.650% | s3t jitter 2.32 2.481%; sprint 2.90 0.283% -> 3.16 0.238% | 2.94 0.396% | 2.79 0.583%. Dominated: bit 65536
# is gone from the shader, so these arms no longer vary.
# MEASURED (budgetjitter1, phase applied to the ray's first step only - erased by the first skip, so no effect): walk s2t 2.45 ms
# 0.760% | s2.5t 2.28 1.957% | s3t 2.07 3.125% | again 2.50 0.760%; sprint s2t 2.92 0.283% | s2.5t 2.68 0.547% | s3t 2.46 0.941%.
