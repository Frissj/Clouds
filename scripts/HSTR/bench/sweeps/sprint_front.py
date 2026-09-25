from sea_config import mk

# Sprint's quality headroom: 0.283% of pixels over 0.02 against the <1% gate (d66196bd default, 2.85 ms), because it holds no
# blocks, so none of its error is parallax. Where the step scale and beam tolerance put it on the cost / error front, before a
# controller spends it. Step scale k: minStepVoxels 2k, maxStepVoxels k (the default is k = 2), transmittance-scaled steps kept.
# budgetjitter1 (old frame, no warp field): sprint k 2 2.92 ms 0.283% | 2.5 2.68 0.547% | 3 2.46 0.941%.
# Step and tolerance resize nothing, so every arm shares one process.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/sprint_front.py --steps 1 --motions "sprint 20 0"


def step(k, **extra):
    return mk(minStepVoxels=2.0 * k, maxStepVoxels=float(k), **extra)


FIRST = [
    ("k2", step(2)),
    ("k2.5", step(2.5)),
    ("k2.75", step(2.75)),
    ("k3", step(3)),
    ("k3.25", step(3.25)),
    ("k2 tol .02", step(2, beamTolerance=0.02)),
    ("k2 tol .03", step(2, beamTolerance=0.03)),
    ("k2 again", step(2)),
]
# MEASURED (sprintfront1; ms, over 0.02, p99.9, max, units marched):
#   k2 2.84 0.282% 0.030 0.64 118k | k2.5 2.77 0.546% | k2.75 2.51 0.716% 0.043 | k3 2.39 0.940% 0.051 | k3.25 2.31 1.228% (fails)
#   k2 tol .02 2.62 0.349% 92k | k2 tol .03 2.53 0.407% 0.036 0.31 82k | k2 again 2.81 0.282%
# Tolerance is the cheaper lever: .03 buys 0.30 ms for +0.13%, where k2.75 buys 0.32 ms for +0.43%. It cuts the units the
# failed tiles march (118k -> 82k) and the resolve's residual; steps cut query and units per ray.
TESTS = [
    ("k2 tol .03", step(2, beamTolerance=0.03)),
    ("k2 tol .04", step(2, beamTolerance=0.04)),
    ("k2 tol .05", step(2, beamTolerance=0.05)),
    ("k2 tol .07", step(2, beamTolerance=0.07)),
    ("k2.5 tol .03", step(2.5, beamTolerance=0.03)),
    ("k2.5 tol .05", step(2.5, beamTolerance=0.05)),
    ("k2.75 tol .03", step(2.75, beamTolerance=0.03)),
    ("k2 tol .03 again", step(2, beamTolerance=0.03)),
]
