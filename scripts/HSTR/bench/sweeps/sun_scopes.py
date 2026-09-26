from sea_config import mk

# DIAGNOSTIC: sunColumns' own passes (sunColumns, sunResample) and the octaves' gather and blur, parked under a moving sun and at
# sprint with a static one. sunpages2 left ~0.45 ms of fineSun unexplained under the moving sun (0.03 at sprint).
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/sun_scopes.py --steps 0
#          --motions "park 0 0 0.01" "sprint 20 0"
TESTS = [
    ("columns", mk(sunColumns=True)),
]
