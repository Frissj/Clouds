from sea_config import mk

# motion --steps 0 with HSTR_CHURN: why a flight frame costs 2.6x a parked one at the same camera positions with 9% more bricks
# resident. Churned then frozen, nothing streams while the timing runs, so this is what the churn left behind. probe 2 walks the
# hierarchy and skips the atlas fetch, probe 6 stops after the parent climb and the octant test. If the gap between parked and
# flying survives probe 2 it is the walk; if it only appears with the fetch it is the atlas's locality.
TESTS = [
    ["base", mk()],
    ["no atlas", mk(cloudCostProbe=2)],
    ["climb", mk(cloudCostProbe=6)],
]
