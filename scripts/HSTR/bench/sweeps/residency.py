from sea_config import mk

# motion --live: the CPU residency cost of a moving camera (run with --motions "static 0 0" "tiny 0.001 0" "fly_2 2 0" "fly_20 20 0").
TESTS = [
    ["ship", mk(beamShip=True)],
]
