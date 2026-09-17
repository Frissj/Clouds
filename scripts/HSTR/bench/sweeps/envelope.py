from sea_config import mk

# motion --live: the residency motion envelope (cloudCutMargin, voxels) against the exact cut, as residency.py
# (run with --motions "static 0 0" "fly_2 2 0" "fly_20 20 0" "rotate 0 0.5").
TESTS = [
    ["ship", mk(beamShip=True)],
    ["margin 16", mk(beamShip=True, cloudCutMargin=16.0)],
    ["margin 4", mk(beamShip=True, cloudCutMargin=4.0)],
    ["margin 8", mk(beamShip=True, cloudCutMargin=8.0)],
]
