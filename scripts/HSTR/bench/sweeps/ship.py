from sea_config import mk

# ab --views near,farside,sea: beamShip against the generic beam programs, before defaulting it.
TESTS = [
    ["generic", mk()],
    ["ship", mk(beamShip=True)],
]
