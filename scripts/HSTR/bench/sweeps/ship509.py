from sea_config import mk

# beamShip with the cost probe left live (mask 509), which ship_combos.py found fast on near and sea.
TESTS = [
    ["generic", mk(beamShipMask=509)],
    ["ship509", mk(beamShip=True, beamShipMask=509)],
    ["ship511", mk(beamShip=True, beamShipMask=511)],
    ["generic end", mk(beamShipMask=509)],
]
