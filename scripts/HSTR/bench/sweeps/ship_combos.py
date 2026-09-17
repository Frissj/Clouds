from sea_config import mk

# ab --views near,sea: HSTR_SHIP folds everything but one group, and the groups that helped alone (ship_bits.py).
BITS = {1: "transfer", 2: "costProbe", 4: "emptySkip", 8: "sunCache", 16: "adaptive", 32: "ends", 64: "footprint", 128: "quadrature",
        256: "sunReuse"}
TESTS = [["generic", mk(beamShipMask=511)], ["best4", mk(beamShip=True, beamShipMask=2 | 16 | 32 | 128)]]
TESTS += [[f"no {name}", mk(beamShip=True, beamShipMask=511 & ~bit)] for bit, name in BITS.items()]
TESTS += [["generic end", mk(beamShipMask=511)]]
