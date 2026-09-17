from sea_config import mk

# ab --views near,sea: which HSTR_SHIP group slows the near view and which speeds up the sea (beamShipMask bits, see
# HSTRCloud.cs.slang).
BITS = {1: "transfer", 2: "costProbe", 4: "emptySkip", 8: "sunCache", 16: "adaptive", 32: "ends", 64: "footprint", 128: "quadrature",
        256: "sunReuse"}
TESTS = [["generic", mk(beamShipMask=511)], ["ship", mk(beamShip=True, beamShipMask=511)]]
TESTS += [[f"only {name}", mk(beamShip=True, beamShipMask=bit)] for bit, name in BITS.items()]
TESTS += [["generic end", mk(beamShipMask=511)]]
