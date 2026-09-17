from sea_config import mk

# The dedicated shipping march loop (HSTR_SHIP bit 512) against the folded generic loop, with the cost probe live (509) and folded.
TESTS = [
    ["ship509", mk(beamShipMask=509)],
    ["loop1021", mk(beamShipMask=1021)],
    ["loop1023", mk(beamShipMask=1023)],
    ["generic", mk(beamShip=False)],
    ["ship509 end", mk(beamShipMask=509)],
]
