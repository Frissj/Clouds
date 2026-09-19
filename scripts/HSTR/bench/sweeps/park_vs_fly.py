from sea_config import mk

# motion --live: the shipped beam view, one arm, flown at three speeds including standing still. Every "moving camera costs 2x"
# claim so far compared a parked frame of one config against a flight of another. With no temporal reuse in the ship config
# (beamRefresh 0, beamTemporal off) a frame has no history to lose, so whatever parked -> flying costs is what the flight puts in
# front of the camera, not what motion invalidated. This measures that difference.
TESTS = [
    ["ship", mk()],
]
