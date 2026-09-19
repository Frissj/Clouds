from sea_config import mk

# motion --live: what the temporal carry (beamRefresh) is worth on a flight, where the frame is spent re-querying beam tiles the
# camera moved. It was measured parked (3.24 -> 2.95 ms at the same error) and at 2 units a frame (error 0.152% -> 0.185%), never
# on the sea's own flights. r is the refresh pattern: a build marches one cell of an r x r pattern and carries the rest.
TESTS = [
    ["off", mk()],
    ["r2", mk(beamRefresh=2)],
    ["r4", mk(beamRefresh=4)],
]
