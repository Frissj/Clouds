from sea_config import mk

# motion: queued root queries (beamQueue) by cost bucket width, alone and with beamRefresh.
S = dict(beamShip=True)
TESTS = [
    ["ship", mk(**S)],
    ["queue 12", mk(beamQueue=True, **S)],
    ["queue 6", mk(beamQueue=True, beamQueueSteps=6.0, **S)],
    ["queue 24", mk(beamQueue=True, beamQueueSteps=24.0, **S)],
    ["q12 r2", mk(beamQueue=True, beamRefresh=2, beamCarryTolerance=0.02, **S)],
    ["q12 r4", mk(beamQueue=True, beamRefresh=4, beamCarryTolerance=0.02, **S)],
    ["q12 r4 px1", mk(beamQueue=True, beamRefresh=4, beamCarryTolerance=0.02, beamParallax=1.0, **S)],
]
