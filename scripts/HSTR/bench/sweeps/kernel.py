from sea_config import mk

# ab: per-ray cost of the beam passes against the full-frame march, the 2D dispatches, and the shipping specialization.
TESTS = [
    ["anchor", mk()],
    ["pixel s2", mk(debugView=8)],
    ["pixel s2 cam", mk(debugView=8, cloudCameraKernel=True)],
    ["grid", mk(beamGridDispatch=True)],
    ["ship", mk(beamShip=True)],
    ["anchor end", mk()],
]
