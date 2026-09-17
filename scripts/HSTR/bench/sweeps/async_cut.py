from sea_config import mk

# motion --live: the residency cut on a worker (cloudCutAsync) against the cut in the frame.
TESTS = [
    ["sync", mk()],
    ["async", mk(cloudCutAsync=True)],
]
