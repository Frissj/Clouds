from sea_config import mk

# Whether the rare live-sun fallback's state belongs in the common camera kernel. Baked brick/ancestor coverage should make the
# fallback unnecessary; quality metrics catch any miss while the timing exposes the register/occupancy cost of merely compiling it.
TESTS = [
    ["live sun", mk(cloudSunLiveMarch=True)],
    ["baked only", mk(cloudSunLiveMarch=False)],
]
