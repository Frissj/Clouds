from sea_config import mk

# sunInvalidateAngle (sunColumns): a moving sun re-marches a tile's beam columns only once the sun has turned this far since its
# last refresh, nearest first, at most cloudSunTilesPerFrame a frame; 0 re-marched the nearest 8 tiles every frame the sun moved.
# suninval2 (4K, 0.57 deg/frame sun, % of pixels over 0.02): parked every frame 3.11 ms 0.362%, 0.6 deg 3.32, 2 deg 2.75, 4 deg
# 2.33 0.363% (query + units 1.40 -> 0.48); walk 3.00 0.363% -> 2.93 / 2.92 at 0.303% (2 and 4 deg; walk's re-march is mostly
# translation). No quality edge up to 4 deg, so this finds it.
# suninval3 (two steps, mean / worst %): parked every frame 3.10 ms 0.367/0.373, 4 deg 2.46 0.368/0.373, 8 deg 2.11 0.434/0.445,
# 16 deg 1.99 same; walk 3.00 0.369/0.375, 4 deg 2.92 0.311/0.320, 8 deg 2.93 0.436/0.596. Shipped at 4 deg (0.07).
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/sun_invalidate.py --steps 2
#          --motions "park 0 0 0.01" "walk 2 0.004 0.01"
TESTS = [
    ("every frame", mk(sunInvalidateAngle=0.0)),
    ("4 deg", mk(sunInvalidateAngle=0.07)),
    ("8 deg", mk(sunInvalidateAngle=0.14)),
    ("16 deg", mk(sunInvalidateAngle=0.28)),
]
