from sea_config import mk

# sunOctaveAngle: the multiply scattered sun octaves (a whole-volume build) follow a moving sun only once it has turned this far
# from their build, and when the cloud changes or the sun stops; the single-scattered field stays exact every frame.
# (suncost1: lagging the field and octaves together by 1 deg took parked 1.78 -> 4.81% of pixels over 0.02, so the field cannot
# lag.) The octave blur also wraps without integer modulos now.
# sunoct1 was void: every arm carried the skipped world cache decay (reverted, cachereg2: 0.364 -> 1.753%).
# worldCacheFrozenBake: with the cache frozen (the harness), a sun move bakes it without the 0.26 ms decay pass.
# sunoct2 (4K, % over 0.02 mean / worst): parked every frame 2.16 ms 0.374/0.376, 2 deg 1.97 0.368/0.373, 4 deg 1.84 same,
# 4 deg + frozen bake 1.57 same; walk 2.69 / 2.48 / 2.45 / 2.20 ms, all 0.312/0.320. Shipped: 4 deg + frozen bake.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/sun_octaves.py --steps 2
#          --motions "park 0 0 0.01" "walk 2 0.004 0.01"
TESTS = [
    ("every frame", mk(sunOctaveAngle=0.0)),
    ("2 deg", mk(sunOctaveAngle=0.035)),
    ("4 deg", mk(sunOctaveAngle=0.07)),
    ("4 deg frozen bake", mk(sunOctaveAngle=0.07, worldCacheFrozenBake=True)),
]
