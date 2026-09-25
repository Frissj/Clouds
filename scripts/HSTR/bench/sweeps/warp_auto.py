from sea_config import mk

# beamWarpAuto: the warp field and the warped resolve run only while the last translating build held at least this share of the
# on-screen guard blocks it classified. Sprint expires every block (6d941dfb: 3.03 ms with the warp, ~2.90 without, 0.283% either
# way), so the question is whether the held share separates walk from sprint cleanly enough to switch on, and what sprint recovers.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/warp_auto.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
TESTS = [
    ("warp", mk(beamWarp=True, beamWarpAuto=0.0)),
    ("no warp", mk(beamWarp=False, beamWarpAuto=0.0)),
    ("auto 0.10", mk(beamWarp=True, beamWarpAuto=0.10)),
    ("auto 0.25", mk(beamWarp=True, beamWarpAuto=0.25)),
    ("warp again", mk(beamWarp=True, beamWarpAuto=0.0)),
]
# MEASURED (warpauto1, a lagged CPU readback of the counts, since replaced by writeBeamWarpArgs on the GPU; ms, over 0.02):
#   walk   warp 2.11 0.937% | no warp 1.98 1.985% | auto 0.10 2.08 0.937% | again 2.08. Held 16,844 of 30,025 classified: 56%.
#   sprint warp 2.97 0.283% | no warp 2.83 0.283% | auto 0.10 2.83 0.283% | again 2.95. Held 266 of 30,027: 0.9%.
# The share separates the two by 60x, and sprint loses nothing with its 266 held blocks unwarped. auto 0.02 only half switched
# (resolve 0.50): the readback lag let it flicker, which is why the decision moved to the GPU.
# MEASURED (warpauto2, decided on the GPU by writeBeamWarpArgs, both resolves indirect; ms, over 0.02, resolve):
#   walk   warp 2.11 0.936% 0.57 | no warp 1.97 1.984% 0.44 | auto 0.10 2.14 | auto 0.25 2.08 0.936% 0.57 | again 2.07. Held 62%.
#   sprint warp 2.97 0.283% 0.58 | no warp 2.84 0.283% 0.45 | auto 0.10 2.86 | auto 0.25 2.85 0.283% 0.46 | again 2.95. Held 0.
# Zero lag, errors bit-identical to the forced setting in both motions; the decision and empty dispatches cost ~0.01-0.02 ms.
# SHIPPED: beamWarpAuto 0.25.
