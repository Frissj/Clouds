from sea_config import mk

# beamDirtyTilesDense (shipped): the dirty tile test as a thread per tile, testing the tiles any listed block's points are read by,
# against the old thread per padded tile of each listed block (36 a block at side 2, ~1 in 36 testing). Same tiles, same units
# (in tile order): errors, tiles tested and units listed must match.
# dense2 (4K ms, identical outputs): walk 1.94 -> 1.85, jog 2.49 -> 2.30, sprint 2.37 -> 2.17 (padded mean of two anchors).
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/dirty_tiles_dense.py --steps 1
#          --motions "walk 2 0.004" "jog 8 0.002" "sprint 20 0"
TESTS = [
    ("padded", mk(beamDirtyTilesDense=False)),
    ("dense", mk()),
    ("padded again", mk(beamDirtyTilesDense=False)),
]
