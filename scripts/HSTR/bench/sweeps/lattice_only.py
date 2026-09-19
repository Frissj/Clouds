from sea_config import mk

# Root-lattice-only architecture. With refinement disabled, cost scales with query count instead of difficult-region pixel count;
# each width must pass the saved path-reference percentile gate before it can replace the finer lattice.
LATTICE = {"beamTolerance": 100.0, "beamEdgeContrast": 0.0}
TESTS = [
    ["8x1 lattice", mk(beamTileSize=8, **LATTICE)],
    ["16x1 lattice", mk(beamTileSize=16, **LATTICE)],
    ["32x1 lattice", mk(beamTileSize=32, **LATTICE)],
]
