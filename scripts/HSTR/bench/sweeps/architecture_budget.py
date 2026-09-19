from sea_config import mk

# Query-count architecture: trade a coarser root lattice against fewer per-pixel fallbacks by changing only the measured surplus
# threshold. Candidates must subsequently pass the saved path-reference percentile gate; this sweep is not permission to ship loss.
TESTS = [
    ["8x1 .05", mk(beamTileSize=8, beamTolerance=0.05)],
    ["8x1 .10", mk(beamTileSize=8, beamTolerance=0.10)],
    ["8x1 .20", mk(beamTileSize=8, beamTolerance=0.20)],
    ["8x1 .50", mk(beamTileSize=8, beamTolerance=0.50)],
    ["16x1 .05", mk(beamTileSize=16, beamTolerance=0.05)],
    ["16x1 .10", mk(beamTileSize=16, beamTolerance=0.10)],
    ["16x1 .20", mk(beamTileSize=16, beamTolerance=0.20)],
    ["32x1 .10", mk(beamTileSize=32, beamTolerance=0.10)],
    ["32x1 .20", mk(beamTileSize=32, beamTolerance=0.20)],
]
