from sea_config import mk

# Cost/quality curve for the full-resolution fallback behind the shipped 8x1 lattice. This isolates whether the expensive fallback
# is buying visible tail quality against the saved path reference or merely matching exact-march detail below that reference's noise.
TESTS = [
    ["8x1 .05", mk(beamTolerance=0.05)],
    ["8x1 .10", mk(beamTolerance=0.10)],
    ["8x1 .20", mk(beamTolerance=0.20)],
    ["8x1 .50", mk(beamTolerance=0.50)],
    ["8x1 no edge", mk(beamTolerance=0.05, beamEdgeContrast=0.0)],
    ["8x1 lattice", mk(beamTolerance=100.0, beamEdgeContrast=0.0)],
]
