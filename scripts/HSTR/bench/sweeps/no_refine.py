from sea_config import mk

# Price the two refinement tests separately and the root lattice alone. The last arm is useful only if its saved-reference tail
# remains inside the quality gate; it deliberately exposes the maximum performance available from the current query topology.
TESTS = [
    ["8x1 ship", mk()],
    ["8x1 no edge", mk(beamEdgeContrast=0.0)],
    ["8x1 lattice", mk(beamTolerance=100.0, beamEdgeContrast=0.0)],
]
