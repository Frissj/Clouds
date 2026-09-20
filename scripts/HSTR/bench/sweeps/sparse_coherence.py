from sea_config import mk

# Sparse basis-ray diagnostic - NOT the final beam architecture.
#
# The current migration path still evaluates beam basis positions with marchBeam(). This is temporary. The final renderer must
# construct BeamTile coefficients from projected HSTR transport nodes and reserve density marching for sparse unresolved residuals.
#
# This experiment demonstrates why sparse independent ray marching is a poor endpoint: at 4K near, parked, 2026-09-20, against the
# same anchor, the dense 4x1 lattice marches 779k coherent rays in 7.1 ms (9.1 ns/query) while the adaptive sparse hierarchy marches
# only 286k but takes 10.1 ms (35 ns/query). Atomic candidate compaction destroys spatial and wave coherence, so fewer rays are
# substantially more expensive individually. The start level isolates it: level 3 emits every root's full 8x8 grid, so the candidate
# list is dense again - 1,046k queries in 8.9 ms, 8.5 ns each, the dense lattice's cost with 3.7x the rays.
#
# Do NOT respond by turning this into a long optimization effort for marchBeam() scheduling. Spatial ordering may be used as a
# temporary migration improvement or later for the residual queue, but normal beam construction is intended to replace these
# independent basis marches with BeamTile x projected-transport-node work.
#
# The important result is therefore: reducing ray count alone is insufficient; independent camera rays remain the wrong unit of work.
COMMON = dict(
    beamTileSize=32,
    beamLevels=4,
    beamSegments=1,
    beamTemporal=False,
    beamRefresh=0,
    beamQueue=False,
    beamGridDispatch=False,
    beamSparseCut=False,
)

# The anchor keeps the shipped dense lattice (beamTileSize 4, beamLevels 1 from BEAM), so it stays comparable with every earlier
# run of this view. Every arm sets every key any other arm sets, because ab_test.py puts nothing back between tests.
TESTS = [
    ("shipping 4x1", mk(beamSparse=False, beamSparseCut=False, beamSparseMinLevel=2, beamQueue=False)),
    ("sparse start 2", mk(beamSparse=True, beamSparseMinLevel=2, **COMMON)),
    ("sparse start 1", mk(beamSparse=True, beamSparseMinLevel=1, **COMMON)),
    ("sparse start 3", mk(beamSparse=True, beamSparseMinLevel=3, **COMMON)),
]
