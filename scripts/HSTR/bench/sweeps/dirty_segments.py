from sea_config import mk

# Split-ray marching for the dirty query.
#
# With the certificate hierarchy in place, a walking octahedral frame spends 0.552 ms of its 0.59 ms beamDirty in the query: the
# ~70 blocks a frame that the turn brings on screen are never-visited directions, and all ~560 of their rays march. That is real
# information, but 560 rays are a few waves on a GPU of thousands of lanes, so the pass costs the latency of its longest ray.
#
# Two changes are measured here. The corner and centre queries now go in ONE dispatch instead of two back to back - every arm has
# that. And beamDirtySegments splits each ray across that many threads, each marching an equal slice of the ray's volume interval
# from unit transmittance, composed front to back by the first. It was rejected once as beamSegments on the million-ray lattice,
# where slices losing early termination is pure extra work; here latency is the whole cost, so it should be the opposite.
#
# Read beamDirty/query, not the frame total: the rest of the frame is the same in every arm. And quality must not move at all -
# the composition is exact, so any change is a bug.
COMMON = dict(
    beamTileSize=4,
    beamLevels=1,
    beamSegments=1,
    beamTemporal=False,
    beamSparse=False,
    beamSparseCut=False,
    beamQueue=False,
    beamGridDispatch=False,
    beamRefreshBlock=4,
    beamCarryTolerance=0.0,
    beamDepthTolerance=0.05,
    beamRefreshDebug=0,
    beamRefFrame=True,
    beamRefresh=256,
    beamGuard=True,
    beamTolerance=0.05,
    beamOct=True,
    beamOctFull=False,
    beamOctScale=1.0,
)

TESTS = [(f"oct seg {s}", mk(**COMMON, beamDirtySegments=s)) for s in (1, 2, 4, 8, 16)]
