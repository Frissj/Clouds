from sea_config import mk

# Is the sea's per-tile sun page rebuild bound by the voxels it touches, or by the fixed cost of the dispatch it takes to touch
# them? dispatchResidualPages binds the whole renderer and issues one dispatch PER TILE, so cloudSunTilesPerFrame scales the work
# but not the overhead. If fineSun barely moves between 1 tile and 64, the throttle cannot help and the loop needs hoisting or
# batching instead. Everything else is held at the cheapest basis configuration so residualPages dominates the frame.
COMMON = dict(
    beamTileSize=4, beamLevels=1, beamSegments=1, beamTemporal=False, beamSparse=False, beamSparseCut=False,
    beamQueue=False, beamGridDispatch=False, beamRefreshBlock=4, beamCarryTolerance=0.0, beamDepthTolerance=0.05,
    beamRefreshDebug=0, beamRefFrame=True, beamRefMargin=0.15, beamRefresh=256,
)

TESTS = [
    ("sun tiles 1", mk(**COMMON, cloudSunTilesPerFrame=1)),
    ("sun tiles 8", mk(**COMMON, cloudSunTilesPerFrame=8)),
    ("sun tiles 64", mk(**COMMON, cloudSunTilesPerFrame=64)),
]
