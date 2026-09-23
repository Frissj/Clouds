import runpy
from pathlib import Path

# beamPushProbe: the push architecture's work count (occupied world cell x BeamTile overlaps, projected into the octahedral beam
# image, binned into per-tile lists, no radiance) beside the shipping frame, which it does not change. Timings are the probe's own
# scopes (push/candidates, clear, mask, project, allocate, scatter); counters are the last scored step's (pushOverlaps etc.).
# Binned at 16 and at 4 beam texels (the current finest tile); pushRect4..32 give the box-only counts at all four sizes either way.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/push_probe.py
#          --steps 2 --motions "walk 2 0.004" "near_park 0 0" "near_walk 2 0.004"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1, cloudSunAncestors=15, beamRepairProbe=False, beamShadowCarry=0,
            cellViews=False, beamPushProbe=False)
# No probe-off arm: the probe writes nothing the frame reads, and each arm's own beamQueries/march scopes are today's dirty cost.
# MEASURED (push1, 4K, one settle, no sort yet), probe ms / overlaps: 16 texels park 0.38 / 952k, walk 0.34 / 951k, sprint 0.46 /
# 961k; 4 texels 2.74-2.82 / ~8.5M (scatter alone 1.5 ms: a visible cell projects ~36 texels, so it lands in ~81 tiles of 4). The
# 4-texel arm is dropped; near-cloud motions ("near ...", sea_motion.py picks the pose) and the sort were added after it.
# MEASURED (push2/push3): the per-tile sort 0.42-0.53 ms (0.78-0.92 near); with the camera in a cloud's occupancy one tile size
# gave 3.3M overlaps and a 62-117 ms probe (near cells spanning thousands of tiles, one thread each). Shells bin each cell at tiles
# that grow with 1 / distance (pushShells, see pushShellOf); "uniform" is the push2/push3 configuration.
PUSH = dict(BASE, beamPushProbe=True, pushTileSize=16)
TESTS = [
    ("shells", dict(PUSH, pushShells=10)),
    ("uniform", dict(PUSH, pushShells=0)),
]
