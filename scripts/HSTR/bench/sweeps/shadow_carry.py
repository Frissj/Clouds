import runpy
from pathlib import Path

# Shadow carry (beamShadowCarry): every dirty ray is still exact-marched, but the blocks a witness policy accepts keep their OLD
# lattice values and guard camera, as a real carry would, so the image prices the carry before anything is skipped. Times are the
# probe's (it marches every ray several times over) and mean nothing; compare the error only.
# Arm value: 1 + signature * 6 + policy. Signatures 0 opacity, 1 +centroid, 2 +quantiles, 3 +sun d50, 4 +sun at all three and
# cache at d50, 5 quantiles at 5%, 6 sun+cache at 5%. Policies 0 one centre, 1 two centres, 2 four, 3 all eight, 4 nearest old
# value, 5 nearest two.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/shadow_carry.py --motions "walk 2 0.004" "sprint 20 0"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1, cloudSunAncestors=15, beamRepairProbe=True, beamShadowCarry=0)


def carry(signature, policy):
    return dict(BASE, beamShadowCarry=1 + signature * 6 + policy)


# Signatures 7-9 are the witness marched in full against its own old radiance: 7 within 0.02, 8 within 0.01, 9 7 with signature 4.
TESTS = [
    ("probe", BASE),
    ("R 2c", carry(7, 1)),
    ("R 4c", carry(7, 2)),
    ("R.01 1c", carry(8, 0)),
    ("R.01 2c", carry(8, 1)),
    ("R+E 2c", carry(9, 1)),
    ("E 1c", carry(4, 0)),
    ("E 2c", carry(4, 1)),
    ("probe again", BASE),
]
