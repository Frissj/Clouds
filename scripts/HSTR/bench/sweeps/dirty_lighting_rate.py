import runpy
from pathlib import Path

# The dirty passes' lighting rate: the march samples density every step, but the world cache (lightingStride contributing steps)
# and the single scatter's sun depth (cloudSunReuse voxels) are held between evaluations. Extinction alone is ~half the march
# (march_components.py: walk 12.9 ms all components, 6.6 extinction only), so this prices the other half with density untouched.
# RESULT (2026-09-22): no - see lightingStride in HSTRCloudTypes.slang. The cache hold saves under 0.1 ms; the sun hold saves
# ~10% of the walk for 30x the pixels over 0.02.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/dirty_lighting_rate.py --motions "walk 2 0.004" "sprint 20 0"
OCT = runpy.run_path(str(Path(__file__).with_name("persistent_residual.py")))["OCT_T001"]
BASE = dict(OCT, beamGuardParallax=1.0, beamDirtySegments=1, cloudSunAncestors=15)
TESTS = [
    ("base", BASE),
    ("stride 2", dict(BASE, lightingStride=2)),
    ("stride 4", dict(BASE, lightingStride=4)),
    ("sun 2", dict(BASE, cloudSunReuse=2.0)),
    ("sun 4", dict(BASE, cloudSunReuse=4.0)),
    ("both 2", dict(BASE, lightingStride=2, cloudSunReuse=2.0)),
    ("both 4", dict(BASE, lightingStride=4, cloudSunReuse=4.0)),
    ("base again", BASE),
]
