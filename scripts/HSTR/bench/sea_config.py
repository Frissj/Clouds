# Shared settings of the cloud sea benchmarks (ab_test.py and sea_motion.py through run_sea.py).
#
# REFERENCE is the exact frame every test is compared against: the per-pixel march at one-voxel steps. It sets every property any
# test sets, so the views settle the same way whatever the tests are. mk() is the beam view as currently shipped, with overrides.

REFERENCE = {
    "debugView": 8, "hstComponents": 15, "cloudSunCache": True, "minStepVoxels": 1, "maxStepVoxels": 1,
    "cloudLongitudinalOracle": 0, "cloudOracleCentroid": False, "cloudZeroSkip": 8, "beamTemporal": False,
    "beamTileSize": 4, "beamLevels": 1, "beamTolerance": 0.05, "cloudThinDepth": 0.0, "cloudEmptySkip": True,
    "cloudTightReject": 0, "cloudCostProbe": 0, "cloudSunReuse": 0.0, "cloudMinTransmittance": 0.001,
    "cloudLocalStep": False, "cloudTrapezoid": False, "cloudSlabClamp": True, "beamSegments": 1,
    "beamEdgeContrast": 0.05, "beamGuide": 0, "beamOracle": 0, "beamOracleBar": 0.02, "beamCentreless": False,
    "beamAdaptiveRoot": False, "beamGridDispatch": False, "cloudCameraKernel": False, "beamRefresh": 0,
    "beamShip": False,
}

BEAM = {
    "debugView": 9, "hstComponents": 15, "beamTemporal": False, "cloudSunCache": True, "cloudZeroSkip": 8,
    "minStepVoxels": 2, "maxStepVoxels": 2, "cloudLongitudinalOracle": 0, "cloudOracleCentroid": False,
    "beamTileSize": 4, "beamLevels": 1, "beamTolerance": 0.05, "cloudThinDepth": 0.05, "cloudEmptySkip": True,
    "cloudTightReject": 0, "cloudCostProbe": 0, "cloudSunReuse": 0.0, "cloudMinTransmittance": 0.02,
    "cloudLocalStep": False, "cloudTrapezoid": False, "cloudSlabClamp": True, "beamSegments": 1,
    "beamEdgeContrast": 0.5, "beamGuide": 0, "beamOracle": 0, "beamOracleBar": 0.02, "beamCentreless": False,
    "beamAdaptiveRoot": False, "beamGridDispatch": False, "cloudCameraKernel": False, "beamRefresh": 0,
    "beamDepthTolerance": 0.05, "beamShip": False, "beamRefreshDebug": 0, "beamParallax": 0.0,
    "beamCarryTolerance": 0.0,
}


def mk(**overrides):
    """The shipped beam view with overrides."""
    return dict(BEAM, **overrides)
