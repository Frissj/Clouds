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
    "minStepVoxels": 2, "maxStepVoxels": 1, "cloudLongitudinalOracle": 0, "cloudOracleCentroid": False,
    "beamTileSize": 4, "beamLevels": 1, "beamTolerance": 0.01, "cloudThinDepth": 0.05, "cloudEmptySkip": True,
    "cloudTightReject": 0, "cloudCostProbe": 0, "cloudSunReuse": 0.0, "cloudMinTransmittance": 0.02,
    "cloudLocalStep": False, "cloudTrapezoid": False, "cloudSlabClamp": True, "beamSegments": 1,
    "beamEdgeContrast": 0.5, "beamGuide": 0, "beamOracle": 0, "beamOracleBar": 0.02, "beamCentreless": False,
    "beamAdaptiveRoot": False, "beamGridDispatch": False, "cloudCameraKernel": False, "beamRefresh": 256,
    "beamDepthTolerance": 0.05, "beamShip": True, "beamRefreshDebug": 0, "beamParallax": 0.0,
    "beamCarryTolerance": 0.0, "beamRefreshBlock": 4, "beamRefreshCentres": 0,
    "cloudCutMargin": 8.0, "beamShipMask": 509, "cloudCutAsync": True,
    # The persistent octahedral beam image (a3d4d427): world-fixed, guarded against translation, residual resolved from the best
    # value of every unit. 4K, GPU ms park / look / flick / walk / sprint 0.49 / 0.51 / 0.72 / 0.92 / 0.56, 0.137% of pixels
    # over 0.02 against the exact march - better than beamScreenResidual's 0.156% at 3.4-7.6 ms. Before it this was the anchored
    # screen-space rectangle at beamTolerance 0.05 and no refresh; sweeps that want that set beamOct / beamRefFrame off.
    "beamRefFrame": True, "beamOct": True, "beamOctFull": False, "beamOctScale": 1.0, "beamGuard": True,
    "beamScreenResidual": False, "beamSparse": False, "beamSparseCut": False, "beamQueue": False,
}


def mk(**overrides):
    """The shipped beam view with overrides."""
    return dict(BEAM, **overrides)
