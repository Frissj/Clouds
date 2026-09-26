from sea_config import mk

# REMOVED: longest-first dirty query, beamDirtyLongSteps (order1) and then beamDirtyOrderClasses (order2), a GPU counting sort of
# the listed blocks by the steps each block's longest ray took when last marched. Never beat the listed order; the results are
# with the removal note in HSTRCloud.cs.slang (after writeBeamDirtyArgs). Errors identical in every arm.
#   order2 ms wall listed / classes 2 / 4 / 8 / 16 / 32 / listed again:
#     walk   1.93 / 1.92 / 1.91 / 1.90 / 1.91 / 1.90 / 1.89
#     jog    2.43 / 2.62 / 2.55 / 2.49 / 2.46 / 2.45 / 2.41
#     sprint 2.26 / 2.55 / 2.50 / 2.40 / 2.36 / 2.36 / 2.26
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/dirty_order.py --steps 1
#          --motions "walk 2 0.004" "jog 8 0.002" "sprint 20 0"
TESTS = [
    ("listed", mk()),
]
