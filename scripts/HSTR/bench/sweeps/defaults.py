from sea_config import mk

# The shipped frame as sea_config.BEAM defines it, alone: run after changing the defaults, to confirm they reproduce.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/defaults.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
TESTS = [("default", mk())]
