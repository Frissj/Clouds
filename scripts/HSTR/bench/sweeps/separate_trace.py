from sea_config import mk

# The separate dirty passes alone, for a GPU Trace to set against fused_trace.py's (same frames, same motion).
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/separate_trace.py --steps 0
#          --motions "sprint 20 0" --ngfx 40 41
TESTS = [("separate", mk())]
