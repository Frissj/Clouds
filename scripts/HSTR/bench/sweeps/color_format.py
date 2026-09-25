from sea_config import mk

# colorFormat: the resolve writes 4K colour at 16 bytes a pixel (RGBA32Float), 133 MB a frame, and the tone mapper reads it back.
# profA1 (walk, Nsight): the resolve's pixel pass is write-bandwidth bound (DRAM write 62%, SM issue 41%), and the tone mapper
# reads at 56% of DRAM. leanhalf1 took RGBA16Float at 5.8 ms and saw the resolve 0.47 -> 0.41 but not the frame; the resolve is
# now ~27% of a 2.1 ms frame. The harness logs the tone mapper too (toneMapper), since it pays the other half.
# R11G11B10Float is 4 bytes but a 6/6/5-bit mantissa (up to ~1.6% relative rounding). Caveat: the scored reference frame is
# written through the same output, so it is quantized too; the arm shows cost, and its error is a lower bound until checked
# against a full-precision reference.
# Each format change reallocates the 133 MB output (a graph recompile), hence the repeated anchor.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/color_format.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
TESTS = [
    ("rgba32f", mk(colorFormat=0)),
    ("rgba16f", mk(colorFormat=1)),
    ("r11g11b10f", mk(colorFormat=2)),
    ("rgba32f again", mk(colorFormat=0)),
]
# MEASURED (colorfmt1; HSTRCloud ms / tone mapper ms, over 0.02, p99.9):
#   walk   32F 2.13 / 0.39 0.939% 0.061 | 16F 2.28* / 0.08 0.938% | 11-11-10 2.11 / 0.06 0.950% | 32F again 2.14 / 0.43
#   sprint 32F 2.90 / 0.39 0.283% 0.030 | 16F 2.85 / 0.07 0.282% | 11-11-10 2.85 / 0.06 0.278% | 32F again 2.89 / 0.39
#   * query and units +12% together in that arm alone, which the format cannot touch: drift. Resolve pixels 0.37 -> 0.36.
# SHIPPED: colorFormat 1 (RGBA16Float) - the tone mapper's cost drops by 0.31 ms, the frame's errors do not move.
