from sea_config import mk

# beamOrderProbe: why longest-first (order1) bought nothing. Over the rays the dirty query marches on each scored frame: steps,
# the longest ray, a log2 histogram, how often the stored step count is missing and how far off it is, and its hit rate on
# "long" rays at the threshold. The query's time comes from the probe-off anchor. Questions: is the query's tail one long ray's
# latency (longest ray vs query time)? Does the stored count predict?
#
# MEASURED (oprobe1, 4K; the probe itself costs the query +0.03-0.08 ms):
#   rays/frame 91-160k walk, 206-228k jog, ~240k sprint; mean steps 14-15 / 11-12 / 10; longest 94-99 / 91-94 / 90-94;
#   0-1 step rays ~20% walk, ~43% sprint. Thresholds of 150 and 400 never reorder anything.
#   The stored count is always present; relative error 6% walk, 11% jog, 20% sprint; at threshold 50 it finds and is right about
#   long rays ~66% of the time at walk, ~50% jog, ~35% sprint (280-770 blocks to the front).
#   query ms listed / probe / order 50 + probe: walk 0.558 / 0.586 / 0.530, jog 0.755 / 0.812 / 0.808, sprint 0.797 / 0.878 / 0.910.
#   So a fixed threshold is 3-20x off the distribution. A threshold-free sort (order2, dirty_order.py) never beat the listed order
#   either, and the ordering was removed.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/order_probe.py --steps 3
#          --motions "walk 2 0.004" "jog 8 0.002" "sprint 20 0"
TESTS = [
    ("listed", mk()),
    ("probe 50", mk(beamOrderProbe=50)),
]
