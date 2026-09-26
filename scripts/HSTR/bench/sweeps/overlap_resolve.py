from sea_config import mk

# beamOverlapResolve: the resolve's pixel pass launched behind the dirty unit march without barriers, to fill the march's draining
# tail (ngfx8: units fills its SMs at launch and then only drains; the last third runs a few long rays on an idle GPU). The warp
# field moves before the unit march, so walk's warp reads the guard depths the query left (the unit march's minima come a frame
# later): the gate decides. Sprint has the warp off, so its arm is the pure overlap. Anchor repeated for drift.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/overlap_resolve.py --steps 3
#          --motions "walk 2 0.004" "jog 8 0.002" "sprint 20 0"
# MEASURED (overlap1; ms, resolve/pixels ms, >0.02): no change anywhere, errors bit-identical (walk included: the warp from the
# query's depths changes nothing).
#   walk   default 2.07 0.37 0.886% | overlap 2.08 0.35 0.886% | again 2.03 0.35
#   jog    default 2.53 0.33 0.829% | overlap 2.55 0.33 0.829% | again 2.54
#   sprint default 2.39        0.473% | overlap 2.41      0.473% | again 2.40
# So the pixel pass still waits for the unit march: see overlap_trace.py.
# ngfx9 (GPU Trace of the overlap arm): 36 ResourceBarrier calls -> 1 between the two dispatches, a UAV -> SRV transition of the
# beam pixels bound again as hstrBeamPixelPrev. Unbound in the overlapped pixel pass:
# MEASURED (overlap2; ms, >0.02): walk 2.08 / 1.89 / 2.03, jog 2.52 / 2.34 / 2.55, sprint 2.40 / 2.21 / 2.39 (default / overlap /
# again), errors identical in every motion (0.886% / 0.829% / 0.473%). DEFAULTED ON; the arms now set it explicitly.
TESTS = [
    ("serial", mk(beamOverlapResolve=False)),
    ("overlap", mk(beamOverlapResolve=True)),
    ("serial again", mk(beamOverlapResolve=False)),
]
