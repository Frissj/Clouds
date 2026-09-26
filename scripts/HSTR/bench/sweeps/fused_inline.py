from sea_config import mk

# beamFusedInline: the separate dirty query (its own program, not persistent) tests the tiles of its own wave's blocks once past
# its march, through the fused chain's per-tile reader join - no queue, no tile consumer, no dirty tile pass. Waves that finish
# early test tiles under the query's tail. Same answers expected: errors, tiles tested and units listed must match separate.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/fused_inline.py --steps 1
#          --motions "walk 2 0.004" "jog 8 0.002" "sprint 20 0"
TESTS = [
    ("separate", mk()),
    ("inline", mk(beamDirtyFused=True, beamFusedInline=True)),
    # DIAGNOSTIC: the inline program with its tile half skipped at run time (nothing tested or listed, so wrong images): its
    # march/fused against separate's query + rebuild is the cost of the program itself (registers, coherent views).
    ("inline no tiles", mk(beamDirtyFused=True, beamFusedInline=True, beamFusedStage=7)),
    ("separate again", mk()),
]
