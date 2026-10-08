import runpy
from pathlib import Path

# The lean beam march (HSTR_SHIP bit 1024, marchBeamLean) against the old shipping march (mask 509): the same samples and values
# with less state live across a step, because the Nsight capture of the shipping walk frame showed the dirty passes
# occupancy-bound (22% / 32% warps in flight, 68% / 63% of warp slots unallocated on active SMs, DRAM read 31% / 22%). One
# controlled difference; the old march is repeated last as the drift anchor. The error columns must match.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_march.py --steps 1
#          --motions "walk 2 0.004" "sprint 20 0"
# MEASURED (lean1, 4K): walk 7.45 / 7.42 -> 6.63 ms, sprint 9.18 / 9.18 -> 8.18 ms, errors identical. Now the default (1533).
BASE = runpy.run_path(str(Path(__file__).with_name("march_cost_split.py")))["BASE"]
LEAN = dict(BASE, beamShipMask=509 | 1024)
OLD = dict(BASE, beamShipMask=509)
TESTS = [
    ("ship", OLD),
    ("lean", LEAN),
    ("ship again", OLD),
]

# The lean march's light terms switched off one at a time (march_cost_split's arms on the lean march). Images wrong on purpose
# except "lean", and so is the dirty set: the arms march 1.9-3.5% of pixels, so read cost per marched ray, not raw ms.
# MEASURED (leansplit1, walk, dirty query + units ms / marched): lean 5.50 / 3.5%, no sun 3.88 / 3.2%, no cache 4.42 / 2.8%,
# density only 2.56 / 1.9% - per marched ray the density chain is most of it, the sun about a quarter, the cache little.
SPLIT = [
    ("lean", LEAN),
    ("lean no sun", dict(LEAN, hstComponents=1 | 4 | 8)),
    ("lean no cache", dict(LEAN, hstComponents=1 | 2)),
    ("lean density only", dict(LEAN, hstComponents=1)),
    ("lean again", LEAN),
]

# The same split on the current default march (BASE carries sea_config's mask), for DRAM traffic per marched ray under
# run_sea.py --nsys --nsys-set ad10x-gfxt: which lookups the frame's VRAM reads come from.
SPLIT_SHIP = [
    ("all", BASE),
    ("no sun", dict(BASE, hstComponents=1 | 4 | 8)),
    ("no cache", dict(BASE, hstComponents=1 | 2)),
    ("density only", dict(BASE, hstComponents=1)),
]

# The resolved level pages (HSTR_SHIP bit 2048, cloudAssetBrickAtFlat): one word per sample instead of the page, the octant test
# and the parent climb; with it, the per-frame resolved sun slots (resolveCloudSunSlots). Exact: the error columns must match.
# MEASURED (leanflat1, pages alone): walk 6.56 / 6.52 -> 6.63 ms, no gain - the climb was not what the march waits on.
# MEASURED (leanflat2, pages + sun slots): walk 6.61 / 6.62 -> 6.06 ms (query 2.60 -> 2.24, units 2.86 -> 2.45), sprint 8.11 /
# 8.22 -> 7.55 ms, errors identical. The sun's key search (16 words a level, up to 15 ancestors) was. Now the default (3581).
FLAT_MASK = 509 | 1024 | 2048
FLAT = [
    ("lean", LEAN),
    ("lean flat", dict(LEAN, beamShipMask=FLAT_MASK)),
    ("lean again", LEAN),
]

# Zero-majorant steps (40% of the lean march's steps, leancount2) crossing whole zero 4-voxel majorant blocks (HSTR_SHIP bit
# 8192, majorantZeroExitFine) instead of one voxel each where the 16-voxel block is not all zero. Exact in what it skips; the later
# samples' phase moves, as with the 16-voxel skip, so the errors are compared, not required identical.
# MEASURED (leanzero1): walk 6.04 / 6.06 -> 5.97 ms, errors identical (sprint the same). Air steps are cheap; kept (11773).
ZERO = [
    ("flat", dict(BASE, beamShipMask=FLAT_MASK)),
    ("flat zero4", dict(BASE, beamShipMask=FLAT_MASK | 8192)),
    ("flat again", dict(BASE, beamShipMask=FLAT_MASK)),
]

# Transmittance-scaled steps (HSTR_SHIP bit 16384: up to 2x, 32768: up to 4x, at 1 / sqrt(T)): fewer lit samples where the pixel
# can no longer see much. An approximation - judged against the path-traced reference.
SHIP_MASK = FLAT_MASK | 8192
TSTEP = [
    ("ship", dict(BASE, beamShipMask=SHIP_MASK)),
    ("tstep 2x", dict(BASE, beamShipMask=SHIP_MASK | 16384)),
    ("tstep 4x", dict(BASE, beamShipMask=SHIP_MASK | 32768)),
    ("ship again", dict(BASE, beamShipMask=SHIP_MASK)),
]

# MEASURED (leantstep1, 4K): walk 5.99 / 5.99 -> 5.44 (2x) / 5.38 (4x) ms for 0.137% -> 0.146% / 0.143% over 0.02 (p99.9 the same,
# max 0.171 -> 0.135); sprint 7.44 / 7.48 -> 6.83 / 6.78 ms for 0.028% -> 0.028% / 0.029%. Opt-in: it trades quality.

# Dispatch order of the dirty passes' waves (HSTR_SHIP bit 262144, shuffledDirtyThread): the same rays per wave, the waves in a
# scrambled order. Exact: the error columns must match. Answers whether the brick refetch (Nsight: ~4x the frame's unique brick
# bytes) is an ordering problem - if scrambling costs little, the listed order already gets what reordering could.
ORDER = [
    ("listed", dict(BASE, beamShipMask=SHIP_MASK)),
    ("shuffled", dict(BASE, beamShipMask=SHIP_MASK | 262144)),
    ("listed again", dict(BASE, beamShipMask=SHIP_MASK)),
]
# MEASURED and REMOVED (leanorder1, 4K, errors identical): walk 6.03 / 6.04 -> 6.63 ms shuffled, sprint 7.51 / 7.55 -> 8.22. The
# worst order costs 10%: the listed order already has the cross-wave reuse, so reordering is not a lever. Bit 262144 is gone.

# Deferred lighting (HSTR_SHIP bits 262144 / 524288: K = 1 / 2 / 4, marchBeamLean): a lit sample's light is parked and the wave
# evaluates the parked samples together once pending * K >= lanes still in the loop. leancount3 (walk): the loop issues 81.4M lane
# steps for 61.5M steps (76%), but only 28% of steps are lit, so the light body ran with ~7 of 32 lanes. Exact: errors must match.
DEFER = [
    ("ship", dict(BASE, beamShipMask=SHIP_MASK)),
    ("defer all", dict(BASE, beamShipMask=SHIP_MASK | 262144)),
    ("defer half", dict(BASE, beamShipMask=SHIP_MASK | 524288)),
    ("defer quarter", dict(BASE, beamShipMask=SHIP_MASK | 786432)),
    ("ship again", dict(BASE, beamShipMask=SHIP_MASK)),
]
# MEASURED and REMOVED (leandefer1, 4K, errors identical): walk 6.01 / 6.03 -> 9.34 / 6.57 / 6.37 ms (all / half / quarter), sprint
# 7.47 / 7.50 -> 11.53 / 8.31 / 8.03. The light is latency-bound, not lane-bound. Bits 262144 / 524288 are gone.

# The lit sample's memory chain shortened, exactly: the sun's far field fetched beside the bake instead of after its opaque test
# (HSTR_SHIP bit 1048576), and the cache fetched ahead of the sun chain instead of behind it (bit 2097152). Errors must match.
CHAIN = [
    ("ship", dict(BASE, beamShipMask=SHIP_MASK)),
    ("far beside", dict(BASE, beamShipMask=SHIP_MASK | 1048576)),
    ("cache first", dict(BASE, beamShipMask=SHIP_MASK | 2097152)),
    ("both", dict(BASE, beamShipMask=SHIP_MASK | 1048576 | 2097152)),
    ("ship again", dict(BASE, beamShipMask=SHIP_MASK)),
]
# MEASURED (leanchain1, 4K, errors identical): walk 6.02 / 6.02 -> far beside 5.97, cache first 6.53, both 6.51 ms; sprint 7.50 /
# 7.51 -> 7.46 / 8.09 / 8.05. The far field beside the bake is now unconditional (leanSunDepth); cache first removed. The bits
# are gone, so these arms no longer vary.

# The sun slot resolve (resolveCloudSunSlots) only when something it reads changed, against every frame (cloudSunResolveAlways).
# Exact: the errors must match. Run parked motion and a moving sun too, since sun scheduling is what re-dirties it.
RESOLVE = [
    ("always", dict(BASE, cloudSunResolveAlways=True)),
    ("on change", dict(BASE, cloudSunResolveAlways=False)),
    ("always again", dict(BASE, cloudSunResolveAlways=True)),
]
# MEASURED (leanresolve1, 4K, errors identical): walk 5.97 / 5.95 -> 5.74 ms, sprint 7.41 / 7.43 -> 7.20. On change is the default.

# The frame written at RGBA16Float (colorHalf) instead of RGBA32Float: the resolve is write-bound (Nsight walk: VRAM 64%, reads
# 9%) and writes 133 MB a 4K frame. A precision change (half keeps ~3 significant digits): judged against the reference.
HALF = [
    ("float", dict(BASE, colorHalf=False)),
    ("half", dict(BASE, colorHalf=True)),
    ("float again", dict(BASE, colorHalf=False)),
]
# Direct-span ceiling (spanProbe): the dirty march records its rays as runs of samples in one brick (or the proxy), and
# evaluateSpans integrates the same samples from the spans alone - no majorant, instance or page lookup, no empty or majorant-zero
# steps - with span production free. Density and single sun only (hstComponents 1 | 2) in every arm, so the march arms time the
# same work: "spans" (timed) against the march arms' query + units; spanMismatch* / spanMaxD* check it is the same integral.
# Gate A: the span march itself <= ~0.7 ms at walk; gate B, once spans come from the projection: build + march <= ~1.2-1.3 ms.
DSUN = dict(BASE, hstComponents=1 | 2, spanProbe=False)
SPAN = [
    ("march dsun", DSUN),
    ("span probe", dict(DSUN, spanProbe=True)),
    ("march dsun again", DSUN),
]
# The span evaluator with batched independent loads (spanGather 1: eight samples' loads before their composite; 2: and the next
# span's header ahead), against the plain evaluator in the same run. Every arm sets spanProbe and spanGather.
GATHER = [
    ("march dsun", dict(DSUN, spanProbe=False, spanGather=0)),
    ("span plain", dict(DSUN, spanProbe=True, spanGather=0)),
    ("span gather8", dict(DSUN, spanProbe=True, spanGather=1)),
    ("span gather8 prefetch", dict(DSUN, spanProbe=True, spanGather=2)),
    ("span gather8 sanity (no sun, must mismatch)", dict(DSUN, spanProbe=True, spanGather=3)),
    ("span headers only (floor, no samples)", dict(DSUN, spanProbe=True, spanGather=4)),
    ("span plain again", dict(DSUN, spanProbe=True, spanGather=0)),
    ("march dsun again", dict(DSUN, spanProbe=False, spanGather=0)),
]
# MEASURED (spangather6, HSTR_RES=1920x1080, --steps 1; at 4K the probe pushes VRAM to 11.7 of 12.3 GB and every pass slows 4-6x):
# march query + units 0.73 + 0.70 / 0.69 + 0.68 ms; spans plain 0.613, gather8 0.620, prefetch 0.613, sanity 0.613 (light
# mismatches 6.1k -> 101.8k), headers only 0.257, plain again 0.613. Batched loads buy nothing on the real evaluator: samples
# 0.356 ms for 1.87M (~5.2 G/s). See evaluateSpanRayGather.
# Where the span evaluator's sample time goes, and whether spans can share transfers. spanEval: one thread a span (each from
# T = 1, composed per ray afterwards: "spans" + "spanCompose"), listed in ray order (load balance alone), grouped by brick
# (cache locality) or scrambled (sanity: must be slower); spanLoop 1 decodes the brick's atlas addresses once a span, 2 skips the
# sun fetches (the density / sun split - its light is wrong by design). spanShare (untimed, every arm): spans keyed by brick, entry
# cell, direction class and sun class, each served its key's first (L, T) - spanShareBad* against spanBad* (the exact evaluator,
# same 0.02 test) is the quality of a transfer cache, spanShareBin* its reuse. spanTimedMismatch: per-span results != per-ray.
# Stop rules: an order or loop arm must cut "spans" (+ compose) by >= 30% of the samples' ~0.356 ms; sharing must leave
# spanShareBad* within ~2x spanBad*.
SPANORDER = [
    ("span per ray", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=True)),
    ("span per span ray order", dict(DSUN, spanProbe=True, spanGather=0, spanEval=1, spanLoop=0, spanShare=True)),
    ("span per span brick order", dict(DSUN, spanProbe=True, spanGather=0, spanEval=2, spanLoop=0, spanShare=True)),
    ("span per span scrambled (sanity, must slow)", dict(DSUN, spanProbe=True, spanGather=0, spanEval=3, spanLoop=0, spanShare=True)),
    ("span resolved loop", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1, spanShare=True)),
    ("span resolved loop brick order", dict(DSUN, spanProbe=True, spanGather=0, spanEval=2, spanLoop=1, spanShare=True)),
    ("span no sun fetch (cost split)", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=2, spanShare=True)),
    ("span per ray again", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=True)),
]
# MEASURED (spanorder1, HSTR_RES=1920x1080, --steps 1): spans [+ compose] per ray 0.60 / 0.61, per span ray order 0.47 + 0.11,
# brick order 0.56 + 0.12, scrambled 1.52 + 0.11 (sanity), resolved loop 0.60, resolved + brick 0.61 + 0.14, no sun fetch 0.39 ms.
# Every order and loop arm misses the 30% rule; the sun fetches are a third of the evaluator. Sharing: 1.52 spans a key, bad rays
# 5.7k -> 20.3k (T), 4.8k -> 52.1k (light) - fails both rules. See evaluateSpan.
# Span-level sun: the anchor's spanLitBin* / spanLitSaveHold / spanLitSaveLinear are the ceiling (sun evaluations one held or two
# interpolated per span would leave out); spanLoop 4 interpolates the sun depth linearly between a span's first and last lit
# samples, 8 holds the first one's (quality oracles - their time includes a first pass and means nothing). spanBad* / spanMismatchL
# against the anchor's are the quality; no sun fetch (2) is the sanity arm (light must break). Stop rules: linear must leave out
# >= 30% of the anchor's sun evaluations and keep spanBadL within 1.2x the anchor's.
SUNSPAN = [
    ("span per ray", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
    ("span sun linear", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=4, spanShare=False)),
    ("span sun held", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=8, spanShare=False)),
    ("span no sun fetch (sanity)", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=2, spanShare=False)),
    ("span per ray again", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
]
# MEASURED (sunspan1, HSTR_RES=1920x1080, --steps 1): VACUOUS - every recorded span holds 0 or 1 lit sample (spanLitBin0 409,293,
# spanLitBin1 568,812 = spanBaked, nothing above), so linear / held left out nothing (spanBaked 568,812 in every arm, spanBad*
# unchanged at 5,730 / 4,753). A recorded span ends where the step changes and the step follows the transmittance, so a span is a
# run of equal steps, not a brick crossing. The crossing's lit samples are counted in the recording instead (spanCross*, CROSS).
# Sanity: no sun fetch, spans 0.39 ms and light mismatches 6,059 -> 121,811.
CROSS = [
    ("span per ray", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
    ("span per ray again", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
]
# The proxy's share of the dirty rays: spanLoop 16 skips the proxy spans, 32 the brick spans (cost split only - each changes the
# light, so both are their own sanity arms); spanProxy* (the anchor's counting pass) say what the proxy samples are and whether
# they matter. Question: is the proxy (smooth, closed-form integrable per cell, cheap to reuse) where the evaluator's time goes?
PROXY = [
    ("span per ray", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
    ("span bricks only (proxy skipped)", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=16, spanShare=False)),
    ("span proxy only (bricks skipped)", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=32, spanShare=False)),
    ("span per ray again", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
]
# Two ceilings in one run. RT + block walking: spanRestart* (the recording) counts per dirty ray the runs of brick samples an RT
# query would have to find, strict (any gap, proxy samples included) and loose (gaps of up to 16 steps walked); stop if the mean
# is much above ~2-2.5 a ray (rtprobe1: ~0.13 ms a query over the walk's dirty rays at 4K). Density + bake in one fetch: its
# ceiling is what skipping the bake fetch saves (128); 64 skips the far-field fetch instead, 2 both (the sun split, against
# spanorder1's 0.39); stop if 128 saves < ~0.1 ms of the 0.35 ms of brick samples.
RESTART = [
    ("span per ray", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
    ("span no bake fetch", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=128, spanShare=False)),
    ("span no far-field fetch", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=64, spanShare=False)),
    ("span no sun fetch", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=2, spanShare=False)),
    ("span per ray again", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
]
# Density + bake in one fetch, the wasteful oracle: spanLoop 256 reads every packable span (the bake resolved at the brick's own
# level: spanPackedSamples) from an RG texture laid out like the sun atlas, density beside each bake, packed untimed every frame.
# Against the plain evaluator and its ceiling (no bake fetch, 128). Light mismatches against the anchor's say how much the one
# shared clamp moves the sun. Stop rule: spans must fall by >= ~0.1 ms (half the bake's 0.22).
PACKED = [
    ("span per ray", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
    ("span packed density + bake", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=256, spanShare=False)),
    ("span packed, density's clamp", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=256 | 512, spanShare=False)),
    ("span no bake fetch (ceiling)", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=128, spanShare=False)),
    ("span packed again", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=256, spanShare=False)),
    ("span per ray again", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=0, spanShare=False)),
]
# The working set of a brick sample: the resolved loop (1, the same integral as per ray, spanorder1) with every density and sun
# slot folded into 16^3 slots (1024: ~4 MB an atlas, fits the 48 MB L2) or 4^3 (2048: ~64 KB, about one SM's L1). Same
# instructions, wrong image (their light / T mismatches against the anchor's are the sanity check). The bake fetch alone (128) and
# the brick-skipped floor (32, the headers and proxy) in the same run. Brick-sample time = arm - floor.
# Stop rule (set before running): if the 16^3 arm's brick-sample time is more than half the anchor's, the working set is not what
# makes a real brick sample slow, and a compact (cache-resident) representation is closed. At a third or less, it is the lever.
WORKSET = [
    ("span resolved loop", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1, spanShare=False)),
    ("span working set 16^3", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1 | 1024, spanShare=False)),
    ("span working set 4^3", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1 | 2048, spanShare=False)),
    ("span no bake fetch", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1 | 128, spanShare=False)),
    ("span floor (bricks skipped)", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1 | 32, spanShare=False)),
    ("span resolved loop again", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1, spanShare=False)),
]
# The cheapest smaller working set: the same bricks without their aprons, 8^3 a slot, aligned to 8 (an untimed copy every frame).
# 4096 reads the density there (exact: its reads never leave a slot's core, so T misses must equal the anchor's), 8192 the sun too
# (at the density's clamp, so the light moves a little at brick faces). The 16^3 fold is the ceiling, the floor as before.
# Stop rule (set before running): brick-sample time (arm - floor) down >= 1.5x with both compact -> the layout alone is worth
# building into the march; under 1.15x -> the working set needs real compression (codebook / block compression), not a layout.
COMPACT = [
    ("span resolved loop", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1, spanShare=False)),
    ("span compact density", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1 | 4096, spanShare=False)),
    ("span compact density + sun", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1 | 4096 | 8192, spanShare=False)),
    ("span working set 16^3", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1 | 1024, spanShare=False)),
    ("span floor (bricks skipped)", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1 | 32, spanShare=False)),
    ("span resolved loop again", dict(DSUN, spanProbe=True, spanGather=0, spanEval=0, spanLoop=1, spanShare=False)),
]
# MEASURED (compact1, HSTR_RES=1920x1080, --steps 1): spans 0.613 / again 0.612, compact density 0.561, compact density + sun
# 0.494, 16^3 0.333, floor 0.240 ms. Brick samples (arm - floor) 0.373 -> 0.321 (1.16x, density alone) -> 0.254 (1.47x, both)
# -> 0.093 (16^3). Density compact is nearly exact (T bad 5,730 -> 5,729, T mismatches 10,358 -> 12,282 from the other texture
# size's sub-texel rounding); the sun at the density's clamp is not (light bad 4,753 -> 23,385). Between the stop rule's lines:
# just under 1.5x with both, and only by an inexact sun. Halving the bytes (1000 -> 512 a slot) bought a third of the 16^3
# ceiling. Caveat: the untimed pack runs inside the march scope right before the spans (march 2.74 -> 5.92 ms), so the copy's
# last writes may sit in L2 and flatter the compact arms.
# MEASURED (workset1, HSTR_RES=1920x1080, --steps 1): spans 0.606 / 16^3 0.343 / 4^3 0.265 / no bake fetch 0.382 / floor 0.254 /
# again 0.606 ms; brick samples (arm - floor) 0.352 -> 0.089 (4.0x) -> 0.011 (~32x); light misses 4,753 -> ~100k in the oracle
# arms. PASSES: the working set, not the instructions, makes a real brick sample slow. See evaluateSpan.
# MEASURED (packed2, edge density replicated into the apron, read at the sun's clamp): spans 0.601 / again 0.604, packed 0.533
# (-0.07), no bake fetch 0.380; light off by > 0.02 4,753 -> 4,791, T 5,730 -> 5,716: exact now, but half packed1's saving.
# MEASURED (packed3, both clamps in one run): spans 0.586 / again 0.587, packed (sun's clamp, exact) 0.492 / again 0.492 (-0.094),
# packed at the density's clamp 0.473 (-0.113, light bad 22,960), no bake fetch 0.383 (-0.203). The wider read costs ~0.02 ms;
# packed2's -0.07 was the packed arm's run-to-run spread (0.533 there, 0.492 here). The exact one-fetch evaluator saves ~0.094 ms
# of 0.586 (16%), ~45% of the bake fetch's ceiling, with the light unchanged (4,792 against 4,753): at the ~0.1 ms line.
# MEASURED (packed1, HSTR_RES=1920x1080, --steps 1): spans per ray 0.623 / again 0.623 ms, packed 0.490 (-0.133), no bake fetch
# 0.400 (-0.223, the ceiling); 603,395 of the ~617k brick samples packable. Time: PASSES (>= 0.1 ms). Quality: rays off the march
# by > 0.02 in T 5,730 -> 5,715 (density unchanged at that test; mismatches > 1e-4 10,359 -> 28,096 from the packed texture's
# other size moving the filter's sub-texel rounding), in light 4,753 -> 22,957 - the one fetch takes the density's clamp (0-7)
# for the sun, whose own is -0.5-7.5. Fix to test: replicate each brick's edge density into its apron in the packed copy and
# fetch both at the sun's clamp (per axis that is exactly the density's clamp). The fill (untimed) costs ~10 ms a frame.
# MEASURED (restart1, HSTR_RES=1920x1080, --steps 1): spans per ray 0.584 / again 0.584 ms, no bake fetch 0.360, no far-field
# fetch 0.584, no sun fetch 0.347 (light mismatches 6,062 -> 102k / 119k / 122k: every arm reaches the code). The bake fetch is
# the whole sun cost (0.22 ms); the far field is free. With the bake free the brick samples cost ~0.11 ms over the 0.248 floor:
# the packed density + bake ceiling passes. Restarts: 132,999 of 215,229 rays have brick samples; strict 190,985 runs (1.44 a
# ray; 1 / 2 / 3 / 4 / 5-8: 92.8k / 27.5k / 9.0k / 2.8k / 1.0k), loose 138,967 (1.04; 127k / 5.9k / 21). With a final miss
# query a ray asks ~1.9 (strict) / ~1.65 (loose) queries: passes the ~2-2.5 rule. See spanRestarts.
# MEASURED (proxyspan1, HSTR_RES=1920x1080, --steps 1): spans 0.601 / 0.603 ms, proxy skipped 0.597, bricks skipped 0.248 (the
# floor). 1.25M of 1.87M samples are proxy samples, 0.5% of them lit: the proxy is free in the evaluator, the 617k brick samples
# (92% lit) are all of its sample time. See spanProxyStats.
# MEASURED (crossspan1, HSTR_RES=1920x1080, --steps 1, both arms identical): 590k brick crossings, 1.28 samples each; lit samples
# per crossing 0 / 1 / 2 / 3: 36k / 409k / 141k / 3.2k. Held per crossing leaves out ~21% of sun evaluations, interpolated 0.5%:
# no ceiling for a span-level sun. See spanCrossing.
# MEASURED (leanspan1, 4K): walk march 3.65 / 3.62 ms (query + units) -> spans 1.77 ms for 80% of 673k rays (20.5% past 12 spans,
# skipped: ~2.2 ms for all), sprint 4.51 / 4.54 -> 2.24. 3.6 samples a span, ~5.1 G samples/s; 0.2-0.3% of rays not bit-exact.
# Gate A (<= 0.7 ms) FAILED: the direct-span evaluator buys ~1.7-2x with spans free, not the ~5x the dirty passes need.

# MEASURED and REMOVED (leanhalf1, 4K, errors identical): resolve 0.47 / 0.49 -> 0.41 ms, frame walk 5.78 / 5.82 -> 5.92, sprint
# 7.28 / 7.27 -> 7.25. Not worth the format change; the colorHalf property is gone.

# MEASURED and REMOVED (leanloads1, errors identical): exact load cuts that cost registers - the proxy only where it answers
# (bits 65536) 6.04 -> 6.16 ms walk, the sea tile instance cached along the ray (131072) 6.04 -> 6.53, both 6.62 (anchor 6.07);
# sprint 7.52 -> 7.68 / 8.12 / 8.22 (anchor 7.58). See leanExtinction.

# The cache held over lightingStride contributing samples (and never across a cache cell): an approximation of the cache, which
# is smooth at cell scale while the samples are a fraction of a cell apart. Judged against the path-traced reference.
# MEASURED and REMOVED (leanstride1, walk): 6.39 / 6.37 / 6.38 / 6.64 ms at stride 1 / 2 / 4 / 8 for 0.137% / 0.153% / 1.048% /
# 4.985% over 0.02. No time to buy; marchBeamLean evaluates the cache at every contributing sample again (the arms no longer vary).
STRIDE = [
    ("stride 1", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=1)),
    ("stride 2", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=2)),
    ("stride 4", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=4)),
    ("stride 8", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=8)),
    ("stride 1 again", dict(BASE, beamShipMask=FLAT_MASK, lightingStride=1)),
]
