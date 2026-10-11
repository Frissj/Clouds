# HSTRCloud: route to the lowest walk-frame cost

Current frame (nowtrace1, GPU trace, 4K sunset walk, defaults at d867571f): HSTRCloud 2.87 ms quiet / 3.23 ms load frame.
dirty query 1.01 (~10% SM busy) | unit march 0.90 (34-37%) | resolve 0.54-0.58 | cloudSea 0.17 / 0.55 | sun scheduling ~0.2 a frame
averaged (~1 ms on every 4th frame) | dirtyTiles 0.08 | worldCacheView 0.055 | warp 0.05.

## Done (uncommitted)

- [x] **Sun stamp by changed-cell prefix sums** (`cloudSunStampFine`, on). Skip-same alone was no faster (stampskip2). stampfine5
      (a lane per (cell, class) union test): stamp ~1.2-1.7 ms a run -> ~0.29 (0.17-0.41 -> ~0.065 ms a frame); exact over 103 runs.
- [x] **worldCacheView every 4th frame moving, none parked** (`worldCacheViewEvery` 4): 0.055-0.065 -> 0.015 ms a frame, motion
      score inside the anchors' spread (cacheview1).
- [x] ~~**Sun stamps beside the dirty query**~~ (`cloudSunBeside`, off): barrier-free after unbinding the slot table / bake list
      views, but traced serial (query + stamps 1.94 = 0.86 + 1.12 ms); stampFine is 58% SM busy, nothing to hide in. Also weighs
      on the second-queue idea: the work it would move is SM-bound, not latency-bound.
- [x] **Bakes mark the beam** (`beamMarkBakes`, on; found while splitting dispatchSunScheduling - they never had): no measurable
      cost or quality change (markbakes1).
- [x] **markBeamChanges: per-asset slot lists** (`beamChangeAssetSlots`, on): 0.029-0.038 ms a run vs 0.047-0.054.

## Next (single queue, small, measured one by one)

- [ ] **Residency fades over the active fades only** (`fades` 0.027 ms every frame, 79% DRAM: a pass over the whole brick capacity
      for ~30-50k active fades). Ceiling ~0.02 ms. A brick is 32 bytes = one sector, so reading only fadeStart saves nothing;
      needs a CPU-built list of fading bricks (~50k entries a frame of CPU walk + upload). Deferred: smallest item left.
- [x] ~~**Warp field beside the dirty query**~~ closed: buildBeamWarpFieldAt reads the live lattice, beam pixels and level map
      (beamLatticeAt / beamPixelRead / beamLevelRead) that this frame's query and tile pass write; only beamWarpHistory (an ORACLE,
      off) switches them to the pre-build snapshots. Its dispatch size also comes from the post-build policy (beamWarpAuto).
- [ ] **Tone mapping folded into the resolve** (ToneMapper 0.1 ms, outside HSTRCloud).

## Open ideas from the discussion (not yet measured)

- [x] ~~**Temporal stochastic march**~~ CLOSED (histblend4, beamHistoryBlendOracle): blending a dirty unit's fresh march with its
      history at 0.5 takes the motion score 5.3-6.4% -> 6.8-10.4% (sanity 0.95: 16-18%), far past the +0.5 stop rule; 60% of
      dirty units' histories are over 0.02 from the fresh march, 28% over 0.1. The guard re-marches exactly where history is wrong.
      Also closes world-anchored sampling with temporal accumulation (below), which needs the same history.
- [ ] **Proxy control variate** (from Volumetric ReSTIR, Lin / Wyman / Yuksel 2021): march the cheap proxy fully, estimate only the
      fine residual sparsely. Closed by reasoning only (the long-step error is rim detail the proxy cannot carry) - not measured.
      Ceiling: long-step error of a proxy-only march vs the full march's long-step error.
- [x] ~~**World-anchored ("hashed", Wyman & McGuire 2017) sample positions with temporal accumulation**~~ closed with the temporal
      march: the accumulation half needs the dirty units' history, which histblend4 shows is wrong where they are dirty.
- [ ] **Fill folded into the shade dispatch** (outside review): the fill-skip oracle passed its stop rule (resolve range -0.13 to
      -0.21 ms, fillskiptrace1 vs halftrace3); a fused fill keeps its ~0.15 ms of work plus a ~30% shaded halo beside the march.
      Build and trace it. Blocker found: the halo (24 shaded pixels of the neighbouring groups per 16 x 8 tile: a row above and
      below, 4 left, 4 right) needs a side-effect-free shade - resolveBeamHalfShade writes color and the basis flag and queues
      residual groups, which the halo must not duplicate. Estimated net 0.07-0.1 ms (fill 0.165 minus ~0.03-0.05 halo shading).
      BUILT and MEASURED out (beamHalfFused, off): the halo is resolveBeamPixel again (same value written twice, verdict ignored).
      Traces: resolve after the march 0.41-0.56 -> 0.24-0.29 ms but the march range +0.3; march + resolve ~1.48 -> ~1.55 ms. Work
      beside the unit march does not hide (it is not idle enough), so the review's "resolve beside the query" has no headroom either.
- [ ] **Dense fill: why slower** (0.151 -> 0.279 ms): unknown without a trace (Shader Pipelines of fillBeamHalf, dense arm).
- [ ] **Sun bake order by sampling** (most-sampled bricks first) to cut the 60-85% ancestor-lit samples while walking.
- [ ] **Bake step 3**: never measured (residency frozen under PT arms stops bake scheduling); needs the arms thawed.

## Big structural (multi-hour, ask first)

- [ ] **Async residency + sun pipeline on a second queue**, inside the dirty query's ~0.9 ms of idle GPU. Hides cloudSea
      (0.17-0.55) and the sun scheduler (~0.2 avg, removes the every-4th-frame ~1 ms spike). Ceiling ~0.4-0.6 ms a frame.
      Needs: a second gfx queue + ComputeContext (slang-gfx has only the Graphics queue type; a second one is possible), fences,
      simultaneous-access density / sun atlases (both queues touch them), one-frame-late publishing of pages and resolved sun
      slots so the query never reads a slot being written.

## Known limits found (not costs, but real)

- While walking, 60-85% of lit samples light from an ancestor's sun bake ~1.5 levels coarser (275-400k bakes always waiting).
  The parked path-trace gate never sees it. Bake *order* (most-sampled first) may matter more than rate.

## Closed this round (numbers in commits 7b935210 / d867571f and the notes beside the code)

- Lockstep waves (7.08 keys a wave at one distance, stop rule 4; sanity counters pass).
- Dense half-width fill buffer (fill 0.151 -> 0.279 ms).
- DLSS / lower internal resolution (resolve -0.31 ms at 1080p before the upscaler's own cost).
- Single-frame interleaved / jittered sampling (budgetjitter2: error -16-25%, +0.25 ms).
- Hardware RT, raster slices, splats, lit volume Form A/B (all in the record).
- Longer sun scheduling interval (every 8: -0.03 ms).
- Query step cap / continuation / slicing (query's own rays are ~0.8 ms; capping moves work into the units).
- Shipped: half-rate resolve, far sky, moving sun bakes 128 -> 64.
