# HSTR 2 ms: handoff after the representation round (2026-10-08)

Follows `HANDOFF_2MS.md` (2026-10-06). Read `CLAUDE.md` first: it is the rule book (one run per question, sanity + anchor arms,
stop rules before running, never monitor running logs, rebuild after every shader edit, removal only on the owner's word).
Latest code: `rt-span-oracle` (`727e596f`); this file sits on top of it on `claude/wizardly-hopper-5utgva`.

## 1. Where the frame is

4K sunset walk (2 units/frame), GPU leaves, `motpack1` (`57499ed5`):

| leaf | walk ms | note |
|---|---|---|
| `march/units` + `beamDirty/query` | 1.97 + 1.57 | the dirty march; 3.5-4.7 ms across runs |
| `march/farSea` | 0.52 | refresh 1/8 a frame, swept and tuned |
| `resolve` | 0.50 | 0.80 parked (overlaps the units' tail in motion, `f2820da2`) |
| sun bake / scan / resolve | 0.36 | sun-depth upkeep |
| `dirtyTiles` | 0.21 | |
| other | ~0.2 | |
| **frame** | **~5.35** | parked ~1.45 (resolve 0.80, query 0.19, dirtyTiles 0.12, warp 0.11) - already under target |

The whole gap is motion. **Without any march, a walking frame is already ~1.8 ms.** 2 ms needs the march cut 4-5x **and** the
non-march leaves cut to ~1 ms. Every experiment this round attacked the march alone.

## 2. Closed this round (do not re-propose; numbers in the commits)

| idea | verdict | why (counter) |
|---|---|---|
| Lit World Volume, Form A (de-instanced world bricks) | rejected, `2889db27` | S0: resampling fails at rims (c 0.65: p99 0.0372, 2.74% of rays > 0.02; rims 3.82% vs interior 0.05%); S1: 2.3-5.5x the turnover of shared bricks |
| Static empty-space fields (asset or world, any resolution) | closed, `ec2c1879`, `af162241`, `c92ee8c9` | empty runs thread gaps between wisps; 96% of empty samples are 0-1 majorant blocks from density |
| Form R: RT ray queries over shared hulls | rejected, `3a576c45` | warp-paid steps -42% but units + query unchanged: the skipped steps were already cheap; queries stall the march, BVH evicts L1 |
| One-read lookup (any merged page / clipmap) | ceiling, `3a576c45` (layeroracle1/2) | -38% per paid step; units + query floors ~2.8 ms |
| Reuse of unchanged rays | ceiling, `edcad809` | a free oracle keeps 60% of walk steps; the old witness validator alone cost 2.7 ms |
| Analytic kernel leaves | closed, `727e596f` `kernel_fit.py` | 16 Gaussians a brick: p99 0.023 vs codec 0.009 |
| Splatting the sea | closed, `727e596f` `splat_rate.py` | 2M splats at 4K: 20-32 ms vs a 1.5 ms kill line |

Stacking every march ceiling as if free (reuse x0.40, one-read x0.62) gives units + query ~1.0-1.2 ms, plus ~1.8 ms of other
leaves: **~2.8-3.0 ms**. No representation alone closes it.

## 3. Form B (open)

`4ace1b72` (S2, frozen residency, off by default: `cloudSeaAlign 0`, `formB 0`): aligned sea + world brick table over shared
packed (density, sun) texels. Units + query 4.05 -> 2.95 ms (-27%), but 7.68% of moving pixels over 0.02 vs ship ~2%, and the
coarse sanity arm is as wrong and as fast. Only 11% of samples take the texel path; 66% are table empties. **If the empties skip
real density, part of the -27% is the bug** (fewer expensive in-cloud samples).

The S2 rule (units + query <= 50% of ship) is unreachable for any one-read structure (layeroracle2 floor, measured after S2's
rule was set). Owner decision, recommended: **one validation run, then decide by this rule, stated before it runs:**

- Run (`formbval1`, walk, frozen residency, 2 cycles, ~5 min): ship / formB 3 (both paths at every sample: empties where the old
  chain finds density, texels off by > 1%, fallback reasons) / **empties only** (texels sent to the old chain) / **texels only**
  (empties sent to the old chain) / formB after the fix if found / ship again. The split arms' image errors locate the bug; their
  times say how much of the -27% is real.
- Keep Form B only if, fixed, it (a) matches ship's moving error against the fresh rebuild within ~0.3 points, (b) cuts units +
  query by >= 15% (~0.6 ms) at frozen residency, and (c) holds that under live residency. Otherwise reject it, keep it off, and
  leave the sea layout alone.
- Even if kept: it changes the sea (every instance at scale 1, cloud scale 0.833 -> 0.825, grid-snapped corners), needs the
  owner's sign-off on the look and a new path-traced reference of the aligned sea, and leaves units + query near 2.6-2.8 ms. It is
  one piece of a 2 ms frame, not the route.

## 4. Next runs, in order (each one launch, stop rule set before running)

1. **Form B validation** - above.
2. **Sun upkeep offline** (needs the owner's answer: is the sun direction fixed in the product?). The per-brick sun bake depends
   only on the asset and its orientation class (8 an asset), not on the camera or neighbours, so it could ship in the package
   like density (packed RG8 already exists: `cloudSunPacked 1`). Ceiling = the 0.36 ms of sun leaves in motion, minus whatever
   slot resolution remains. Cost: up to 8 sun bakes a brick on disk and in streaming. Check first that nothing in bake / scan /
   resolve depends on the instance placement; if it does, this idea shrinks to that part.
3. **Resolve floor** (`resolvefloor1`, parked + walk, arms: ship / write-only (constant colour, no beam reads) / ship again). If
   the write-only floor is within 0.15 ms of ship, the resolve is not bandwidth-bound and formats will not help: close. Otherwise
   try a narrower output format and fewer inputs, scored against the path trace.
4. **Depth-binned reuse probe** (counts only, extends `beamRepairProbe` from `edcad809`). Each re-marched ray also records its
   cloud in depth bins; a later frame rebuilds it by warping each bin by its own parallax from its capture camera and compositing
   front to back. Bin width from the parallax budget: within-bin parallax spread `delta * sin(a) * width / d^2` <= 0.5 px
   (2e-4 rad) gives `width = 2.9e-4 d^2` at walk speed (delta 2, sin a ~0.35): ~12 bins from 256 to 2240 per frame of reuse, ~48
   to reuse 4 frames. Unlike the rejected near-segment split (`fe2a6378`: far parallax had to stay within a texel unwarped), each
   bin is warped. Count: march steps whose bins rebuild within 0.02 log radiance at F = 1 / 2 / 4 frames, and bins a frame (the
   rebuild touches every non-empty bin, so it costs more than its reads suggest). Stop rule: close if <= 75% of walk steps are
   reusable at F = 2 (60% today). Then disocclusion (content behind T < 1e-3 was never marched) is the next counter.

Realistic best case if 1-4 all pay: march ~1 ms, other leaves ~1.2-1.4 ms: **~2.2-2.5 ms**. The rest needs an owner decision (§5).

## 5. Questions only the owner can answer

- **Motion per frame**: dirty work scales with travel per frame. Must 2 ms hold at walk (2 units/frame) and sprint (20)? One run
  at v0.5 / v1 / v2 would show the scaling.
- **Bounded staleness in motion**: re-march the dirty set over 2-4 frames and reproject the rest, scored per moving frame against
  the path trace at p99 like everything else. The only lever not at a measured ceiling.
- **Sun fixed?** (decides §4.2.)
- **Form B's sea change** (§3), if it survives validation.

## 6. Traps found this round

- ngfx auto-export writes every capture to the same `ngfx/BASE` folder: copy the first export away before the second trace.
- A rule missed by 1.6% is a miss (`3a576c45` took R0's 0.508 vs 0.5 ms as a pass). State the margin with the rule instead.
- `hstComponents` multiplies after the reads (`HSTRCloud.cs.slang` ~2487): it cannot be a timing oracle. Compile the reads out.
- `727e596f` committed `graphify-out/cache/stat-index.json` (a tool cache) under the message "scripts": keep caches out and write
  the full commit message CLAUDE.md asks for.
- Counted steps are not time: Form R removed 42% of paid steps for -8% of instructions. Price a step class (GPU trace or timed
  oracle) before chasing its count.

## 7. State

- Branch `rt-span-oracle` at `727e596f`: Form R (`formR`, off) and Form B (`cloudSeaAlign`, `formB`, off) in the renderer; results
  `HSTR_results/formr1-3`, `ngfx/formr4{on,off}_export`, `formbs2`.
- `LIT_VOLUME.md` holds the Lit World Volume design (sections 1-7) and Form R (6c).
- Offline tools in `scripts/HSTR/bench/litvolume/`: `hstrlib_dense` (exact `.hstrlib` decoder), S0 / S1 gates, Form R scene + trace.
