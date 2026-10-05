# HSTR: the road to 2 ms at 4K - where it stands and why it is hard

Handoff written 2026-10-06 on branch `march-empty-skip`. It covers the motion guard, reuse and quality-trade work since `af162241` and
the measured limits of every route to the target. Read `CLAUDE.md` first for the workflow rules, and grep the `MEASURED` comments
before proposing anything: almost every idea below has numbers beside the code.

## 1. The target and where the frame is

Target: under 2 ms of GPU time at 4K, at path-tracer quality (CLAUDE.md: 99th-percentile quality against the saved path trace).

The shipped 4K sunset walk (2 units/frame), GPU leaves (`gpuleaves1`, commit `bb630686`; sum 6.19 ms):

| Leaf | ms | What it is |
|---|---|---|
| `march/units` | 2.13 | exact march of the beam units whose tiles failed, in blocks translation invalidated |
| `beamDirty/query` | 1.77 | lattice queries (tile corners) of those blocks |
| `march/farSea` | 0.54 | the far cloud layer past the near march's view distance |
| `resolve/pixels` + `residual` | 0.53 | beam image to screen, warped |
| `dirtyTiles` | 0.20 | tile classification of dirty blocks |
| sun bake / scan / resolve | 0.40 | sun-depth cache upkeep |
| warp, skirt masks, tone map, rest | ~0.6 | |

Units + query (~3.6-4.2 ms depending on run and thermal state) is the problem: everything else together is already ~2.3 ms. A 2 ms
frame needs units + query well under 1 ms **and** the rest roughly halved.

Rates: the dirty work is ~740k rays and ~47M march steps a walk frame. The shipping march runs ~7.3 G steps/s; a 2 ms frame needs
~50-70 G steps/s at today's step count (`brickshare1`, note at `HSTRCloud.cs.slang` "MEASURED and REJECTED (brickshare1").

## 2. Architecture in one paragraph

A world-fixed octahedral **beam image** holds radiance per direction (two slices: cache/multiple-scattering and single-scattering
basis, plus depth). Rotation re-marches nothing. Translation is handled per 8-texel **guard block**: a parallax certificate
(`beamCellMotionCertified`, isotropic bound at 8 texels, a 4x4 min-depth pyramid) decides whether a block is held (read through a
backward **warp field** built from lattice depths and each block's capture camera) or listed (its lattice points re-queried and its
failed tiles' units re-marched). Tiles are reconstructed bilinearly from corner queries (`beamReconstruct`); tiles failing the
curvature / centre test (`beamTileRefines`, tolerance 0.4) fall to per-unit exact marching (the residual). The far sea is a separate
half-resolution layer marched 1/8 per frame over a dense extinction texture and reprojected.

## 3. What was tried this round, and why each failed

### 3.1 Holding more blocks: the OR motion guard (closed)

The directional guard (`beamGuardMotion` 1: shift = |w x d| + reach|d|) OR'd with the isotropic one holds ~3k more blocks a frame:
units + query ~3.8 -> ~3.1 ms (-0.7 ms) on the walk. But motion error rises from ~2.2% to ~3.0-3.4% of pixels over 0.02 (against a
fresh rebuild, `HSTR_MOTION_SCORE=2`), visible as chips.

Every repair was built and measured (commits `af162241`, `cf94361d`, `d1af32e8`, `f3743349`; notes beside each parameter in
`HSTRCloudTypes.slang`):

| Attempt | Result |
|---|---|
| far-parallax guard, fold / fine warp, edge-tile repair (`beamWarpEdge`) | lose the gain or no effect |
| sparse signed gap repair (`beamWarpGap`) | quality half returns, +2 ms of scattered repair units |
| occlusion-aware warp field (`beamWarpCover`) | fires (16-45k points re-pointed) and makes the image **worse** |
| mode 1 only at level 0 / levels 0-1 (`beamGuardModeLevels`) | level 0 only = iso cost; the gain and the error are both levels >= 1 |
| level budget scale (`beamGuardLevelScale`) | slides along the iso p8 / p4 frontier |
| double-buffered pre-march history (`beamWarpHistory`, Nehab / Sitthi-amorn style) | live (69k overwritten reads a frame) but no quality change |
| list held blocks beside nearer content (`beamGuardNeighbour`) | even the ceiling (any nearer neighbour) leaves the error |
| per-pixel path attribution (`beamPathDump`, `HSTR_MOTION_PATH`) | **diagnosis**: OR-only blocks are simply staler (5-15x the bad rate of iso-held ones) and leak ~1 block into neighbours through shared lattice corners / the interpolated warp |
| offline delta / shift oracle (`HSTR_results/delta_oracle.py`) + control on iso tiles | probe-ray brightness fits make it worse; the best oracle shift + offset removes 40% from OR tiles **and** from ordinary iso tiles: nothing OR-specific to correct |

Conclusion: the blocks the guard could skip are exactly the ones missing new radiance and visibility. No resolve-side, warp-side,
history or probe-and-correct scheme recovers them; any fix re-marches them. **iso p8 stays.**

### 3.2 Making each dirty ray cheaper: already measured ceilings

| Route | Measured ceiling | Where |
|---|---|---|
| precomposed transport per (brick, entry cell, direction class) | reuse 4.55x near / 2.46x sea -> ~1.26x frame; deeper nodes ~1.03x | `b4437f68` |
| per-cell / node x tile payloads | hard 14% of interactions hold 45% of steps at any resolution -> ~2x dirty work | `share1-5` notes |
| world-space per-cell transport cache (cell views) | measured slower (10.05 vs 7.57 ms) | `51c2dace` |
| temporal reuse of ray results | 94% of steps reusable at the exact ray, 58% once snapped to a beam texel; ray jets recover part, not cheaply | `share6-8`, `ray_jet_fit.py` |
| witnessed carry (probe rays per block) | witness pass alone 2.71 ms; walk 7.46 -> 8.82 ms | `51c2dace` |
| wavefront / lane refill | divergence overhead only 1.29x units / 1.55x query; refill destroyed L1 locality (77 -> 53% hit), slower | `210c2f51`, note at `beamUnitRefill` |
| cooperative brick-resident kernel | 17-30 G steps/s ceiling vs 50-70 needed | `9f61a15a` |
| ray start / end / layer hints | perfect start -21% units, end -8% more, layer hint 4.8% | `bb630686` |
| far-sea mip chain | far texels span at most ~3 voxels, so ~0.05 ms; far sea is tail-bound by 20k-unit grazing rays | reasoning in chat 2026-10-06; `bb630686` counters |

None of these individually or together reaches the ~3x units + query needs. The march is L1-bound; register / occupancy cuts,
prefetch, step-in-flight and load cuts were all measured without gain (CLAUDE.md list).

### 3.3 Spending quality: the frontier against the path trace (this session)

Scored per pixel against the 326-spp path trace `sunset_hill_0_0_3840x2160_960x540` (squared log error, trace noise subtracted;
anchor excess 0.020228, reproduced to the digit at both ends of every run). Timed on the 4K walk with shipped anchors interleaved
between every arm (the first arm of a run reads high and timings drift ~0.4 ms across a run - never compare against one anchor).

| Setting | Units + query change | PT excess change | Motion error (vs ~2.0-2.1% anchors) | Look |
|---|---|---|---|---|
| step 5 (`minStepVoxels`) | -0.43 ms | +0.5% | ~2.5-3.2% | as shipped |
| step 6 | -0.9 ms | +2.4% | ~4.0% | as shipped |
| tolerance 0.6 | ~-0.5 ms | +0.6% | ~2.8% | **straight tile-diagonal facets on outlines** |
| step 5 + tolerance 0.6 | -0.79 ms | +1.5% | ~2.5% | facets |
| step 6 + tolerance 0.6 | -1.2 ms | +3.6% | ~3.3% | facets |
| tolerance 0.2 | +0.47 ms | -0.8% | ~2.2% | **cleanest (owner's pick)** |
| tolerance 0.2 + step 5 | -0.24 ms | 0.0% | ~3.8% | clean |
| tolerance 0.2 + step 6 | -0.28 ms | +1.5% | ~3.3% | clean |
| outline-only 0.2 (`beamEdgeTolerance`, range 0.05) | ~0 (+-0.15) | -0.25% | noisy | about a third of 0.2's change |
| `beamOctScale` 0.42 / 0.35 | -0.67 / -1.1 ms | +4.1 / +7.4% | worse | blockier |
| `beamOctScale` 0.75 / 1.0 | (costlier; old note: 0.5 -> 1.0 ~+1.45 ms) | -4.0 / -4.9% | | sharper outlines |

Lessons:
- **The repo's "<1% of pixels over 0.02" gate on 8x8 blocks is blind here**: it scores every arm 0.000% (even `beamOctScale` 0.25 at
  +18.7% excess). Use `compareSquared` per pixel (`sunset_hill.py` `HSTR_HILL_PT_ARMS`, `launcherBeam`, `beamPolicy` False,
  `beamReset` True) **and look at images**: the owner sees tile blockiness that the squared excess barely registers (+0.6%).
- Longer steps cost motion stability (held values drift further from a fresh march), not static quality.
- The visible blockiness is coarse tiles accepted by the 0.4 tolerance. Outline-only strictness (opacity range over the corners)
  catches only a third of it; the rest is likely lighting edges inside clouds. Next step there: flag tiles by log-radiance range
  across the corners too, not just transmittance.
- Even the best acceptable trade found (~-0.8 to -1.2 ms) leaves units + query near 2.6-3.0 ms.

## 4. Why 2 ms is hard - the honest arithmetic

- Units + query must go from ~3.8 ms to well under 1 ms (~4-5x). Measured ceilings: guard/reuse closed; operators ~1.26-2x;
  kernel/divergence ~1.35x; quality trades ~1.3x before the image visibly degrades.
- Multiplying the optimistic ceilings that are compatible (kernel 1.35x x quality 1.3x) gives ~1.75x: units + query ~2.2 ms.
- The other ~2.3 ms (far sea, resolve, sun upkeep, tiles, warp, skirt, tone map) must also roughly halve; each leaf there is
  0.1-0.5 ms and has been tuned already (far sea alone: footprint step, refresh 1/8, scale 2, start hints all measured).
- So on this hardware (RTX 4080 Laptop) with this representation, every measured route lands at roughly 4-5 ms. 2 ms needs a new
  idea that cuts **steps** by ~3x without changing what the pixels see, or a decision to accept a different quality bar.

## 5. Open leads, ranked

1. **Lighting-edge-aware tile test** (cheap, in progress): extend `beamEdgeTolerance` from transmittance range to log-radiance
   range across the corners; target = the look of tolerance 0.2 at outline-only cost. Then pair with step 5 if the motion cost is
   accepted. Quality/look win more than speed.
2. **"Other" leaves**: sun upkeep (0.4), resolve (0.5), dirtyTiles (0.2), skirt masks (0.1) have not had the units/query level of
   scrutiny this session; a 2 ms frame cannot carry ~2.3 ms of them.
3. **Coherent dirty kernel**: only with intra-queue ordering that keeps neighbouring units together (refill showed locality is
   worth more than occupancy); ceiling ~1.35x on units + query.
4. **Owner decision on the quality bar**: if the target is negotiable (e.g. 3 ms, or per-distance quality), the frontier table above
   says what each step costs.

## 6. Measurement traps (each cost runs)

- First arm of a run reads high; timings drift through a run. Interleave anchors (`base1 / cand / base2 / ...`).
- One profiled frame per arm: +-10-20%. Repeat candidates.
- Motion score 2 compares against a rebuild at the **same** settings: it shows motion artefacts, not resolution loss.
- Occasional outlier chunks (a cumulus present in the rebuild but not the moving frame) push single arms up ~1-2 points; read the
  worst-chunk column.
- Laptop must be plugged in (an unplugged run was discarded).
- `HSTR_results/path_attr.py` (string keys) is slow on 8M pixels a frame; vectorise before reuse.

## 7. State of the tree

Committed through `f3743349`. Uncommitted: `beamEdgeTolerance` / `beamEdgeRange` (off by default) in `beamTileRefines`, the delta
oracle note in `HSTRCloudTypes.slang`, and the quality-frontier comments in `scripts/HSTR/IntelCloudSeaHalf.py` (defaults unchanged).
Results: `C:/Users/Friss/Documents/HSTR_results/` - `pathattr1`, `qualfront1-3`, `qualtime1-4`, `blocky1`, `tol02`, `tol02time`,
`edgetol1`, `edgetime1`, and the zoom images `*/compare_zoom*.png`, `blocky1/blocky_zoom.png`, `tol02/tol02_zoom.png`,
`edgetol1/edgetol_zoom.png`.
