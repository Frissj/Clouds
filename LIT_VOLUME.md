# Lit World Volume (LWV): cache the medium translation preserves, not the integrals it destroys

Proposal written 2026-10-07 on `claude/wizardly-hopper-5utgva`, after `HANDOFF_2MS.md` and the "2 ms handoff" note. Nothing here is
built into the renderer yet. The offline gates in section 6 come first, per CLAUDE.md ("measure the ceiling before building").

## 1. The idea in one paragraph

The beam is fast under rotation because it caches what rotation leaves unchanged: radiance per direction at a point. Under
translation it caches the wrong thing. Line integrals change with every camera step: `share6-8` (`fb18d0b4`) measured 58% of
steps reusable once snapped to a beam texel, and the guard lists 17.4k of ~30k on-screen blocks a walk frame. What translation
leaves unchanged is the **lit medium**: extinction and sun-lit source at world points. HSTR re-derives that medium for every
dirty sample of every frame through two instanced layer chains, a resolved sun slot, a far-field fetch, an octave texture and an
SH cache. The Lit World Volume stores it once, in world space, pre-lit and with the layers summed. Each march step then becomes:

```
today (per lit sample, per layer x 2):  tile -> instance -> asset level page -> brick record -> cell test -> octant skirt -> atlas
                                         + resolved sun slot -> sun atlas, + far field (fineSunAt), + octave spread texture,
                                         + world cache SH (+ its own fineSunAt modulation)
LWV (per sample):                        level from footprint (ALU) -> world page entry -> lit atlas texel (sigma, sun depth)
                                         + one coarse view-resolved light texel (independent of the chain, issued in parallel)
```

The instance transform, layer sum, skirts, fades, cell tests and the sun all move into a per-brick **bake**, paid when a world brick
becomes resident or its sources change. They are no longer paid per sample per frame.

## 2. Why this escapes the measurements that closed earlier routes

| Earlier result | Why it does not bound LWV |
|---|---|
| One-read density oracle (`layeroracle2`, handoff): 0.273 -> 0.169 ns a step, "density bounded" | It removed the density chain and **left the lighting chain in place**. Lighting was ~49% of the generic dirty cost (`chainsplit1`, `926eb8d5`) and sun 30% + cache 15% of a lean sample (`strip2`). LWV removes both. |
| Packed density + sun (`a8d343c2`, `918e26d1`, `57499ed5`): neutral | It fused two fetches at the **end of the same chain** (the slot still came from the resolved-sun lookup; far field and cache stayed). The chain is the cost, not the fetch count (Nsight `unitsreg1`: march latency-bound in the dependent lookup chain). |
| Combined sea-space page table blocked (`fcbef650`): "instances are scaled signed permutations, a sea-space table still needs the instance transform" | LWV does not share bricks across instances. It **bakes** each world brick from the instances once, so the per-sample path never sees a transform. The price is memory and bake turnover (gates S1/S3). |
| Second layer ~40% of the dirty march, two chains a step (`04675421`) | Layers are summed in the bake: one lookup per sample. |
| ~1.3 samples a brick crossing (`b686db3e`): per-crossing setup cannot amortise | LWV has no per-crossing setup. Every sample pays the same two dependent loads. |
| Lighting not smooth over 2 voxels (`f8c51661`: sun hold 2 voxels 0.136% -> 4.04%) | LWV keeps a lighting value **per sample**, at the march's own spacing. It makes each lighting evaluation cost what the density read costs, instead of evaluating fewer of them. |
| Transfer varies at texel scale (`fb18d0b4` share2/3/8, jets `bb74ec69`, certification `9f61a15a`) | LWV caches no transfer. The medium is interpolable at voxel scale because it *is* the voxel field. |
| Cooperative brick-resident kernel 17-30 G/s (`9f61a15a`) | LWV is not a staging scheme. It is the data layout the flat kernels below assume. |

The handoff's framing ("lighting evaluations are the expensive quantity, so evaluate fewer") only holds while a lighting evaluation
is a separate sparse chain. LWV keeps the number of evaluations and makes each one almost free.

## 3. Ceiling, from numbers already recorded

- Shipping 4K sunset walk (`motpack1`, `57499ed5`): units 1.97 + query 1.57 = **3.54 ms** of dirty march. The frame without it is
  ~1.8 ms walking and ~1.45 ms parked.
- A one-fetch-per-step loop over the same rays: `chainsplit1`'s proxy-only arm ran the whole step loop with one dense fetch at
  **27%** of the shipping cost (1.9 of 6.94 ms). Older single-layer sea; attribution, not an exact split.
- LWV adds one dependent load (the page entry) to that loop: estimate **27-40%** of today, so the dirty march falls to ~0.95-1.4 ms
  and the walk frame to ~3.0-3.3 ms with nothing else changed.
- The steps themselves: 74-76% of dirty-march steps land on an **empty** fine brick (`chainsplit1` pushWalk), exactly skippable by
  a brick-level DDA. The instanced, two-layer structure cannot skip them (`ec2c1879`: empty boxes <= ~2.4 sea voxels, runs average
  ~45). A merged, world-aligned occupancy can: each empty page entry carries its own Chebyshev skip distance.
- The flat kernel this enables was already timed offline on this GPU (`a3adf5ff`, NVIDIA Warp, RTX 4080 Laptop): direct page
  table + brick DDA over a packed (density, sun) atlas, **1M rays in 0.14 ms coherent / 0.62 ms shuffled**. The batched gather from
  the first non-empty brick ran at 40 G samples/s coherent, and tile order was worth 4.7x. For ~740k dirty rays that ceiling is
  **~0.1-0.5 ms**. The instanced representation is what has kept the renderer away from it.

**Honest projection.** LWV alone: walk ~3.0-3.3 ms **plus its bake turnover** (section 5.3, not yet measured: it can erase the
gain in Form A). LWV plus DDA skipping and tile-coherent ray order (the `a3adf5ff` kernel): dirty march ~0.2-0.5 ms, walk ~2.0-2.4
ms plus turnover. Below 2 ms also needs the ~1.8 ms "rest" trimmed. LWV contributes there too: its coarse levels can serve the far
sea (0.52 ms), and its bake subsumes the sun bake / scan / resolve upkeep (0.36 ms).

## 4. The representation

**Levels.** World level `l` has voxel `W_l = W_0 * 2^l` world units. Bricks are 8^3 voxels with a one-voxel apron (10^3 texels,
as today's atlas, so hardware trilinear needs no cross-brick logic). `W_0` is a free parameter set by gate S0. It is the
de-instancing resolution: instance source voxels are `0.25 * fitScale * scale`, with scale 0.65-1 per tile (`CloudSea::makeTile`),
so no single world grid aligns with every instance. A sample picks `l` from its footprint, as `footprintLevel` does today, and
reads the finest resident level at or above it. The coarsest level is the existing domain volume (`hstrExtinction`,
`hstrFineSun`), which is always resident.

**Texel.** RG8. `r` = summed extinction of both layers, with the brick's fade, octant skirt and density floor already applied, coded
like the atlas (`min + range * (code / 255)^2`, min and range in the page entry). `g` = total sun optical depth, near plus far, at
8 bits (`cloudSunDecode`, measured neutral against the path trace in `57499ed5`). This is two bytes a texel, like today's R16F sun
atlas, but it replaces the density atlas, the sun atlas and the far-field read.

**The coarse light texel.** A per-frame, view-resolved 3D texture at world-cache cell resolution (`worldCacheCellVoxels`, 4 sea
voxels), RGBA16F. `rgb` = the world-cache SH evaluated toward the camera at the cell centre, times its sun modulation. `a` = the
octave 1-3 spread term (`residualAtDepth`'s `multiple`, phase-weighted for the cell's view direction). The view direction varies
across a 20-unit cell by cell / distance (~0.02 rad at 1 km), which is negligible for SH. One trilinear fetch, independent of the
fine chain, replaces `worldCacheRadianceTextured`, `shWindow`, the octave texture and both `fineSunAt` reads. Rebuilt each frame
over cells in the view: ~1M cells, an estimated <0.05 ms.

**Page entries.** Per level, a camera-centred toroidal 3D table at brick granularity, covering the cloud layer's height and the
level's useful radius (`W_l * 8 / pixelAngle`-scaled, so every level is about the same number of entries across). One `uint` per
entry:
- `0x00000000 | d`: empty. `d` = Chebyshev distance in bricks to the nearest non-empty brick, so the march skips the whole cube
  in one step. This is `cloudLayerRunDistance`'s field, but over the merged layers at every level.
- `0x80000000 | slot (20 bits) | scale (11 bits)`: resident. `scale` is a log-coded range for the extinction code.
- `0x7FFFFFFF`: wanted but not resident. Read the next coarser level and append to the bake request list.

**Bake.** One 1000-thread group per world brick. Each texel evaluates the existing exact path at its centre: both layers'
`cloudInstanceAt` -> `cloudAssetBrickAtFlat` -> `cloudSampleDensity` (fade, skirt, floor), and `leanSunDepth` for near + far sun.
So LWV holds exactly what the shipping march would read at those points, and the only new approximation is resampling at the
world grid (gate S0). Bake inputs that change re-request the brick: the residency's brick changes and fades, sun bakes landing
(the beam already invalidates on both: `5543f950`), and a sun direction change (levels drain coarse-first, as sun bakes do now).

**March.** `marchBeamLit` replaces `marchBeamLean` in the dirty query and units under a define, so the shipped kernel carries no
dead path. Per step: level from footprint, page entry, then either a DDA jump (empty), a coarser fallback (not resident), or one
`SampleLevel` on the lit atlas plus the coarse light texel. Composite `sigma * (phase0 * exp(-tauSun) * E + light.rgb +
phaseOct * light.a)`. The step rule, transmittance boost and opacity cut are unchanged, so images stay comparable.

### 4b. Form B: aligned instances, shared lit bricks (if S0 or S1 fails Form A)

Form A de-instances because the sea's instances do not line up with any world grid. That is a property of the layout
(`CloudSea::makeTile`: continuous scale 0.65-1, free offsets), not of the clouds. If every instance has one source-voxel size and
its asset origin is snapped to a multiple of a level-2 brick (8 x 4 source voxels, ~8 world units at s = 0.25), then:
- every asset brick of levels 0-2 lands exactly on one world brick, under quarter turns and mirrors too (padded asset dims are
  multiples of 128);
- a world page entry can name the **shared** lit brick `(asset brick, orientation class)` plus the class's signed permutation (3
  bits). The texel is shared across instances like today's sun bake, so there is no resampling (no S0) and no de-instancing
  turnover (S1 = today's shared turnover);
- per step it is still one page entry plus one texel. The far sun field joins the coarse light texel as a factor
  `exp(-tau_far(x + near s))`, since the near bake is per class and the far field is world.
- layers 1 and 0 rarely hold density in the same world brick (0.08% of samples, `04675421`). Those pages name a de-instanced
  merged brick, which is rare enough not to matter for turnover.
Levels 3 and up (coarse, few, dense) can be de-instanced as in Form A at negligible cost. The price of Form B is the scene: the sea
loses its continuous instance scale, and offsets move by up to ~4 units. That is an owner decision, and it needs new path-traced
references for the new layout.

## 5. What could kill it (each is a gate below)

1. **Resampling.** Point-sampling a trilinear field on a misaligned grid, then interpolating again, low-passes rims by up to
   about a voxel. A finer `W_0` reduces this at `(s / W_0)^3` memory. Backlit rims are where the owner looks.
2. **Memory.** LWV is per world location, not shared across instances or orientation classes. Today: density atlas ~256 MB
   (243k-brick cap, already saturated) + sun atlas ~1056 MB (~600k bakes desired). LWV at RG8 is 2 KB a brick, so 650k bricks
   fit in the same ~1.3 GB.
3. **Bake turnover in motion: the most likely killer.** De-instancing gives up the sharing that bounds today's turnover. An
   asset brick serves every instance that wants it at that level, so the shared set saturates (5 assets). A world set has to be
   rebuilt along every LOD shell the camera sweeps. For the finest level, that is ~`v / (512 pixelAngle^2 W_0)` brick volumes of
   frustum shell a frame before occupancy: ~110k at the walk's 2 units a frame. The S1 smoke run on synthetic clouds (rays every
   16 px, bricks kept 4 frames unused) shows the mechanism: the shared asset set stayed put (3-32 new bricks a frame of 31.5k),
   while the world set needed 5.5-8.9k new of 183k and the per-instance set 8.5-14k. The values mean nothing; the gap is the point.
   A bake costs ~1,000 texel
   evaluations, so LWV only pays if (a) bricks bake on first entry only, with LOD hysteresis (serve from the resident level above
   or below until the bake lands, as the residency already does), and (b) a baked texel is several times cheaper than a march
   sample (brick-local, its one or two source asset bricks L1-resident, no ray divergence). The amortisation is: bake texels per
   frame = new bricks x 1000; march samples saved per frame ~ the dirty march's ~10-16M. Gate S1 measures the first number.
4. **The coarse light texel.** It approximates per-sample SH by the cell-centre view direction, and the octave spread by its
   cell value (the spread is already a 2-voxel-cell texture). A/B'd against per-sample lighting in S2.

## 6. Gates, in order, with stop rules stated before running

**S0 - resampling quality, offline** (`scripts/HSTR/bench/litvolume/lit_volume_fit.py`, real Intel bricks). Each crop of a real
asset is placed with the sea's instance transforms (scale 0.65-1, random sub-voxel phase), resampled to a world grid of voxel
`W = c * s_max`, and scored against the instanced field. Two metrics: the compiler's metric (axis columns, `|exp(-tau) -
exp(-tau')|`, p99) and backlit single scatter along oblique rays, with the sun depth integrated through the instance field and,
for the world grid, point-sampled from it as the bake would (`|dL|` against the crop's 90th-percentile ray, share over 0.02). Arms:
`c` = 1.0, 0.8, 0.65, 0.5; prefiltered vs point; **sanity**: aligned grid (must be exactly 0) and `c` = 2 (must be clearly worse);
**anchor** repeated last (must repeat to the digit).
*Stop rule:* if no `c >= 0.65` (memory <= 3.6x of matched) keeps column p99 at or below the packages' own compression p99 (0.0365,
`IntelSea/manifest.json`) **and** `|dL|/L90` over 0.02 in at most 1% of rays, close LWV's world-aligned form. The fallback would be
per-instance lit bricks: no resampling, but two lookups per sample.

**S1 - working set and turnover, offline** (`lit_volume_budget.py`). The sea layout is ported from `CloudSea::makeTile`, occupancy
comes from the same decoded assets, the LOD rule from `footprintLevel`, and cameras are the walk / sprint harness paths at the
sunset pose. Reports world bricks wanted per level, MB at RG8, and new bricks a frame.
*Stop rule:* close if the walk needs > 2.6 GB (2x today's atlases) or > 4k new bricks a frame at the chosen `c`.

**S2 - GPU timing ceiling, frozen** (renderer, built only after S0/S1 pass). LWV baked once after the settle for a frozen residency,
no streaming, with `marchBeamLit` in the dirty passes. One launch, `--motions "walk 2 0.004"`, arms: ship / LWV / LWV + DDA
skip / ship again. Sanity arm: LWV with sun depth forced to 0 (must change the image). Leaf scopes `march/units`, `beamDirty/query`.
*Stop rule:* continue only if LWV units + query is <= 50% of the interleaved anchors.

**S3 - live** (streaming bakes, invalidation, coarse light texel). Gates: `sunset_hill.py` `HSTR_HILL_PT_ARMS` squared excess
within +1% of the anchor (0.020228) with images looked at, motion `HSTR_MOTION_SCORE=2` walk within the anchors' spread, and walk
/ sprint frame time with bake cost counted.

## 6b. Results (2026-10-07, real Intel half-resolution clouds, clouds-v1 package)

**S0 - Form A fails its stop rule.** Asset 0 at level 1, sea fit scale 0.784 (level-1 instance voxel 0.392 world units at scale
1), 48 crops (12 each rim / interior / wispy / any) x 2 instance transforms, 768 rays and 4,800 columns a crop (`s0_run1`, 410 s).
Harness checks hold: SANITY aligned 0 exactly in every metric, SANITY c = 2 clearly worse, anchor identical to the digit.

| arm | memory vs instanced | column p99 | columns > 0.0365 | rays \|dL\|/L90 > 0.02 | rim rays > 0.02 | interior rays > 0.02 |
|---|---|---|---|---|---|---|
| point c = 1.0 | 0.58x | 0.0549 | 2.32% | 8.29% | 14.25% | 0.31% |
| point c = 0.8 | 1.14x | 0.0444 | 1.55% | 4.85% | 7.68% | 0.14% |
| point c = 0.65 | 2.13x | 0.0372 | 1.04% | 2.74% | 3.82% | 0.05% |
| point c = 0.5 | 4.67x | 0.0292 | 0.58% | 1.21% | 1.43% | 0.01% |
| prefilter c = 1.0 | 0.58x | 0.0408 | 1.27% | 4.90% | 8.09% | 0.22% |
| prefilter c = 0.8 | 1.14x | 0.0314 | 0.71% | 2.11% | 2.92% | 0.12% |
| SANITY c = 2 | 0.07x | 0.1102 | 6.91% | 25.21% | 43.88% | 2.75% |

The rule needed column p99 <= 0.0365 **and** <= 1% of rays over 0.02 at some c >= 0.65. No arm meets both. Even c = 0.5 at 4.7x
memory puts 1.21% of rays over. The error is the second interpolation at rims: at c = 0.65, rim rays 3.82% and rim column p99
0.0527, against interior 0.05% / 0.0016. The wispy class's ray share (17-23%) is inflated by its tiny L90 and is not what
fails the rule: rim and "any" fail it alone. Not tested: prefilter at c = 0.65.

**S1 - reduced run (walk only, 3 frames, rays every 8 px, `s1_run2`).** The full run (4 px, 6 walk + 4 sprint frames) needs over
an hour on the 4-core container and was stopped. Frame 2 (bricks kept 4 frames, so the history is not yet full):

| set | bricks | MB at 2 KB | new this frame |
|---|---|---|---|
| world c = 1.0 | 1.80M | 3,428 | 162k |
| world c = 0.65 | 2.69M | 5,136 | 384k |
| per-instance | 1.85M | 3,527 | 174k |
| shared asset bricks (today's) | 1.40M | 2,666 | 70k |

Absolute sizes are not the renderer's. The same counting puts today's shared set at 1.4M bricks and 70k new a frame, but the
renderer caps at 243k and streams <= 1,024 a frame (and is starved in flight, `57499ed5`: 102-136k held of 243k wanted). The sim
counts every brick a sample in cloud touches, through a coarse level-4 transmittance with no empty-box or majorant skipping. The
ratios under identical counting are the result: world c = 1.0 costs 1.3x the shared set's memory and 2.3x its turnover; c = 0.65,
the setting S0 would need, costs 1.9x and 5.5x. By the stated rule (> 2.6 GB or > 4k new a frame) Form A also fails S1. So does the
shared baseline under this counting, so S1's absolute threshold cannot separate the forms; the ratios can.

**Reading.** Form A (de-instanced, resampled world bricks) fails S0 outright and costs 2-5x the turnover of shared bricks. Form B
(aligned instances, shared lit bricks) is untouched by both: no resampling (S0 is exactly its SANITY aligned arm: 0), and its
memory and turnover are the shared line. It needs the owner's decision on the scene layout before the renderer gates (S2 / S3).

## 6c. Form R: cross empty space by hardware ray queries over shared asset hulls (proposed 2026-10-07)

An outside review proposed "Form C": keep values native in asset space behind the instance transform, and skip empty space with a
per-asset Chebyshev distance field. The first half stands and replaces Form B's reason to change the sea: the transform is cheap
(`fcbef650`, chainsplit1), so shared lit bricks need no aligned layout, only the transform. The navigation half is already
measured:
- Its oracle exists: `4f900bbe` nav_probe1, perfect empty-run skips cut warp-paid steps 14.77M -> 5.76M (2.57x).
- No static isotropic structure reaches it (`ec2c1879`): 54% of empty samples lie in a 4-voxel majorant block with density and
  42% one block away; the octant-mask oracle cut warp-paid steps 2.8%. The runs (~45 sea voxels) thread gaps between wisps, and a
  Chebyshev cube is bounded by the clearance across the ray, not the run along it. Asset coordinates do not change the cube.
- Ray-specific but 8 px apart is too coarse (`af162241`: lattice depth masks, <= 17% of steps for 12% of units wrong); a finer
  loop in the lookup costs more than it saves (`c92ee8c9`: 4.3 -> 6.1 ms).
- Entries free, density + single sun alone project ~1.3 ms at 4K (`db20cf3b`); a brick crossing is 1.28 samples (`b686db3e`).

The gaps are per ray, so the navigation must be too. Form R asks the RT cores:
- **Hulls**: one BLAS per (asset, level), the closed boundary of the octant cells (4 voxels at the level) holding any decoded
  density, dilated one cell. Shared by every instance, as the density atlas is. Measured on the decoded Intel assets (per asset):
  L4 ~3.4k triangles, L3 ~12k, L2 ~43-48k, L1 ~207k (a0); L0 undecoded (dense L0 is ~3.2 GB an asset). Built from the decoded
  bricks, so it also removes the dropped density of `fcbef650`'s layer bits (0.085%, built from source maxima).
- **TLAS**: one instance per (sea tile, layer) in range, its transform makeTile's scaled signed permutation; rebuilt only when
  tiles change (`seaTilesChanged` 0 on profiled flights). One query sees both layers.
- **Overlap**: both layers fill the same slab (y -80..80, the second shifted half a tile in x / z), so hulls overlap, and a jump
  is exact only from outside every hull. The union's inside is a count (+1 front face, -1 back face), and every face crossing
  costs a query: not one query per empty run. The CPU estimate on the walk frame's lattice: 5.9 faces a ray (p50 5, p90 12,
  max 28) with L1 / L2 hulls, ~6.9 queries a ray with the final miss, ~2.4M for the 342k lattice rays.
- **March**: outside every hull, take the query's front-face hit and jump there; inside, today's lean chain steps, and the
  pending query's face says when the count changes. No image change if the dilation covers the reconstruction's reach
  (trilinear, fade parent, skirts): the renderer must take the margin from the codec, not R0's one cell.
- **Ceiling**: at most nav_probe1's 2.57x warp-paid, less query cost and lane divergence. It does not lower `db20cf3b`'s floor
  for samples in cloud, so 2 ms also needs the lit-sample arm (packed density + sun from shared bricks, one coarse world light
  fetch for far field / cache / octaves), measured alone against the path trace.

Gates, stop rules set before any run:
- **R0 query cost** (`form_r_scene.py` + `form_r_trace.py`, slangpy on D3D12, minutes, no renderer change): every lattice ray of
  the 4K walk frame, every face crossing. Stop if it costs more than 0.5 ms. Sanity: shuffled ray order must be slower with
  identical counts; anchor within ~5%; the facing convention must leave no first hit on a back face; first entries must fall in
  the CPU reference interval.
- **R1 renderer arm** (walk only, one launch, off / on / off / on): queries replace empty stepping; sanity arm with undilated
  hulls must change the image. Stop if units + query fall less than 15%.
- **R2 lit-sample arm**: its own run against the saved path trace.

R0 so far (this container, no GPU): the scene half ran. Hulls are closed and outward (signed volume equals cell volume exactly,
directed edges balanced), 345 instances lie within the near view distance, the far rule needs levels 0-2 (27 instances want
level 0 and get the L1 hull, or L2 where L1 is undecoded), 342,002 lattice rays reach the slab, and 1,811 of 2,048 check rays
enter a hull. The GPU half has not run; its Slang source type-checks (the CPU target rejects only the ray-query capability). On
the 4080 Laptop, decode L1 for every asset first, or R0 times L2 hulls for four of the five and understates the cost:

```
for i in 0 1 2 3 4: hstrlib_dense IntelSea/half/intelCloudLib_dense.$i.L.hstrlib 1 a${i}_L1.npy
                    hstrlib_dense IntelSea/half/intelCloudLib_dense.$i.L.hstrlib 2 a${i}_L2.npy
python form_r_scene.py --volumes . --out form_r_walk.npz            # ~20 s; --level-rule near for finer, heavier hulls
pip install slangpy
python form_r_trace.py form_r_walk.npz                              # prints facing, CPU check, counts, four timed arms, verdict
```

R0 on the 4080 Laptop (L1 hulls for all five assets, 1.13M triangles, 75.5 MB of BLAS): facing default leaves 0 of 288,185 first
hits on a back face; 1,693 of 1,727 check rays enter within the CPU interval, 34 early, 4 GPU only; 7.3 queries a ray, 2.50M a
frame. Full 0.508 ms (0.204 ms per 1M queries), shuffled 1.059, first entry 0.070, anchor 0.506: STOP by 1.6% on the 0.5 ms rule,
taken as a pass.

R1 (renderer, `formR`, `form_r_hulls.py`; HSTRCloud.h updateHullTlas) - FAILED its stop rule, Form R REJECTED:
- The hulls are not exact: dilated by one cell they still miss 38k brick density samples a build (formR 2's count; undilated
  124k), falling to 3.2k only two levels coarser, where the outside share of steps falls from 51% to 26%. Samples read coarse
  resident ancestors whose reach passes a finer hull.
- The jump removes 42% of warp-paid steps but only 8% of the units' instructions: the steps outside the hulls were already the
  cheap ones (majorant-zero and layer-mask skips). Units + query 3.86 / 3.70 -> 3.90 / 3.50 ms; GPU Trace units 2.76 -> 2.91 ms
  (occupancy unchanged, SM throughput 35 -> 31%, L1 hit 78 -> 74%: traversal waits and BVH traffic), query 2.05 -> 1.82 ms.
- What remains is the brick chain and lighting at samples inside the clouds, which no empty-space structure removes - the same
  conclusion as `layeroracle1`. 2 ms needs the per-sample cost cut (R2), not the empty space.

## 7. In this commit

- `LIT_VOLUME.md` (this file).
- `scripts/HSTR/bench/litvolume/hstrlib_dense.cpp` + `build_hstrlib_dense.sh`: offline `.hstrlib` decoder (exact
  `reconstructBrick`), writes a level of an asset as a dense float16 `.npy`. It needs the reference GDeflate decoder
  (microsoft/DirectStorage `GDeflate/` + NVIDIA/libdeflate), cloned and built by the script's instructions.
- `scripts/HSTR/bench/litvolume/lit_volume_fit.py` (S0) and `lit_volume_budget.py` (S1). Both are smoke-tested on a synthetic
  volume only (`--synthetic`). S0's sanity arm scores exactly 0, its coarse arm is clearly worse, and its anchor repeats to the
  digit. The real-data runs have not happened yet.
- `scripts/HSTR/bench/litvolume/form_r_scene.py`, `form_r_trace.py`, `form_r_trace.slang`: Form R gate R0 (section 6c).

To run them on the half-resolution Intel sea (`IntelSea/half`, five assets):

```
hstrlib_dense IntelSea/half/intelCloudLib_dense.0.L.hstrlib 1 a0_L1.npy                 # S0 input (~400 MB), one asset is enough
python lit_volume_fit.py --volume a0_L1.npy --level 1 --crops 48 --trials 2
for i in 0 1 2 3 4: hstrlib_dense IntelSea/half/intelCloudLib_dense.$i.L.hstrlib 4 a${i}_L4.npy
                    hstrlib_dense IntelSea/half/intelCloudLib_dense.$i.L.hstrlib 2 a${i}_L2.npy
python lit_volume_budget.py --assets a0_L4.npy,...,a4_L4.npy --occupancy a0_L2.npy,...,a4_L2.npy --spacing 4
```

`--spacing 4` keeps rays under half a LOD-matched brick's screen width. Wider spacing undersamples far bricks: the touched set
then flickers between frames and reads as turnover (the synthetic run's set grew 77k -> 183k from 32 px to 16 px).
