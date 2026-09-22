# Handoff: translation cost, cell views (branch `cell-views-handoff`)

Read README.md and CLAUDE.md first. Target: every motion (park, look, walk, sprint) well under 2 ms at 4K, ideally ~1 ms, with
quality held at today's level against the exact march and the saved path-traced reference.

## Where the renderer stands (4K, OCT_T001, beamGuardParallax 1, one settle, exact-march gate)

| motion | GPU ms | >0.02 vs exact march | where the time goes |
|---|---|---|---|
| park   | ~0.55 | 0.137% | resolve 0.46 |
| look   | ~0.6  | ~0.13% | |
| walk   | 7.5   | 0.136% | dirty lattice query 2.9 (414k rays), dirty units 3.4 (280k units), rest ~1.1 |
| sprint | 9.3   | 0.025% | query 3.9, units 3.8 |

Rotation and parking are solved by the world-fixed octahedral beam image. Translation is not: the per-block parallax guard
(1 texel) lists ~40% of the lattice every walk frame, and every listed ray and unit is re-marched with `marchBeam()`.

Measured (probe `beamRepairProbe`, dirty_march_probe.py): 91.5% of the re-marched lattice rays and 45% of units were within 0.02 in
place. Even an oracle that re-marched only the ones that changed would still march ~185k rays a frame (~2 ms). **No selection
scheme over camera rays reaches the target; the evaluator itself has to stop being a per-ray volume walk.**

## What this branch contains

1. **Runtime witnessed carry removed** (`beamWitness*`). Rejected on its ceiling: even carrying every block (a wrong image) only
   reached 6.1 ms walk; the real arms were 8.8-10.1 ms. The witness-matrix and shadow-carry PROBES are kept
   (`beamProbeWitnesses`, `beamShadowCarry`, `sweeps/shadow_carry.py`) with the numbers in a source comment. Old witness sweeps are
   archived in `HSTR_results/removed_witness/`.
2. **Cell views** (`cellViews`, off by default; nothing ships with it). A world-space transport cache:
   - Cell = domain voxel `[i-0.5, i+0.5)^3`. A view = the cell's (cache rgb, transmittance) and (single rgb, opacity-centroid
     fraction) for a `cellViewEdge^2` grid of rays over its projection, marched along each ray's chord of the cell from the camera
     that built it. Slots in a ring (`hstrCellViews`, atlas `hstrCellTexels*`), map `hstrCellMap` (wrapped cell -> slot+1).
   - Validity: lateral misregistration `|camera shift across the ray| / distance * half chord <= cellViewDrift`, footprint-level
     band (`cellViewBand`), sea-fade ratio (corrected on lookup), age, content invalidation through `hstrBeamInvalidations`.
   - `hstrCellOccupancy`: per cell, whether any of its 27 voxels has a nonzero conservative maximum (`buildCellOccupancy`).
   - Build pass `buildCellViews` (one 8x8 group per queued cell), `writeCellViewArgs`, `invalidateCellViews`.
   - Composition is exact given the views: splitting a ray at cells and marching each chord matched the plain march
     (0.134% vs 0.137%). **Bug found on the way, kept fixed**: `marchSegmentMoment` restarted every call with `fineVoxels = 1`, so a
     resumed march's first step crossed a whole domain voxel on one sample; it now has an overload taking `fineVoxels`.
3. **`composeBeam()` - WRONG ARCHITECTURE, DELETE IT.** It walks each dirty ray cell by cell and reads views instead of density.
   That is exactly the README red flag. Measured (4K walk, cvdebug5): 10.05 ms vs 7.57 base, 0.297% over 0.02; per ray ~46 cells
   visited, ~10 occupied; builds 2.0 ms at 16k/frame; ~58k distinct occupied cells touched per frame. Per-ray composition costs
   about what marching costs because the traversal, not the density read, dominates. Remove `composeBeam`, `marchBeamDirty`,
   `HSTR_CELL_VIEWS`, `cellViewDebug`, and `sweeps/cell_views_debug.py`; keep the view data structures if the next design uses them.

## What to build next (the README's final work unit)

Projected transport node x BeamTile, with no per-ray volume walk:
- Project visible transport cells (occupied cells with valid views, occlusion-culled against the previous frame's opacity/depth)
  into screen/oct-image tiles; bin and sort front to back (per-tile lists; uniform-grid cells sort correctly by centre distance).
- Tile/lattice values composite their tile's node list front to back with early termination; a node without a valid view is the
  residual (march its chord only). Cost then follows the visible cloud surface on screen, not cloud count or density - required
  since the user wants many more clouds.
- Also worth measuring: lattice density by depth. Near cloud has ~18 px voxels but gets the same 4 px lattice as the horizon, and
  the near region is where most of the walk's dirty work is.
- Open questions: view memory at scale (58k cells touched per walk frame now), rebuild rate, view resolution at near silhouettes.

## Harness notes

- `python scripts/HSTR/bench/run_sea.py motion TAG SWEEP --motions "walk 2 0.004" "sprint 20 0"`; results in
  `C:/Users/Friss/Documents/HSTR_results/TAG_test.txt`. A 3-4 arm walk run takes ~6-10 minutes; run it in the background.
- `sea_motion.py` now logs the cell view counters of the last scored step (`cellView*` in cloudStats).
- Rebuild before every run: shaders are only live once the build has copied them into `bin/`.
- Result tags from this session: cellviews1, cvdebug1 (both ran a stale shader, ignore), cvdebug2-5.
