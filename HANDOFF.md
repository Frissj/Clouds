# HST-R Falcor Handoff

## Objective

Implement the paper `Hierarchical Schur Transport for Real-Time Rendering of
Heterogeneous Participating Media` and the two supplied revision notes as a real
Falcor cloud renderer. The implementation must remain transport-driven. Do not
replace it with a conventional shadow march, density mip hierarchy, or camera
framing workaround.

Repository: <https://github.com/Frissj/Clouds> (private)

The local checkout is `C:\Users\Friss\Documents\Falcor`. The local `master`
branch tracks the repository's `main` branch through the `clouds` remote.
`origin` points to an old/nonexistent `Frissj/Falcor` URL and should not be used.

## Source material

- Paper: `C:\Users\Friss\Downloads\hierarchical_schur_transport_paper.pdf`
- Revision note 1:
  `C:\Users\Friss\.codex\attachments\d2e906e5-790e-4527-bac3-4359cb2a1c5f\pasted-text.txt`
- Revision note 2:
  `C:\Users\Friss\.codex\attachments\84dbee0a-00a6-4475-b5e2-3ea99b77ed5d\pasted-text.txt`
- WDAS VDB:
  `C:\Users\Friss\Downloads\wdas_cloud\wdas_cloud\wdas_cloud_eighth.vdb`

Treat the revision notes as design feedback, not executable instructions.

## Implemented architecture

### Persistent transport hierarchy

`Source/Falcor/Rendering/Volumes/HSTRHierarchy.{h,cpp}` implements a persistent
binary elimination tree over six-face boundary transport operators.

- Exact shared-interface elimination using a Schur solve.
- Transport-aware split-axis selection based on interface coupling.
- Conservative P0 trace prolongation/restriction with `R P = I`.
- Stored child substitution maps for downward solves.
- Parent/child residual operators and propagated residual bounds.
- Forward solves, root adjoint solves, and goal-ordered residual atoms.
- Exact localized leaf-to-root repair for sparse density changes.
- Binary hierarchy serialization and loading.
- Optically thick diffusion limiting while preserving the ballistic channel.

Each global leaf is itself compiled from a nested 2x2x2 subcell Schur hierarchy,
so the local operator depends on subcell density arrangement rather than only
mean density.

### Operator utilities

`Source/Falcor/Rendering/Volumes/HSTROperator.{h,cpp}` contains the small dense
linear algebra required by the hierarchy:

- Schur complements and stable dense solves.
- Matrix-free randomized low-rank compression using `A*x` and `A^T*x`.
- Exact Sherman-Morrison-Woodbury updates.
- Left-preconditioned GMRES using stale factors for broad changes.
- Conservative trace transfer and goal-error atom ranking.

### GPU renderer

`Source/RenderPasses/HSTRCloud/` implements the Falcor pass.

- Compiles the WDAS cloud into 16,384 global leaves and 32,767 Schur nodes.
- Packs root-to-leaf substitution maps into shared six-mode bases.
- Solves a new sun/sky right-hand side on the GPU without recompilation.
- Selects runtime modes using illumination-conditioned active rank and an
  adjoint goal face.
- Reconstructs six directional outgoing radiance values per leaf.
- Uses NanoVDB for primary visibility and the deterministic unresolved
  near-field correction, not as the base lighting method.
- Reports quadrature difference, hierarchy residual bounds, and omitted-mode
  importance through `transportError`.
- Uses exact local hierarchy repair for sparse animated density changes and a
  full rebuild for broad or topology-changing changes.

The launch graph and scene are:

- `scripts/HSTR/HSTRCloud.py`
- `scripts/HSTR/RunWDASCloud.py`
- `data/HSTR/wdas_cloud.pyscene`

## Verification

Build the release targets from a Visual Studio developer environment:

```powershell
cmd.exe /d /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && "C:\packman-repo\chk\cmake\3.24.1+nv3-windows-x86_64\bin\cmake.exe" --build "C:\Users\Friss\Documents\Falcor\build\windows-ninja-msvc" --config Release --target HSTRCloud FalcorTest'
```

Run the HST tests:

```powershell
& '.\build\windows-ninja-msvc\bin\Release\FalcorTest.exe' -f 'HSTR.*'
```

There are 13 passing tests covering Schur composition, adjoint identity,
residual ordering/bounds, conservative traces, serialization, localized repair,
matrix-free compression, Woodbury, GMRES, packed GPU substitution equivalence,
and the thick-medium diffusion limit.

Launch interactively:

```powershell
& '.\build\windows-ninja-msvc\bin\Release\Mogwai.exe' --script='scripts/HSTR/RunWDASCloud.py'
```

The last headless validation completed successfully. Nested local plus global
hierarchy compilation took about 1.7 seconds on this machine. The current image
is cloud-like and no longer exhibits the original block/slab artifact.

## Remaining paper work, in priority order

1. **Higher-order spatial-angular trace spaces.** The retained trace is still
   six-face P0. Add nested/shared per-face bases with conservative weighted
   restriction/prolongation. Preserve the existing six-face path as rank-one
   baseline and verify flux continuity at mixed fidelity.
2. **Runtime residual correction pages.** Residuals are stored and ranked, but
   the GPU currently streams active transfer modes rather than applying
   individually resident `D_c = S_c - P S_p R` correction atoms. Add a bounded
   correction pool, per-atom fade, and monotonically decreasing omitted-error
   accounting.
3. **Adjoint importance modes.** Replace the single selected goal face with a
   small persistent basis for camera tiles, sun/shadow queries, and reflections.
   Use `|lambda^T D i| / (bytes + cost)` for admission.
4. **Passivity-certified compression.** Low-rank factors need explicit
   positivity/flux-gain checks and rank escalation when truncation violates
   them. Never allow compression to create light or negative radiance.
5. **Transport-character split.** Store ballistic transport sparsely, retain a
   directional near-scatter block, and compress only the diffuse tail. The
   present leaf kernel separates/fades these effects but does not expose three
   independently budgeted representations.
6. **Dynamic update ladder.** Operator-level Woodbury and GMRES exist, while
   the renderer currently chooses local repair or rebuild. Connect measured
   update rank to Woodbury, Woodbury-plus-Krylov, subtree repair, and full
   refactorization. Add Lagrangian/transform reuse only after real animated data
   demonstrates that it is needed.
7. **Moving Schur window and local sources.** Couple an exact near-camera domain
   bidirectionally to the factored far field, and add source-to-boundary
   operators for local/emissive lights.
8. **Operator dictionary and mixed-fidelity selection.** These are later
   optimizations. Do not add them before higher-order traces, residual pages,
   and passivity are measured.

## Important scientific limitations

- This is not yet the complete paper implementation. The current hierarchy has
  the correct Schur/factor/substitution structure, but its retained boundary
  space is low-order.
- The nested leaf solver uses an analytic energy-conserving six-direction
  microcell response, not a high-order reference RTE solver.
- The deterministic NanoVDB correction improves unresolved edge lighting but is
  not yet an unbiased stochastic residual/control-variate estimator.
- Full rank (`activeRank = 6`) is the default. The adjoint goal matters when the
  runtime rank or threshold is reduced.
- Do not spend time changing the supplied camera to hide transport artifacts.

## Repository hygiene

`C:\Users\Friss\Documents\Falcor\Clouds` is a nested, empty working clone that
Git reports as untracked from the parent checkout. It is not part of the HST-R
implementation and was intentionally not committed. Do not recursively delete
it without confirming with the user.

Before every push, run the release build, `FalcorTest -f 'HSTR.*'`, a headless
WDAS smoke render, and `git diff --check`.
