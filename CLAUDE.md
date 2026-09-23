# HSTR performance work

HSTR is a quality-constrained real-time renderer. The target is below 2 ms at 4K without surrendering opportunities for performance, while preserving at least the required 99th-percentile quality against the original path-traced result.

## STOP: architecture gate before writing any renderer code

README.md defines the final architecture. It is not background reading. Before designing or writing ANY new evaluator, pass, cache or
optimization, write down in chat, explicitly:

1. Which README "Final Architecture" component this builds.
2. The unit of work it runs per: if the answer is "per camera ray", "per lattice point", "per basis ray" or "per pixel", and the
   work walks the volume (steps, cells, bricks, nodes along the ray), STOP. That is a ray marcher by another name, whatever it reads.
   The README's final work unit is projected transport node x BeamTile.
3. Whether it still exists once `marchBeam()` is only the residual path. If not, do not build it without explicit instruction.
4. Which README "Automatic Red Flags" it resembles. Replacing the density a ray reads with a cached representation, while still
   traversing per ray, is the red flag "improve the density lookup for every Beam query".

If you cannot answer these in a way that matches the README, do not write the code. Ask. This rule exists because an agent spent
an hour building a per-ray cell traversal (`composeBeam`) that the README explicitly rules out.

Also:
- The build copies shaders into `build/windows-ninja-msvc/bin/Release/shaders`. A shader edit is NOT live until the build command
  has run. Rebuild after every shader edit, before every benchmark run, or the run measures the old shader.
- Edit files with the Edit tool. Do not rewrite source through python/sed/heredoc scripts.

## Non-negotiable workflow

- Run Falcor/Mogwai headless only. The benchmark wrappers already pass `--headless`; do not open an interactive renderer.
- Reuse saved references under `C:/Users/Friss/Documents/HSTR_results/references` through `loadReference`. Do not path trace a scene again when a matching saved reference exists.
- Treat the saved path-traced image as the quality authority. Fast exact-march A/B tests are useful for iteration, but they do not replace the final saved-reference gate. Report mean, threshold share, tail percentile, and maximum when available.
- Measure parked and live motion. A static win does not ship if residency churn, fallback work, or reconstruction makes moving frames slower.
- Build immediately after C++ or shader edits with the documented HSTRCloud build command. Do not hunt for another command.
- Preserve unrelated working-tree changes and benchmark artifacts.

## Benchmark runs: a few minutes each, one per question

- Every run must finish in a few minutes. Each Mogwai launch pays a ~1,200-frame sea settle before it measures anything, so the settle, not the measurement, is the cost. Never loop a harness over several processes (one per settle count, one per repeat): put every arm and variant in ONE run's TESTS or property sweep, so they share one settle.
- One run per question. Before launching, state what the run answers and roughly how long it takes. If it cannot answer the question, add the logging it needs first; do not launch repeats hoping a pattern appears.
- Run only the motions the question needs (e.g. `--motions "sprint 20 0"` alone), with the fewest arms and steps that answer it.
- Log the state needed to explain the result (counters, settle state) in the same run, so a surprising number never needs a second run just to find out what happened.
- Compare A/B arms within one run. Numbers from different runs are not comparable until the harness is shown to be deterministic.

## Optimization philosophy

- Pursue both sides of the budget: fewer expensive queries and cheaper queries. Micro-optimizing one million queries is not a credible route to the target by itself.
- Separate stable spatial identity from physical residency. Hot camera lookups should use coherent virtual addressing; allocators must not make churn permanently degrade every later query.
- Build and maintain GPU-hot structures on the GPU when practical. Do not trade the frame target for large CPU expansion or a reduced-quality shortcut.
- Keep rare pathological work out of common SIMD kernels. If a fallback inflates registers or occupancy even when untaken, isolate it in another dispatch/queue or compile it out only after coverage and quality are measured.
- Exploit spatial, temporal, and wave coherence only when measurements validate the complete combination. A disappointing first implementation is diagnostic: locate its cost and try complementary changes before rejecting the underlying idea.
- Never retain an optimization merely because it is architecturally attractive. Test settled, churned, and moving cases; remove variants that remain slower after reasonable combinations are explored.
- Do not achieve speed by silently weakening density integration, reference quality, resolution, or the percentile gate. Any deliberate approximation must be explicit and separately measured.

## Evidence and documentation

- Add concise comments beside non-obvious performance decisions with the benchmark conditions and before/after numbers. Explain rejected alternatives when a future maintainer might reasonably retry them.
- Keep reusable sweeps in `scripts/HSTR/bench/sweeps/`; give each arm one controlled difference and retain anchor arms where drift matters.
- Use distinct result tags and preserve the generated logs/JSON outside the repository. Compare like-for-like configurations and note thermal or residency-state confounders.
- Commit only measured changes, with a detailed message covering architecture, quality evidence, parked/live timings, and known remaining limits.

## Build command

Run from the Falcor root with PowerShell:

```powershell
cmd.exe /d /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul 2>nul && "C:\packman-repo\chk\cmake\3.24.1+nv3-windows-x86_64\bin\cmake.exe" --build "C:\Users\Friss\Documents\Falcor\build\windows-ninja-msvc" --config Release --target HSTRCloud 2>&1' | Select-String -Pattern "error|FAILED|Linking" | Select-Object -First 10
```

## Nsight Systems (GPU metrics) - the ONLY way to do it

```bash
python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/oct_ship.py --steps 0 --motions "walk 2 0.004" --nsys
```

- Any sweep and motions work; `--steps 0` skips the scored steps. Mogwai runs under `nsys launch` (API tracing armed, nothing
  recorded); sea_motion.py sends `nsys start` (GPU metrics) right before each arm's 48-frame timed flight and `nsys stop` right
  after, so there is one small report per arm x motion: `HSTR_results/nsight/TAG_<motion>_<arm>.nsys-rep`. Never capture the
  whole run (the settle is not the question). A capture finishes within seconds of its stop: if nsys is still busy minutes
  later it is hung, not importing. Log `HSTR_results/TAG.log`.
  Run it in the background once; you are notified when it exits. Do not poll.
- Mogwai is always `--headless`. `nsys.exe` is set to run as administrator in its compatibility settings (GPU metrics need admin).
  Windows only starts such an exe through ShellExecute: `subprocess`/CreateProcess fails with WinError 740. `--nsys` therefore
  re-runs `run_sea.py` elevated (ShellExecuteEx "runas" on pythonw.exe) and waits; the elevated copy sets the HSTR_* environment
  itself, starts nsys with CREATE_NO_WINDOW, and everything under it is elevated. No window may appear: a console program started
  through ShellExecute gets a console window even with SW_HIDE (Windows Terminal ignores it), hence pythonw + CREATE_NO_WINDOW. Do not use ProfileNsight.ps1, PowerShell, a .vbs wrapper, or a UAC helper of your own.
- Read it: `python scripts/HSTR/bench/nsys_export.py <report>.nsys-rep` (nsys needs admin for export too; the script elevates the
  same way) writes `<report>.sqlite` in seconds. FALCOR_PROFILE scopes are GPU ranges in DX12_WORKLOAD (textId -> StringIds, e.g.
  `query`, `units`, `resolve`); GPU_METRICS (names in TARGET_INFO_GPU_METRICS) averaged over those ranges gives SMs Active, SM
  Issue, Compute Warps in Flight, Unallocated Warps, DRAM Read/Write. GPU metrics inflate frame time (walk 7.5 -> 11.0 ms): compare
  ratios, not milliseconds.

When I say commit, you fucking commit on the spot instantly without hesitation unless I give additional instructions. What I say goes. 

DO NOT MONITOR THE LOGS OF RUNNING BUILDS IN THE BACKGROUND, THIS USES UP MY TOKENS AND THEREFORE MY MONEY WHICH IS BOLD OF YOU TO DO WITHOUT MY PERMISSION. WHEN THE RUNNING TASK FINISHES, YOU AUTOMATICALLY GET AN UPDATE ANYWAY SO JUST SHUT UP AND STOP RUNNING IN THE BACKGROUND TO PRESERVE TOKENS.

Always edit like a normal person, you are fucking unbearable when you dont
