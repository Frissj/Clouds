# HSTR performance work

HSTR is a quality-constrained real-time renderer. The target is below 2 ms at 4K without surrendering opportunities for performance, while preserving at least the required 99th-percentile quality against the original path-traced result.

## Memories are orders

- The memories in MEMORY.md are my instructions, learned from past mistakes. Obey them like this file. Before every decision
  (what to run next, what a result means, what to keep or remove), check which memories apply and follow them.
- Never remove, revert or reject an experiment on your own. Removal only on my word. Everything else - adding logging,
  diagnostics, oracle arms, the next run - just do it without asking; do not stop to wait for my answer.
- Every explanation of why something failed must cite a counter from the run. If no counter answers it, say "unknown", add
  the logging that would answer it and run it.

## Editing and building

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

## Check what was already tried - before suggesting anything

- Hundreds of experiments are recorded. Before proposing any optimisation, diagnostic or "missing option", search them:
  commit message bodies (`git log --format='%h %s%n%b' | grep -i -B2 -A8 <keyword>`), the MEASURED / MEASURED and REMOVED /
  REJECTED comments beside the code (`grep -rn MEASURED Source/RenderPasses/HSTRCloud scripts/HSTR`).
- Cite what you found ("tried in <commit>: result") and propose only what is genuinely new. Already measured, among many: register /
  occupancy cuts on the march and dirty query (no gain - L1-bound), prefetch / steps in flight, load cuts, deferred lighting, query
  group size, witness / carry reuse and its oracle ceiling, longer / footprint / adaptive steps, certificate loosening, tile size 8,
  reconstruction bases. Profiler samples are stall attribution, not removable time.

## Measuring correctly - lessons that cost runs

- Defaults go in the launcher the benchmark actually loads. The sunset benches run `SunsetCloudSea.py`, which runs
  `IntelCloudSeaHalf.py` - not `CloudSea.py`. Check the exec chain before changing a default.
- Beam quality against the path trace: the 8 x 8 block gate cannot see tile-scale interpolation error (a bilinear tile keeps its
  block means). Score per pixel with `compareSquared` (`sunset_hill.py` `HSTR_HILL_PT_ARMS`: squared log error with the trace's
  noise variance subtracted). Hill beam arms need `beamPolicy=False` (otherwise the GPU policy, not `beamTolerance`, sets the
  tolerance) and `beamReset=True`. Usable references: `sunset_hill_0_0_3840x2160_960x540` (326 spp, beamOct on) and the crop
  `sunset_hill_513_930_140x100_500x357` (696 spp, beamOct off). The 3840-wide 700x400 crop has 86 spp: too noisy to use.
- Every arm list needs a sanity arm that must change the result (a known-bad setting) and an anchor repeated at the end. Arms
  identical to the last digit mean the parameter is not reaching the code (wrong launcher, policy override, state leaking between
  arms) or the metric is blind - find out which before concluding anything.
- Motion quality: `HSTR_MOTION_SCORE=2` (moving frame against a fresh rebuild at that camera). When a number looks wrong, rerun with
  `HSTR_MOTION_OUT=<dir>` and look at the difference images before theorising. Per-build counters (repair probe, dirty reasons)
  go through `HSTR_MOTION_COUNT=<frames>`; as chunk deltas they read 0.
- Timing: compare profiled leaf scopes (`march/units`, `beamDirty/query`, ...). Whole-frame means on the live walk are dominated
  by CPU residency and swing +-20%. Sprint quality scores are noisy (single chunks past 30%): judge on the walk first.
- Measure the ceiling before building: a non-deciding diagnostic or oracle arm, with a stop rule stated in advance (e.g. stop if it
  rescues under 5%), settles most ideas in one run before any real implementation.

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

## Nsight Graphics GPU Trace: ONE trace per launch

- `run_sea.py --ngfx START STOP` produces exactly ONE trace per Mogwai launch. Pass ONE motion and a sweep with ONE arm. NEVER
  launch a run that would take two or more traces (several `--motions`, or several TESTS arms).
- Why: with `--auto-export`, ngfx exports the first report, prints "Terminating process..." and kills Mogwai; the kill hangs
  (Mogwai frozen at 0% CPU, elevated, needing a UAC taskkill) and no later trace is ever taken (shipgfx1: walk traced, sprint
  never, ~30 minutes lost).
- A second motion or arm is a second launch, after the first has exited.

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
