# HSTR performance work

HSTR is a quality-constrained real-time renderer. The target is below 2 ms at 4K without surrendering opportunities for performance, while preserving at least the required 99th-percentile quality against the original path-traced result.

## Non-negotiable workflow

- Run Falcor/Mogwai headless only. The benchmark wrappers already pass `--headless`; do not open an interactive renderer.
- Reuse saved references under `C:/Users/Friss/Documents/HSTR_results/references` through `loadReference`. Do not path trace a scene again when a matching saved reference exists.
- Treat the saved path-traced image as the quality authority. Fast exact-march A/B tests are useful for iteration, but they do not replace the final saved-reference gate. Report mean, threshold share, tail percentile, and maximum when available.
- Measure parked and live motion. A static win does not ship if residency churn, fallback work, or reconstruction makes moving frames slower.
- Build immediately after C++ or shader edits with the documented HSTRCloud build command. Do not hunt for another command.
- Preserve unrelated working-tree changes and benchmark artifacts.

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

When I say commit, you fucking commit on the spot instantly without hesitation unless I give additional instructions. What I say goes. 

DO NOT MONITOR THE LOGS OF RUNNING BUILDS IN THE BACKGROUND, THIS USES UP MY TOKENS AND THEREFORE MY MONEY WHICH IS BOLD OF YOU TO DO WITHOUT MY PERMISSION. WHEN THE RUNNING TASK FINISHES, YOU AUTOMATICALLY GET AN UPDATE ANYWAY SO JUST SHUT UP AND STOP RUNNING IN THE BACKGROUND TO PRESERVE TOKENS.
