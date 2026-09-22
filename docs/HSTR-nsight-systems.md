# Profiling HSTR with NVIDIA Nsight Systems

`ProfileHSTRCloud.bat` records a bounded, headless Mogwai run with NVIDIA Nsight Systems. It discovers the installed Nsight Systems CLI automatically and writes timestamped `.nsys-rep` reports to `C:\Users\Friss\Documents\HSTR_results\nsight`, outside the repository.

From the Falcor root, capture 300 presented frames with D3D12 tracing:

```powershell
.\ProfileHSTRCloud.bat
```

Capture 600 frames and open the completed report in the Nsight Systems UI:

```powershell
.\ProfileHSTRCloud.bat -Frames 600 -Open
```

Profile a terminating benchmark script instead of the default cloud scene:

```powershell
.\ProfileHSTRCloud.bat -Script scripts/HSTR/bench/steady_state.py -Frames 300 -Open
```

Use Vulkan tracing when Mogwai is running on Vulkan:

```powershell
.\ProfileHSTRCloud.bat -Api vulkan -Frames 300
```

The capture includes API calls, individual GPU workloads, graphics annotations, and NVTX events when present. CPU instruction-pointer sampling is disabled so the launcher works without administrator privileges and avoids sampling overhead. Nsight stops after the selected number of presented frames and terminates the otherwise interactive default Mogwai process.

To inspect the resolved executables and command without starting Mogwai:

```powershell
.\ProfileHSTRCloud.bat -DryRun
```
