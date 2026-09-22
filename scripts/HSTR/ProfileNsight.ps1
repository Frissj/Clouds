[CmdletBinding()]
param(
    # 0: no frame limit - the Mogwai script decides when to exit (a benchmark that settles first).
    [ValidateRange(0, 1000000)]
    [int]$Frames = 300,

    [ValidateSet("d3d12", "vulkan")]
    [string]$Api = "d3d12",

    [string]$Script = "scripts/HSTR/RunWDASCloud.py",

    [string]$OutputDirectory = "C:\Users\Friss\Documents\HSTR_results\nsight",

    [string]$Name = ("hstr-{0}" -f (Get-Date -Format "yyyyMMdd-HHmmss")),

    [string[]]$MogwaiArgs = @(),

    # Samples SM activity, warp occupancy and DRAM throughput over time, to tell a kernel with low occupancy from one waiting on
    # its longest waves.
    [switch]$GpuMetrics,

    [switch]$Open,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$mogwai = Join-Path $root "build\windows-ninja-msvc\bin\Release\Mogwai.exe"
if (-not (Test-Path -LiteralPath $mogwai -PathType Leaf)) {
    throw "Mogwai is not built at '$mogwai'. Build the Release HSTRCloud target first."
}

$scriptPath = if ([System.IO.Path]::IsPathRooted($Script)) { $Script } else { Join-Path $root $Script }
if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "Mogwai script not found: '$scriptPath'."
}

$nsys = Get-Command nsys.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
if (-not $nsys) {
    $nvidiaRoot = Join-Path $env:ProgramFiles "NVIDIA Corporation"
    $nsys = Get-ChildItem -LiteralPath $nvidiaRoot -Directory -Filter "Nsight Systems *" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object { Join-Path $_.FullName "target-windows-x64\nsys.exe" } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}
if (-not $nsys) {
    throw "NVIDIA Nsight Systems was not found in PATH or under '$env:ProgramFiles\NVIDIA Corporation'."
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$outputBase = Join-Path (Resolve-Path -LiteralPath $OutputDirectory).Path $Name
$trace = if ($Api -eq "d3d12") { "dx12,dx12-annotations,nvtx" } else { "vulkan,vulkan-annotations,nvtx" }

$profileArgs = @(
    "profile"
    "--trace=$trace"
    "--sample=none"
)
if ($Frames -gt 0) {
    $profileArgs += @("--duration-frames=$Frames", "--kill=true")
}
if ($GpuMetrics) {
    $profileArgs += @("--gpu-metrics-devices=all", "--gpu-metrics-frequency=20000")
}
$profileArgs += @(
    "--wait=all"
    "--stats=true"
    "--show-output=true"
    "--force-overwrite=false"
    "--output=$outputBase"
    $mogwai
    "--headless"
    "--device-type=$Api"
    "--script=$Script"
) + $MogwaiArgs

Write-Host "Nsight Systems: $nsys"
Write-Host "Report:         $outputBase.nsys-rep"
Write-Host "Frames:         $Frames"
Write-Host "Command:        & '$nsys' $($profileArgs | ForEach-Object { "'$_'" })"

if ($DryRun) {
    return
}

Push-Location $root
try {
    & $nsys @profileArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Nsight Systems exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

$report = "$outputBase.nsys-rep"
if (-not (Test-Path -LiteralPath $report -PathType Leaf)) {
    throw "Nsight Systems completed but did not create '$report'."
}

Write-Host "Capture complete: $report"
if ($Open) {
    $ui = Join-Path (Split-Path (Split-Path $nsys -Parent) -Parent) "host-windows-x64\nsys-ui.exe"
    if (-not (Test-Path -LiteralPath $ui -PathType Leaf)) {
        throw "Nsight Systems UI not found at '$ui'."
    }
    Start-Process -FilePath $ui -ArgumentList @($report)
}
