import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

# Runs a cloud sea sweep in headless Mogwai and prints its results.
#   python scripts/HSTR/bench/run_sea.py ab TAG scripts/HSTR/bench/sweeps/oracle.py
#   python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/refresh.py --motions "static 0 0" "yaw 0 0.004"
#   python scripts/HSTR/bench/run_sea.py truth TAG scripts/HSTR/bench/sweeps/truth_step.py --views farside --spp 256
# A sweep file defines TESTS, a list of [label, properties] built with sea_config.mk (the first is the anchor for ab). ab is
# ab_test.py on the settled sea view (parked camera, interleaved drift anchors); motion is sea_motion.py (frozen residency unless
# --live, then flights); truth is sea_truth.py (the 4K path tracer as the reference).
BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[2]
RESULTS = Path("C:/Users/Friss/Documents/HSTR_results")
MOGWAI = ROOT / "build/windows-ninja-msvc/bin/Release/Mogwai.exe"

parser = argparse.ArgumentParser()
parser.add_argument("harness", choices=["ab", "motion", "truth"])
parser.add_argument("tag")
parser.add_argument("sweep")
parser.add_argument("--motions", nargs="*", help='"label forward yaw" per motion (motion harness)')
parser.add_argument("--steps", type=int, default=6, help="compared steps per flight (motion harness)")
parser.add_argument("--live", action="store_true", help="leave residency running (motion harness)")
parser.add_argument("--spp", type=int, default=256, help="path-traced reference samples (truth harness)")
parser.add_argument("--components", default="15", help="comma-separated hstComponents masks, one config each (ab harness)")
parser.add_argument("--views", default="sea", help="comma-separated views of ab_test.py: near, farside, sea (ab harness)")
parser.add_argument("--nsys", action="store_true",
                    help="wrap the headless Mogwai in Nsight Systems with GPU metrics; report at HSTR_results/nsight/TAG.nsys-rep "
                         "(nsys.exe is set to run as administrator in its compatibility settings, which GPU metrics need)")
parser.add_argument("--nsys-set", default="", help="--nsys: the GPU metric set (nsys --gpu-metrics-set), e.g. ad10x-gfxt; default set if empty")
parser.add_argument("--ngfx", nargs=2, type=int, metavar=("START", "STOP"),
                    help="Nsight Graphics GPU Trace of flight frames [START, STOP) (0..47) of each timed flight (motion harness); "
                         "reports at HSTR_results/ngfx/TAG_<motion>_<arm>_f<START>-<STOP>.ngfx-gputrace")
parser.add_argument("--ngfx-metrics", default="Throughput Metrics", help="--ngfx: the Ada metric set name")
args = parser.parse_args()

sys.path.insert(0, str(BENCH))
from elevate import NGFX, NSYS, elevate, run_hidden  # noqa: E402

if args.nsys or args.ngfx:
    elevate(__file__, ROOT)  # nsys / ngfx need admin for GPU counters; see elevate.py. Mogwai stays --headless.
from sea_config import REFERENCE  # noqa: E402

tests = runpy.run_path(args.sweep)["TESTS"]
env = dict(os.environ, HSTR_TAG=args.tag, HSTR_BASE=json.dumps(REFERENCE), HSTR_TESTS=json.dumps(tests))
for suffix in ("_test.txt", "_test.jsonl"):
    (RESULTS / f"{args.tag}{suffix}").unlink(missing_ok=True)
if args.harness == "ab":
    configs = [[f"beam c{c}", {"hstComponents": int(c)}] for c in args.components.split(",")]
    env.update(HSTR_VIEWS=args.views, HSTR_CAPTURE="0", HSTR_INTERLEAVE="1", HSTR_CONFIGS=json.dumps(configs))
    script = "scripts/HSTR/bench/ab_test.py"
elif args.harness == "truth":
    env.update(HSTR_VIEWS=args.views, HSTR_SPP=str(args.spp))
    script = "scripts/HSTR/bench/sea_truth.py"
else:
    # "label forward yaw" or "label forward yaw sunRadiansPerFrame" - the sun rate is optional and per motion, so one run can
    # hold a static-sun control beside a moving-sun arm instead of comparing across runs.
    motions = [[m.split()[0].replace("_", " ")] + [float(v) for v in m.split()[1:]] for m in args.motions] if args.motions else None
    if motions:
        env["HSTR_MOTIONS"] = json.dumps(motions)
    env.update(HSTR_STEPS=str(args.steps), HSTR_FREEZE="0" if args.live else "1")
    script = "scripts/HSTR/bench/sea_motion.py"
log = RESULTS / f"{args.tag}.log"
command = [str(MOGWAI), "--headless", f"--script={script}"]
if args.nsys:
    # Mogwai starts under `nsys launch`, which traces nothing until told: sea_motion.py sends `nsys start` just before each timed
    # flight and `nsys stop` after it, so each report holds that flight's frames and not the ~1,200-frame settle (a whole-run
    # capture with GPU metrics was an 807 MB stream that took 20 GB and many minutes to import).
    (RESULTS / "nsight").mkdir(exist_ok=True)
    session = f"hstr_{args.tag}"
    env.update(HSTR_NSYS=NSYS, HSTR_NSYS_SESSION=session, HSTR_NSYS_OUTPUT=str(RESULTS / "nsight" / args.tag),
               HSTR_NSYS_SET=args.nsys_set)
    command = [NSYS, "launch", f"--session-new={session}", "--trace=dx12,dx12-annotations,nvtx", "--wait=all"] + command
if args.ngfx:
    # Nsight Graphics GPU Trace: ngfx launches Mogwai with the trace injected and idle; sea_motion.py starts it before flight frame
    # START and stops it after frame STOP-1 through the SDK (Mogwai's gpuTraceStart/Stop), so only those frames are traced.
    # No --platform: ngfx is a Qt program and Qt takes --platform as its own option ("no Qt platform plugin could be initialized").
    (RESULTS / "ngfx").mkdir(exist_ok=True)
    env["HSTR_NGFX"] = f"{args.ngfx[0]} {args.ngfx[1]}"
    command = [NGFX, "--activity", "GPU Trace Profiler", "--exe", str(MOGWAI), "--dir", str(ROOT),
               "--args", subprocess.list2cmdline(command[1:]), "--output-dir", str(RESULTS / "ngfx"), "--no-timeout",
               "--start-with-ngfx-sdk", "--stop-with-ngfx-sdk", "--architecture", "Ada", "--metric-set-name", args.ngfx_metrics,
               "--auto-export"]
with open(log, "w") as f:
    (run_hidden if args.nsys or args.ngfx else subprocess.run)(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
errors = [line for line in open(log, errors="replace") if "(Error)" in line or "Exception" in line or "Error when loading" in line or "RuntimeError" in line]
if errors:
    print("".join(errors[:5]))
if args.harness in ("ab", "truth"):
    text = RESULTS / f"{args.tag}_test.txt"
    print(text.read_text() if text.exists() else "no results")
else:
    from sea_summary import summarize  # noqa: E402

    summarize(args.tag)
