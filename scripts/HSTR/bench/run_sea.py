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
args = parser.parse_args()

sys.path.insert(0, str(BENCH))
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
    motions = [[m.split()[0].replace("_", " "), float(m.split()[1]), float(m.split()[2])] for m in args.motions] if args.motions else None
    if motions:
        env["HSTR_MOTIONS"] = json.dumps(motions)
    env.update(HSTR_STEPS=str(args.steps), HSTR_FREEZE="0" if args.live else "1")
    script = "scripts/HSTR/bench/sea_motion.py"
log = RESULTS / f"{args.tag}.log"
with open(log, "w") as f:
    subprocess.run([str(MOGWAI), "--headless", f"--script={script}"], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
errors = [line for line in open(log, errors="replace") if "(Error)" in line or "Exception" in line or "Error when loading" in line or "RuntimeError" in line]
if errors:
    print("".join(errors[:5]))
if args.harness in ("ab", "truth"):
    text = RESULTS / f"{args.tag}_test.txt"
    print(text.read_text() if text.exists() else "no results")
else:
    from sea_summary import summarize  # noqa: E402

    summarize(args.tag)
