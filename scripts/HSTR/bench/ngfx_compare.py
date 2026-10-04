# Side-by-side per-range metrics of Nsight Graphics GPU Trace auto-exports (GPUTRACE_REGIMES.xls + D3DPERF_EVENTS.xls).
# Usage: python scripts/HSTR/bench/ngfx_compare.py RANGE_SUFFIX DIR_A DIR_B ...
#   e.g. python scripts/HSTR/bench/ngfx_compare.py march/units HSTR_results/ngfx/refillgfx0_BASE HSTR_results/ngfx/refillgfx1_BASE
# Every range whose name ends with RANGE_SUFFIX (one per traced frame) is shown: duration, compute warps active, L1 sector hit
# rate, register-allocation launch stalls, long-scoreboard (L1TEX) stalls, SM throughput, DRAM throughput, instructions executed.
import csv
import sys
from pathlib import Path

METRICS = [
    ("warps/cyc", "tpc__warps_active_shader_cs_realtime.avg.per_cycle_elapsed"),
    ("L1 hit %", "l1tex__t_sector_hit_rate.pct"),
    ("regalloc %", "warp_launch_cycles_stalled_shader_cs_reason_register_allocation.avg.pct_of_peak_sustained_elapsed"),
    ("LGSB %", "tpc__warps_issue_stalled_long_scoreboard_pipe_l1tex.avg.pct_of_peak_sustained_elapsed"),
    ("SM thr %", "GPUTrace.sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM thr %", "dramc__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("inst", "smsp__inst_executed.sum"),
]


def table(directory, name):
    rows = list(csv.reader(open(Path(directory) / name, encoding="utf-8", errors="replace"), delimiter="\t"))
    return rows[0], rows[1:]


def column(header, key):
    hits = [i for i, c in enumerate(header) if c.endswith(key)]
    return hits[0] if hits else None


suffix = sys.argv[1]
for directory in sys.argv[2:]:
    header, rows = table(directory, "GPUTRACE_REGIMES.xls")
    eheader, erows = table(directory, "D3DPERF_EVENTS.xls")
    # The events table is an indented tree of bare range names.
    durations = [r[1] for r in erows if r and r[0].strip() == suffix.split("/")[-1]]
    print(f"== {directory}")
    print("   ms: " + ", ".join(durations))
    for r in (r for r in rows if r and r[0].endswith(suffix)):
        values = []
        for label, key in METRICS:
            i = column(header, key)
            values.append(f"{label} {float(r[i]):.4g}" if i is not None and r[i] else f"{label} -")
        print("   " + " | ".join(values))
