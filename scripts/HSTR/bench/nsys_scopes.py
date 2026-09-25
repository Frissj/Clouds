# Per FALCOR_PROFILE scope of an nsys_export.py sqlite: GPU ms per frame and GPU metrics averaged over the scope's ranges.
# Usage: python scripts/HSTR/bench/nsys_scopes.py <report>.sqlite [frames=48]
import bisect, sqlite3, sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 48
names = dict(db.execute("select id, value from StringIds"))
ranges = defaultdict(list)
for start, end, text in db.execute("select start, end, textId from DX12_WORKLOAD where textId is not null"):
    ranges[names.get(text, str(text))].append((start, end))
wanted = ["SMs Active", "SM Issue", "Compute Warps in Flight", "Unallocated Warps in Active SMs", "DRAM Read Bandwidth",
          "DRAM Write Bandwidth", "L2 Hit Rate", "L1 Hit Rate"]
metrics = {n: i for i, n in db.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")}
series = {}
for n in wanted:
    key = next((k for k in metrics if k.startswith(n)), None)
    if key is not None:
        rows = db.execute("select timestamp, value from GPU_METRICS where metricId = ? order by timestamp", (metrics[key],)).fetchall()
        series[key] = ([r[0] for r in rows], [r[1] for r in rows])
print(f"{'scope':44s} {'ms/f':>6s} {'n/f':>5s} " + " ".join(f"{k[:14]:>14s}" for k in series))
for name, spans in sorted(ranges.items(), key=lambda kv: -sum(e - s for s, e in kv[1])):
    total = sum(e - s for s, e in spans) / 1e6 / frames
    if total < 0.01:
        continue
    cols = []
    for ts, vs in series.values():
        acc = cnt = 0
        for s, e in spans:
            i, j = bisect.bisect_left(ts, s), bisect.bisect_right(ts, e)
            acc += sum(vs[i:j]); cnt += j - i
        cols.append(f"{acc / cnt:14.1f}" if cnt else f"{'-':>14s}")
    print(f"{name[:44]:44s} {total:6.3f} {len(spans) / frames:5.1f} " + " ".join(cols))
