import json
import sys

# One line per motion and test of sea_motion.py runs: python sea_summary.py TAG [TAG...]
RESULTS = "C:/Users/Friss/Documents/HSTR_results"


def summarize(tag):
    print("==", tag)
    for line in open(f"{RESULTS}/{tag}_test.jsonl"):
        d = json.loads(line)
        g = d["gpu"]
        e = d["errors"]
        mean = lambda k: sum(x[k] for x in e) / len(e)
        r = g.get("residency", {})
        cpu = g.get("cpu", {})
        print(f'{d["motion"]:8s} {d["test"]:18s} {g["HSTRCloud"]:6.2f} ms wall {g["wall"]:6.1f} '
              f'q0 {g.get("beamQueries/beamLevel0/queries0", 0):5.2f} march {g.get("resolve/march", 0):5.2f} | '
              f'>.02 {100 * mean("over02"):.3f}% worst {100 * max(x["over02"] for x in e):.3f}% p999 {mean("p999"):.3f} '
              f'carried {mean("carriedPoints"):.0f}/{mean("carriedPixels"):.0f} | residency {cpu.get("residency", 0):.1f} '
              f'cut {cpu.get("residency/cut", 0):.1f} bakes {cpu.get("residency/apply/sunBakes", 0):.1f} '
              f'pops {r.get("cutPops")} ordered {r.get("cutOrdered")} loaded {r.get("loaded")} pending {r.get("pending")}')


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        summarize(tag)
