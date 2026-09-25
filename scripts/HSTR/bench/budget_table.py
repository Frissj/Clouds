# One line per arm of a sea_motion run: frame ms, the dirty passes, the tile tests, the resolve, and the gate's statistics.
# Usage: python scripts/HSTR/bench/budget_table.py TAG
import json, sys
for line in open(f"C:/Users/Friss/Documents/HSTR_results/{sys.argv[1]}_test.jsonl"):
    r = json.loads(line); g = r["gpu"]; e = r["errors"][0]
    q = g.get("beamQueries/beamLevel0/queries0/beamDirty/query", 0); u = g.get("march/units", 0)
    tiles = g.get("beamQueries/beamLevel0", 0) - g.get("beamQueries/beamLevel0/queries0", 0)
    other = g.get("beamQueries/beamLevel0/queries0/beamDirty", 0) - q
    print(f'{r["motion"]:7s} {r["test"]:18s} {g.get("HSTRCloud", 0):5.2f} ms | query {q:.2f} units {u:.2f} tiles {tiles:.2f} '
          f'classify+ {other:.2f} resolve {g.get("resolve", 0):.2f} | >0.02 {100 * e["over02"]:.3f}% p99.9 {e["p999"]:.3f} '
          f'max {e["max"]:.3f} marchTiles {e["marchTiles"]}')
