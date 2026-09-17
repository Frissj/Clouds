# Compact table of a sea_motion.py run: python sea_table.py TAG [TAG ...]
import json
import sys

for tag in sys.argv[1:]:
    for line in open(f"C:/Users/Friss/Documents/HSTR_results/{tag}_test.jsonl"):
        r = json.loads(line)
        g, e = r["gpu"], r["errors"]
        mean = lambda k: sum(x.get(k, 0) for x in e) / len(e)
        print(f"{r['motion']:7s} {r['test']:12s} {g.get('HSTRCloud', 0):5.2f} q {g.get('beamQueries', 0):5.2f} "
              f"march {g.get('resolve/march', 0):5.2f} wall {g['wall']:6.2f} cp {mean('carriedPoints'):7.0f} px {mean('carriedPixels'):7.0f} "
              f"steps m {mean('marchedSteps'):9.0f} c {mean('carriedSteps'):9.0f} err {100 * mean('over02'):.3f} "
              f"worst {100 * max(x['over02'] for x in e):.3f}")
