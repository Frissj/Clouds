import sys
from pathlib import Path

# Exports an Nsight Systems report to sqlite beside it (nsys needs admin: see elevate.py), for reading with python's sqlite3.
#   python scripts/HSTR/bench/nsys_export.py C:/Users/Friss/Documents/HSTR_results/nsight/TAG_walk_ship.nsys-rep
sys.path.insert(0, str(Path(__file__).resolve().parent))
from elevate import NSYS, elevate, run_hidden  # noqa: E402

report = Path(sys.argv[1]).resolve()
elevate(__file__, report.parent)
with open(report.with_suffix(".export.log"), "w") as log:
    run_hidden([NSYS, "export", "--type=sqlite", "--force-overwrite=true", f"--output={report.with_suffix('.sqlite')}", str(report)],
               stdout=log, stderr=log)
