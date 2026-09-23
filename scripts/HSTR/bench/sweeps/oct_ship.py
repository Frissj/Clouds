import runpy
from pathlib import Path

# The shipping octahedral configuration alone (OCT_T001, parallax guard 1), for captures: run_sea.py motion TAG this --nsys.
TESTS = [("ship", runpy.run_path(str(Path(__file__).with_name("march_cost_split.py")))["BASE"])]
