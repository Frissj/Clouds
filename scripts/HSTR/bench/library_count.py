# Loads the sunset launcher's graph, renders one frame and exits: the startup log carries the library's brick count against the atlas.
# Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/library_count.py
from pathlib import Path
from falcor import *

launcher = Path(__file__).resolve().parents[1] / "SunsetCloudSea.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
m.renderFrame()
exit()
