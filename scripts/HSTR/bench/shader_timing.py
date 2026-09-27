# Load time of the Intel sea's shaders: loads the launcher's graph and scene with HSTR_SHADER_TIMING=1 (HSTRCloud logs the front end
# and every kernel's compile), then exits. Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/shader_timing.py
import os
from pathlib import Path
from falcor import *

os.environ["HSTR_SHADER_TIMING"] = "1"
launcher = Path(__file__).resolve().parents[1] / f"IntelCloudSea{os.environ.get('HSTR_INTEL_SET', 'half').capitalize()}.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
exit()
