# Two-capture pick for sunset_motion.py's capture oracle (HSTR_MOTION_CAPTURE_ORACLE with HSTR_MOTION_OUT): per stop, each pixel of
# the view synthesized from the capture h behind ("old") and h ahead ("future") is scored against the fresh rebuild at the camera,
# and best-of-two takes whichever is closer per pixel (no blending: an information ceiling, not a selector). The moving frame that
# score() saved at the same camera is today's iso p8 baseline.
# Metric: |log luminance| difference of the tone-mapped images over THRESHOLD - an 8-bit proxy for referenceNoiseError; the
# single-capture shares are printed beside the in-engine ones so the proxy can be read against them.
# Usage: python scripts/HSTR/bench/capture_oracle.py C:/Users/Friss/Documents/HSTR_results/capture_oracle1
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THRESHOLD = 0.02
folder = Path(sys.argv[1])


def load(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return np.log(a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32) + 1e-3)


def find(label):
    hits = sorted(folder.glob(f"sunset_motion_{label}.*.png"))
    return hits[-1] if hits else None


stops = sorted({int(m.group(1)) for p in folder.glob("sunset_motion_oracle*_fresh.*.png")
                if (m := re.search(r"oracle(\d+)_fresh", p.name))})
for n in stops:
    fresh = load(find(f"oracle{n}_fresh"))
    # score1 is the parked score; stop n's moving frame is score n + 1.
    moving = find(f"score{n + 1}_moving")
    errors = {}
    for p in folder.glob(f"sunset_motion_oracle{n}_h*.png"):
        h = float(re.search(r"_h(-?[\d.]+)\.", p.name).group(1))
        errors[h] = np.abs(load(p) - fresh)
    line = f"stop {n}:"
    if moving:
        line += f" moving (iso p8) {100 * np.mean(np.abs(load(moving) - fresh) > THRESHOLD):.3f}%"
    print(line)
    print(f"  single captures: " + ", ".join(f"h {h:g} {100 * np.mean(e > THRESHOLD):.3f}%" for h, e in sorted(errors.items())))
    for h in sorted(k for k in errors if k > 0 and -k in errors):
        old, future = errors[-h] > THRESHOLD, errors[h] > THRESHOLD
        best = np.minimum(errors[-h], errors[h]) > THRESHOLD
        bad = old | future
        total = max(int(bad.sum()), 1)
        print(f"  h {h:g}: old {100 * old.mean():.3f}% future {100 * future.mean():.3f}% best-of-two {100 * best.mean():.3f}%"
              f" | of {int(bad.sum())} pixels bad in either: old-only bad {100 * (old & ~future).sum() / total:.1f}%"
              f" (future rescues), future-only bad {100 * (future & ~old).sum() / total:.1f}% (old rescues),"
              f" both bad {100 * (old & future).sum() / total:.1f}%")
