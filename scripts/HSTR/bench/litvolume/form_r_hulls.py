"""Form R gate R1 (LIT_VOLUME.md section 6c): the shared occupancy hulls as files HSTRCloud loads (property formRHulls = DIR).

Every a{i}_L{l}.npy in --volumes (hstrlib_dense, float16 (z, y, x)) gives hull_a{i}_L{l}.bin (dilated by one cell, as R0) and
hull_a{i}_L{l}_u.bin (undilated: R1's sanity arm, which must change the image). form_r_scene.hull_mesh: vertices in the asset's
source voxels (voxel k centred at k, the renderer's instance-local x), triangles counter-clockwise seen from outside.
File: uint32 vertex count, uint32 triangle count, float32 x y z per vertex, uint32 i j k per triangle.

    python form_r_hulls.py --volumes DIR --out DIR
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from form_r_scene import hull_cells, hull_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(args.volumes, "a*_L*.npy"))):
        m = re.fullmatch(r"a(\d+)_L(\d+)\.npy", os.path.basename(f))
        if not m:
            continue
        asset, level = int(m.group(1)), int(m.group(2))
        volume = np.load(f, mmap_mode="r")
        for dilate, suffix in ((True, ""), (False, "_u")):
            verts, idx = hull_mesh(hull_cells(volume, dilate), level)
            path = os.path.join(args.out, f"hull_a{asset}_L{level}{suffix}.bin")
            with open(path, "wb") as out:
                np.array([len(verts), len(idx)], np.uint32).tofile(out)
                verts.astype(np.float32).tofile(out)
                idx.astype(np.uint32).tofile(out)
            print(f"{path}: {len(idx)} triangles, {len(verts)} vertices", flush=True)


if __name__ == "__main__":
    main()
