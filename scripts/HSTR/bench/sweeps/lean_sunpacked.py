import os
import runpy
from pathlib import Path

# The packed sun atlas in the shipping march (cloudSunPacked 2: RG16F, the bake unchanged in r, its brick's density in g, written by
# bakeCloudSun): with cloudSunPackedRead every lit brick sample whose instance's bake is at the brick's own level reads density and
# bake from one fetch, and its sun skips the resolved lookup and the bake fetch (the span probe: one fetch instead of two saved
# ~0.094 ms of 0.586 at 1080p, packed3). The atlas must exist from the first frame, so it is a launch property:
#     HSTR_LAUNCH_PROPS='{"cloudSunPacked": 2}' python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/lean_sunpacked.py
#         --steps 1 --motions "walk 2 0.004" "sprint 20 0"
# Interleaved anchors; errors must move a little (filter rounding of the density read from another texture) but not jump; the sanity
# arm (no sun term) must change everything.
# MEASURED (sunpacked1): DEVICE_REMOVED in the first 4K reference frame (nvlddmkm 153, a timeout). sunpacked2: both packed arms the
# anchors' numbers to the last digit - cloudSunPacked set after load never made its atlas (now in cloudSettings). sunpacked3 / 4
# (videoUsedMB, FALCOR_ALLOC_LOG): 13,926 of an 11,227 MB budget, 4.85 GB of it beam lists sized for four levels at beamLevels 1.
# MEASURED (sunpacked5, lists sized by level, 8,389 of 11,227 MB): walk units 1.53 / 1.53 -> packed 1.69 / 1.67 ms, sprint 1.57 /
# 1.57 -> 1.61 / 1.61, query 2.02 unchanged in both; errors unchanged within noise (walk 8.10 / 8.10 -> 7.98 / 8.20% over 0.02) -
# but every arm's error is ~60x leantstep1's 0.137% (p99.9 0.819 against 0.0256), see LISTS.
BASE = runpy.run_path(str(Path(__file__).with_name("lean_march.py")))["BASE"]
TESTS = [
    ("march", dict(BASE, cloudSunPackedRead=False, beamListsFull=False)),
    ("march packed", dict(BASE, cloudSunPackedRead=True, beamListsFull=False)),
    ("march again", dict(BASE, cloudSunPackedRead=False, beamListsFull=False)),
    ("march packed again", dict(BASE, cloudSunPackedRead=True, beamListsFull=False)),
    ("sanity: no sun term", dict(BASE, cloudSunPackedRead=False, beamListsFull=False, hstComponents=1 | 4 | 8)),
]
# HSTR_PACKED_SWEEP=lists, launched without the packed atlas (the 85-per-tile lists add 4.85 GB): is the error jump the lists'
# resize? The old size between two anchors at the new one; the sanity arm last.
LISTS = [
    ("march", dict(BASE, cloudSunPackedRead=False, beamListsFull=False)),
    ("march lists 85", dict(BASE, cloudSunPackedRead=False, beamListsFull=True)),
    ("march again", dict(BASE, cloudSunPackedRead=False, beamListsFull=False)),
    ("sanity: no sun term", dict(BASE, cloudSunPackedRead=False, beamListsFull=False, hstComponents=1 | 4 | 8)),
]
# MEASURED (lists1, walk): march 8.098% / lists 85 7.977% / again 8.097% over 0.02 - sunpacked5's numbers arm for arm, so the
# lists' resize is not the error, and the 2nd arm's 7.977 is its position, not packed. p99.9 0.819 in every arm, sanity too.
# HSTR_PACKED_SWEEP=image with HSTR_MOTION_OUT=<dir>: the scored frames and their references, to see where the error is.
IMAGE = [
    ("march", dict(BASE, cloudSunPackedRead=False, beamListsFull=False)),
    ("sanity: no sun term", dict(BASE, cloudSunPackedRead=False, beamListsFull=False, hstComponents=1 | 4 | 8)),
]
# MEASURED (image2): the far sea active and valid in both frames; the exact view's miss path (a ray past the near window's box)
# wrote sky alone. Fixed (execute, the world-cache view): image3 walk 8.10 -> 4.95% over 0.02, p99.9 0.819 -> 0.290. What remains:
# silhouettes and blocky speckle sheets in the gaps under mid-distance clouds, beam arm only (rows 1080-1295: 12.5% over 0.02).
# HSTR_PACKED_SWEEP=speckle with HSTR_MOTION_OUT: which part draws the sheets - each arm takes one suspect away (the lean march's
# lookups, the guard's held blocks, the far sea; the reference inherits seaFarField, BASE does not set it).
MARCH = dict(BASE, cloudSunPackedRead=False, beamListsFull=False)
SPECKLE = [
    ("march", MARCH),
    ("not lean", dict(MARCH, beamShipMask=BASE["beamShipMask"] & ~1024)),
    ("no guard", dict(MARCH, beamGuard=False)),
    ("no far sea", dict(MARCH, seaFarField=False)),
    ("march again", MARCH),
    ("sanity: no sun term", dict(MARCH, hstComponents=1 | 4 | 8)),
]
# MEASURED (speckle1, walk, PNG band = rows 1080-1295 over 0.02): march 4.95% over 0.02 (p99.9 0.290; band 12.5%), not lean 4.61%
# (11.8%), no far sea 1.30% (p99.9 0.061; band 1.6%), march again 5.12%, sanity 14.0%; no guard 34.2% (a broken frame, not a
# suspect cleared). The sheets are the far sea's. The reference's frames follow a property change (mFarSeaDirty): it reads a layer
# recomputed whole, the moving arm one refreshed 1/8 a run (seaFarRefresh 8) and reprojected for the rest.
# HSTR_PACKED_SWEEP=farsea: ORACLE - every texel every run (seaFarRefresh 1), and 2; is the reprojection what draws them?
FARSEA = [
    ("march", MARCH),
    ("far refresh 1", dict(MARCH, seaFarRefresh=1)),
    ("far refresh 2", dict(MARCH, seaFarRefresh=2)),
    ("march again", MARCH),
    ("sanity: no sun term", dict(MARCH, hstComponents=1 | 4 | 8)),
]
# MEASURED (farsea1, walk): seaFarRefresh 8 / 1 / 2 / 8 again: farSea 0.45 / 1.14 / 0.64 / 0.45 ms, over 0.02 4.95 / 2.74 / 2.79 /
# 4.99% (p99.9 0.290 / 0.086 / 0.086 / 0.290; PNG band 12.5 / 3.0 / 3.1 / 11.8%), sanity 14.0%. The reprojected texels draw them;
# 2 recovers nearly all of 1 for +0.19 ms. Noted beside farSeaDecision.
# HSTR_PACKED_SWEEP=farfix: the two suspects as probes (seaFarProbe bit 7: reproject through the source texel's distance; bit 8:
# march again every run the texels whose last march lit the near / far fade), alone and together, against refresh 2; one counting
# arm (bit 6, stalls - its time is not comparable) gives how many texels bit 8 marches again.
FAR = dict(MARCH, seaFarProbe=0, seaFarRefresh=8)
FARFIX = [
    ("march", FAR),
    ("far source distance", dict(FAR, seaFarProbe=128)),
    ("far band refresh", dict(FAR, seaFarProbe=256)),
    ("far both", dict(FAR, seaFarProbe=128 | 256)),
    ("far refresh 2", dict(FAR, seaFarRefresh=2)),
    ("far both counted", dict(FAR, seaFarProbe=128 | 256 | 64)),
    ("march again", FAR),
    ("sanity: no sun term", dict(FAR, hstComponents=1 | 4 | 8)),
]
# MEASURED (farsea2, walk; farSea ms / over 0.02 / p99.9 / PNG all, band): march 0.46 / 4.95% / 0.290 / 3.65, 12.5%; source
# distance (bit 7) 0.46 / 4.39% / 0.244 / 3.07, 8.1%; band refresh (bit 8) 0.46 / 3.96% / 0.172 / 2.40, 3.4%; both 0.47 / 3.97% /
# 0.145 / 2.39, 3.2%; refresh 2 0.66 / 2.80% / 0.086 / 1.88, 3.0%; march again 0.46 / 4.88%; sanity 14.0%. Counted: bit 8 marches
# 45-63k texels again a run beside the ~120k scheduled segments, for no measurable time (they start inside cloud and end soon).
# What both still lack against refresh 2: crack-thin far cloud edges (rows 648-1079) - the bilinear chain of reprojections.
# HSTR_PACKED_SWEEP=faredge: on top of both, nearest reprojection (bit 9) and edge texels marched (bit 10).
BOTH = dict(FAR, seaFarProbe=128 | 256)
FAREDGE = [
    ("march", FAR),
    ("far both", BOTH),
    ("far both nearest", dict(BOTH, seaFarProbe=128 | 256 | 512)),
    ("far both edges", dict(BOTH, seaFarProbe=128 | 256 | 1024)),
    ("far both edges nearest", dict(BOTH, seaFarProbe=128 | 256 | 512 | 1024)),
    ("far both refresh 4", dict(BOTH, seaFarRefresh=4)),
    ("far refresh 2", dict(FAR, seaFarRefresh=2)),
    ("march again", FAR),
    ("sanity: no sun term", dict(FAR, hstComponents=1 | 4 | 8)),
]
TESTS = {"lists": LISTS, "image": IMAGE, "speckle": SPECKLE, "farsea": FARSEA, "farfix": FARFIX, "faredge": FAREDGE}.get(os.environ.get("HSTR_PACKED_SWEEP"), TESTS)
