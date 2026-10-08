# Why moving is slow on the sunset sea: SunsetCloudSea.py's graph exactly as the launcher runs it (two layers, residency live, sun
# reach 8), headless. sea_motion.py cannot answer this: it freezes residency and runs one layer. One settle, then per motion (+z,
# units a frame): a live flight with per-frame wall times and counter deltas, a profiled live flight (GPU and CPU leaf scopes), and
# arms alternating every CHUNK frames along one continued path: live, residency frozen, beam stream invalidation off.
# Usage: Mogwai.exe --headless --script=scripts/HSTR/bench/sunset_motion.py
#   HSTR_RES=WxH, HSTR_MOTION_SPEEDS=comma list of units a frame (default "2,20"), HSTR_MOTION_SETTLE=settle frames (default 1800).
import os
import re
import time
from pathlib import Path
from falcor import *

root = Path(__file__).resolve().parents[3]
launcher = root / "scripts" / "HSTR" / "SunsetCloudSea.py"
exec(launcher.read_text(), {**globals(), "__file__": str(launcher)})
m.resizeFrameBuffer(*[int(v) for v in os.environ.get("HSTR_RES", "3840x2160").split("x")])
hstr = m.activeGraph.getPass("HSTRCloud")
# HSTR_MOTION_PROPS: a Python dict literal of HSTRCloud properties applied before the settle (e.g. {'skyModel': 0} for scoring, as
# the quality gates run: the exact view and the beam resolve do not apply the aerial perspective alike).
if os.environ.get("HSTR_MOTION_PROPS"):
    hstr.set_properties(eval(os.environ["HSTR_MOTION_PROPS"]))
SCORE = os.environ.get("HSTR_MOTION_SCORE", "0") != "0"
if SCORE:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sea_config import REFERENCE
cam = m.scene.camera
position = [float3(cam.position.x, cam.position.y, cam.position.z)]
VIEW = float3(cam.target.x, cam.target.y, cam.target.z) - position[0]
COUNTERS = ("beamDirtyBlocks", "beamDirtyOwnMarched", "beamDirtyApronMarched", "beamWarpHeld", "seaTilesChanged", "sunBakeFrames",
            "densityChangedFrames", "cuts", "cutTotalMs", "cutPops", "skirtMaskChecks", "skirtMaskMismatches",
            "dirtyStatFrames", "dirtyStatBlocks", "dirtyStatUnits")
LEVELS = ("desired", "mapped", "pending", "mapBacklog", "sunWaiting", "sunBakesFrame", "sunSlotsFree", "sunStale","beamWarpOn", "beamPolicyToleranceNow",
          "beamMarchTiles", "bindCpuMs") + tuple(  # rtSpanProbe's last frame (present only while it is on).
          f"rtProbe{n}" for n in ("Boxes", "Instances", "Rays", "HitRays", "Candidates", "Max", "Overflow", "Covered")) + tuple(
          f"rtProbeBin{b}" for b in range(8))
CHUNK = int(os.environ.get("HSTR_MOTION_CHUNK", "20"))
DROP = CHUNK // 4  # Frames at the start of each chunk carrying the previous arm's state.


DIRTY = ("beamDirtyBlocks", "beamDirtyUnverified", "beamDirtyOwnMarched", "beamDirtyApronMarched", "beamWarpEdgeUnits", "beamHistoryHeld", "beamHistoryLanded", "beamHistoryChanged", "beamCoverPoints", "beamCoverNearer", "beamCoverChanged", "beamGuardNeighbourListed", "beamWarpHeld", "beamWarpListed",
         "beamWarpOn", "beamPolicyToleranceNow", "beamFrameDim") + tuple(
         f"beamProbe{kind}{name}" for kind in ("Rays", "Units", "RaySteps", "UnitSteps")
         for name in ("Scored", "InPlace", "Reprojected", "Either")) + (  # beamRepairProbe: per build, so counted (HSTR_MOTION_COUNT)
         # beamGuardMotion: blocks listed by first failing pyramid level, by the zoom cap; held that the isotropic bound would list.
         tuple(f"beamGuardFail{level}" for level in range(12)) + ("beamGuardRescued", "beamGuardHeld") +
         # beamLayerProbe: the layered lookups of the dirty marches by outcome, density samples by layers with density, and the
         # marches' lane steps, warp-paid steps and warps.
         tuple(f"beamLayer{n}" for n in ("NoCloud", "OutOfBox", "BrickEmpty", "Density", "Proxy", "Dense0", "Dense1", "Dense2",
                                         "LaneSteps", "PaidSteps", "Warps", "TightZero", "MaskMismatch", "MaskEmpty0", "MaskEmpty1",
                                         "VoxelEmpty", "VoxelDense", "TightRun", "BlockDense", "BlockDenseSum", "BlockDenseEdge",
                                         "Miss0", "Miss1", "Miss2", "Miss3", "Miss4", "Miss5", "Miss6", "Miss7")) +
         # ... per layer by what answered, and layer 1's lookups after a dense layer 0.
         tuple(f"beamSplit{layer}{n}" for layer in (0, 1)
               for n in ("BlockSkip", "VoxelSkip", "NoCloud", "OutOfBox", "BrickEmpty", "Density", "Proxy")) +
         ("beamSplit1EmptyAfterDense", "beamSplit1DenseAfterDense", "beamLayerUnitLaneSteps", "beamLayerUnitPaidSteps",
          "beamLayerUnitWarps") +
         # ... and a perfect ray start's oracle (kBeamLeadSteps): leading empty steps, warp-paid steps with that run as one step,
         # rays with no density and their steps, for the dirty query and the units.
         tuple(f"beamLead{p}{n}" for p in ("Query", "Unit") for n in ("Empty", "IdealPaid", "NoneRays", "NoneSteps")) +
         # ... and the dirty query's waves by their longest ray (half-octave bins) and their cost by quarter of the listed order.
         tuple(f"beamQueryWaveHist{k}" for k in range(20)) + tuple(f"beamQueryWaveQuarter{k}" for k in range(4)) +
         # ... and beamRepairProbe's re-marched samples by distance: scored, steps, steps whose old value held within 0.02.
         tuple(f"beamProbeDist{kind}{b}{n}" for kind in ("Unit", "Ray") for b in range(12) for n in ("Scored", "Steps", "Stable")) +
         # ... and beamUnitStart's misses (kBeamUnitStartProbe).
         ("beamUnitStartHinted", "beamUnitStartMissed", "beamUnitStartLost", "beamUnitStartMissedVoxels16",
          "beamUnitStartRadianceOff", "beamUnitStartTransmittanceOff", "beamUnitStartOffNoMiss", "beamUnitStartStepsSaved") +
         # ... and a perfect ray end as well (kBeamTailSteps): trailing empty steps, warp-paid steps with both runs as one step.
         tuple(f"beamTail{p}{n}" for p in ("Query", "Unit") for n in ("Empty", "IdealPaid")) +
         # ... and a per-ray layer hint's (kBeamLayerRayHint): brick walks in layers the ray found no density in, those ray-layers.
         tuple(f"beamLayerHint{p}{n}" for p in ("Query", "Unit") for n in ("Walks", "Layers")) +
         # ... and the layer-free run skip's (kBeamRunSkip): runs, blocks crossed, those in a layer-free 16-voxel parent, capped runs.
         ("beamRunSkipRuns", "beamRunSkipBlocks", "beamRunSkipCoarseBlocks", "beamRunSkipCapped") +
         # ... and beamUnitSpan's (kBeamUnitSpanProbe): units with a mask, jumps, against the unmasked march from the same start
         # units off by > 0.02 in log radiance / in transmittance, steps saved.
         ("beamUnitSpanUnits", "beamUnitSpanJumps", "beamUnitSpanRadianceOff", "beamUnitSpanTransmittanceOff",
          "beamUnitSpanStepsSaved") +
         # ... and formR 2's (kFormRProbe): hull queries, steps outside every hull, density samples there (bricks / proxy), back
         # faces met outside (drift), steps, rays; the hull structure's instances and triangles.
         tuple(f"formR{n}" for n in ("Queries", "OutsideSteps", "Missed", "MissedProxy", "Drift", "Steps", "Rays", "Instances",
                                     "Triangles")) +
         # ... and formB 2's (kFormBProbe): lean samples answered by a table texel / a table empty, sent to the old chain by a
         # fallback word / by being outside the table or level 3+; the last table build's counts.
         tuple(f"formB{n}" for n in ("Texels", "Empties", "Fallbacks", "Outside", "EmptyMissed", "TexelOff", "TexelExtra",
                                     "TexelCompared", "EmptyAlong", "TexelOffOtherLevel", "TexelOffSameLevel", "TexelOtherLevel",
                                     "FallbackL0", "FallbackL1", "FallbackL2", "FallbackCell",
                                     "BuildRecords", "BuildChildren", "BuildFallbackCells", "BuildEmptyCells",
                                     "BuildTwoLayerCells", "BuildWhyUnmapped", "BuildWhySkirt", "BuildWhyCoarser", "BuildWhyFade",
                                     "BuildWhyConstant", "BuildWhyNoBake", "BuildWhyBakeLevel", "BuildWhyTop", "Builds")))


def score():
    # sea_motion.py's step, without moving: the frame in flight is stored and compared with a frame of the same camera. SCORE 1: the
    # exact per-pixel view (sea_config.REFERENCE) - on this sea it differs from the launcher's own settled beam by ~17% of pixels
    # over 0.02 parked, so it cannot judge motion here. SCORE 2: the launcher's own beam view rebuilt from nothing at that camera
    # (beamReset, then parked frames), i.e. what a parked camera shows there - the error that moving adds. Either way the beam
    # history is dropped, so the next frames re-march everything.
    fresh = os.environ.get("HSTR_MOTION_SCORE") == "2"
    saved = {k: hstr.properties[k] for k in REFERENCE if k in hstr.properties}
    hstr.set_properties({"storeExact": True, "compareReference": True, "compareExact": True, "compareBlock": 1})
    # HSTR_MOTION_PATH (with HSTR_MOTION_OUT): the moving frame's per-pixel resolve path beside its image (beamPathDump).
    if os.environ.get("HSTR_MOTION_PATH") and os.environ.get("HSTR_MOTION_OUT"):
        n = getattr(score, "taken", 0) + 1
        hstr.set_properties({"beamPathDump": os.path.join(os.environ["HSTR_MOTION_OUT"], f"sunset_motion_score{n}_path.bin")})
    m.renderFrame()
    # HSTR_MOTION_OUT: the frame in flight and, below, what it is scored against, numbered per score.
    score.taken = getattr(score, "taken", 0) + 1
    capture(f"score{score.taken}_moving")
    s = dict(cloud_stats())
    marched = float(hstr.properties["beamMarchedFraction"])
    if fresh:
        hstr.set_properties({"storeExact": False, "compareReference": False, "compareExact": False, "beamReset": True})
        for _ in range(4):
            m.renderFrame()
    else:
        hstr.set_properties(dict(REFERENCE, compareReference=False, compareExact=False))
        m.renderFrame()
    capture(f"score{score.taken}_against")
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    p = hstr.properties
    result = {"over02": float(p["referenceNoiseError"]), "p999": float(p["referenceLogP999"]), "max": float(p["referenceLogMax"]),
              "marched": marched, **{k: s.get(k) for k in DIRTY}}
    hstr.set_properties(dict(saved, storeExact=False, compareReference=False, compareExact=False))
    return result


COUNT = int(os.environ.get("HSTR_MOTION_COUNT", "0"))
# HSTR_MOTION_CAPTURE_ORACLE: comma list of capture offsets h in world units along the motion (+ ahead, - behind). At each of
# HSTR_MOTION_ORACLE_STOPS cameras P on the first flight (CHUNK frames apart) the beam is rebuilt from nothing at P and stored, then
# for each h rebuilt from nothing at P + h * direction, the camera put back at P and one frame resolved with every block held
# (beamGuardHoldAll): P's view synthesized from one capture h away, scored against the fresh rebuild at P. Residency is frozen
# throughout so every capture sees P's resident set. With HSTR_MOTION_OUT each image is saved for the two-capture pick offline.
ORACLE = [float(v) for v in os.environ.get("HSTR_MOTION_CAPTURE_ORACLE", "").split(",") if v.strip()]


def place(p):
    cam.position = p
    cam.target = p + VIEW


def capture_oracle():
    here = position[0]
    frozen = hstr.properties["cloudResidencyFrozen"]
    n = capture_oracle.taken = getattr(capture_oracle, "taken", 0) + 1
    hstr.set_properties({"cloudResidencyFrozen": True, "beamReset": True})
    for _ in range(4):
        m.renderFrame()
    hstr.set_properties({"storeExact": True, "compareReference": True, "compareExact": True, "compareBlock": 1})
    m.renderFrame()
    capture(f"oracle{n}_fresh")
    hstr.set_properties({"storeExact": False, "compareReference": False, "compareExact": False})
    rows = {}
    for h in ORACLE:
        place(here + float3(*DIRECTION) * h)
        hstr.set_properties({"beamReset": True})
        for _ in range(4):
            m.renderFrame()
        place(here)
        hstr.set_properties({"beamGuardHoldAll": True, "compareReference": True, "compareExact": True, "compareBlock": 1})
        m.renderFrame()
        p, s = hstr.properties, cloud_stats()
        rows[h] = {"over02": float(p["referenceNoiseError"]), "p999": float(p["referenceLogP999"]),
                   "max": float(p["referenceLogMax"]), "dirtyBlocks": s.get("beamDirtyBlocks"),
                   "unverified": s.get("beamDirtyUnverified"), "warpHeld": s.get("beamWarpHeld"), "warpOn": s.get("beamWarpOn")}
        capture(f"oracle{n}_h{h:g}")
        hstr.set_properties({"beamGuardHoldAll": False, "compareReference": False, "compareExact": False})
    # Back to a fresh beam at P, so the flight continues as if it had been parked here.
    hstr.set_properties({"cloudResidencyFrozen": frozen, "beamReset": True})
    for _ in range(4):
        m.renderFrame()
    return rows


def counted(frames, speed):
    # The beam counters are read back only on compared frames (synchronous readbacks, so these frames are not timed): a short
    # continued flight comparing each frame with whatever frame was stored, only for its dirty counts.
    hstr.set_properties({"storeExact": True})
    m.renderFrame()
    hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
    rows = []
    for _ in range(frames):
        fly(1, speed)
        s = cloud_stats()
        rows.append(dict({k: s.get(k) for k in DIRTY if k != "beamFrameDim"}, marched=float(hstr.properties["beamMarchedFraction"]),
                         marchTiles=s.get("beamMarchTiles")))
    hstr.set_properties({"compareReference": False, "compareExact": False})
    return rows


def capture(label):
    if os.environ.get("HSTR_MOTION_OUT"):
        m.frameCapture.outputDir = os.environ["HSTR_MOTION_OUT"]
        m.frameCapture.baseFilename = f"sunset_motion_{label}"
        m.frameCapture.capture()


def cloud_stats():
    return hstr.properties.get("cloudStats", {})


# run_sea.py sunset --ngfx START STOP: an Nsight Graphics GPU Trace of frames [START, STOP) of the first live flight, once (ngfx
# terminates Mogwai after exporting it, so there is never a second).
NGFX = [int(v) for v in os.environ["HSTR_NGFX"].split()] if os.environ.get("HSTR_NGFX") else None
# HSTR_MOTION_DIRECTION: "x,y,z", the world direction a unit of speed moves the camera (default +z, the walk; "1,0,0" strafes).
DIRECTION = [float(v) for v in os.environ.get("HSTR_MOTION_DIRECTION", "0,0,1").split(",")]


def fly(frames, speed, stats=False, trace=False):
    walls, per_frame = [], []
    for i in range(frames):
        position[0] = position[0] + float3(*DIRECTION) * speed
        cam.position = position[0]
        cam.target = position[0] + VIEW
        if NGFX and trace and i == NGFX[0]:
            m.gpuTraceStart()
        t0 = time.perf_counter()
        m.renderFrame()
        walls.append((time.perf_counter() - t0) * 1000.0)
        if NGFX and trace and i == NGFX[1] - 1:
            from pathlib import Path
            import shutil
            report = Path(m.gpuTraceStop())
            named = report.with_name(f"{os.environ.get('HSTR_TAG', 'sunset')}_f{NGFX[0]}-{NGFX[1]}{report.suffix}")
            shutil.copyfile(report, named)  # A copy: ngfx opens the original for --auto-export after this.
            print(f"MOTION ngfx: frames {NGFX[0]}..{NGFX[1] - 1} -> {named}", flush=True)
        if stats:
            per_frame.append(cloud_stats())
    return walls, per_frame


def summary(walls):
    v = sorted(walls)
    return f"mean {sum(v) / len(v):.2f} ms, p50 {v[len(v) // 2]:.2f}, p95 {v[int(0.95 * (len(v) - 1))]:.2f}, max {v[-1]:.2f}"


def delta(before, after):
    out = {}
    for k in COUNTERS:
        a, b = before.get(k), after.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = round(b - a, 2)
    return out


def levels(s):
    return {k: s.get(k) for k in LEVELS if k in s}


def profile(tag, frames, speed):
    m.profiler.enabled = True
    m.profiler.start_capture()
    fly(frames, speed)
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    for kind in ("gpu_time", "cpu_time"):
        lanes = {name: lane for name, lane in capture["events"].items() if name.endswith("/" + kind)}
        paths = [name.rsplit("/", 1)[0] for name in lanes]
        leaves = {n: l for n, l in lanes.items() if not any(p.startswith(n.rsplit("/", 1)[0] + "/") for p in paths)}
        means = sorted(((sum(l["records"]) / max(len(l["records"]), 1), n) for n, l in leaves.items()), reverse=True)
        frame = next((sum(l["records"]) / len(l["records"]) for n, l in lanes.items() if n.endswith("/onFrameRender/" + kind)), 0.0)
        print(f"MOTION {tag} profiled frame {kind} {frame:.2f} ms; top leaves:", flush=True)
        for value, name in means[:12]:
            print(f"MOTION {tag}   {value:6.3f} ms  {name.rsplit('/', 1)[0].split('/onFrameRender/')[-1]}", flush=True)
        # HSTR_MOTION_PROFILE_MATCH (a regex): these leaves always, with the frames they ran in and their mean over all frames - a
        # pass that runs on some frames only (the sun stamp) drops out of the top list or reads its mean over its own frames.
        match = os.environ.get("HSTR_MOTION_PROFILE_MATCH")
        if match and kind == "gpu_time":
            for n, l in leaves.items():
                if re.search(match, n):
                    records = l["records"]
                    print(f"MOTION {tag}   match {sum(records) / frames:6.3f} ms a frame ({len(records)} of {frames} frames, "
                          f"{sum(records) / max(len(records), 1):.3f} when run)  {n.rsplit('/', 1)[0].split('/onFrameRender/')[-1]}",
                          flush=True)


# HSTR_MOTION_ARMS: a Python list literal of (name, property dict) pairs; the first is also the state of the live flights.
ARMS = eval(os.environ.get("HSTR_MOTION_ARMS", "None")) or [
    ("live", {"cloudResidencyFrozen": False, "beamInvalidate": True}),
    ("frozen", {"cloudResidencyFrozen": True, "beamInvalidate": True}),
    ("noInval", {"cloudResidencyFrozen": False, "beamInvalidate": False})]

settle = int(os.environ.get("HSTR_MOTION_SETTLE", "1800"))
for checkpoint in range(0, settle, 300):
    if checkpoint == 300:
        # Every arm's programs compile during the settle, not in a measured chunk.
        for _, props in ARMS[::-1]:
            hstr.set_properties(props)
            fly(3, 0.0)
    walls, _ = fly(300, 0.0)
    print(f"MOTION settle{checkpoint // 300}: {summary(walls)} {levels(cloud_stats())}", flush=True)
walls, _ = fly(60, 0.0)
print(f"MOTION parked: {summary(walls)} {levels(cloud_stats())}", flush=True)
profile("parked", 60, 0.0)
if COUNT:
    d = counted(COUNT, 0.0)
    print(f"MOTION parked dirty per frame (counted): "
          f"{ {k: round(sum(float(x.get(k) or 0) for x in d) / len(d), 1) for k in d[0]} }", flush=True)
if SCORE:
    print(f"MOTION parked quality {score()}", flush=True)
    fly(30, 0.0)

speeds = [float(v) for v in os.environ.get("HSTR_MOTION_SPEEDS", "2,20").split(",")]
for speed in speeds:
    tag = f"v{speed:g}"
    before = cloud_stats()
    walls, per_frame = fly(120, speed, stats=True, trace=speed == speeds[0])
    print(f"MOTION {tag} live: {summary(walls)} counters {delta(before, per_frame[-1])} {levels(per_frame[-1])}", flush=True)
    capture(f"{tag}_live")
    # The ten worst frames with what the frame did.
    prev = [before] + per_frame[:-1]
    for i in sorted(range(len(walls)), key=lambda i: -walls[i])[:10]:
        print(f"MOTION {tag}   frame {i}: {walls[i]:.2f} ms {delta(prev[i], per_frame[i])} {levels(per_frame[i])}", flush=True)
    profile(f"{tag} live", 60, speed)
    if ORACLE:
        # Today's moving frame (score) and the capture oracle at the same cameras; no arms.
        for stop in range(int(os.environ.get("HSTR_MOTION_ORACLE_STOPS", "3"))):
            fly(CHUNK, speed)
            if SCORE:
                print(f"MOTION {tag} oracle stop {stop} moving {score()}", flush=True)
            print(f"MOTION {tag} oracle stop {stop} capture {capture_oracle()}", flush=True)
        continue
    # Arms alternating along the continued path; the first DROP frames of each chunk carry the previous arm's backlog and are dropped.
    arm_walls = {name: [] for name, _ in ARMS}
    arm_counts = {name: {} for name, _ in ARMS}
    arm_scores = {name: [] for name, _ in ARMS}
    arm_dirty = {name: [] for name, _ in ARMS}
    arm_levels = {name: [] for name, _ in ARMS}
    for cycle in range(int(os.environ.get("HSTR_MOTION_CYCLES", "4"))):
        for name, props in ARMS:
            hstr.set_properties(props)
            before = cloud_stats()
            walls, _ = fly(CHUNK, speed)
            arm_walls[name] += walls[DROP:]
            for k, v in delta(before, cloud_stats()).items():
                arm_counts[name][k] = round(arm_counts[name].get(k, 0) + v, 2)
            arm_levels[name].append(levels(cloud_stats()))
            if COUNT:
                arm_dirty[name] += counted(COUNT, speed)
            if SCORE:
                arm_scores[name].append(score())
    for name, _ in ARMS:
        print(f"MOTION {tag} arm {name}: {summary(arm_walls[name])} counters {arm_counts[name]}", flush=True)
        ends = arm_levels[name]
        print(f"MOTION {tag} arm {name} chunk ends: " + str({k: round(sum(float(e.get(k) or 0) for e in ends) / len(ends))
                                                             for k in ("sunWaiting", "sunStale", "desired", "mapped")}), flush=True)
        if arm_dirty[name]:
            d = arm_dirty[name]
            means = {k: round(sum(float(x.get(k) or 0) for x in d) / len(d), 1) for k in d[0]}
            print(f"MOTION {tag} arm {name} dirty per frame (counted, {len(d)} frames): {means}", flush=True)
        if arm_scores[name]:
            e = arm_scores[name]
            print(f"MOTION {tag} arm {name} quality: >0.02 mean {100 * sum(x['over02'] for x in e) / len(e):.3f}% worst "
                  f"{100 * max(x['over02'] for x in e):.3f}%, p99.9 {max(x['p999'] for x in e):.3g}, marched "
                  f"{100 * sum(x['marched'] for x in e) / len(e):.1f}%; per score {e}", flush=True)
    if os.environ.get("HSTR_MOTION_ARM_PROFILE", "0") != "0":
        for name, props in ARMS:
            hstr.set_properties(props)
            fly(5, speed)
            profile(f"{tag} arm {name}", int(os.environ.get("HSTR_MOTION_PROFILE_FRAMES", "20")), speed)
    hstr.set_properties(ARMS[0][1])
    # Back to parked: how long the backlog takes to drain.
    for window in range(3):
        walls, _ = fly(60, 0.0)
        print(f"MOTION {tag} parkedAfter{window}: {summary(walls)} {levels(cloud_stats())}", flush=True)
exit()
