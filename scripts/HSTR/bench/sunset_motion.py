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
start_position = float3(cam.position.x, cam.position.y, cam.position.z)  # HSTR_MOTION_REPARK returns here.
VIEW = float3(cam.target.x, cam.target.y, cam.target.z) - position[0]
COUNTERS = ("beamDirtyBlocks", "beamDirtyOwnMarched", "beamDirtyApronMarched", "beamWarpHeld", "seaTilesChanged", "sunBakeFrames",
            "densityChangedFrames", "cuts", "cutTotalMs", "cutPops", "skirtMaskChecks", "skirtMaskMismatches",
            "dirtyStatFrames", "dirtyStatBlocks", "dirtyStatUnits", "skirtFrames", "skirtRegionFrames", "skirtRegionPages",
            "skirtFullPages", "skirtDirtyRegions", "skirtDirtyWork", "sunResolveFrames", "sunResolveTouchFrames", "sunTouchChecks",
            "sunTouchMismatches", "gpuTimeSum", "gpuTimeFrames", "uploadRuns", "uploadBytes", "scatterJobs", "scatterChecks", "scatterMismatches", "commitChecks", "commitMismatches", "sunResolveListFrames", "sunResolveListBricks", "sunResolveListChecks",
            "sunResolveListMismatches", "sunStampRuns", "sunStampRaw", "sunStampChanges", "sunStampCheckTotal", "sunStampCheckRuns", "sunBesideRuns", "sunSerialRuns", "asyncProbeRan", "asyncResidencyFrames", "walkTouchedCalls", "walkTouchedListed", "updateUs", "workerUpdateUs", "takeWaitUs", "replayUs", "workerRunCycles", "workerWallCycles", "workerKernelUs", "workerUserUs", *[f"step{k}" for k in range(40)], "workerUpdates", "submitMainUs", "submitResidencyUs", "residencySkips", "fadeParallelBuckets", "fadeParallelEnds", "mapParallelRuns","phaseMergeUs", "phaseLoadsUs", "phaseEvictUs", "phaseMapsUs", "phaseStageUs", "phaseFadesUs", "phaseSunUs",
            "phaseUploadUs", "phaseLaunchUs", "fadeStarts", "fadeEnds", "fadeVoid", "fadeEndIns", "maps", "unmaps",
            "beamHistoryProbeUnits", "beamHistoryProbe02", "beamHistoryProbe10",
            # World cache photon updates run, and those focused (worldCacheFocusFar) with their probability-1 / starving tiles summed.
            "worldCacheUpdatesRun", "worldCacheFocusUpdates", "worldCacheFocusNear", "worldCacheFocusHungry",
            # worldCacheAsync: updates recorded on the compute queue, frames a pending one waited, ones that fell back to the main queue.
            "worldCacheAsyncRuns", "worldCacheAsyncWaits", "worldCacheAsyncFallbacks",
            # worldCacheUrgentTarget: main-queue whole-batch updates run while a tile was under it.
            "worldCacheUrgentRuns", "seaApplyEvents",
            # worldCacheBudgetMs: frames whose wanted continuation waited for the frame budget's credit.
            "worldCacheBudgetSkips",
            # Sun-column refreshes, resumed ones, and whole walks by cause (new field, residual dirty, sun moved, no changed tile).
            "sunColumnRefreshes", "sunColumnResumes", "sunColumnWholeField", "sunColumnWholeDirty", "sunColumnWholeSun",
            "sunColumnWholeNoChange",
            # worldCacheAsyncSpan: main-queue writes that waited for a running slice, slices that followed a running one, bakes put off.
            "worldCacheAsyncSpanWaits", "worldCacheAsyncBusy", "worldCacheBakesPutOff", "worldCacheBakes",
            # Async slices timed on their own queue (hstrPhotonProbe): summed span in us, how many, how many over 4 ms.
            "worldCacheSliceUs", "worldCacheSlicesTimed", "worldCacheSlicesOver4Ms",
            # Their waves and photons (hstrPhotonProbe, traceWorldCachePhotons).
            "photonWaveUs", "photonWaves", "photonLongestWaveUs", "photonLongestWaveFlights", "photonFlights", "photonsTraced",
            "photonSunFlights", "photonSunTraced", "photonCells", "photonDeposited", "photonSteps", "photonFlushLanes",
            "photonFlushGroups",
            # The fewest baked batches among the tiles within 3 of the camera, summed over frames, and frames under 10 / 26.
            "worldCacheNearLowSum", "worldCacheNearFrames", "worldCacheNearUnder10", "worldCacheNearUnder26",
            "sunFineRuns", "sunFineFallbacks", "sunFineCells", "sunFineUploadBytes", "cutOrderedCount", "cutEntered", "cutLeft", "cutReentered") + tuple(
            f"cutEnteredL{k}" for k in range(8))
LEVELS = ("desired", "mapped", "pending", "mapBacklog", "sunWaiting", "sunBakesFrame", "sunSlotsFree", "sunStale","beamWarpOn", "beamPolicyToleranceNow",
          "photonCostUs", "frameBaseUs",  # worldCacheBudgetMs: an update's smoothed GPU cost, the frame's without photons.
          "beamMarchTiles", "bindCpuMs", "loaded", "slotsUsed", "freeSlots", "freeBricks", "committed", "commitFails",
          "evictCalls", "evictFreed", "evictCandidates", "evictScans", "loadHeld", "loadScanned", "undesiredFadingOut", "undesiredHeld",
          "undesiredIdle", "activeFades", "cuts", "cutEntered", "sunStampStores", "sunStampFresh", "sunStampCheckStores",
          "sunStampCheckFresh", "sunFineCellsPeak", "sunFineWaves", "sunFinePassed", "sunFineTested") + tuple(  # rtSpanProbe's last frame (present only while it is on).
          f"rtProbe{n}" for n in ("Boxes", "Instances", "Rays", "HitRays", "Candidates", "Max", "Overflow", "Covered")) + tuple(
          f"rtProbeBin{b}" for b in range(8)) + tuple(  # rasterProbe's last frame (present only while it is on).
          f"raster{n}" for n in ("Boxes", "Items", "Fragments", "Culled", "Empty", "Unbaked", "Subdivided", "NoSlice", "QueueOverflow",
                                 "BoxOverflow", "ItemOverflow", "Instances", "Seeds", "SeedLevel")) + tuple(
          f"rasterLevelBoxes{k}" for k in range(16)) + (  # worksetProbe's last frame.
          "worksetBrickReads", "worksetBricks", "worksetBakeReads", "worksetBakes")
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
         tuple(f"beamSun{n}" for n in ("Resolved", "Unbaked", "CoarseOrNoNear", "NoBrick", "Ancestor", "AncestorLevels")) +
         tuple(f"beamLookup{n}" for n in ("Lanes", "SameInstance", "SamePage")) +
         # Lockstep oracle: lookup waves, distinct (instance, level, page) keys now and at the wave's mean camera distance.
         tuple(f"beamLockstep{n}" for n in ("Waves", "KeysNow", "KeysLockstep", "KeysMin", "KeysFirst", "TransformChecked",
                                             "TransformMiss")) +
         # beamHalfResolve: filled pixels, exact ones (a neighbour not from a basis / past the contrast), the shaded half's basis pixels.
         tuple(f"beamHalf{n}" for n in ("Filled", "NotBasis", "Contrast", "ShadedBasis", "FillWaves", "FillWavesExact")) +
         # beamOrderProbe: the dirty query's rays, steps, max steps (2) and log2 step histogram (6-17).
         tuple(f"beamOrderProbe{i}" for i in range(21)) +
         tuple(f"beamLayer{n}" for n in ("NoCloud", "OutOfBox", "BrickEmpty", "Density", "Proxy", "Dense0", "Dense1", "Dense2",
                                         "LaneSteps", "PaidSteps", "Warps", "TightZero", "MaskMismatch", "MaskEmpty0", "MaskEmpty1",
                                         "VoxelEmpty", "VoxelDense", "TightRun", "BlockDense", "BlockDenseSum", "BlockDenseEdge",
                                         "Miss0", "Miss1", "Miss2", "Miss3", "Miss4", "Miss5", "Miss6", "Miss7")) +
         # ... per layer by what answered, and layer 1's lookups after a dense layer 0.
         tuple(f"beamSplit{layer}{n}" for layer in (0, 1)
               for n in ("BlockSkip", "VoxelSkip", "NoCloud", "OutOfBox", "BrickEmpty", "Density", "Proxy")) +
         ("beamSplit1EmptyAfterDense", "beamSplit1DenseAfterDense", "beamLayerUnitLaneSteps", "beamLayerUnitPaidSteps",
          "beamLayerUnitWarps") +
         # ... and the layer loop's waves: walking a chain, two chains one after the other, a lane needing both; lanes needing both.
         tuple(f"beamLayerWarp{n}" for n in ("Waves", "TwoChains", "SomeBoth", "LanesBoth")) +
         # ... and beamQueryPairCheck: paired query rays, and those whose serial march differs in steps / beyond 1e-5 / in any bit.
         tuple(f"beamPairCheck{n}" for n in ("Rays", "Steps", "Far", "Bits")) +
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
    fresh = os.environ.get("HSTR_MOTION_SCORE") in ("2", "3")
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
    if os.environ.get("HSTR_MOTION_SCORE") == "3":
        # SCORE 3: score 2's fresh beam reads the cache as the flight left it, so it cannot see refill lag. Here the same moving
        # frame is scored again after the world cache has converged in place (sync full batches until no tile is short of the
        # target), the beam rebuilt from nothing over it: "over02Converged" minus "over02" is what the cache's refill lag costs.
        # Residency frozen meanwhile: FOUND (lagscore3) ~67 parked frames of live streaming changed the resident bricks too, and
        # every arm, refilling or not, scored 9-12% against that frame instead of 4-6%.
        # FOUND (lagscore4, residency frozen): still not a refill gate. Converging adds batches to every tile, not only the
        # restarted ones, so the second score compares two noise realizations of the cache: ship 5.57 -> 12.18%, no refill 4.72 ->
        # 9.40%, span4 5.42 -> 9.23%, the difference signed ~0 and spread over 14-40% of the band's pixels (difference images).
        # worldCacheBudgetMs off while converging: its skips read as idle frames and ended the convergence after 4 frames (lagbudget1).
        keys = ("worldCacheUpdates", "worldCacheAsync", "worldCacheAsyncSpan", "worldCacheSlices", "cloudResidencyFrozen", "worldCacheBudgetMs")
        arm = {k: hstr.properties[k] for k in keys if k in hstr.properties}
        hstr.set_properties({"compareReference": False, "compareExact": False, "worldCacheUpdates": 1, "worldCacheAsync": False,
                             "worldCacheAsyncSpan": False, "worldCacheSlices": 1, "cloudResidencyFrozen": True, "worldCacheBudgetMs": 0.0})
        last, idle, converge = None, 0, 0
        while idle < 3 and converge < 150:
            m.renderFrame()
            converge += 1
            run = int(cloud_stats().get("worldCacheUpdatesRun") or 0)
            idle = idle + 1 if run == last else 0
            last = run
        hstr.set_properties({"beamReset": True})
        for _ in range(4):
            m.renderFrame()
        capture(f"score{score.taken}_converged")
        hstr.set_properties({"compareReference": True, "compareExact": True, "compareBlock": 1})
        m.renderFrame()
        p = hstr.properties
        result.update(over02Converged=float(p["referenceNoiseError"]), p999Converged=float(p["referenceLogP999"]),
                      convergeFrames=converge)
        hstr.set_properties(arm)
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
        # FOUND (lagscore1 / lagscore2): a directory that did not exist ended Mogwai silently a few frames after the capture.
        os.makedirs(os.environ["HSTR_MOTION_OUT"], exist_ok=True)
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


NSYS = os.environ.get("HSTR_NSYS")  # run_sea.py --nsys: an nsys report of each live flight's profiled frames, and only those.
ARM_NSYS = NSYS and os.environ.get("HSTR_MOTION_ARM_PROFILE", "0") != "0"


def nsys(verb, *options):
    import subprocess
    subprocess.run([NSYS, verb, f"--session={os.environ['HSTR_NSYS_SESSION']}", *options], creationflags=subprocess.CREATE_NO_WINDOW)


def nsys_start(report):
    metric_set = [f"--gpu-metrics-set={os.environ['HSTR_NSYS_SET']}"] if os.environ.get("HSTR_NSYS_SET") else []
    # HSTR_NSYS_SAMPLE=1: CPU call-stack samples of the process instead of GPU metrics (where the CPU frame goes, by function).
    what = (["--sample=process-tree", "--sampling-frequency=8000"] if os.environ.get("HSTR_NSYS_SAMPLE") else
            ["--sample=none", "--gpu-metrics-devices=all", "--gpu-metrics-frequency=20000", *metric_set])
    nsys("start", *what, "--force-overwrite=true", f"--output={os.environ['HSTR_NSYS_OUTPUT']}_{report}")


def profile(tag, frames, speed, report=None):
    if NSYS and report:
        nsys_start(report)
        if os.environ.get("HSTR_NSYS_SAMPLE"):
            # CPU samples of the shipping frame: Falcor's profiler off (its endFrame alone was 12% of the frame thread's samples).
            fly(frames, speed)
            nsys("stop")
            return
    m.profiler.paused = False
    m.profiler.enabled = True
    m.profiler.start_capture()
    fly(frames, speed)
    capture = m.profiler.end_capture()
    m.profiler.enabled = False
    if NSYS and report:
        nsys("stop")  # After end_capture, which has waited for the flight's GPU work.
    spike_frames = None
    for kind in ("gpu_time", "cpu_time"):
        lanes = {name: lane for name, lane in capture["events"].items() if name.endswith("/" + kind)}
        paths = [name.rsplit("/", 1)[0] for name in lanes]
        leaves = {n: l for n, l in lanes.items() if not any(p.startswith(n.rsplit("/", 1)[0] + "/") for p in paths)}
        means = sorted(((sum(l["records"]) / max(len(l["records"]), 1), n) for n, l in leaves.items()), reverse=True)
        frame = next((sum(l["records"]) / len(l["records"]) for n, l in lanes.items() if n.endswith("/onFrameRender/" + kind)), 0.0)
        # HSTR_MOTION_PROFILE_LEAVES: how many leaves to print (default 12), with the sum of all leaves.
        print(f"MOTION {tag} profiled frame {kind} {frame:.2f} ms; leaves sum {sum(v for v, _ in means):.2f} ms; top leaves:",
              flush=True)
        for value, name in means[:int(os.environ.get("HSTR_MOTION_PROFILE_LEAVES", "12"))]:
            print(f"MOTION {tag}   {value:6.3f} ms  {name.rsplit('/', 1)[0].split('/onFrameRender/')[-1]}", flush=True)
        if os.environ.get("HSTR_MOTION_PROFILE_LEAVES"):
            # Self time of every scope with children: its mean less its children's - work recorded outside any child scope (or a
            # wait inside it), which no leaf shows.
            mean_of = {n.rsplit("/", 1)[0]: sum(l["records"]) / max(len(l["records"]), 1) for n, l in lanes.items()}
            selfs = []
            for path, value in mean_of.items():
                children = [v for p, v in mean_of.items() if p.rsplit("/", 1)[0] == path]
                if children:
                    selfs.append((value - sum(children), path))
            for value, path in sorted(selfs, reverse=True)[:12]:
                print(f"MOTION {tag}   self {value:6.3f} ms  {path.split('/onFrameRender/')[-1]}", flush=True)
        # HSTR_MOTION_PROFILE_SPIKES=<ms>: the frames whose GPU frame time exceeds ms against the rest - each leaf's mean over the
        # spike frames less its mean over the others, the largest first: what a spike spends that a normal frame does not. Records
        # are aligned from the end (a scope first seen mid-capture has fewer). Under 1 it is the slowest fraction of the frames
        # instead (0.1: the slowest 10%) - profiled frames run slower than unprofiled ones, so a fixed threshold misleads.
        # The GPU's spike frames are reused for cpu_time. Beside the leaves, the self time of every scope with children per frame
        # (recorded outside any child scope - a wait, a gap between dispatches, untimed work): the part of a spike no leaf shows
        # (the frame's gpu_time includes the GPU's wait for the CPU's recording).
        spike_ms = float(os.environ.get("HSTR_MOTION_PROFILE_SPIKES", "0"))
        if spike_ms > 0.0:
            totals = next((l["records"] for n, l in lanes.items() if n.endswith("/onFrameRender/" + kind)), [])
            count = len(totals)
            if kind == "gpu_time":
                if 0.0 < spike_ms < 1.0 and count:
                    spike_ms = sorted(totals)[min(count - 1, int((1.0 - spike_ms) * count))] - 1e-9
                spikes = [i for i in range(count) if totals[i] > spike_ms]
                others = [i for i in range(count) if totals[i] <= spike_ms]
                spike_frames = (spikes, others, count)
            elif spike_frames and spike_frames[2] == count:
                spikes, others, _ = spike_frames
            else:
                spikes, others = [], []
            if spikes and others:
                def at(records, i):
                    j = i - (count - len(records))
                    return records[j] if 0 <= j < len(records) else 0.0
                series = {n.rsplit("/", 1)[0]: (lambda i, r=l["records"]: at(r, i)) for n, l in leaves.items()}
                records_of = {n.rsplit("/", 1)[0]: l["records"] for n, l in lanes.items()}
                for path, records in records_of.items():
                    children = [r for p, r in records_of.items() if p.rsplit("/", 1)[0] == path]
                    if children:
                        series[path + " (self)"] = (lambda i, r=records, c=children: at(r, i) - sum(at(x, i) for x in c))
                excess = sorted(((sum(f(i) for i in spikes) / len(spikes) - sum(f(i) for i in others) / len(others), n)
                                 for n, f in series.items()), reverse=True)
                print(f"MOTION {tag} spikes over {spike_ms:g} ms ({kind}): {len(spikes)} of {count} frames, frame "
                      f"{sum(totals[i] for i in spikes) / len(spikes):.2f} against {sum(totals[i] for i in others) / len(others):.2f} ms; "
                      f"leaves over the other frames:", flush=True)
                for value, name in excess[:14]:
                    print(f"MOTION {tag}   spike +{value:6.3f} ms  {name.split('/onFrameRender/')[-1]}", flush=True)
        # HSTR_MOTION_PROFILE_MATCH (a regex): these scopes always, with the frames they ran in and their mean over all frames - a
        # pass that runs on some frames only (the sun stamp) drops out of the top list or reads its mean over its own frames.
        # Any scope, not only leaves: the profiler keeps every event it has seen, so a scope whose children stopped running in
        # this arm (march with seaFarOverlapScopes off) is never a leaf.
        match = os.environ.get("HSTR_MOTION_PROFILE_MATCH")
        if match:
            for n, l in lanes.items():
                if re.search(match, n):
                    # Falcor records 0 for an event on a frame it did not run: count the frames it ran by their non-zero records.
                    ran = [r for r in l["records"] if r > 0.0]
                    print(f"MOTION {tag}   match {kind[:3]} {sum(ran) / frames:6.3f} ms a frame ({len(ran)} of {frames} frames, "
                          f"{sum(ran) / max(len(ran), 1):.3f} when run)  {n.rsplit('/', 1)[0].split('/onFrameRender/')[-1]}",
                          flush=True)


# HSTR_MOTION_ARMS: a Python list literal of (name, property dict) pairs; the first is also the state of the live flights.
ARMS = eval(os.environ.get("HSTR_MOTION_ARMS", "None")) or [
    ("live", {"cloudResidencyFrozen": False, "beamInvalidate": True}),
    ("frozen", {"cloudResidencyFrozen": True, "beamInvalidate": True}),
    ("noInval", {"cloudResidencyFrozen": False, "beamInvalidate": False})]

# An arm's "_res": "WxH" renders it at that frame size (every arm without one at HSTR_RES), so resolutions share one settle. A resize
# drops the beam image: time such arms after a lead (HSTR_MOTION_PROFILE_LEAD) long enough for the moving frame to settle again.
RES_DEFAULT = os.environ.get("HSTR_RES", "3840x2160")
res_now = [RES_DEFAULT]


def set_arm(props):
    res = props.get("_res", RES_DEFAULT)
    if res != res_now[0]:
        m.resizeFrameBuffer(*[int(v) for v in res.split("x")])
        res_now[0] = res
    # _profilerPaused: Falcor's Profiler::endFrame returns at once (off, it still waits for the last frame's fence and submits).
    m.profiler.paused = bool(props.get("_profilerPaused", False))
    # _ui: Mogwai's UI (headless keeps showUI on: every frame builds the whole UI and renders it with ImGui into the target). Off
    # by default: it is Falcor's debug UI, not the renderer. MEASURED (uicost1, 4K sunset walk, worker on, arms alternating): walk
    # p50 3.87 ms off against 4.16 / 4.36 on; GPU 2.84 against 2.76 / 2.89 ms (means within the arms' spread).
    m.ui = bool(props.get("_ui", False))
    hstr.set_properties({k: v for k, v in props.items() if not k.startswith("_")})

settle = int(os.environ.get("HSTR_MOTION_SETTLE", "1800"))
for checkpoint in range(0, settle, 300):
    if checkpoint == 300:
        # Every arm's programs compile during the settle, not in a measured chunk. Then the settle goes on in its own state: it
        # used to keep the first arm's, so an arm with cloudResidencyFrozen froze residency 300 frames into the settle, and every
        # frozen run measured with ~271k mapped bricks never sun-baked (sunWaiting / sunSlotsFree identical in every later settle
        # chunk; sunBakesFrame is the last run's count). Found 2026-10-09 (packview1).
        before = {k: hstr.properties.get(k) for _, props in ARMS for k in props}
        for _, props in ARMS[::-1]:
            set_arm(props)
            fly(3, 0.0)
        hstr.set_properties({k: v for k, v in before.items() if v is not None and not k.startswith("_")})
        set_arm({})  # Back to the launch size.
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
# The live flights (and the --ngfx trace taken in the first) run in the first arm's state. Since the settle restores its own state
# after compiling the arms (packview1), they had run in the launcher's: halftrace1 traced resolveBeam with beamHalfResolve armed.
set_arm(ARMS[0][1])
for speed_i, speed in enumerate(speeds):
    tag = f"v{speed:g}"
    before = cloud_stats()
    walls, per_frame = fly(120, speed, stats=True, trace=speed == speeds[0])
    print(f"MOTION {tag} live: {summary(walls)} counters {delta(before, per_frame[-1])} {levels(per_frame[-1])}", flush=True)
    capture(f"{tag}_live")
    # The ten worst frames with what the frame did.
    prev = [before] + per_frame[:-1]
    for i in sorted(range(len(walls)), key=lambda i: -walls[i])[:10]:
        print(f"MOTION {tag}   frame {i}: {walls[i]:.2f} ms {delta(prev[i], per_frame[i])} {levels(per_frame[i])}", flush=True)
    # Under --nsys with HSTR_MOTION_ARM_PROFILE the arms take the launch's one report (below), so the live flight takes none.
    profile(f"{tag} live", 60, speed, report=None if ARM_NSYS else f"{tag}_live")
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
    walks_before = cloud_stats()
    for cycle in range(int(os.environ.get("HSTR_MOTION_CYCLES", "4"))):
        for name, props in ARMS:
            set_arm(props)
            before = cloud_stats()
            # HSTR_MOTION_TRACE=N: the chunk flown N frames at a time, the world cache's and the sea window's state printed after each.
            trace_every = int(os.environ.get("HSTR_MOTION_TRACE", "0")) or CHUNK
            walls = []
            for start in range(0, CHUNK, trace_every):
                piece, _ = fly(min(trace_every, CHUNK - start), speed)
                walls += piece
                if trace_every < CHUNK:
                    s = cloud_stats()
                    print(f"MOTION {tag} trace {name} frame {start + len(piece)}: " + str({k: s.get(k) for k in (
                        "cameraTileZ", "seaPendingTiles", "seaReadyTiles", "seaHoldFrames", "seaHoldBusyFrames", "seaHoldLongest",
                        "seaTilesChanged", "seaApplyEvents", "desired", "mapped", "cuts",
                        "cutBusyFrames", "worldCacheUpdatesRun", "worldCacheStarving", "worldCacheEmpty", "worldCacheNearLow",
                        "worldCacheBakeLowBatches")}), flush=True)
            arm_walls[name] += walls[DROP:]
            # One read: cloudStats resets desiredLow on read, so a second read in the same frame sees 0.
            now = cloud_stats()
            for k, v in delta(before, now).items():
                arm_counts[name][k] = round(arm_counts[name].get(k, 0) + v, 2)
            arm_levels[name].append(levels(now))
            # The frames over 6 ms and what they ran (cloudStats spike*), against how many of all frames ran each event.
            events = ("Tiles", "Cut", "SunRun", "Upload", "Photons", "Bake", "SunBakes")
            sd = lambda k: int(float(now.get(k) or 0) - float(before.get(k) or 0))
            print(f"MOTION {tag} chunk {cycle} {name} spikes {sd('spikeFrames')} (none of these {sd('spikeQuiet')}): " +
                  ", ".join(f"{e} {sd('spikeWith' + e)} of {sd('framesWith' + e)}" for e in events), flush=True)
            # Per chunk: residency and the timestamp pair's GPU time, so a slow drift and an arm's own effect can be told apart.
            d = {k: (now.get(k) or 0) - (before.get(k) or 0) for k in ("gpuTimeSum", "gpuTimeFrames", "committedTotal",
                                                                      "committedCapped", "cuts", "cutEntered", "evictFreed",
                                                                      "cutWalkMsTotal", "cutBusyFrames", "cutIdleFrames",
                                                                      "cutParallelMsTotal", "cutHeapMsTotal", "cutPriorityRises",
                                                                      "cutCoarseMsTotal", "cutFineMsTotal", "cutDesireMsTotal", "cutTailMsTotal",
                                                                      "cutVisitsTotal", "cutRefreshesTotal", "cutRaysTotal",
                                                                      "gpuVisRequests", "gpuVisApplied", "gpuVisBatches", "gpuVisDropped",
                                                                      "cutVisAsyncRuns", "cutVisMainRuns", "cutVisDeferred",
                                                                      "dispatchCount", "dispatchPrepareCycles", "dispatchPipelineCycles",
                                                                      "dispatchBindCycles", "dispatchCycles", "bindCpuMs", "executeFrames",
                                                                      "bindSceneCycles", "bindParamsCycles", "bindFieldsCycles", "bindCalls",
                                                                      "cutOrderedCount", "cutThresholdTrims", "cutUnderfilled",
                                                                      "cutCheckBricks", "cutCheckMismatches", "uploadBytes", "uploadRuns",
                                                                      "scatterJobs", "scatterChecks", "scatterMismatches",
                                                                      "seaTilesChanged", "residencySkips", "workerUpdateUs", "takeWaitUs",
                                                                      "seaHoldFrames", "walkSlotChanges", "seaApplyEvents",
                                                                      "photonBesideRuns", "photonBesideFallbacks",
                                                                      "desiredSum", "mappedSum", "desiredFrames", "gpuTimeOver2Ms",
                                                                      "gpuTimeOver3Ms", "gpuTimeOver4Ms", "gpuTimeOver6Ms")}
            gpu = float(d.get("gpuTimeSum") or 0) / max(float(d.get("gpuTimeFrames") or 0), 1.0) / 1000.0
            print(f"MOTION {tag} chunk {cycle} {name}: gpu {gpu:.3f} ms ({d.get('gpuTimeFrames')} frames; over 2 / 3 / 4 / 6 ms "
                  f"{d.get('gpuTimeOver2Ms')} / {d.get('gpuTimeOver3Ms')} / {d.get('gpuTimeOver4Ms')} / {d.get('gpuTimeOver6Ms')}, "
                  f"longest {float(now.get('gpuTimeMaxUs') or 0) / 1000.0:.2f}), wall p50 "
                  f"{sorted(walls)[len(walls) // 2]:.2f} ms mean {sum(walls[DROP:]) / max(len(walls) - DROP, 1):.2f}, tiles changed "
                  f"{d.get('seaTilesChanged')} in {d.get('seaApplyEvents')} applies (held {d.get('seaHoldFrames')} frames, "
                  f"{d.get('walkSlotChanges')} slots beside a walk; photons beside the query {d.get('photonBesideRuns')}, not "
                  f"{d.get('photonBesideFallbacks')}), skips {d.get('residencySkips')}, worker {float(d.get('workerUpdateUs') or 0) / 1e3:.0f} ms "
                  f"wait {float(d.get('takeWaitUs') or 0) / 1e3:.0f} ms, desired {now.get('desired')} mapped {now.get('mapped')} backlog {now.get('mapBacklog')} "
                  f"(chunk mean desired {float(d.get('desiredSum') or 0) / max(float(d.get('desiredFrames') or 0), 1):.0f} mapped "
                  f"{float(d.get('mappedSum') or 0) / max(float(d.get('desiredFrames') or 0), 1):.0f}, fewest desired {now.get('desiredLow')}) "
                  f"pending {now.get('pending')} committed {d.get('committedTotal')} ({d.get('committedCapped')} frames at the cap), "
                  f"cuts {d.get('cuts')} entered {d.get('cutEntered')} evicted {d.get('evictFreed')} "
                  f"fading out {now.get('undesiredFadingOut')} free {now.get('freeSlots')}, walk {d.get('cutWalkMsTotal'):.0f} ms "
                  f"worker, {d.get('cutBusyFrames')} frames busy, {d.get('cutIdleFrames')} idle; parallel "
                  f"{d.get('cutParallelMsTotal'):.0f} ms (coarse {d.get('cutCoarseMsTotal'):.0f}, fine {d.get('cutFineMsTotal'):.0f}, "
                  f"desire {d.get('cutDesireMsTotal'):.0f}; visits {d.get('cutVisitsTotal')} refreshes {d.get('cutRefreshesTotal')} "
                  f"rays {d.get('cutRaysTotal')}; gpu visibility {d.get('gpuVisRequests')} sent, {d.get('gpuVisApplied')} applied, "
                  f"{d.get('gpuVisDropped')} dropped in {d.get('gpuVisBatches')} batches, {d.get('cutVisAsyncRuns')} on the compute queue "
                  f"{d.get('cutVisMainRuns')} main, {d.get('cutVisDeferred')} deferred), tail {d.get('cutTailMsTotal'):.0f} ms, heap {d.get('cutHeapMsTotal'):.0f} ms, ordered {d.get('cutOrderedCount')}, "
                  f"unordered desired {now.get('cutUnorderedDesired')}, priority rises {d.get('cutPriorityRises')}; threshold trims "
                  f"{d.get('cutThresholdTrims')} underfilled {d.get('cutUnderfilled')} check {d.get('cutCheckMismatches')} of "
                  f"{d.get('cutCheckBricks')}; upload {float(d.get('uploadBytes') or 0) / 1e6:.1f} MB in {d.get('uploadRuns')} runs + "
                  f"{d.get('scatterJobs')} scatters, checks {d.get('scatterChecks')} mismatches {d.get('scatterMismatches')}", flush=True)
            # Dispatch recording on the CPU (Falcor's DispatchTiming, rdtsc ~2.5 GHz on this rig): per frame and per dispatch.
            frames = max(float(d.get("executeFrames") or 0), 1.0)
            n = max(float(d.get("dispatchCount") or 0), 1.0)
            stages = {s: float(d.get(f"dispatch{s}Cycles") or 0) for s in ("Prepare", "Pipeline", "Bind", "")}
            print(f"MOTION {tag} chunk {cycle} {name} recording: {n / frames:.0f} dispatches a frame, k cycles each: prepare "
                  f"{stages['Prepare'] / n / 1e3:.1f} pipeline {stages['Pipeline'] / n / 1e3:.1f} bind {stages['Bind'] / n / 1e3:.1f} "
                  f"dispatch {stages[''] / n / 1e3:.1f}; M cycles a frame {sum(stages.values()) / frames / 1e6:.2f}; bindRenderer "
                  f"{float(d.get('bindCpuMs') or 0) / frames:.3f} ms a frame in {float(d.get('bindCalls') or 0) / frames:.1f} calls, k cycles "
                  f"a call: scene {float(d.get('bindSceneCycles') or 0) / max(float(d.get('bindCalls') or 0), 1) / 1e3:.1f} params "
                  f"{float(d.get('bindParamsCycles') or 0) / max(float(d.get('bindCalls') or 0), 1) / 1e3:.1f} fields "
                  f"{float(d.get('bindFieldsCycles') or 0) / max(float(d.get('bindCalls') or 0), 1) / 1e3:.1f}", flush=True)
            if COUNT:
                arm_dirty[name] += counted(COUNT, speed)
            if SCORE:
                arm_scores[name].append(score())
    # Cut walks by the configuration they ran under (cutWaves | cutTailParallel << 1 | ray mode << 2, 0 reference, 1 stepped,
    # 2 blocked, 3 cut off | cutLowPriority << 4): walks straddle arms.
    walks_after = cloud_stats()
    for k in range(32):
        w = lambda key: float(walks_after.get(f"cutConfig{k}{key}") or 0) - float(walks_before.get(f"cutConfig{k}{key}") or 0)
        if w("Walks") > 0:
            print(f"MOTION {tag} walk config {k}: {w('Walks'):.0f} walks, {w('Ms') / w('Walks'):.1f} ms a walk (coarse "
                  f"{w('CoarseMs') / w('Walks'):.1f}, fine {w('FineMs') / w('Walks'):.1f}, desire {w('DesireMs') / w('Walks'):.1f}, "
                  f"mid {w('MidMs') / w('Walks'):.1f}, tail {w('TailMs') / w('Walks'):.1f}),{w('Rays') / w('Walks'):.0f} rays, coarse "
                  f"{1000 * w('CoarseMs') / max(w('Rays'), 1):.2f} ms per 1k rays; coarse thread Gcycles a walk: busy "
                  f"{w('BusyCycles') / w('Walks') / 1e9:.3f}, brickPriority {w('PriorityCycles') / w('Walks') / 1e9:.3f}, rays "
                  f"{w('RayCycles') / w('Walks') / 1e9:.3f}, waves serial {w('SerialCycles') / w('Walks') / 1e9:.3f}, critical "
                  f"{w('CriticalCycles') / w('Walks') / 1e9:.3f}; {w('Refreshes') / w('Walks'):.0f} refreshes, "
                  f"{w('RayCycles') / max(w('Refreshes'), 1) / 1e3:.1f}k ray cycles a refresh", flush=True)
    print(f"MOTION {tag} ray mismatches {float(walks_after.get('cutRayMismatchesTotal') or 0) - float(walks_before.get('cutRayMismatchesTotal') or 0):.0f}"
          f" of {float(walks_after.get('cutRaysTotal') or 0) - float(walks_before.get('cutRaysTotal') or 0):.0f} rays", flush=True)
    for name, _ in ARMS:
        print(f"MOTION {tag} arm {name}: {summary(arm_walls[name])} counters {arm_counts[name]}", flush=True)
        ends = arm_levels[name]
        print(f"MOTION {tag} arm {name} chunk ends: " + str({k: round(sum(float(e.get(k) or 0) for e in ends) / len(ends))
                                                             for k in ("sunWaiting", "sunStale", "desired", "mapped", "sunStampStores",
                                                                       "sunStampFresh", "sunStampCheckStores",
                                                                       "sunStampCheckFresh", "sunFineWaves", "sunFinePassed",
                                                                       "sunFineTested")}), flush=True)
        if arm_dirty[name]:
            d = arm_dirty[name]
            means = {k: round(sum(float(x.get(k) or 0) for x in d) / len(d), 1) for k in d[0]}
            print(f"MOTION {tag} arm {name} dirty per frame (counted, {len(d)} frames): {means}", flush=True)
        if arm_scores[name]:
            e = arm_scores[name]
            print(f"MOTION {tag} arm {name} quality: >0.02 mean {100 * sum(x['over02'] for x in e) / len(e):.3f}% worst "
                  f"{100 * max(x['over02'] for x in e):.3f}%, p99.9 {max(x['p999'] for x in e):.3g}, marched "
                  f"{100 * sum(x['marched'] for x in e) / len(e):.1f}%; per score {e}", flush=True)
            if "over02Converged" in e[0]:
                print(f"MOTION {tag} arm {name} against a converged cache: >0.02 mean "
                      f"{100 * sum(x['over02Converged'] for x in e) / len(e):.3f}% worst "
                      f"{100 * max(x['over02Converged'] for x in e):.3f}%, p99.9 {max(x['p999Converged'] for x in e):.3g}, "
                      f"converge frames {[x['convergeFrames'] for x in e]}", flush=True)
    if os.environ.get("HSTR_MOTION_ARM_PROFILE", "0") != "0":
        # Under --nsys every arm's profiled frames go into ONE report (TAG_<speed>_arms): a later nsys session in the same launch
        # records no DX12 GPU workloads (sunevery1: five per-arm reports held only Map / Unmap calls; the launch's first, the live
        # flight, held them). The lead frames carry scope ranges too (sunevery2: 265 = 5 x (5 + 48)), so the arms split by count, in
        # ARMS order (and round by round with HSTR_MOTION_PROFILE_ROUNDS).
        if ARM_NSYS:
            nsys_start(f"{tag}_arms")
        # HSTR_MOTION_PROFILE_ROUNDS: the arms' profiles taken round-robin in that many rounds of PROFILE_FRAMES each, so every arm
        # is timed along the same stretches of the path (layerloop1: one profile an arm, ship 3.58 against ship again 2.68 ms -
        # each arm had flown a different stretch).
        rounds = int(os.environ.get("HSTR_MOTION_PROFILE_ROUNDS", "1"))
        for r in range(rounds):
            for name, props in ARMS:
                set_arm(props)
                # HSTR_MOTION_PROFILE_LEAD: frames flown before the profile, so an arm whose properties rebuild the beam (beamOctScale)
                # is timed after its rebuild, not during it.
                fly(int(os.environ.get("HSTR_MOTION_PROFILE_LEAD", "5")), speed)
                profile(f"{tag} arm {name}" + (f" round {r}" if rounds > 1 else ""), int(os.environ.get("HSTR_MOTION_PROFILE_FRAMES", "20")),
                        speed)
        if ARM_NSYS:
            nsys("stop")
    # HSTR_MOTION_PARK_ARM: the parked windows in that arm's state (default the first's); HSTR_MOTION_PARK_WINDOWS of them, each
    # with its GPU time and the world cache's refill state (does an async continuation reach worldCacheTarget parked, at what cost).
    # A comma list names one arm per entry of HSTR_MOTION_SPEEDS (two drains A/B in one run: HSTR_MOTION_SPEEDS=20,20).
    park_names = os.environ.get("HSTR_MOTION_PARK_ARM", ARMS[0][0]).split(",")
    park_arm = dict(ARMS).get(park_names[min(speed_i, len(park_names) - 1)], ARMS[0][1])
    set_arm(park_arm)
    # Back to parked: how long the backlog takes to drain.
    for window in range(int(os.environ.get("HSTR_MOTION_PARK_WINDOWS", "3"))):
        before = cloud_stats()
        walls, _ = fly(60, 0.0)
        s = cloud_stats()
        churn = ("maps", "unmaps", "fadeStarts", "fadeEnds", "cuts", "cutEntered", "uploadRuns", "seaTilesChanged", "sunBakeFrames",
                 "evictFreed", "committedTotal", "beamInvalidatedBuilds", "beamInvalidateAllBuilds", "beamSphereBuilds",
                 "beamSpheresTotal", "beamChangeBuilds", "beamChangeInputsTotal", "beamChangeBakesTotal", "beamChangeOverflows")
        d = {k: float(s.get(k) or 0) - float(before.get(k) or 0) for k in (
            "gpuTimeSum", "gpuTimeFrames", "worldCacheUpdatesRun", "worldCacheUrgentRuns", "worldCacheAsyncRuns", "worldCacheBakes") + churn}
        print(f"MOTION {tag} parkedAfter{window}: gpu {d['gpuTimeSum'] / max(d['gpuTimeFrames'], 1.0) / 1000.0:.3f} ms, updates "
              f"{d['worldCacheUpdatesRun']:.0f} (urgent {d['worldCacheUrgentRuns']:.0f}, async {d['worldCacheAsyncRuns']:.0f}), bakes "
              f"{d['worldCacheBakes']:.0f}, starving {s.get('worldCacheStarving')} empty {s.get('worldCacheEmpty')} near low "
              f"{s.get('worldCacheNearLow')} bake low {s.get('worldCacheBakeLowBatches')}; churn { {k: int(d[k]) for k in churn} }; "
              f"{summary(walls)} {levels(s)}", flush=True)
        # beamRefineStep: what the parked sweep did in the window (builds, blocks, finished and restarted sweeps, why it restarted).
        r = {k: float(s.get(k) or 0) - float(before.get(k) or 0) for k in (
            "beamRefineFrames", "beamRefineBlocks", "beamRefineSweeps", "beamRefineRestarts", "beamInvalidatedBuilds",
            "beamInvalidateAllBuilds", "beamPolicyRestores", "beamRefineRewinds")}
        print(f"MOTION {tag} parkedAfter{window} refine: { {k: int(v) for k, v in r.items()} } cursor {s.get('beamRefineCursor')}/"
              f"{s.get('beamRefineBoxBlocks')} left {s.get('beamRefineLeft')} slice {s.get('beamRefineSlice')} base "
              f"{s.get('beamRefineBaseUs')} us ({s.get('beamRefineBaseSamples')}) build {s.get('beamRefineBuildUs')} us credit "
              f"{s.get('beamRefineCreditUs')} us", flush=True)
        # HSTR_MOTION_PARK_PROFILE=N: each parked window followed by N profiled frames (what the drain after a flight costs).
        if int(os.environ.get("HSTR_MOTION_PARK_PROFILE", "0")):
            profile(f"{tag} parkedAfter{window}", int(os.environ["HSTR_MOTION_PARK_PROFILE"]), 0.0)
        # With HSTR_MOTION_COUNT: what the parked beam re-marches, and why (dirty reasons), window by window.
        if COUNT:
            c = counted(COUNT, 0.0)
            print(f"MOTION {tag} parkedAfter{window} dirty per frame (counted): "
                  f"{ {k: round(sum(float(x.get(k) or 0) for x in c) / len(c), 1) for k in c[0]} }", flush=True)
    # HSTR_MOTION_REPARK=1: back at the parked camera after the flights, settled, and profiled as "parked" was - the same view, so a
    # GPU time above parked's is the run's clocks or accumulated state, not the content the walk reached.
    if os.environ.get("HSTR_MOTION_REPARK", "0") != "0":
        position[0] = start_position
        for window in range(int(os.environ.get("HSTR_MOTION_REPARK_SETTLE", "300")) // 60):
            walls, _ = fly(60, 0.0)
            print(f"MOTION {tag} repark{window}: {summary(walls)} {levels(cloud_stats())}", flush=True)
        profile(f"{tag} reparked", 60, 0.0)
exit()
