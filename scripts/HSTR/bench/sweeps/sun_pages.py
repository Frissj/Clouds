from sea_config import mk

# Sun page cost: where residualPages (fineSun + sunOctaves) goes under streaming and under a moving sun, and sunColumns - the
# sea's sun field by sun ray (the whole domain each refresh) against the per-voxel march of the stale tiles.
# Logs per frame: sunPageTiles (tiles re-sunned), sunPageChanged (tiles whose cloud changed; each also re-suns every tile it
# shadows) and sunPageQueued (tiles the sun-move queue released). The exact frame keeps the per-voxel field (REFERENCE), rebuilt
# whole when an arm switches method, so a columns arm is scored against a fresh per-voxel field at the same sun.
# sunpages1 (4K, per-voxel): sprint fineSun 0.17 ms for 2 tiles a frame; parked 0.57 deg/frame sun 1.10 ms (8 queued tiles, of
# which sunOctaves 0.53), total 3.40 against ~1.9 static.
# Run: python scripts/HSTR/bench/run_sea.py motion TAG scripts/HSTR/bench/sweeps/sun_pages.py --steps 1
#          --motions "park 0 0 0.01" "walk 2 0.004 0.01" "sprint 20 0"
# sunpages2 (4K, columns resampled into hstrFineSun): parked moving sun 3.42 -> 3.31 ms at 0.501 -> 0.360% over 0.02; sprint
# 2.035 -> 1.96 at 0.358 -> 0.361%. sunscopes1: a refresh cost columns 0.28, resample 0.22, octave gather 0.19, blur 0.31 ms.
# Now read in place (no resample), and sunOctavesHalf stores the octaves in RGBA16F.
TESTS = [
    ("per voxel", mk()),
    ("columns", mk(sunColumns=True)),
    ("columns half octaves", mk(sunColumns=True, sunOctavesHalf=True)),
    ("per voxel again", mk()),
]
