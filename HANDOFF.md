# Handoff — night shift → next shift

Written 2026-09-20, ~06:00 UTC. Everything below was run on Shubh's Linux
laptop (RTX 4070, 8 GB), which is **the demo machine**.

Branch to continue from: **`integration`** (pushed to `origin/integration`).
It is `origin/FINAL-BRANCH` + `origin/arnav-elevenlabs` + 5 commits of fixes.

---

## 1. Start here

```bash
cd OneVoice_VTHacks
git fetch origin && git checkout integration && git pull

# the venv lives in the repo now
.venv/bin/python -m pytest tests/ -q          # expect 647 tests, 642 pass, 0 fail
node --test tests/demo/web_ui.test.cjs tests/demo/explain_ui.test.cjs   # 15/15
```

**The working demo command** (flags matter — see §4):

```bash
.venv/bin/python demo/tap_to_select.py \
    --config configs/experiments/dolphin_tap.yaml \
    --live --camera-index 2 --camera-size 1280x720 --input-device 5
```

Then open the URL it prints. Press **Start listening**, wait for faces, click a
person.

---

## 2. The one thing that still breaks the demo

**Separation works. Holding on to the person does not.**

An 86-second live recording with two people talking is at
`runs/sessions/clip_20260920_054831_hvw3x21_/`. Its `manifest.json` tells the
whole story:

```
  t(s)     n   target     ep  muted  conditioning  ready
   0.4   377   person-1   10  False  visual        True    <- working, 7.5s of audio
  13.0  1783   None       -1  True   none          False   <- target lost, 35s of silence
  30.3   263   person-2   11  False  visual        True    <- working again, 5.3s
  35.8     2   person-2   11  True   visual        False   <- lost again
```

Whenever Dolphin has a target it emits real isolated audio: `conditioning:
visual`, `output_ready: true`, no fallback, no passthrough. Measured on the
wavs: input 99.9 % non-silent, **output only 24.4 %**, and all output stops at
39 s.

So the model is fine. What fails is that the selected face track dies and the
selection is orphaned.

### What I already did about it

`video.max_age` was 30 frames = **1.0 s** at 30 fps. Looking down, turning your
head, or a hand crossing your face for one second deletes the track. Raised to
90 frames (3 s), `reid_window_ms` 2500 → 6000, and switched the demo to 720p so
faces are 164–189 px instead of 90–136 px.

Measured effect in the same scene: **first target loss moved from 13 s to 27 s.**
Better, not solved.

### The decision waiting for you — this is the big one

Once a target is lost it stays lost until the listener clicks the person again.
That is **deliberate**, not a bug. It is pinned by
`tests/demo/test_session_state.py::test_lost_target_stays_lost_until_explicit_reselection`,
present since the first prototype commit. `SessionState._lost` is cleared only
in `_reset()` and `select()`, never when the track reappears.

The reason is sound: a track id freed by one person can later be handed to
another, so silently resuming would put a stranger's voice in the listener's
ear.

But as it stands a judge will watch the audio cut out and need a re-click every
~25 s. Three options, in the order I would try them:

1. **Auto-resume only when identity matching confirms the same face.** Keeps the
   safety property and fixes the demo. `identity_matching: true` is already on
   and `SessionFaceTracker` + `LocalFaceEncoder` exist. **Unverified — I never
   got to test whether identity matching actually re-links a returning face.**
   That is the single highest-value next experiment.
2. **Relax the gate outright** — clear `_lost` in `observe_tracks` when the
   requested id is visible again. One line, and it deletes a deliberate safety
   property plus fails an existing test. Only do this as a conscious tradeoff.
3. **Leave it and demo around it** — re-click when it drops. Safest, looks worst.

I deliberately did not choose. It is a product decision.

---

## 3. Problems found and fixed (all committed on `integration`)

| # | Problem | Fix | Commit |
|---|---|---|---|
| 1 | **Fire-alarm safety existed but was not wired into the product.** `SafetyMonitor`/`YamnetClassifier` were reachable only from `demo/safety_demo.py`, which ran a `FakeClassifier` and printed "no real detector wired in yet". The real session had no safety at all. | Wired into `SessionRunner`: raw mic tapped ahead of isolation, `SessionSink` plays raw audio at full gain while an alarm sounds, session stops when the alarm *clears* (so the passthrough is actually audible) and needs a manual restart, with a reason on the stopped card. | `4f9fa3f` |
| 2 | **Tests imported a different checkout.** An editable install pointed at `/home/shubh/onevoice`, so `import onevoice` silently resolved outside the repo. **16 of 20 apparent test failures were this, not real defects.** | Added root `conftest.py` that puts this tree first and *verifies* the import landed here, raising a named error if not. | `f41569f` |
| 3 | `demo/safety_demo.py` printed "no real detector wired in yet", which became false once #1 landed. | Added `--real-detector` (uses YAMNet) and a notice that names the detector actually in use. | `5da3580` |
| 4 | **False positives were untested.** The documented true-negative fixture (`data/dolphin_tier_a/.../mix.wav`) does not exist in the repo. | Added `data/safety_test_clips/speech_dialect_wikimedia_pd.ogg` (45 s conversational speech, public domain, Library of Congress). | `4848573` |
| 5 | **Lip gate could not start.** `lip_gate.enabled: true` but nothing fetched `checkpoints/pause/face_landmarker.task`; a live session died with `FileNotFoundError`. | Added `scripts/fetch_face_landmarker.py` (SHA-256 verified) and fetched it. | `a8cc010` |
| 6 | **`video.device_index: 700` is Windows-only.** 700 is `cv2.CAP_DSHOW`, not a camera index. Correct on the teammate's Windows box, fatal on Linux. | Set to `0`; use `--camera-index 2` for the C270 (see §4). | `a8cc010` |
| 7 | **Captions OOM'd the GPU.** Config asked for `large-v3` on cuda; Dolphin takes ~3.9 GB of the 8 GB card, so it failed and fell back to cpu/small after 3.2 s. | Pinned `captions: {device: cpu, model_size: small}` — same model, 0.9 s, deterministic. | `a8cc010` |
| 8 | **YAMNet cached in `/tmp/tfhub_modules`**, which a reboot clears — the alarm detector would re-download at the venue and, on bad wifi, never arrive, with `SafetyMonitor` swallowing it as "no alarm". | `_persist_tfhub_cache()` points `TFHUB_CACHE_DIR` at `checkpoints/tfhub`. | `a8cc010` |
| 9 | Face tracks died after 1 s of missed detections (§2). | `max_age` 30 → 90, `reid_window_ms` 2500 → 6000. | `fbe98c2` |

---

## 4. Hardware gotchas — these cost us time, do not rediscover them

- **The C270 is camera index 2**, not 0 or 1. The integrated camera occupies
  `/dev/video0` and `/dev/video1` (video1 is metadata-only and will not open).
  The config default of `0` is the integrated camera. **Pass `--camera-index 2`.**
- **`--input-device C270` fails.** It matches both the ALSA device (5) and a
  JACK duplicate (23); the app correctly refuses an ambiguous match. The
  `--help` text suggests the name, which does not work here. **Use
  `--input-device 5`.**
- **Earphones** are the 3.5 mm jack = output device 0 (`HDA Intel PCH ALC257`).
  The OS default routes there when plugged in; no flag needed.
- **Point the webcam at faces.** Two of our test runs produced zero detections
  because the C270 was aimed at a whiteboard and then at a desk. Check
  `/api/frame` if `faces=0`.
- **cuDNN is broken on this machine** — `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.
  `dolphin_loader` detects it and disables cuDNN, so Dolphin runs correctly on
  CUDA but slower. Only 2 overdue-window warnings in 86 s, so not currently
  fatal. Pre-dates the venv rebuild; not worth touching before the deadline.

---

## 5. Environment — self-contained, do not depend on the old checkout

`/home/shubh/onevoice` is a **separate, stale clone**. The repo no longer
depends on it in any way (verified: no `.pth`, no path reference, editable
install repointed). It can be deleted once its 7.6 GB venv is no longer wanted.

Rebuilt inside this repo, all gitignored:

| Path | Size | Source |
|---|---|---|
| `.venv/` | 8.3 GB | 147 pinned packages, CUDA torch 2.13.0+cu130 |
| `vendor/dolphin/` | 815 MB | `scripts/vendor_dolphin.py` |
| `checkpoints/gtcrn/` | 536 KB | `scripts/fetch_gtcrn.py` |
| `checkpoints/pause/` | 3.6 MB | `scripts/fetch_face_landmarker.py` |
| `checkpoints/tfhub/` | 18 MB | YAMNet, auto |

Model caches are **warm — nothing should download at the venue**: Dolphin 28 MB
and faster-whisper-small 464 MB in `~/.cache/huggingface`, YAMNet in
`checkpoints/tfhub`.

> **Landmine:** rebuilding the venv from `pyproject.toml` pulls in `triton` as a
> torch dependency. It is not in the known-good set and **segfaults** the test
> run when loaded after TensorFlow initialises CUDA (died at 61 %). If you
> rebuild: `pip uninstall triton`. Torch and CUDA are unaffected.

---

## 6. Safety feature — verified, with one gap

```bash
.venv/bin/python scripts/verify_safety_classifier.py --positive data/safety_test_clips/civil_defense_siren_wikimedia_pd.ogg
.venv/bin/python scripts/verify_safety_classifier.py --negative data/safety_test_clips/speech_dialect_wikimedia_pd.ogg
```

- Siren → `Siren 1.00`, **PASS**. Through a real `SafetyMonitor`: activates at
  0.93, then release → session stop.
- Speech → **0.00 on every monitored class**, PASS. Conversation does not trip it.
- `siren_wikimedia_pd.ogg` scores 0.03 and does **not** trigger — that is known
  and deliberate, documented by whoever added the clips.

**Gap: `max('Smoke detector, smoke alarm')` is 0.00 in every run.** Both fixtures
are sirens; the actual fire-alarm class has never been exercised. **If you will
demo with a phone playing a smoke alarm, run that exact clip through
`--positive` first.** ~2 minutes, and it is the difference between the feature
firing on stage or not.

---

## 7. Also verified working

- Web UI: all endpoints 200; start → 3 tracks → select → record → stop → quit,
  clean exit, camera and mic released.
- Recording writes `manifest.json` + `input_audio.wav` + `output_audio.wav`.
  The manifest's per-chunk metadata is excellent for debugging — that is how §2
  was diagnosed.
- ElevenLabs and Gemini degrade cleanly with no API key: HTTP 502 and an
  actionable message. Keys go in `.env` (see `.env.example`); `.env` is
  gitignored and nothing sensitive is committed.
- Preflight passes 7/7 on both `dolphin_live.yaml` and `dolphin_tap.yaml`.

---

## 8. Branches

- **`origin/integration`** — continue here. Contains everything.
- `origin/FINAL-BRANCH` — team trunk. `integration` is 10 commits ahead of it and
  merges cleanly; merge when you are happy.
- `origin/arnav-elevenlabs` — already merged into `integration` (0 conflicts).
- `origin/dev-shubh` — the safety feature alone; already in FINAL-BRANCH.
- `integration-old-a8ba643` — **local only**, not pushed. An earlier integration
  attempt whose webcam-fake and CPU-pin commits were superseded by better
  versions from the team. Ignore it; delete when convenient.

---

## 9. Suggested order for the next shift

1. **Test whether identity matching re-links a returning face** (§2 option 1).
   Highest value: it decides the target-loss fix.
2. **Verify the real smoke-alarm clip** you will demo with (§6).
3. Decide the target-loss policy and implement it.
4. Merge `integration` → `FINAL-BRANCH`.

Do a dry run end to end before judging, with the exact command in §1.
