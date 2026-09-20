

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext, SeparatorAdapter
from onevoice.separation.config import SeparationConfig
from onevoice.separation.dolphin_loader import MOUTH_FPS, MOUTH_SIZE, resolve_dtype
from onevoice.separation.utilities import (
    chunk_to_float_list,
    import_numpy,
    import_torch,
)

logger = logging.getLogger(__name__)

@dataclass
class _TrackWindowState:
    audio_buf: Any = None
    mouth_buf: list = field(
        default_factory=list
    )
    frame_buf: list = field(
        default_factory=list
    )
    samples_since_infer: int = 0
    audio_start_sample: int = 0
    total_samples: int = 0
    next_window_end: int = 0
    origin_timestamp_ms: float | None = None
    pending_window_start: int | None = None
    pending_visual_coverage: float = 0.0
    pending_visual_max_age_ms: float | None = None
    ola_next_start: int | None = None
    fifo_start_sample: int | None = None
    recent_tail: Any = None
    last_visual_coverage: float = 0.0
    last_visual_max_age_ms: float | None = None
    dropped_hops: int = 0
    audio_discontinuities: int = 0
    audio_resyncs: int = 0
    resynced_samples: int = 0
    ola_acc: Any = (
        None
    )
    ola_norm: Any = None
    fifo: Any = None
    warmed_up: bool = False
    last_conditioning: str = "buffering"
    last_seen_ms: float = 0.0
    pending_future: Future | None = (
        None
    )

@dataclass
class _CallBundle:
    samples: Any = None
    n_samples: int = 0
    target_track_id: str | None = None
    lip_patch: Any = None
    frame_bgr: Any = None
    audio_timestamp_ms: float = 0.0
    visual_timestamp_ms: float | None = None
    source_timestamp_ms: float | None = None
    buffered_delay_ms: float | None = None
    visual_coverage: float = 0.0
    visual_max_age_ms: float | None = None
    dropped_hops: int = 0
    audio_discontinuities: int = 0
    audio_resyncs: int = 0
    conditioning_this_call: str = "buffering"
    output_ready: bool = False

class DolphinAdapter(SeparatorAdapter):

    def __init__(self, config: SeparationConfig) -> None:
        np = import_numpy()
        self._np = np
        params = config.params
        self._window_s = float(params.get("window_s", 0.5))
        self._hop_s = float(params.get("hop_s", self._window_s / 2.0))
        self._mouth_size = int(params.get("mouth_size", MOUTH_SIZE))
        self._mouth_fps = int(params.get("mouth_fps", MOUTH_FPS))
        self._sample_rate = config.sample_rate
        self._window_samples = max(1, int(round(self._window_s * self._sample_rate)))
        self._hop_samples = max(1, int(round(self._hop_s * self._sample_rate)))
        self._window_frames = max(1, int(round(self._window_s * self._mouth_fps)))
        if not (
            math.isfinite(self._window_s)
            and math.isfinite(self._hop_s)
            and 0 < self._hop_s <= self._window_s
            and self._mouth_fps > 0
        ):
            raise ValueError("Dolphin requires 0 < hop_s <= window_s and mouth_fps > 0")
        self._visual_max_gap_ms = float(params.get("visual_max_gap_ms", 120.0))
        self._min_visual_coverage = float(params.get("min_visual_coverage", 0.9))
        self._max_audio_gap_ms = float(params.get("max_audio_gap_ms", 120.0))

        self._max_resync_gap_ms = float(
            params.get("max_resync_gap_ms", 1000.0 * self._window_s)
        )
        self._output_gain = float(params.get("output_gain", 1.0))
        self._output_mode = str(params.get("output_mode", "overlap_add"))
        self._lookahead_s = float(params.get("lookahead_s", 0.5))
        self._crossfade_s = float(params.get("crossfade_s", 0.02))
        if self._output_mode not in ("overlap_add", "recent"):
            raise ValueError("output_mode must be overlap_add or recent")
        if not (
            math.isfinite(self._lookahead_s)
            and self._lookahead_s >= 0
            and math.isfinite(self._crossfade_s)
            and self._crossfade_s >= 0
        ):
            raise ValueError("lookahead_s and crossfade_s must be finite and >= 0")
        self._lookahead_samples = round(self._lookahead_s * self._sample_rate)
        self._crossfade_samples = round(self._crossfade_s * self._sample_rate)
        if self._output_mode == "recent" and (
            self._hop_samples + self._lookahead_samples + self._crossfade_samples
            > self._window_samples
            or self._crossfade_samples > self._hop_samples
        ):
            raise ValueError(
                "Recent output needs hop + lookahead + crossfade <= window "
                "and crossfade <= hop"
            )
        if not math.isfinite(self._output_gain) or self._output_gain < 0:
            raise ValueError("output_gain must be finite and >= 0")
        if not (
            math.isfinite(self._visual_max_gap_ms)
            and self._visual_max_gap_ms > 0
            and math.isfinite(self._max_audio_gap_ms)
            and self._max_audio_gap_ms > 0
            and math.isfinite(self._min_visual_coverage)
            and 0 < self._min_visual_coverage <= 1
        ):
            raise ValueError("Invalid visual coverage or A/V timestamp gap limits")
        self._hann = np.hanning(self._window_samples).astype(np.float32)

        self._hann = np.clip(self._hann, 1e-3, None)

        self._auto_gain = bool(params.get("auto_gain", True))
        gain_cap = params.get("max_auto_gain")
        self._max_auto_gain = None if gain_cap is None else float(gain_cap)
        if self._max_auto_gain is not None and (
            not math.isfinite(self._max_auto_gain) or self._max_auto_gain < 1.0
        ):
            raise ValueError("max_auto_gain must be null or a finite value >= 1")
        self._noise_floor = float(params.get("noise_floor", 1e-4))

        self._visual_aligner = str(params.get("visual_aligner", "mediapipe")).lower()
        self._align_window_margin = int(params.get("align_window_margin", 12))
        self._aligner: Any = None
        self._aligner_failed = False
        self._track_states: dict[str, _TrackWindowState] = {}
        self._active_track_id: str | None = None
        self._active_selection_epoch: int | None = None
        self._call = _CallBundle()

        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="dolphin-window"
        )

    def to_backend(
        self,
        model: Any,
        audio_chunk: AudioChunk,
        target: TargetSelection,
        all_tracks: list[SpeakerTrack],
        ctx: AdapterContext,
    ) -> Any:
        np = self._np
        samples = np.asarray(chunk_to_float_list(audio_chunk.data), dtype=np.float32)
        if audio_chunk.channels > 1 and samples.size:
            samples = samples.reshape(-1, audio_chunk.channels).mean(axis=1)

        track_id: str | None = None
        lip_patch = None
        frame_bgr = None
        visual_timestamp_ms = None
        speaker = target.selected_speaker
        if speaker is not None:
            track_id = speaker.track_id
            self._maybe_reset_state(
                track_id, speaker.metadata.get("ui_selection_epoch")
            )
            lip_patch = speaker.metadata.get("lip_patch")
            frame_bgr = speaker.metadata.get("frame_bgr")
            visual_timestamp_ms = speaker.metadata.get("lip_roi_ts_ms")
            state = self._track_states.setdefault(track_id, self._new_state())

            timestamp = float(audio_chunk.timestamp_ms)
            if not math.isfinite(timestamp):
                raise ValueError("Audio timestamp must be finite")
            if state.origin_timestamp_ms is not None and timestamp != 0.0:
                expected = (
                    state.origin_timestamp_ms
                    + 1000 * state.total_samples / self._sample_rate
                )
                gap_ms = timestamp - expected
                if abs(gap_ms) > self._max_audio_gap_ms:
                    if 0.0 < gap_ms <= self._max_resync_gap_ms:

                        missing = int(round(gap_ms * self._sample_rate / 1000.0))
                        if missing > 0:
                            self._push_audio(
                                state, np.zeros(missing, dtype=np.float32)
                            )
                            state.audio_resyncs += 1
                            state.resynced_samples += missing
                            logger.info(
                                "Dolphin audio gap %.0f ms on %s; zero-filled "
                                "(%d samples), window preserved",
                                gap_ms,
                                track_id,
                                missing,
                            )
                    else:
                        self._warn_if_discarding_in_flight_window(
                            state, track_id, track_id
                        )
                        fresh = self._new_state()
                        fresh.audio_discontinuities = state.audio_discontinuities + 1
                        fresh.dropped_hops = state.dropped_hops
                        fresh.audio_resyncs = state.audio_resyncs
                        fresh.resynced_samples = state.resynced_samples
                        self._track_states[track_id] = state = fresh
                        logger.warning(
                            "Dolphin audio timestamp discontinuity %.0f ms "
                            "(beyond %.0f ms resync limit); rebuffering %s",
                            gap_ms,
                            self._max_resync_gap_ms,
                            track_id,
                        )
            state.last_seen_ms = time.monotonic() * 1000.0

        self._call = _CallBundle(
            samples=samples,
            n_samples=int(samples.shape[0]),
            target_track_id=track_id,
            lip_patch=lip_patch,
            frame_bgr=frame_bgr,
            audio_timestamp_ms=float(audio_chunk.timestamp_ms),
            visual_timestamp_ms=visual_timestamp_ms,
        )
        return self._call

    def infer(self, model: Any, backend_input: Any, ctx: AdapterContext) -> Any:
        call: _CallBundle = backend_input
        track_id = call.target_track_id
        if track_id is None:
            return call.samples

        state = self._track_states.setdefault(track_id, self._new_state())
        if state.origin_timestamp_ms is None:
            state.origin_timestamp_ms = call.audio_timestamp_ms
        audio_start_ms = (
            state.origin_timestamp_ms + 1000 * state.total_samples / self._sample_rate
        )
        self._push_audio(state, call.samples)
        visual_ts = call.visual_timestamp_ms
        if visual_ts is None:

            visual_ts = audio_start_ms
        visual_ts = float(visual_ts)
        if not math.isfinite(visual_ts):
            raise ValueError("lip_roi_ts_ms must be finite")
        if self._visual_aligner == "retinaface":
            self._push_frame(state, call.frame_bgr, visual_ts)

            self._push_mouth(state, call.lip_patch, visual_ts)
        else:
            self._push_mouth(state, call.lip_patch, visual_ts)

        self._maybe_collect_finished(state)

        if state.total_samples >= self._window_samples:
            state.warmed_up = True
            if (
                state.total_samples >= state.next_window_end
                and state.pending_future is None
            ):
                self._submit_window(model, state, ctx)

        n_needed = call.n_samples

        call.output_ready = n_needed > 0 and state.fifo.shape[0] >= n_needed
        if state.fifo.size and state.fifo_start_sample is not None:
            call.source_timestamp_ms = (
                state.origin_timestamp_ms
                + 1000 * state.fifo_start_sample / self._sample_rate
            )
            call.buffered_delay_ms = max(0.0, audio_start_ms - call.source_timestamp_ms)
        call.visual_coverage = state.last_visual_coverage
        call.visual_max_age_ms = state.last_visual_max_age_ms
        call.dropped_hops = state.dropped_hops
        call.audio_discontinuities = state.audio_discontinuities
        call.audio_resyncs = state.audio_resyncs
        out = self._pop_fifo(state, n_needed)
        call.conditioning_this_call = (
            state.last_conditioning if state.warmed_up else "buffering"
        )
        return out

    def flush_pending(self, track_id: str, timeout_s: float = 30.0) -> None:

        state = self._track_states.get(track_id)
        if state is None or state.pending_future is None:
            return
        state.pending_future.result(timeout=timeout_s)
        self._maybe_collect_finished(state)

    def drain_and_warn_on_shutdown(self, timeout_s: float = 2.0) -> None:

        for track_id, state in list(self._track_states.items()):
            future = state.pending_future
            if future is None:
                continue
            try:
                window_out, _had_visual, _window_audio = future.result(
                    timeout=timeout_s
                )
            except TimeoutError:
                cancelled = future.cancel()
                logger.warning(
                    "Dolphin: shutdown with an in-flight window still computing for "
                    "track %s after %.1fs; giving up (cancelled=%s) "
                    "— that audio is lost.",
                    track_id,
                    timeout_s,
                    cancelled,
                )
                continue
            except Exception as exc:  # noqa: BLE001 - just reporting, not handling
                logger.warning(
                    "Dolphin: shutdown collected a failed window for track %s: %s",
                    track_id,
                    exc,
                )
                continue
            np = self._np
            logger.warning(
                "Dolphin: shutdown collected a completed window for track %s "
                "(max_abs=%.4g) that arrived too late to be played/recorded — that "
                "audio is lost. Consider a longer run duration to avoid this.",
                track_id,
                float(np.abs(window_out).max()) if window_out.size else 0.0,
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def from_backend(
        self, backend_output: Any, reference: AudioChunk, ctx: AdapterContext
    ) -> AudioChunk:
        np = self._np
        samples = np.asarray(backend_output, dtype=np.float32).reshape(-1).tolist()
        conditioning = self._call.conditioning_this_call
        target_track_id = self._call.target_track_id
        metadata = {
            **reference.metadata,
            "backend": ctx.config.name,
            "conditioning": conditioning,
            "target_track_id": target_track_id,
            "output_ready": self._call.output_ready,
            "fallback": False,
            "source_timestamp_ms": self._call.source_timestamp_ms,
            "buffered_delay_ms": self._call.buffered_delay_ms,
            "visual_coverage": self._call.visual_coverage,
            "visual_max_age_ms": self._call.visual_max_age_ms,
            "dropped_hops": self._call.dropped_hops,
            "audio_discontinuities": self._call.audio_discontinuities,
            "audio_resyncs": self._call.audio_resyncs,
        }
        return AudioChunk(
            timestamp_ms=reference.timestamp_ms,
            data=samples,
            sample_rate=reference.sample_rate,
            channels=reference.channels,
            metadata=metadata,
        )

    def _new_state(self) -> _TrackWindowState:
        np = self._np
        return _TrackWindowState(
            ola_acc=np.zeros(self._window_samples, dtype=np.float32),
            ola_norm=np.zeros(self._window_samples, dtype=np.float32),
            fifo=np.zeros(0, dtype=np.float32),
            next_window_end=self._window_samples,
        )

    def _push_audio(self, state: _TrackWindowState, samples: Any) -> None:
        np = self._np
        if state.audio_buf is None:
            state.audio_buf = samples.copy()
        else:
            state.audio_buf = np.concatenate([state.audio_buf, samples])
        state.total_samples += int(samples.shape[0])

        capacity = self._window_samples + 2 * self._hop_samples
        if state.audio_buf.shape[0] > capacity:
            state.audio_buf = state.audio_buf[-capacity:]
        state.audio_start_sample = state.total_samples - state.audio_buf.shape[0]
        state.samples_since_infer += int(samples.shape[0])

    def _push_mouth(
        self, state: _TrackWindowState, lip_patch: Any, timestamp_ms: float
    ) -> None:
        frame = self._patch_to_frame(lip_patch) if lip_patch is not None else None
        if frame is not None and not self._np.all(self._np.isfinite(frame)):
            frame = None
        state.mouth_buf = self._append_visual(
            state, state.mouth_buf, timestamp_ms, frame
        )

    def _push_frame(
        self, state: _TrackWindowState, frame_bgr: Any, timestamp_ms: float
    ) -> None:

        np = self._np
        arr = None if frame_bgr is None else np.asarray(frame_bgr)
        if arr is not None and (arr.ndim != 3 or arr.shape[2] < 3):
            arr = None
        state.frame_buf = self._append_visual(state, state.frame_buf, timestamp_ms, arr)

    def _append_visual(self, state, buffer, timestamp_ms, frame):

        if buffer and timestamp_ms < buffer[-1][0]:
            return buffer
        if buffer and timestamp_ms == buffer[-1][0]:
            buffer = buffer[:-1]
        buffer = [*buffer, (timestamp_ms, frame)]
        now_ms = (
            state.origin_timestamp_ms + 1000 * state.total_samples / self._sample_rate
        )
        history_ms = (
            1000 * (self._window_samples + 2 * self._hop_samples) / self._sample_rate
        )
        oldest = now_ms - history_ms - self._visual_max_gap_ms

        return [item for item in buffer if item[0] >= oldest][-2048:]

    def _sample_visual(self, buffer, start_ms, *, mouth=True):

        np = self._np
        blank = np.zeros((self._mouth_size, self._mouth_size), dtype=np.float32)
        times = np.asarray([entry[0] for entry in buffer], dtype=np.float64)
        frames, ages, valid = [], [], []
        for index in range(self._window_frames):
            wanted = start_ms + index * 1000.0 / self._mouth_fps
            position = int(np.searchsorted(times, wanted))
            candidates = [j for j in (position - 1, position) if 0 <= j < len(buffer)]
            nearest = (
                min(candidates, key=lambda j: abs(times[j] - wanted))
                if candidates
                else None
            )
            age = abs(times[nearest] - wanted) if nearest is not None else None
            frame = buffer[nearest][1] if nearest is not None else None
            usable = frame is not None and age <= self._visual_max_gap_ms
            frames.append(frame if usable else (blank if mouth else None))
            valid.append(usable)
            if usable:
                ages.append(float(age))
        return frames, float(np.mean(valid)), max(ages) if ages else None

    def _get_aligner(self) -> Any | None:
        if self._aligner_failed:
            return None
        if self._aligner is not None:
            return self._aligner
        try:
            from onevoice.video.aligners import RetinaFaceAligner

            self._aligner = RetinaFaceAligner()
            logger.info("Dolphin: RetinaFaceAligner loaded (visual_aligner=retinaface)")
            return self._aligner
        except Exception as exc:  # noqa: BLE001
            self._aligner_failed = True
            logger.warning(
                "Dolphin: RetinaFaceAligner unavailable (%s); falling back to "
                "MediaPipe lip_patch for this session",
                exc,
            )
            return None

    def _patch_to_frame(self, lip_patch: Any) -> Any:

        np = self._np
        size = self._mouth_size
        if lip_patch is None:
            return np.zeros((size, size), dtype=np.float32)
        arr = np.asarray(lip_patch, dtype=np.float32)
        if arr.ndim == 2:
            if arr.size == 0:
                return np.zeros((size, size), dtype=np.float32)
            if arr.shape == (size, size):
                return arr
            return self._resize_2d(arr, size)
        arr = arr.reshape(-1)
        if arr.size == 0:
            return np.zeros((size, size), dtype=np.float32)
        target_len = size * size
        if arr.size < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[: arr.size] = arr
            arr = padded
        elif arr.size > target_len:
            arr = arr[:target_len]
        return arr.reshape(size, size)

    def _resize_2d(self, patch: Any, size: int) -> Any:

        np = self._np
        src_h, src_w = patch.shape
        row_idx = (np.arange(size) * src_h / size).astype(np.int64).clip(0, src_h - 1)
        col_idx = (np.arange(size) * src_w / size).astype(np.int64).clip(0, src_w - 1)
        return patch[row_idx][:, col_idx].astype(np.float32)

    def _submit_window(
        self, model: Any, state: _TrackWindowState, ctx: AdapterContext
    ) -> None:

        latest_end = (
            self._window_samples
            + ((state.total_samples - self._window_samples) // self._hop_samples)
            * self._hop_samples
        )
        if state.next_window_end < latest_end:
            skipped = (latest_end - state.next_window_end) // self._hop_samples
            state.dropped_hops += skipped
            state.next_window_end = latest_end
            self._reset_output(state)
            logger.warning("Dolphin skipped %d overdue hop(s); reset overlap", skipped)
        end = state.next_window_end
        start = end - self._window_samples
        offset = start - state.audio_start_sample
        window_audio = state.audio_buf[offset : offset + self._window_samples].copy()
        if offset < 0 or len(window_audio) != self._window_samples:
            raise RuntimeError("Dolphin audio window history is not contiguous")
        start_ms = state.origin_timestamp_ms + 1000 * start / self._sample_rate
        mouth_frames, coverage, age = self._sample_visual(state.mouth_buf, start_ms)
        frame_bgrs = None
        if self._visual_aligner == "retinaface":
            frames, frame_coverage, _ = self._sample_visual(
                state.frame_buf, start_ms, mouth=False
            )

            if frame_coverage == 1.0:
                frame_bgrs = frames
        state.pending_window_start = start
        state.pending_visual_coverage = coverage
        state.pending_visual_max_age_ms = age
        state.next_window_end = end + self._hop_samples
        state.samples_since_infer = state.total_samples - end
        state.pending_future = self._executor.submit(
            self._infer_window_forward,
            model,
            window_audio,
            mouth_frames,
            ctx,
            frame_bgrs,
        )

    def _maybe_collect_finished(self, state: _TrackWindowState) -> None:
        future = state.pending_future
        if future is None or not future.done():
            return
        state.pending_future = None
        try:
            window_out, had_visual, window_audio = future.result()
        except Exception as exc:  # noqa: BLE001 - degrade to silence, keep streaming
            logger.warning(
                "Dolphin window inference failed, dropping this hop: %s", exc
            )
            state.dropped_hops += 1
            self._reset_output(state)
            return
        start = state.pending_window_start
        if start is not None:
            lateness = state.total_samples - (start + self._window_samples)
            if lateness > self._hop_samples:
                state.dropped_hops += 1
                self._reset_output(state)
                logger.warning(
                    "Dolphin discarded an overdue result; rebuffering current audio"
                )
                return
            if state.ola_next_start is not None and state.ola_next_start != start:
                self._reset_output(state)
            state.ola_next_start = start + self._hop_samples
        had_visual = (
            had_visual and state.pending_visual_coverage >= self._min_visual_coverage
        )
        state.last_visual_coverage = state.pending_visual_coverage
        state.last_visual_max_age_ms = state.pending_visual_max_age_ms
        if not had_visual:

            self._reset_output(state)
            state.last_conditioning = "audio_only"
            return
        if not state.fifo.size:
            state.fifo_start_sample = start
        self._integrate_window_result(state, window_out, had_visual, window_audio)

    def _reset_output(self, state: _TrackWindowState) -> None:
        state.ola_acc = self._np.zeros(self._window_samples, dtype=self._np.float32)
        state.ola_norm = self._np.zeros(self._window_samples, dtype=self._np.float32)
        state.ola_next_start = None
        state.fifo = self._np.zeros(0, dtype=self._np.float32)
        state.fifo_start_sample = None
        state.recent_tail = None
        state.last_conditioning = "buffering"

    def _infer_window_forward(
        self,
        model: Any,
        window_audio: Any,
        mouth_frames: list,
        ctx: AdapterContext,
        frame_bgrs: list | None = None,
    ) -> tuple[Any, bool, Any]:

        np = self._np
        torch = import_torch()

        if window_audio.shape[0] < self._window_samples:
            pad = np.zeros(
                self._window_samples - window_audio.shape[0], dtype=np.float32
            )
            window_audio = np.concatenate([pad, window_audio])

        if frame_bgrs:
            aligner = self._get_aligner()
            if aligner is not None:
                try:
                    mouth_stack = aligner.align_window(
                        frame_bgrs, window_margin=self._align_window_margin
                    )
                    mouth_frames = [mouth_stack[i] for i in range(mouth_stack.shape[0])]
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Dolphin RetinaFace align_window failed (%s); "
                        "using lip_patch buffer",
                        exc,
                    )

        had_visual = any(np.any(f) for f in mouth_frames)
        if len(mouth_frames) < self._window_frames:
            pad_frame = np.zeros((self._mouth_size, self._mouth_size), dtype=np.float32)
            mouth_frames = [pad_frame] * (
                self._window_frames - len(mouth_frames)
            ) + mouth_frames
        elif len(mouth_frames) > self._window_frames:
            mouth_frames = mouth_frames[-self._window_frames :]
        mouth_stack = np.stack(mouth_frames, axis=0)

        dtype = resolve_dtype(ctx.config, torch)
        mix = torch.from_numpy(np.ascontiguousarray(window_audio)).unsqueeze(0)
        mouth = (
            torch.from_numpy(np.ascontiguousarray(mouth_stack))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        mix = mix.to(device=ctx.device, dtype=dtype)
        mouth = mouth.to(device=ctx.device, dtype=dtype)

        with torch.no_grad():
            est = model(mix, mouth)
            if str(ctx.device).startswith("cuda"):
                torch.cuda.synchronize()
        window_out = est.squeeze().detach().float().cpu().numpy()
        window_out = np.asarray(window_out, dtype=np.float32).reshape(-1)
        logger.debug(
            "Dolphin window computed: max_abs=%.6g mean_abs=%.6g shape=%s",
            float(np.abs(window_out).max()) if window_out.size else 0.0,
            float(np.abs(window_out).mean()) if window_out.size else 0.0,
            window_out.shape,
        )
        if window_out.shape[0] != self._window_samples:

            fitted = np.zeros(self._window_samples, dtype=np.float32)
            n = min(window_out.shape[0], self._window_samples)
            fitted[:n] = window_out[:n]
            window_out = fitted

        return window_out, had_visual, window_audio

    def _apply_auto_gain(self, segment: Any, reference_audio: Any) -> Any:

        np = self._np
        input_rms = float(np.sqrt(np.mean(np.square(reference_audio))))
        output_rms = float(np.sqrt(np.mean(np.square(segment))))
        if output_rms < self._noise_floor or input_rms < 1e-8:
            return segment
        gain = input_rms / output_rms
        if self._max_auto_gain is not None:
            gain = min(gain, self._max_auto_gain)
        return np.clip(segment * gain, -1.0, 1.0).astype(np.float32)

    def _integrate_window_result(
        self,
        state: _TrackWindowState,
        window_out: Any,
        had_visual: bool,
        window_audio: Any,
    ) -> None:

        if self._output_mode == "recent":
            self._integrate_recent(state, window_out, had_visual, window_audio)
            return
        np = self._np
        state.ola_acc += window_out * self._hann
        state.ola_norm += self._hann

        finalized = state.ola_acc[: self._hop_samples] / np.maximum(
            state.ola_norm[: self._hop_samples], 1e-6
        )
        if self._auto_gain:

            finalized = self._apply_auto_gain(
                finalized, window_audio[: self._hop_samples]
            )
        if self._output_gain != 1.0:
            finalized = np.clip(finalized * self._output_gain, -1.0, 1.0)
        state.fifo = np.concatenate([state.fifo, finalized.astype(np.float32)])

        if state.fifo.size > 2 * self._hop_samples:
            removed = state.fifo.size - 2 * self._hop_samples
            state.fifo = state.fifo[removed:]
            if state.fifo_start_sample is not None:
                state.fifo_start_sample += removed

        state.ola_acc = np.concatenate(
            [
                state.ola_acc[self._hop_samples :],
                np.zeros(self._hop_samples, dtype=np.float32),
            ]
        )
        state.ola_norm = np.concatenate(
            [
                state.ola_norm[self._hop_samples :],
                np.zeros(self._hop_samples, dtype=np.float32),
            ]
        )
        state.last_conditioning = "visual" if had_visual else "audio_only"

    def _integrate_recent(self, state, window_out, had_visual, window_audio):

        np = self._np
        overlap = self._crossfade_samples
        end = self._window_samples - self._lookahead_samples
        start = end - self._hop_samples
        prefix = window_out[start - overlap : start].copy()
        if overlap and state.recent_tail is not None:
            alpha = np.linspace(0, 1, overlap, dtype=np.float32)
            prefix = state.recent_tail * (1 - alpha) + prefix * alpha
        finalized = np.concatenate((prefix, window_out[start : end - overlap]))
        state.recent_tail = window_out[end - overlap : end].copy() if overlap else None
        if not state.fifo.size:
            state.fifo_start_sample = (
                (state.pending_window_start or 0) + start - overlap
            )
        if self._auto_gain:
            finalized = self._apply_auto_gain(
                finalized,
                window_audio[start - overlap : end - overlap],
            )
        if self._output_gain != 1.0:
            finalized = np.clip(finalized * self._output_gain, -1.0, 1.0)
        state.fifo = np.concatenate((state.fifo, finalized.astype(np.float32)))
        if state.fifo.size > 2 * self._hop_samples:
            removed = state.fifo.size - 2 * self._hop_samples
            state.fifo = state.fifo[removed:]
            if state.fifo_start_sample is not None:
                state.fifo_start_sample += removed
        state.last_conditioning = "visual" if had_visual else "audio_only"

    def _pop_fifo(self, state: _TrackWindowState, n: int) -> Any:
        np = self._np
        available = state.fifo.shape[0]
        if state.fifo_start_sample is not None:
            state.fifo_start_sample += min(n, available)
        if available >= n:
            out, state.fifo = state.fifo[:n], state.fifo[n:]
            return out

        out = np.zeros(n, dtype=np.float32)
        out[:available] = state.fifo
        state.fifo = np.zeros(0, dtype=np.float32)
        return out

    def _maybe_reset_state(
        self, track_id: str, selection_epoch: int | None = None
    ) -> None:
        if self._active_track_id is not None and (
            self._active_track_id != track_id
            or self._active_selection_epoch != selection_epoch
        ):
            old_state = self._track_states.pop(self._active_track_id, None)
            self._warn_if_discarding_in_flight_window(
                old_state, self._active_track_id, track_id
            )
            logger.info(
                "Dolphin streaming window reset on target switch %s -> %s "
                "(selection epoch %s -> %s)",
                self._active_track_id,
                track_id,
                self._active_selection_epoch,
                selection_epoch,
            )
        self._active_track_id = track_id
        self._active_selection_epoch = selection_epoch

    def _warn_if_discarding_in_flight_window(
        self, old_state: _TrackWindowState | None, old_track_id: str, new_track_id: str
    ) -> None:

        if old_state is None:
            return
        future = old_state.pending_future
        if future is None:
            return
        if future.done():
            logger.warning(
                "Dolphin: discarding a COMPLETED separated window for track %s "
                "(switched to %s before it was collected) — that audio is lost.",
                old_track_id,
                new_track_id,
            )
        else:
            cancelled = future.cancel()
            logger.warning(
                "Dolphin: discarding an IN-FLIGHT separated window for track %s "
                "(switched to %s mid-computation, cancelled=%s) — that audio is lost.",
                old_track_id,
                new_track_id,
                cancelled,
            )

    def evict_stale_tracks(self, max_age_ms: float = 5000.0) -> None:
        now = time.monotonic() * 1000.0
        stale = [
            tid
            for tid, st in self._track_states.items()
            if now - st.last_seen_ms > max_age_ms
        ]
        for tid in stale:
            self._track_states.pop(tid, None)
