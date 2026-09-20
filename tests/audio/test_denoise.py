from types import SimpleNamespace

import numpy as np
import pytest

from onevoice.audio import denoise
from onevoice.core.models.audio_chunk import AudioChunk

class IdentitySession:

    def __init__(self):
        self.seen_states = []
        self.fail = False

    def get_inputs(self):
        return [
            SimpleNamespace(name=name, shape=list(shape), type="tensor(float)")
            for name, shape in denoise._INPUT_SHAPES.items()
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(name=name, shape=list(shape), type="tensor(float)")
            for name, shape in zip(
                denoise._OUTPUT_NAMES, denoise._INPUT_SHAPES.values()
            )
        ]

    def run(self, names, values):
        if self.fail:
            raise RuntimeError("model failed")
        self.seen_states.append(values["conv_cache"].copy())
        return [values["mix"].copy()] + [
            values[name] + 1 for name in list(denoise._INPUT_SHAPES)[1:]
        ]

@pytest.fixture
def factory(monkeypatch, tmp_path):
    path = tmp_path / "model.onnx"
    path.touch()

    def create():
        session = IdentitySession()
        monkeypatch.setattr(denoise, "_load_session", lambda *args: session)
        return denoise.GtcrnDenoiser(path), session

    return create

def audio(data, epoch=1, target="speaker-a", sample_rate=16000, channels=1):
    return AudioChunk(
        100.0,
        data,
        sample_rate,
        channels,
        {
            "target_track_id": target,
            "ui_selection_epoch": epoch,
            "output_ready": True,
        },
    )

def test_arbitrary_chunk_sizes_have_exact_fixed_delay_and_no_lost_samples(factory):
    source = np.random.default_rng(3).normal(0, 0.1, 4096).astype(np.float32)
    source_copy = source.copy()
    expected = np.concatenate((np.zeros(512), source))
    padded = np.concatenate((source, np.zeros(512)))
    for split in ([320] * 14, [1, 257, 1000, 3, 512, 799], [len(padded)]):
        processor, _ = factory()
        offset = 0
        actual = []
        for size in split + [len(padded)]:
            part = padded[offset : offset + size]
            if not part.size:
                break
            result = processor.process(audio(part))
            assert len(result.data) == len(part)
            assert result.timestamp_ms == 100.0
            assert result.metadata["output_ready"] is True
            assert result.metadata["denoise_delay_ms"] == 32.0
            actual.extend(result.data)
            offset += len(part)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(source, source_copy)

@pytest.mark.parametrize("change", ["epoch", "target", "reset"])
def test_reset_drops_old_audio_and_all_model_state(factory, change):
    processor, session = factory()
    processor.process(audio(np.ones(1600, dtype=np.float32) * 0.5))
    assert np.max(session.seen_states[-1]) > 0
    if change == "reset":
        processor.reset()
    args = {}
    if change == "epoch":
        args["epoch"] = 2
    elif change == "target":
        args["target"] = "speaker-b"
    before = len(session.seen_states)
    result = processor.process(audio(np.zeros(1600), **args))
    assert not any(result.data)
    assert not np.any(session.seen_states[before])

def test_inference_failure_raises_and_resets_instead_of_passing_input(factory):
    processor, session = factory()
    processor.process(audio(np.ones(320) * 0.25))
    session.fail = True
    with pytest.raises(RuntimeError, match="noise suppression failed"):
        processor.process(audio(np.ones(320) * 0.25))
    session.fail = False
    assert not any(processor.process(audio(np.zeros(1024))).data)

def test_rejects_missing_model_before_audio_opens(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_gtcrn"):
        denoise.GtcrnDenoiser(tmp_path / "missing.onnx")

@pytest.mark.parametrize(
    "samples,kwargs",
    [
        ([float("nan")], {}),
        ([0.0], {"channels": 2}),
        ([0.0], {"sample_rate": 48000}),
    ],
)
def test_invalid_audio_is_rejected(factory, samples, kwargs):
    processor, _ = factory()
    with pytest.raises(ValueError):
        processor.process(audio(samples, **kwargs))

def test_wrong_model_interface_is_rejected_at_construction(monkeypatch, tmp_path):
    path = tmp_path / "wrong.onnx"
    path.touch()
    session = IdentitySession()
    monkeypatch.setattr(session, "get_inputs", lambda: [])
    monkeypatch.setattr(denoise, "_load_session", lambda *args: session)
    with pytest.raises(ValueError, match="unsupported input/output"):
        denoise.GtcrnDenoiser(path)

def test_non_finite_model_fails_during_construction(monkeypatch, tmp_path):
    path = tmp_path / "broken.onnx"
    path.touch()
    session = IdentitySession()
    monkeypatch.setattr(
        session,
        "run",
        lambda names, data: [
            np.full(shape, np.nan) for shape in denoise._INPUT_SHAPES.values()
        ],
    )
    monkeypatch.setattr(denoise, "_load_session", lambda *args: session)
    with pytest.raises(RuntimeError, match="returned invalid"):
        denoise.GtcrnDenoiser(path)
