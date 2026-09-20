import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from demo.safety import YamnetClassifier


def test_alarm_load_and_inference_use_cpu(monkeypatch, tmp_path):
    class_map = tmp_path / "classes.csv"
    class_map.write_text("index,mid,display_name\n0,/siren,Siren\n")
    events = []

    @contextmanager
    def device(name):
        events.append(("enter", name))
        yield
        events.append(("exit", name))

    def infer(_):
        assert events[-1] == ("enter", "/CPU:0")
        return SimpleNamespace(numpy=lambda: np.array([[0.9]])), None, None

    model = Mock(side_effect=infer)
    model.class_map_path.return_value.numpy.return_value = str(class_map).encode()

    def load(_):
        assert events == [("hide", [], "GPU"), ("enter", "/CPU:0")]
        return model

    tf = SimpleNamespace(
        config=SimpleNamespace(
            set_visible_devices=lambda devices, kind: events.append(
                ("hide", devices, kind)
            )
        ),
        device=device,
    )
    monkeypatch.setitem(sys.modules, "tensorflow", tf)
    monkeypatch.setitem(sys.modules, "tensorflow_hub", SimpleNamespace(load=load))
    classifier = YamnetClassifier({"Siren"})
    assert classifier.classify(np.zeros(16000), 16000).confidence == 0.9
