import pytest

from onevoice.separation.exceptions import BackendNotFoundError
from onevoice.separation.passthrough import PassthroughSeparator
from onevoice.separation.registry import (
    available_backends,
    create_separator,
    register_backend,
    resolve_backend_class,
)

def test_builtin_backends_registered():
    names = available_backends()
    for expected in [
        "passthrough",
        "raven",
        "clearvoice",
        "speechbrain",
        "asteroid",
        "avtse",
        "look_once_to_hear",
    ]:
        assert expected in names

def test_resolve_passthrough():
    assert resolve_backend_class("passthrough") is PassthroughSeparator

def test_unknown_backend_raises():
    with pytest.raises(BackendNotFoundError):
        resolve_backend_class("does-not-exist")

def test_create_from_string():
    sep = create_separator("passthrough")
    assert isinstance(sep, PassthroughSeparator)

def test_create_from_dict():
    sep = create_separator({"name": "passthrough"})
    assert isinstance(sep, PassthroughSeparator)

def test_create_from_none_defaults_passthrough():
    assert isinstance(create_separator(None), PassthroughSeparator)

def test_register_custom_backend():
    register_backend(
        "custom-passthrough",
        "onevoice.separation.passthrough:PassthroughSeparator",
    )
    assert isinstance(create_separator("custom-passthrough"), PassthroughSeparator)
