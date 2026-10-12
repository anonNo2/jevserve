import pytest

from jevserve.artifacts.registry import ModelRegistry


def test_unregistered_model_is_not_downloaded_or_substituted():
    with pytest.raises(ValueError, match="not in"):
        ModelRegistry({}).resolve("ZefanCai/Open-Jev-2B")
