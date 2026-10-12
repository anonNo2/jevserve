import pytest
from pydantic import ValidationError

from jevserve.config import ServerConfig


def test_safe_defaults():
    config = ServerConfig()
    assert (config.body_limit_bytes, config.max_questions) == (1048576, 16)
    assert (config.max_waiting, config.max_active, config.deadline_ms) == (64, 16, 30000)
    assert config.api_workers == 1
    assert config.drain_seconds == 30
    assert config.host == "127.0.0.1"


@pytest.mark.parametrize(
    "override",
    [
        {"api_workers": 2},
        {"max_active": 0},
        {"max_waiting": -1},
        {"deadline_ms": 30001},
        {"max_deadline_ms": 1000},
        {"drain_seconds": 31},
        {"body_limit_bytes": 1048577},
        {"max_questions": 17},
        {"port": True},
        {"typo": 1},
    ],
)
def test_invalid_config(override):
    with pytest.raises(ValidationError):
        ServerConfig(**override)


def test_zero_queue_is_valid():
    assert ServerConfig(max_waiting=0).max_waiting == 0
