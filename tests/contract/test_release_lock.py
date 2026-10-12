import copy
import json
from pathlib import Path

import pytest

from validation.locks import validate_baseline, validate_release

ROOT = Path(__file__).resolve().parents[2]


def baseline():
    return tuple(
        json.loads((ROOT / path).read_text())
        for path in ("compat/engine-lock.json", "models/locks/open-jev-2b.json")
    )


def test_observed_baseline_has_immutable_sources():
    validate_baseline(*baseline())


@pytest.mark.parametrize("field", ["revision", "reference_revision"])
def test_mutable_model_revision_rejected(field):
    engine, model = baseline()
    model[field] = "main"
    with pytest.raises(ValueError):
        validate_baseline(engine, model)


def test_pending_baseline_cannot_be_released():
    with pytest.raises(ValueError, match="certif"):
        validate_release(*baseline(), ROOT)


def test_bad_image_digest_rejected():
    engine, model = baseline()
    engine["image_digest"] = "sha256:invented"
    with pytest.raises(ValueError):
        validate_baseline(engine, model)


def test_duplicate_and_escaping_files_rejected():
    engine, model = baseline()
    for path in ["../escape", "/absolute", "a/../b", "a\\b"]:
        bad = copy.deepcopy(model)
        bad["files"][0]["path"] = path
        with pytest.raises(ValueError):
            validate_baseline(engine, bad)
    model["files"].append(model["files"][0])
    with pytest.raises(ValueError):
        validate_baseline(engine, model)
