"""Derive a vLLM text checkpoint without merging, rounding or changing tensor values."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from jevserve.artifacts.manifest import PreparedModel
from jevserve.artifacts.prepare import _lock
from jevserve.artifacts.verify import ArtifactCorrupt, verify_package
from jevserve.runtime.registration import ARCHITECTURE

RECIPE = "qwen35-unmerged-v1"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_compiled(path: Path, source_identity: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ArtifactCorrupt("invalid compiled model path")
    marker = path / "compiled.json"
    if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 1048576:
        raise ArtifactCorrupt("compiled manifest missing or unsafe")
    inventory = json.loads(marker.read_text())
    if inventory.get("recipe") != RECIPE or inventory.get("source_identity") != source_identity:
        raise ArtifactCorrupt("compiled source identity mismatch")
    files = inventory.get("files", {})
    expected = set(files) | {"compiled.json"}
    actual = set()
    for file in path.rglob("*"):
        if file.is_symlink():
            raise ArtifactCorrupt("compiled symlink rejected")
        if file.is_file():
            actual.add(str(file.relative_to(path)))
    if actual != expected:
        raise ArtifactCorrupt("compiled file set mismatch")
    for name, checksum in files.items():
        if digest(path / name) != checksum:
            raise ArtifactCorrupt("compiled file checksum mismatch")


def compile_open_jev(model: PreparedModel, cache: Path) -> Path:
    verify_package(model.path, model.manifest)
    identity = model.manifest.identity
    cache.mkdir(parents=True, exist_ok=True)
    final = cache / (identity + "-" + RECIPE)
    with _lock(cache / (identity + "-compile.lock"), 1800):
        if final.exists() or final.is_symlink():
            verify_compiled(final, identity)
            return final
        stage = Path(tempfile.mkdtemp(prefix="compile-", dir=cache))
        try:
            _convert(model.path, stage)
            inventory = {
                "recipe": RECIPE,
                "source_identity": identity,
                "files": {
                    str(p.relative_to(stage)): digest(p)
                    for p in sorted(stage.rglob("*"))
                    if p.is_file()
                },
            }
            (stage / "compiled.json").write_text(json.dumps(inventory, indent=2) + "\n")
            verify_compiled(stage, identity)
            os.rename(stage, final)
            return final
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def _convert(source: Path, target: Path) -> None:
    # Dependencies are supplied only by the pinned runtime image/environment.
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    base = source / "base"
    checkpoint = source / "checkpoint/package/checkpoint"
    raw_config = json.loads((base / "config.json").read_text())
    config = raw_config["text_config"]
    config.update(
        architectures=[ARCHITECTURE],
        num_labels=1,
        head_dtype="float32",
        id2label={"0": "jev_score"},
        label2id={"jev_score": 0},
    )
    if config["model_type"] != "qwen3_5_text":
        raise ValueError("unsupported text backbone")
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "merges.txt",
        "vocab.json",
    ):
        shutil.copyfile(base / filename, target / filename)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    prefix = "model.language_model."
    weight_map = {}
    total = 0
    for i, filename in enumerate(sorted(set(index["weight_map"].values()))):
        with safe_open(base / filename, framework="pt", device="cpu") as f:
            tensors = {
                "model." + name.removeprefix(prefix): f.get_tensor(name)
                for name in f.keys()
                if name.startswith(prefix)
            }
        if tensors:
            shard = f"backbone-{i:04d}.safetensors"
            save_file(tensors, target / shard)
            weight_map.update({name: shard for name in tensors})
            total += sum(t.numel() * t.element_size() for t in tensors.values())
        del tensors
    head = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    if head["weight"].dtype != torch.float32 or head["weight"].shape != (1, config["hidden_size"]):
        raise ValueError("unexpected decision head")
    tensors = {"score.weight": head["weight"].contiguous(), "score.bias": head["bias"].contiguous()}
    save_file(tensors, target / "score.safetensors")
    weight_map.update({name: "score.safetensors" for name in tensors})
    total += sum(t.numel() * t.element_size() for t in tensors.values())
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n"
    )
    # PEFT was trained on the inner text model. vLLM owns that under `model`.
    lora = target / "lora"
    lora.mkdir()
    values = load_file(checkpoint / "adapter/adapter_model.safetensors")
    renamed = {}
    for name, tensor in values.items():
        if not name.startswith("base_model.model.layers.") or tensor.dtype != torch.float32:
            raise ValueError("unexpected LoRA tensor namespace or precision")
        renamed[name.replace("base_model.model.layers.", "base_model.model.model.layers.", 1)] = (
            tensor
        )
    save_file(renamed, lora / "adapter_model.safetensors")
    shutil.copyfile(checkpoint / "adapter/adapter_config.json", lora / "adapter_config.json")
