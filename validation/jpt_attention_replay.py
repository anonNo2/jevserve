"""Fixed-input layer-3 projection replay; no engine, production patch, or certification."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path

from validation import jpt_gdn_probe as tensors
from validation import jpt_layer_probe as layers


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_pair(hf, runtime, hf_digest):
    for key in (
        "model_id",
        "revision",
        "source_identity",
        "upstream_revision",
        "temperature",
        "reference_sha256",
        "fixture",
        "selection",
    ):
        if hf.get(key) != runtime.get(key):
            raise ValueError("attention provenance mismatch: " + key)
    if (
        hf.get("mode") != "hf_attention"
        or runtime.get("mode") != "runtime_attention"
        or runtime.get("attention_reference_sha256") != hf_digest
        or runtime.get("valid_trial") is not True
        or not hf.get("frozen_logit_comparison", {}).get("exact_match")
    ):
        raise ValueError("requires valid linked actual attention reports")
    if hf["selection"] != {"case_index": 0, "row_index": 0, "case": 0, "layer_index": 3}:
        raise ValueError("only original first fixture layer 3 supported")
    for report in (hf, runtime):
        for record in report["capture"]["boundaries"].values():
            tensors.decode_tensor(record)
    for name in ("attention_input", "projected_q", "gate"):
        a, b = [r["capture"]["boundaries"][name] for r in (hf, runtime)]
        if (a["shape"], a["dtype"], a["sha256"]) != (b["shape"], b["dtype"], b["sha256"]):
            raise ValueError("requires exact observed " + name)


def restore(record, device):
    import torch

    tensors.decode_tensor(record)
    dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}[record["dtype"]]
    return (
        torch.frombuffer(bytearray(base64.b64decode(record["data_base64"])), dtype=dtype)
        .reshape(record["shape"])
        .to(device)
    )


def run(args):
    import torch
    from safetensors import safe_open

    hf_raw, rt_raw = args.hf.read_bytes(), args.runtime.read_bytes()
    hf, rt = json.loads(hf_raw), json.loads(rt_raw)
    verify_pair(hf, rt, digest(hf_raw))
    selected = layers.profile(args.model)
    model = layers.load_model_directory(args.model_dir, layers.source_manifest(args.model))
    if (
        hf["model_id"] != args.model
        or hf["revision"] != selected["revision"]
        or hf["source_identity"] != model.manifest.identity
    ):
        raise ValueError("verified checkpoint provenance differs")
    if importlib.metadata.version("torch").split("+")[0] != "2.13.0":
        raise ValueError("requires pinned engine Torch 2.13.0")
    names = ("q_proj", "k_proj", "v_proj")
    keys = {f"model.language_model.layers.3.self_attn.{name}.weight": name for name in names}
    weights, proofs = {}, {}
    for path in sorted(args.model_dir.rglob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as source:
            for key in keys.keys() & set(source.keys()):
                name = keys[key]
                if name in weights:
                    raise ValueError("duplicate checkpoint key")
                original = source.get_tensor(key)
                weight = original.to(torch.bfloat16).contiguous()
                proofs[name] = {
                    "key": key,
                    "shape": list(weight.shape),
                    "source_dtype": str(original.dtype),
                    "effective_dtype": str(weight.dtype),
                    "effective_sha256": digest(weight.view(torch.uint8).numpy().tobytes()),
                }
                weights[name] = weight.to(args.device)
    if set(weights) != set(names):
        raise ValueError("missing original projection weights")
    bounds = hf["capture"]["boundaries"]
    x = restore(bounds["attention_input"], args.device)
    qwidth, kwidth, vwidth = [weights[n].shape[0] for n in names]
    if (
        qwidth != 2 * bounds["projected_q"]["shape"][1]
        or kwidth != bounds["projected_k"]["shape"][1]
        or vwidth != bounds["projected_v"]["shape"][1]
    ):
        raise ValueError("checkpoint/capture projection shape mismatch")
    config = json.loads((args.model_dir / "config.json").read_text())
    config = config.get("text_config", config)
    dim = config["head_dim"]

    def unpack(outputs):
        q, gate = outputs[0].reshape(x.shape[0], -1, 2 * dim).chunk(2, -1)
        return dict(
            zip(
                ("projected_q", "gate", "projected_k", "projected_v"),
                (q, gate, outputs[1], outputs[2]),
                strict=True,
            )
        )

    with torch.no_grad():
        separate = unpack(
            [torch.nn.functional.linear(x.unsqueeze(0), weights[n]).squeeze(0) for n in names]
        )
        packed = unpack(
            torch.nn.functional.linear(x, torch.cat([weights[n] for n in names], 0)).split(
                [qwidth, kwidth, vwidth], -1
            )
        )
    results = {}
    for label, values in [("separate", separate), ("packed", packed)]:
        encoded = {k: tensors.encode_tensor(v, x.shape[0]) for k, v in values.items()}
        results[label] = {
            target: tensors.compare_captures(
                encoded, {k: report["capture"]["boundaries"][k] for k in encoded}
            )
            for target, report in [("hf", hf), ("runtime", rt)]
        }
    # Alignment of observed operands is CPU arithmetic, independent of projection GPU replay.
    haux, raux = hf["capture"]["rotary_operands"], rt["capture"]["rotary_operands"]
    cos, sin, cache = [
        restore(r, "cpu") for r in (haux["rope_cos"], haux["rope_sin"], raux["rope_cache_rows"])
    ]
    c, s = cache.chunk(2, -1)
    rotary = {
        "hf_cos_dtype": str(cos.dtype),
        "runtime_cache_dtype": str(cache.dtype),
        "cos_exact": torch.equal(cos, torch.cat([c, c], -1)),
        "sin_exact": torch.equal(sin, torch.cat([s, s], -1)),
    }
    reproduction = all(
        results[label][target]["first_unequal_boundary"] is None
        for label, target in [("separate", "hf"), ("packed", "runtime")]
    )
    return {
        "schema_version": 1,
        "mode": "attention_projection_replay",
        "certified": False,
        "production_changed": False,
        "device": args.device,
        "model_id": args.model,
        "source_identity": model.manifest.identity,
        "hf_sha256": digest(hf_raw),
        "runtime_sha256": digest(rt_raw),
        "probe_sha256": digest(Path(__file__).read_bytes()),
        "torch": torch.__version__,
        "weights": proofs,
        "actual_runtime_weights_captured": False,
        "scope": "Reconstructed checkpoint weights and fixed observed input; replay is not an observed engine invocation.",
        "comparison": results,
        "observed_rotary_alignment": rotary,
        "both_observed_projection_paths_reproduced_exactly": reproduction,
        "interpretation": "Supports packed versus separate projection arithmetic for this fixture; actual runtime weight equality remains unverified."
        if reproduction
        else "No attribution: both observed paths were not reproduced exactly.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    for flag in ("model-dir", "hf", "runtime", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be fresh")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
