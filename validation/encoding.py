"""Compare native adapter encoding against pinned author code and a real tokenizer.

No GPU/model weights are needed. This checks encoding only, never inference.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from jevserve.adapters.open_jev import OpenJevAdapter
from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.requests import DecisionRequest
from validation.reference import cases, verify_files


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "models/locks/open-jev-2b.json").read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        revision != lock["reference_revision"]
        or subprocess.check_output(
            ["git", "-C", str(args.upstream), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    ):
        raise ValueError("author source must be clean and pinned")
    tokenizer_files = [
        item
        for item in lock["base"]["files"]
        if item["path"]
        in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "merges.txt",
            "vocab.json",
        )
    ]
    verify_files(args.tokenizer, tokenizer_files)
    from transformers import AutoTokenizer

    sys.path.insert(0, str(args.upstream.resolve()))
    from jev.api import candidate_prompts, compile_request

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    adapter = OpenJevAdapter(
        tokenizer,
        temperature=lock["temperature"],
        capabilities=Capabilities(
            question_types=["choice", "noul", "score"],
            max_choice_candidates=255,
            max_score_levels=10,
            max_tokens_per_candidate=4096,
            max_candidate_sequences=256,
            max_total_input_tokens=65536,
        ),
    )
    results = []
    for case in cases():
        if case.get("scope") == "author_accepts_native_rejects":
            continue
        request = DecisionRequest.model_validate(case["request"])
        records = compile_request(case["request"]["state"], case["request"]["questions"])
        prompts = [prompt for record in records for prompt in candidate_prompts(record)]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        encoded = tokenizer(rendered, padding=True, truncation=False)
        expected = [
            tuple(token for token, mask in zip(row, masks, strict=True) if mask)
            for row, masks in zip(encoded["input_ids"], encoded["attention_mask"], strict=True)
        ]
        actual = adapter.encode(request)
        if actual.engine_inputs != tuple(expected):
            raise ValueError("encoding mismatch: " + case["id"])
        if [group.keys for group in actual.groups] != [
            tuple(record["answer_keys"]) for record in records
        ]:
            raise ValueError("readout order mismatch: " + case["id"])
        results.append(
            {
                **case,
                "prompts": prompts,
                "engine_inputs": expected,
                "readout_positions": actual.readout_positions,
                "question_ids": actual.question_ids,
                "answer_keys": [record["answer_keys"] for record in records],
            }
        )
    report = {
        "scope": "CPU encoding only; no model inference",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "tokenizers", "jinja2")
        },
        "model_revision": lock["revision"],
        "reference_revision": revision,
        "tokenizer_digest": lock["tokenizer_digest"],
        "template_digest": lock["template_digest"],
        "case_count": len(results),
        "failures": 0,
        "results_sha256": hashlib.sha256(
            json.dumps(results, ensure_ascii=False).encode()
        ).hexdigest(),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(len(results), "real-tokenizer author encoding cases passed")


if __name__ == "__main__":
    main()
