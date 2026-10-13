"""One-fixture normalized layer capture under explicit diagnostic arithmetic modes.

Use the jpt_layer_probe runtime CLI plus --arithmetic-mode. The HF reference is
an original, unmodified hf_layers report. This is not certification evidence.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from validation import jpt_layer_probe as layers
from validation.jpt_arithmetic_probe import (
    ENGINE_PROFILE,
    MODES,
    ArithmeticProbeWorker,
    ArithmeticRuntime,
    trial_scope,
    validate_trial,
)


class ArithmeticLayerWorker(layers.LayerProbeWorker, ArithmeticProbeWorker):
    """Named RPCs combine arithmetic patches with independent layer hooks."""


class ArithmeticLayerRuntime(ArithmeticRuntime):
    def _create_engine(self, path):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=str(path),
                runner="generate",
                trust_remote_code=False,
                gpu_memory_utilization=self.gpu_memory_utilization,
                cpu_offload_gb=0,
                max_logprobs=255,
                logprobs_mode="processed_logits",
                generation_config="vllm",
                limit_mm_per_prompt={"image": 0, "video": 0},
                seed=0,
                enable_log_requests=False,
                disable_log_stats=True,
                worker_extension_cls="validation.jpt_arithmetic_layers.ArithmeticLayerWorker",
                **ENGINE_PROFILE,
            )
        )


def finalize_report(report, metadata, mode):
    if report.get("mode") != "runtime_layers":
        raise ValueError("expected original layer runtime report")
    if not validate_trial(metadata, 1, mode):
        raise ValueError("one-question arithmetic coverage/restoration failed")
    report.update(
        mode="runtime_arithmetic_layers",
        arithmetic_mode=mode,
        arithmetic=metadata,
        valid_trial=True,
        scope="one synthetic fixture normalized layer boundaries; " + trial_scope(mode),
        runtime_profile=dict(ENGINE_PROFILE),
        production_changed=False,
        certified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--arithmetic-mode", choices=MODES, required=True)
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print(__doc__)
        print("Arithmetic modes: " + ", ".join(MODES))
        original = sys.argv
        try:
            sys.argv = [original[0], "--help"]
            layers.main()
        finally:
            sys.argv = original
        return
    options, remaining = parser.parse_known_args()
    if not remaining or remaining[0] != "runtime":
        parser.error("only runtime mode supported; use original hf_layers reference")
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, without_output = output_parser.parse_known_args(remaining)
    if output.output.exists() or output.output.is_symlink():
        parser.error("output already exists; choose a new path")
    # Preserve original CLI's output/input separation even though it writes staging.
    inputs = argparse.ArgumentParser(add_help=False)
    inputs.add_argument("--reference", type=Path, required=True)
    inputs.add_argument("--layer-reference", type=Path, required=True)
    paths, _ = inputs.parse_known_args(remaining)
    if output.output.resolve() in {paths.reference.resolve(), paths.layer_reference.resolve()}:
        parser.error("output must not overwrite input references")
    instances = []

    def factory(*args, **kwargs):
        runtime = ArithmeticLayerRuntime(*args, arithmetic_mode=options.arithmetic_mode, **kwargs)
        instances.append(runtime)
        return runtime

    output.output.parent.mkdir(parents=True, exist_ok=True)
    original_runtime, original_argv = layers.LayerRuntime, sys.argv
    with tempfile.TemporaryDirectory(
        prefix=".arithmetic-layers-", dir=output.output.parent
    ) as stage:
        intermediate = Path(stage) / "layers.json"
        try:
            layers.LayerRuntime = factory
            sys.argv = [original_argv[0], *without_output, "--output", str(intermediate)]
            layers.main()
        finally:
            layers.LayerRuntime, sys.argv = original_runtime, original_argv
        report = json.loads(intermediate.read_text())
    if len(instances) != 1:
        raise ValueError("expected one arithmetic runtime")
    finalize_report(report, instances[0].arithmetic_metadata, options.arithmetic_mode)
    with output.output.open("x") as destination:
        json.dump(report, destination, indent=2, ensure_ascii=False, allow_nan=False)
        destination.write("\n")
    print("Saved arithmetic layer diagnostic", output.output)


if __name__ == "__main__":
    main()
