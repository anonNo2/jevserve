"""Explicit NeoHorse reference backend in one isolated GPU process.

Cancellation of active Torch work terminates the worker and makes the service
unhealthy. It never abandons a GPU thread or restarts/retries automatically.
"""

import asyncio
import math
import multiprocessing
import time
from pathlib import Path
from typing import Any

from jevserve.adapters.base import EncodedDecision, RawDecision
from jevserve.artifacts.manifest import PreparedModel
from jevserve.runtime.vllm_backend import EngineFailed


def _worker(connection: Any, model_path: str, memory_fraction: float, model_id: str) -> None:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import importlib.metadata
        import json
        import math

        for name, version in {
            "torch": "2.13.0",
            "transformers": "5.10.4",
            "peft": "0.20.0",
            "safetensors": "0.8.0",
        }.items():
            if importlib.metadata.version(name).split("+")[0] != version:
                raise RuntimeError(
                    "runtime package versions differ from the pinned pointer profile"
                )

        import torch
        from safetensors.torch import load_file
        from transformers import AutoModel

        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        root = Path(model_path)
        from jevserve.artifacts.catalog import profile

        selected = profile(model_id)
        if selected["adapter"] == "neohorse":
            meta = json.loads((root / "model_manifest.json").read_text())
            if (
                meta["backbone_class"] != "Qwen3_5Model"
                or meta["head_dim"] != 256
                or meta["option_isolation"]
                or not meta["hybrid"]
                or meta["temperature"] != 1.0
            ):
                raise ValueError("unsupported pointer checkpoint profile")
            model, info = AutoModel.from_pretrained(
                root / "backbone",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
                trust_remote_code=False,
                output_loading_info=True,
            )
            if any(
                info.get(k)
                for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            ):
                raise ValueError("backbone weights do not match configuration")
            model = model.to(device="cuda", dtype=torch.bfloat16).eval()
            lm = model.language_model
            weights = load_file(str(root / "pointer_head.safetensors"))
        elif selected["adapter"] == "kev":
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            meta = torch.load(root / "checkpoint/head.pt", map_location="cpu", weights_only=True)
            if (
                meta["base"] != selected["base"]["model_id"]
                or meta["base_revision"] != selected["base"]["revision"]
                or meta["head_dim"] != 256
                or meta["option_isolation"]
                or meta["special_embeddings"]
                or meta["weights_dtype"] != "fp32"
                or meta["lora_placement"] != "full"
                or meta["temperature"] != selected["temperature"]
            ):
                raise ValueError("unsupported Kev checkpoint metadata")
            # Explicit FP32 reference profile: no BF16 rounding or quantization.
            backbone = AutoModelForCausalLM.from_pretrained(
                root / "base",
                dtype=torch.float32,
                attn_implementation="sdpa",
                local_files_only=True,
                trust_remote_code=False,
            ).model
            lm = (
                PeftModel.from_pretrained(
                    backbone, root / "checkpoint", local_files_only=True, is_trainable=False
                )
                .merge_and_unload()
                .to("cuda")
                .eval()
            )
            weights = meta["head"]
        else:
            raise ValueError("unsupported pointer backend")
        q = torch.nn.Linear(lm.config.hidden_size, 256, dtype=torch.float32, device="cuda")
        k = torch.nn.Linear(lm.config.hidden_size, 256, dtype=torch.float32, device="cuda")
        q.load_state_dict(
            {name[2:]: value for name, value in weights.items() if name.startswith("q.")}
        )
        k.load_state_dict(
            {name[2:]: value for name, value in weights.items() if name.startswith("k.")}
        )
        q.eval()
        k.eval()
        connection.send(("ready", None))
        while True:
            encoded = connection.recv()
            if encoded is None:
                return
            length = max(map(len, encoded.engine_inputs))
            shape = (len(encoded.engine_inputs), length)
            ids = torch.full(shape, lm.config.pad_token_id or 0, dtype=torch.long, device="cuda")
            positions = torch.zeros(shape, dtype=torch.long, device="cuda")
            attention = torch.zeros(shape, dtype=torch.long, device="cuda")
            for i, row in enumerate(encoded.engine_inputs):
                ids[i, : len(row)] = torch.tensor(row, device="cuda")
                positions[i, : len(row)] = torch.arange(len(row), device="cuda")
                attention[i, : len(row)] = 1
            with torch.inference_mode():
                hidden = lm(
                    input_ids=ids, position_ids=positions, attention_mask=attention, use_cache=False
                ).last_hidden_state.float()
                result = tuple(
                    tuple(
                        float(v)
                        for v in (
                            k(hidden[i, list(opts)])
                            @ q(hidden[i, encoded.readout_positions[i]])
                            / math.sqrt(256)
                        ).cpu()
                    )
                    for i, opts in enumerate(encoded.candidate_positions)
                )
            connection.send(("result", result))
    except BaseException as error:
        try:
            connection.send(("error", type(error).__name__))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class TorchPointerRuntime:
    def __init__(self, *, gpu_memory_utilization: float = 0.5) -> None:
        if not math.isfinite(gpu_memory_utilization) or not 0.05 <= gpu_memory_utilization <= 0.95:
            raise ValueError("GPU memory utilization must be between .05 and .95")
        self._memory_fraction = gpu_memory_utilization
        self._process: Any = None
        self._connection: Any = None
        self._ready = False
        self._closed = False
        self._lock = asyncio.Lock()
        self._current: str | None = None

    def _receive(self, deadline: float) -> Any:
        while time.monotonic() < deadline:
            if self._connection.poll(0.1):
                return self._connection.recv()
            if self._closed or not self._process.is_alive():
                raise EngineFailed("Torch worker stopped")
        raise TimeoutError("Torch worker deadline exceeded")

    async def start(self, model: PreparedModel) -> None:
        if self._process is not None or self._closed:
            raise RuntimeError("runtime cannot be started twice")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_worker,
            args=(child, str(model.path), self._memory_fraction, model.manifest.model_id),
            daemon=True,
        )
        self._process.start()
        child.close()
        try:
            status, _ = await asyncio.to_thread(self._receive, time.monotonic() + 1800)
            if status != "ready":
                raise EngineFailed("Torch checkpoint initialization failed")
            self._ready = True
        except BaseException:
            await self.close()
            raise

    def healthy(self) -> bool:
        return (
            self._ready
            and not self._closed
            and self._process is not None
            and self._process.is_alive()
        )

    async def infer(
        self, encoded: EncodedDecision, request_id: str, deadline: float
    ) -> RawDecision:
        if (
            not encoded.engine_inputs
            or len(encoded.candidate_positions) != len(encoded.engine_inputs)
            or len(encoded.readout_positions) != len(encoded.engine_inputs)
        ):
            raise ValueError("invalid pointer inputs")
        async with asyncio.timeout_at(deadline), self._lock:
            if not self.healthy():
                raise EngineFailed("Torch worker is not healthy")
            self._current = request_id
            try:
                await asyncio.to_thread(self._connection.send, encoded)
                status, rows = await asyncio.to_thread(self._receive, deadline)
                if status != "result":
                    raise EngineFailed("Torch inference failed")
                return RawDecision(encoded, (), rows)
            except BaseException:
                await self.abort(request_id)
                raise
            finally:
                self._current = None

    async def abort(self, request_id: str) -> None:
        if self._current == request_id:
            self._ready = False
            if self._process is not None and self._process.is_alive():
                self._process.terminate()
                await asyncio.to_thread(self._process.join, 5)
                if self._process.is_alive():
                    self._process.kill()
                    await asyncio.to_thread(self._process.join, 5)

    async def close(self) -> None:
        self._ready = False
        self._closed = True
        if self._process is not None and self._process.is_alive():
            if self._current is None:
                try:
                    self._connection.send(None)
                except (OSError, EOFError):
                    pass
                await asyncio.to_thread(self._process.join, 5)
            if self._process.is_alive():
                self._process.terminate()
                await asyncio.to_thread(self._process.join, 5)
            if self._process.is_alive():
                self._process.kill()
                await asyncio.to_thread(self._process.join, 5)
        if self._connection is not None:
            self._connection.close()
