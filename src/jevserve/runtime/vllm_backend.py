"""Pinned vLLM 0.29 pooling bridge, with request-wide abort and no fallback engine."""

import asyncio
import importlib.metadata
import math
import os
from pathlib import Path
from typing import Any

from jevserve.adapters.base import EncodedDecision, RawDecision
from jevserve.artifacts.compile import compile_open_jev
from jevserve.artifacts.manifest import PreparedModel


class EngineFailed(RuntimeError):
    pass


class VllmRuntime:
    def __init__(
        self,
        cache_dir: Path = Path(".cache/jevserve"),
        *,
        gpu_memory_utilization: float = 0.5,
        engine: Any = None,
        pooling_params: Any = None,
        lora_request: Any = None,
    ):
        if not 0.05 <= gpu_memory_utilization <= 0.95:
            raise ValueError("GPU memory utilization must be between .05 and .95")
        self.cache_dir = cache_dir
        self.gpu_memory_utilization = gpu_memory_utilization
        self._engine = engine
        self._pooling = pooling_params
        self._lora = lora_request
        self._requests: dict[str, tuple[str, ...]] = {}
        self._closed = False
        self.compiled_path: Path | None = None

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(self._requests)

    async def start(self, model: PreparedModel) -> None:
        if self._engine is not None or self._closed:
            raise RuntimeError("runtime cannot be started twice")
        expected = {"vllm": "0.29.0", "torch": "2.13.0", "transformers": "5.10.4", "peft": "0.20.0"}
        for name, version in expected.items():
            if importlib.metadata.version(name).split("+")[0] != version:
                raise EngineFailed("runtime package versions differ from tested profile")
        self.compiled_path = await asyncio.to_thread(
            compile_open_jev, model, self.cache_dir / "compiled"
        )
        root = self.cache_dir.resolve()
        os.environ.update(
            HF_HUB_OFFLINE="1",
            HF_HOME=str(root / "hf"),
            VLLM_NO_USAGE_STATS="1",
            VLLM_CACHE_ROOT=str(root / "vllm"),
            TRITON_CACHE_DIR=str(root / "triton"),
            VLLM_PLUGINS="jevserve",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
        )
        self._engine = await asyncio.to_thread(self._create_engine, self.compiled_path)
        from vllm import PoolingParams

        self._pooling = PoolingParams(
            task="classify", use_activation=False, skip_reading_prefix_cache=True
        )
        self._lora = None

    def _create_engine(self, path: Path) -> Any:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        args = AsyncEngineArgs(
            model=str(path),
            runner="pooling",
            convert="none",
            dtype="bfloat16",
            max_model_len=4096,
            trust_remote_code=False,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=False,
            mamba_ssm_cache_dtype="float32",
            enforce_eager=True,
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            seed=0,
            enable_log_requests=False,
            disable_log_stats=True,
            pooler_config={"task": "classify", "seq_pooling_type": "LAST", "use_activation": False},
        )
        return AsyncLLM.from_engine_args(args)

    def healthy(self) -> bool:
        return self._engine is not None and not self._closed and not bool(self._engine.errored)

    async def infer(
        self, encoded: EncodedDecision, request_id: str, deadline: float
    ) -> RawDecision:
        if not self.healthy():
            raise EngineFailed("engine is not healthy")
        if request_id in self._requests or not encoded.engine_inputs:
            raise ValueError("duplicate request or empty engine inputs")
        ids = tuple(f"{request_id}:{i}" for i in range(len(encoded.engine_inputs)))
        self._requests[request_id] = ids
        complete = False

        async def score(index: int) -> float:
            result = None
            async for output in self._engine.encode(
                prompt={"prompt_token_ids": list(encoded.engine_inputs[index])},
                pooling_params=self._pooling,
                request_id=ids[index],
                lora_request=self._lora,
            ):
                result = output
            if result is None or not result.finished:
                raise EngineFailed("engine returned no final pooling result")
            values = result.outputs.data
            if len(values) != 1 or not math.isfinite(float(values[0])):
                raise EngineFailed("engine returned invalid scalar output")
            return float(values[0])

        tasks = []
        try:
            async with asyncio.timeout_at(deadline):
                tasks = [asyncio.create_task(score(i)) for i in range(len(ids))]
                scores = await asyncio.gather(*tasks)
            complete = True
            return RawDecision(encoded, tuple(scores))
        finally:
            try:
                if not complete:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await self.abort(request_id)
            finally:
                self._requests.pop(request_id, None)

    async def abort(self, request_id: str) -> None:
        ids = self._requests.get(request_id, ())
        if ids and self._engine is not None:
            try:
                async with asyncio.timeout(5):
                    await self._engine.abort(list(ids))
            except Exception:
                # A dead engine cannot accept abort; it must remain unhealthy.
                self._closed = True

    async def close(self) -> None:
        for request_id in tuple(self._requests):
            await self.abort(request_id)
        self._closed = True
        if self._engine is not None:
            engine, self._engine = self._engine, None
            await asyncio.to_thread(engine.shutdown, timeout=30)
