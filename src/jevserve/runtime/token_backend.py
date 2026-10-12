"""vLLM one-token candidate readout without top-k truncation.

Mask only non-candidates and request N processed logprobs for all N allowed tokens.
This avoids the pinned engine's 128-entry logprob_token_ids limit. Temperature is
1 here; the checkpoint temperature is applied once by the adapter. Conditional
log-softmax differs from full-vocabulary log-softmax only by a shared constant.
"""

import asyncio
import importlib.metadata
import math
import os
from pathlib import Path
from typing import Any

from jevserve.adapters.base import EncodedDecision, RawDecision
from jevserve.artifacts.manifest import PreparedModel
from jevserve.runtime.vllm_backend import EngineFailed, VllmRuntime


class VllmTokenRuntime(VllmRuntime):
    def __init__(
        self,
        cache_dir: Path,
        *,
        gpu_memory_utilization: float = 0.5,
        cpu_offload_gb: float = 0,
        engine: Any = None,
        sampling_params: Any = None,
    ):
        super().__init__(cache_dir, gpu_memory_utilization=gpu_memory_utilization, engine=engine)
        if not math.isfinite(cpu_offload_gb) or not 0 <= cpu_offload_gb <= 128:
            raise ValueError("CPU offload must be an explicit 0..128 GiB budget")
        self.cpu_offload_gb = cpu_offload_gb
        self._sampling_params = sampling_params

    async def start(self, model: PreparedModel) -> None:
        if self._engine is not None or self._closed:
            raise RuntimeError("runtime cannot be started twice")
        for name, version in {
            "vllm": "0.29.0",
            "torch": "2.13.0",
            "transformers": "5.10.4",
        }.items():
            if importlib.metadata.version(name).split("+")[0] != version:
                raise EngineFailed("runtime package versions differ from tested profile")
        root = self.cache_dir.resolve()
        os.environ.update(
            HF_HUB_OFFLINE="1",
            HF_HOME=str(root / "hf"),
            VLLM_NO_USAGE_STATS="1",
            VLLM_CACHE_ROOT=str(root / "vllm"),
            TRITON_CACHE_DIR=str(root / "triton"),
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
        )
        self._engine = await asyncio.to_thread(self._create_engine, model.path)
        from vllm import SamplingParams

        self._sampling_params = SamplingParams

    def _create_engine(self, path: Path) -> Any:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        return AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=str(path),
                runner="generate",
                dtype="bfloat16",
                max_model_len=4096,
                trust_remote_code=False,
                gpu_memory_utilization=self.gpu_memory_utilization,
                cpu_offload_gb=self.cpu_offload_gb,
                enable_prefix_caching=False,
                mamba_ssm_cache_dtype="float32",
                enforce_eager=True,
                max_num_seqs=8,
                max_num_batched_tokens=4096,
                max_logprobs=255,
                logprobs_mode="processed_logprobs",
                generation_config="vllm",
                limit_mm_per_prompt={"image": 0, "video": 0},
                seed=0,
                enable_log_requests=False,
                disable_log_stats=True,
            )
        )

    async def infer(
        self, encoded: EncodedDecision, request_id: str, deadline: float
    ) -> RawDecision:
        if not self.healthy():
            raise EngineFailed("engine is not healthy")
        if (
            request_id in self._requests
            or not encoded.engine_inputs
            or len(encoded.engine_inputs) != len(encoded.candidate_token_ids)
        ):
            raise ValueError("invalid token-readout inputs or duplicate request")
        ids = tuple(f"{request_id}:{i}" for i in range(len(encoded.engine_inputs)))
        self._requests[request_id] = ids
        complete = False

        async def score(index: int) -> tuple[float, ...]:
            candidates = encoded.candidate_token_ids[index]
            if not 1 <= len(candidates) <= 255 or len(set(candidates)) != len(candidates):
                raise ValueError("invalid candidate token IDs")
            params = self._sampling_params(
                max_tokens=1,
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                allowed_token_ids=list(candidates),
                logprobs=len(candidates),
                detokenize=False,
                seed=0,
            )
            result = None
            async for output in self._engine.generate(
                prompt={"prompt_token_ids": list(encoded.engine_inputs[index])},
                sampling_params=params,
                request_id=ids[index],
            ):
                result = output
            if result is None or not result.finished or len(result.outputs) != 1:
                raise EngineFailed("engine returned no final candidate result")
            rows = result.outputs[0].logprobs
            if rows is None or len(rows) != 1 or any(token not in rows[0] for token in candidates):
                raise EngineFailed("engine omitted a requested candidate")
            values = tuple(float(rows[0][token].logprob) for token in candidates)
            if not all(math.isfinite(v) for v in values):
                raise EngineFailed("engine returned nonfinite candidate logprobs")
            return values

        tasks = []
        try:
            async with asyncio.timeout_at(deadline):
                tasks = [asyncio.create_task(score(i)) for i in range(len(ids))]
                rows = await asyncio.gather(*tasks)
            complete = True
            return RawDecision(encoded, (), tuple(rows))
        finally:
            try:
                if not complete:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await self.abort(request_id)
            finally:
                self._requests.pop(request_id, None)
