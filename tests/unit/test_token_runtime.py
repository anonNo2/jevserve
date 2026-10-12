import asyncio
import time
from types import SimpleNamespace

import pytest

from jevserve.adapters.base import EncodedDecision
from jevserve.runtime.token_backend import VllmTokenRuntime
from jevserve.runtime.vllm_backend import EngineFailed


class TokenEngine:
    errored = False

    def __init__(self, delay=0, missing=False):
        self.aborted = []
        self.delay, self.missing = delay, missing

    async def generate(self, prompt, sampling_params, request_id):
        await asyncio.sleep(self.delay)
        assert sampling_params.max_tokens == 1
        assert sampling_params.top_k == -1 and sampling_params.top_p == 1
        ids = sampling_params.allowed_token_ids
        assert sampling_params.logprobs == len(ids)
        values = {i: SimpleNamespace(logprob=-float(i)) for i in ids}
        if self.missing:
            values.pop(ids[-1])
        yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(logprobs=[values])])

    async def abort(self, ids):
        self.aborted.extend(ids)

    def shutdown(self, **kwargs):
        pass


def inputs():
    return EncodedDecision(("q", "r"), ((1, 2), (3,)), (), 3, 4, (1, 0), ((99, 2), (4, 8)))


@pytest.mark.asyncio
async def test_all_requested_candidates_preserve_order(tmp_path):
    runtime = VllmTokenRuntime(tmp_path, engine=TokenEngine(), sampling_params=SimpleNamespace)
    result = await runtime.infer(inputs(), "a", time.monotonic() + 1)
    assert result.candidate_logits == ((-99.0, -2.0), (-4.0, -8.0))
    assert not runtime.active_request_ids
    await runtime.close()


@pytest.mark.asyncio
async def test_missing_low_probability_candidate_fails_closed(tmp_path):
    runtime = VllmTokenRuntime(
        tmp_path, engine=TokenEngine(missing=True), sampling_params=SimpleNamespace
    )
    with pytest.raises(EngineFailed, match="omitted"):
        await runtime.infer(inputs(), "missing", time.monotonic() + 1)
    assert set(runtime._engine.aborted) == {"missing:0", "missing:1"}


@pytest.mark.asyncio
async def test_deadline_and_cancellation_abort_all_questions(tmp_path):
    runtime = VllmTokenRuntime(
        tmp_path, engine=TokenEngine(delay=1), sampling_params=SimpleNamespace
    )
    with pytest.raises(TimeoutError):
        await runtime.infer(inputs(), "deadline", time.monotonic() + 0.01)
    task = asyncio.create_task(runtime.infer(inputs(), "cancel", time.monotonic() + 2))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert set(runtime._engine.aborted) == {"deadline:0", "deadline:1", "cancel:0", "cancel:1"}
    assert not runtime.active_request_ids


@pytest.mark.asyncio
async def test_all_255_candidates_fit_the_explicit_mask_path(tmp_path):
    runtime = VllmTokenRuntime(tmp_path, engine=TokenEngine(), sampling_params=SimpleNamespace)
    encoded = EncodedDecision(("q",), ((1, 2),), (), 2, 255, (1,), (tuple(range(255)),))
    result = await runtime.infer(encoded, "wide", time.monotonic() + 1)
    assert result.candidate_logits == (tuple(-float(i) for i in range(255)),)
    await runtime.close()
