import asyncio
import time
from types import SimpleNamespace

import pytest

from jevserve.adapters.base import EncodedDecision
from jevserve.runtime.vllm_backend import EngineFailed, VllmRuntime


class Engine:
    errored = False
    is_running = True

    def __init__(self):
        self.ids = []
        self.aborted = []
        self.closed = False
        self.delay = 0

    async def encode(self, prompt, pooling_params, request_id, **kwargs):
        self.ids.append(request_id)
        await asyncio.sleep(self.delay)
        yield SimpleNamespace(
            finished=True, outputs=SimpleNamespace(data=[float(prompt["prompt_token_ids"][0])])
        )

    async def abort(self, ids):
        self.aborted.extend(ids if isinstance(ids, list) else [ids])

    def shutdown(self, timeout=None):
        self.closed = True
        self.is_running = False


def encoded():
    return EncodedDecision(("q",), ((1, 2), (3,)), (), 3, 2, (1, 0))


@pytest.mark.asyncio
async def test_real_bridge_port_retains_sequence_order():
    engine = Engine()
    runtime = VllmRuntime(engine=engine)
    result = await runtime.infer(encoded(), "one", time.monotonic() + 1)
    assert result.scores == (1.0, 3.0)
    assert len(set(engine.ids)) == 2
    await runtime.close()
    assert engine.closed


@pytest.mark.asyncio
async def test_deadline_aborts_every_subrequest_and_cleans_mapping():
    engine = Engine()
    engine.delay = 1
    runtime = VllmRuntime(engine=engine)
    with pytest.raises(TimeoutError):
        await runtime.infer(encoded(), "deadline", time.monotonic() + 0.01)
    assert set(engine.ids) <= set(engine.aborted)
    assert not runtime.active_request_ids


@pytest.mark.asyncio
async def test_cancellation_and_engine_failure():
    engine = Engine()
    engine.delay = 1
    runtime = VllmRuntime(engine=engine)
    task = asyncio.create_task(runtime.infer(encoded(), "cancel", time.monotonic() + 2))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert set(engine.ids) <= set(engine.aborted)
    engine.errored = True
    assert not runtime.healthy()
    with pytest.raises(EngineFailed):
        await runtime.infer(encoded(), "failed", time.monotonic() + 1)
