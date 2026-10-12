import asyncio
import json
import multiprocessing
import time
from pathlib import Path

import pytest

from jevserve.adapters.base import EncodedDecision, RawDecision
from jevserve.adapters.neohorse import NeoHorseAdapter
from jevserve.contracts.capabilities import Capabilities
from jevserve.contracts.requests import DecisionRequest
from jevserve.runtime.torch_pointer import TorchPointerRuntime

FIXTURE = json.loads(Path("validation/reports/neohorse-encoding.json").read_text())


class RecordedTokenizer:
    unk_token_id = None

    def convert_tokens_to_ids(self, text):
        return FIXTURE["special_tokens"].get(text)

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return FIXTURE["user_tokens"][text]


def adapter(temperature=1.0):
    return NeoHorseAdapter(
        RecordedTokenizer(),
        Capabilities(
            question_types=["choice", "noul", "score"],
            max_choice_candidates=255,
            max_score_levels=255,
            max_tokens_per_candidate=4096,
            max_candidate_sequences=256,
            max_total_input_tokens=8192,
            max_padded_input_tokens=8192,
        ),
        temperature=temperature,
    )


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_pointer_encoding_and_candidate_positions(case):
    result = adapter().encode(DecisionRequest.model_validate(case["request"]))
    assert result.engine_inputs == tuple(tuple(r) for r in case["tokens"])
    assert result.candidate_positions == tuple(tuple(r) for r in case["candidate_positions"])
    assert result.readout_positions == tuple(len(r) - 1 for r in case["tokens"])


def test_noul_false_first_and_exact_stored_temperature():
    model = adapter(temperature=2.406050072164233)
    encoded = model.encode(DecisionRequest.model_validate(FIXTURE["cases"][2]["request"]))
    assert encoded.groups[0].keys == ("false", "true")
    value = model.decode(RawDecision(encoded, (), ((0.0, 2.406050072164233),)))
    assert next(iter(value.values())).noul == pytest.approx(0.7310585786300049)


def test_padded_gpu_work_is_bounded():
    caps = adapter().capabilities()
    with pytest.raises(ValueError, match="padded"):
        caps.validate_token_lengths([4096, 10, 10])


def _cpu_worker(connection):
    try:
        while True:
            message = connection.recv()
            if message is None:
                return
            time.sleep(0.25)
            connection.send(("result", ((1.0, 2.0),)))
    finally:
        connection.close()


def runtime():
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_cpu_worker, args=(child,))
    process.start()
    child.close()
    result = TorchPointerRuntime()
    result._connection = parent
    result._process = process
    result._ready = True
    return result


def encoded():
    return EncodedDecision(("q",), ((1, 2, 3),), (), 3, 2, (2,), candidate_positions=((0, 1),))


@pytest.mark.asyncio
async def test_active_cancellation_terminates_worker_without_restart():
    model = runtime()
    task = asyncio.create_task(model.infer(encoded(), "active", time.monotonic() + 5))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not model.healthy()
    assert not model._process.is_alive()
    assert model._process.exitcode is not None
    await model.close()


@pytest.mark.asyncio
async def test_queued_cancellation_does_not_kill_another_active_request():
    model = runtime()
    active = asyncio.create_task(model.infer(encoded(), "active", time.monotonic() + 5))
    await asyncio.sleep(0.05)
    with pytest.raises(TimeoutError):
        await model.infer(encoded(), "waiting", time.monotonic() + 0.01)
    result = await active
    assert result.candidate_logits == ((1.0, 2.0),)
    assert model.healthy()
    await model.close()
    assert not model._process.is_alive()
    assert model._process.exitcode == 0
