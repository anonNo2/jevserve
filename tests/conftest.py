import asyncio

from jevserve.adapters.base import RawDecision


class FakeRuntime:
    """CPU protocol fixture only; never part of the installed serving package."""

    def __init__(self):
        self.is_healthy = True
        self.delay = 0
        self.aborted_ids = []
        self.closed = False

    async def start(self, model):
        pass

    async def infer(self, encoded, request_id, deadline):
        await asyncio.sleep(self.delay)
        return RawDecision(encoded, tuple(0.0 for _ in encoded.engine_inputs))

    async def abort(self, request_id):
        self.aborted_ids.append(request_id)

    def healthy(self):
        return self.is_healthy and not self.closed

    async def close(self):
        self.closed = True


class TestTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def encode(self, text, **kwargs):
        return list(text.encode())
