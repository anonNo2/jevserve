"""One serialized daemon encoder; cancelled callers never free running work.

Python cannot preempt tokenizer code. A cancelled encode retains the worker slot
and disables readiness until it really completes. The daemon worker allows process
shutdown even if native tokenizer code never returns; no replacement worker is
started. Waiting callers are bounded by the server's existing admission limits.
"""

import asyncio
import queue
import threading
from concurrent.futures import Future

from jevserve.adapters.base import EncodedDecision, ModelAdapter
from jevserve.contracts.requests import DecisionRequest
from jevserve.server.admission import Draining


class Encoder:
    def __init__(self, adapter: ModelAdapter):
        self._adapter = adapter
        self._lock = asyncio.Lock()
        self._jobs: queue.Queue[tuple[DecisionRequest, Future[EncodedDecision]]] = queue.Queue(1)
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: asyncio.Future[EncodedDecision] | None = None
        self._orphaned = False

    @property
    def active(self) -> int:
        return int(self._pending is not None and not self._pending.done())

    def healthy(self) -> bool:
        return not self._closed.is_set() and not (self._orphaned and self.active)

    def _run(self) -> None:
        while True:
            try:
                parsed, result = self._jobs.get(timeout=0.1)
            except queue.Empty:
                if self._closed.is_set():
                    return
                continue
            try:
                result.set_result(self._adapter.encode(parsed))
            except BaseException as exc:
                result.set_exception(exc)
            finally:
                del parsed, result
            if self._closed.is_set():
                return

    async def encode(self, parsed: DecisionRequest) -> EncodedDecision:
        await self._lock.acquire()
        if self._closed.is_set():
            self._lock.release()
            raise Draining
        result: Future[EncodedDecision] = Future()
        pending = asyncio.wrap_future(result)
        self._pending = pending
        self._orphaned = False

        def completed(future: asyncio.Future[EncodedDecision]) -> None:
            # Consume late exceptions without logging their potentially private text.
            if not future.cancelled():
                future.exception()
            self._pending = None
            self._lock.release()

        pending.add_done_callback(completed)
        try:
            if self._thread is None:
                thread = threading.Thread(target=self._run, name="jevserve-encoder", daemon=True)
                thread.start()
                self._thread = thread
            self._jobs.put_nowait((parsed, result))
        except Exception as exc:
            result.set_exception(exc)
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            self._orphaned = True
            raise

    async def close(self, deadline: float) -> None:
        self._closed.set()
        if self._pending is not None:
            try:
                async with asyncio.timeout_at(deadline):
                    await asyncio.shield(self._pending)
            except Exception:
                # Shutdown neither waits forever nor exposes a late encoder error.
                pass
