"""Event-loop admission with bounded waiting and absolute deadlines."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class Overloaded(Exception):
    pass


class Draining(Exception):
    pass


class Admission:
    def __init__(self, max_active: int, max_waiting: int):
        if max_active < 1 or max_waiting < 0:
            raise ValueError("invalid admission capacity")
        self.max_active = max_active
        self.max_waiting = max_waiting
        self.active = 0
        self.waiting = 0
        self._closed = False
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def lease(self, deadline: float) -> AsyncIterator[None]:
        async with asyncio.timeout_at(deadline):
            async with self._condition:
                if self._closed:
                    raise Draining
                if self.active >= self.max_active:
                    if self.waiting >= self.max_waiting:
                        raise Overloaded
                    self.waiting += 1
                    try:
                        await self._condition.wait_for(
                            lambda: self._closed or self.active < self.max_active
                        )
                    finally:
                        self.waiting -= 1
                    if self._closed:
                        raise Draining
                self.active += 1
        try:
            yield
        finally:
            async with self._condition:
                self.active -= 1
                self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

    async def wait_empty(self, deadline: float) -> None:
        async with asyncio.timeout_at(deadline):
            async with self._condition:
                await self._condition.wait_for(lambda: self.active == 0)
