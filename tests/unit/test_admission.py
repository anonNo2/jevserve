import asyncio
import time

import pytest

from jevserve.server.admission import Admission, Draining, Overloaded


@pytest.mark.asyncio
async def test_queue_bound_and_release():
    admission = Admission(1, 1)
    async with admission.lease(time.monotonic() + 1):

        async def queued():
            async with admission.lease(time.monotonic() + 1):
                pass

        task = asyncio.create_task(queued())
        await asyncio.sleep(0.01)
        assert admission.waiting == 1
        with pytest.raises(Overloaded):
            async with admission.lease(time.monotonic() + 1):
                pass
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert admission.waiting == 0
    assert admission.active == 0


@pytest.mark.asyncio
async def test_wait_deadline_and_drain_wake_waiters():
    admission = Admission(1, 1)
    async with admission.lease(time.monotonic() + 1):
        with pytest.raises(TimeoutError):
            async with admission.lease(time.monotonic() + 0.01):
                pass
        assert admission.waiting == 0
        await admission.close()
        with pytest.raises(Draining):
            async with admission.lease(time.monotonic() + 1):
                pass
    assert admission.active == 0
