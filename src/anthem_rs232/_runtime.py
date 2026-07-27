"""Shared runtime plumbing for both receiver generations.

serialkit owns the wire and holds no device data, so the pieces that are about
*this device* — the state object, the subscriber list, and when a batch of
changes is worth telling anyone about — live here. Both generations need the
same ones, so they are written once.

Anthem receivers are publishers: they report state on their own schedule, and
a frame arrives whether or not anything was sent. State is therefore applied in
``on_frame`` and flushed in ``on_turn``, which the link calls once after every
frame in a read chunk has been dispatched. That is the coalescing point — a
burst of reports becomes one subscriber callback rather than one per frame.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from serialkit import Backoff, Framer, IdleProbe, Pacing, SerialLink

_LOGGER = logging.getLogger(__name__)


class _CopyableState(Protocol):
    """A receiver state object: subscribers get a snapshot, not the live one."""

    def copy(self) -> Any: ...


class ReceiverRuntime[S: _CopyableState]:
    """Owns a :class:`serialkit.SerialLink` and the device model.

    Subclasses supply the wire configuration and implement ``on_frame`` and
    ``on_connect``; everything about subscribers and lifecycle is here.
    """

    def __init__(
        self,
        *,
        state: S,
        connect: Callable[[], Awaitable[tuple[Any, Any]]],
        framer: Framer,
        pacing: Pacing | None = None,
        liveness: IdleProbe | None = None,
        backoff: Backoff | None = None,
    ) -> None:
        self.state: S = state
        self.link = SerialLink(
            connect=connect,
            framer=framer,
            handler=self,
            pacing=pacing,
            liveness=liveness,
            **({"backoff": backoff} if backoff is not None else {}),  # type: ignore[arg-type]
        )
        self._subscribers: list[Callable[[Any], None]] = []
        self._dirty = False
        self._flush_scheduled = False
        self._batch_depth = 0

    # -- Connection lifecycle ------------------------------------------------

    @property
    def connected(self) -> bool:
        return self.link.connected

    @property
    def frame_errors(self) -> deque[tuple[bytes, Exception]]:
        """Recent per-frame dispatch failures, newest last (bounded)."""
        return self.link.frame_errors

    async def start(self) -> None:
        await self.link.start()

    async def stop(self) -> None:
        await self.link.stop()

    # -- Subscribers ---------------------------------------------------------

    def subscribe(self, callback: Callable[[Any], None]) -> Callable[[], None]:
        """Register ``callback`` for state snapshots; returns an unsubscribe fn.

        The callback receives a copy of the state when something changes, and
        ``None`` when the connection drops.
        """
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def notify(self) -> None:
        """Mark the state changed.

        Inside a dispatch turn this is flushed by :meth:`on_turn`; from a
        caller task it is flushed on the next loop iteration. Either way a run
        of changes produces one callback.
        """
        self._dirty = True
        if not self._flush_scheduled:
            self._flush_scheduled = True
            asyncio.get_running_loop().call_soon(self._flush)

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Hold notifications for the block, then deliver one."""
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self._flush()

    def _flush(self) -> None:
        self._flush_scheduled = False
        if self._batch_depth or not self._dirty:
            return
        self._dirty = False
        if not self.link.connected:
            return  # disconnected since the change; None was already delivered
        self._deliver(self.state.copy())

    def _deliver(self, snapshot: Any) -> None:
        for callback in list(self._subscribers):
            try:
                callback(snapshot)
            except Exception:
                _LOGGER.exception("State subscriber raised")

    # -- serialkit DeviceHandler callbacks ----------------------------------

    def on_frame(self, frame: bytes) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    async def on_connect(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def on_turn(self) -> None:
        """One read chunk has been fully dispatched: deliver at most once."""
        self._flush()

    def on_disconnect(self, exc: Exception | None) -> None:
        self._dirty = False  # a pending snapshot must not outrun None
        self._deliver(None)
