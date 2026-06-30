from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict, List

from app.models.schemas import SessionEvent


class SessionStore:
    def __init__(self, max_events: int = 300) -> None:
        self.max_events = max_events
        self._events: Dict[str, Deque[SessionEvent]] = defaultdict(deque)

    def add(self, session_id: str, event_type: str, payload: dict) -> SessionEvent:
        event = SessionEvent(session_id=session_id, ts=time.time(), event_type=event_type, payload=payload)
        queue = self._events[session_id]
        queue.append(event)
        while len(queue) > self.max_events:
            queue.popleft()
        return event

    def list_events(self, session_id: str) -> List[SessionEvent]:
        return list(self._events[session_id])
