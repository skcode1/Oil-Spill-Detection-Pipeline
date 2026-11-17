"""Thread-safe incident store for the latest pipeline results.

Provides a bounded in-memory deque so the ``GET /incidents/latest`` endpoint
always returns the most recent completed pipeline run without growing
unbounded.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from app.schemas import PipelineResponse

logger = logging.getLogger(__name__)

#: Maximum number of pipeline results to retain in memory.
MAX_INCIDENTS = 50


@dataclass
class IncidentStore:
    """Thread-safe bounded store for pipeline results."""

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _incidents: Deque[PipelineResponse] = field(default_factory=deque, init=False)

    def add(self, response: PipelineResponse) -> None:
        """Record a completed pipeline response."""
        with self._lock:
            self._incidents.append(response)
            if len(self._incidents) > MAX_INCIDENTS:
                self._incidents.popleft()
            logger.debug(
                "Incident stored (total=%d, max=%d)", len(self._incidents), MAX_INCIDENTS
            )

    def latest(self) -> Optional[PipelineResponse]:
        """Return the most recent pipeline response, or ``None`` if empty."""
        with self._lock:
            return self._incidents[-1] if self._incidents else None

    def clear(self) -> None:
        """Remove all stored incidents."""
        with self._lock:
            self._incidents.clear()
            logger.debug("Incident store cleared")

    def __len__(self) -> int:
        with self._lock:
            return len(self._incidents)

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._incidents)


# Global instance for the application lifespan
_incident_store: Optional[IncidentStore] = None


def get_incident_store() -> IncidentStore:
    """Return the singleton incident store instance."""
    global _incident_store
    if _incident_store is None:
        _incident_store = IncidentStore()
    return _incident_store


def reset_incident_store() -> None:
    """Reset the singleton (used in tests)."""
    global _incident_store
    _incident_store = None
