"""Base Connector interface (Milestone 2).

All connectors must implement ``fetch_new()`` so they are swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawNotice:
    """A raw notice file ready for ingestion."""

    path: Path
    original_filename: str


class Connector(ABC):
    """Abstract connector interface.

    Implement this to provide new sources of notices (e.g., folder, ERP).
    """

    @abstractmethod
    def fetch_new(self) -> list[RawNotice]:
        """Yield notices that have not been ingested yet.

        Returns:
            A list of :class:`RawNotice` objects to process.
        """
        ...
