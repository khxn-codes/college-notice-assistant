"""Base Connector interface.

All connectors must implement ``fetch_new()`` so they are swappable.
The folder connector (current) and any future ERP connector both extend this.
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

    Implement this to add new notice sources (e.g., folder, ERP).
    The rest of the ingestion pipeline only depends on this interface.
    """

    @abstractmethod
    def fetch_new(self) -> list[RawNotice]:
        """Return files that have not been ingested yet.

        Returns:
            A list of :class:`RawNotice` objects to process.
        """
        ...
