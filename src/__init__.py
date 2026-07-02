"""NeuroLog — edge-native, zero-shot semantic video search."""

from .ingest import NeuroLogIngestor
from .search import NeuroLogSearch

__version__ = "1.0.0"
__all__ = ["NeuroLogIngestor", "NeuroLogSearch"]
