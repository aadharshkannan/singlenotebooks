from dataclasses import dataclass
from typing import Callable, List, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class CapiEmbedding:
    index: int
    vector: Sequence[float]


class CapiTransport(Protocol):
    def embed(self, model: str, texts: Sequence[str]) -> Sequence[CapiEmbedding]: ...


class CapiEmbedder:
    """CAPI-compatible embedder around an injected authenticated transport."""

    def __init__(self, transport: CapiTransport, model: str, dimensions: int):
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        self._transport = transport
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: List[str]) -> np.ndarray:
        results = list(self._transport.embed(self.model, texts))
        by_index = {}
        for result in results:
            if result.index in by_index:
                raise ValueError(f"duplicate CAPI embedding index: {result.index}")
            by_index[result.index] = result.vector
        expected = set(range(len(texts)))
        if set(by_index) != expected:
            raise ValueError("CAPI response indices do not match the input batch")
        vectors = np.asarray([by_index[index] for index in range(len(texts))], dtype=np.float32)
        if vectors.shape != (len(texts), self.dimensions):
            raise ValueError(
                f"CAPI returned shape {vectors.shape}; expected {(len(texts), self.dimensions)}"
            )
        return vectors


class CapiTokenizer:
    """Exact CAPI token counter supplied by the eventual CAPI client contract."""

    def __init__(self, counter: Callable[[str, str], int], model: str, version: str):
        self._counter = counter
        self.model = model
        self.name = model
        self.version = version

    def count(self, text: str) -> int:
        count = int(self._counter(self.model, text))
        if count < 0:
            raise ValueError("CAPI token counter returned a negative count")
        return count
