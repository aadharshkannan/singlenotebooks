from __future__ import annotations

from dataclasses import dataclass

from trace_sampling.representation import (
    CANONICAL_POLICY,
    CANONICAL_VERSION,
    DEFAULT_MAX_UTF8_BYTES,
)


@dataclass(frozen=True)
class MinHashConfig:
    ngram_size: int = 3
    permutations: int = 128
    seed: int = 13
    lsh_bands: int | None = None
    lsh_rows: int | None = None
    similarity_threshold: float = 0.50
    max_shingles: int = 4096
    cache_size: int = 4096
    ttl_s: float = 60.0
    purge_every: int = 200
    max_clusters_per_agent: int = 256
    max_clusters_total: int = 4096
    staleness_k: float = 16.0
    iat_alpha: float = 0.3
    representation_policy: str = CANONICAL_POLICY
    representation_version: str = CANONICAL_VERSION
    representation_max_utf8_bytes: int = DEFAULT_MAX_UTF8_BYTES
    retain_debug_shingles: bool = False

    def __post_init__(self) -> None:
        if self.ngram_size <= 0:
            raise ValueError("ngram_size must be > 0")
        if self.permutations <= 0:
            raise ValueError("permutations must be > 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.lsh_bands is None and self.lsh_rows is None:
            preferred_bands = min(32, self.permutations)
            while self.permutations % preferred_bands != 0:
                preferred_bands -= 1
            object.__setattr__(self, "lsh_bands", preferred_bands)
            object.__setattr__(self, "lsh_rows", self.permutations // preferred_bands)
        elif self.lsh_bands is None:
            if self.lsh_rows is None or self.lsh_rows <= 0 or self.permutations % self.lsh_rows != 0:
                raise ValueError("lsh_rows must be a positive divisor of permutations")
            object.__setattr__(self, "lsh_bands", self.permutations // self.lsh_rows)
        elif self.lsh_rows is None:
            if self.lsh_bands <= 0 or self.permutations % self.lsh_bands != 0:
                raise ValueError("lsh_bands must be a positive divisor of permutations")
            object.__setattr__(self, "lsh_rows", self.permutations // self.lsh_bands)

        if self.lsh_bands is None or self.lsh_bands <= 0:
            raise ValueError("lsh_bands must be > 0")
        if self.lsh_rows is None or self.lsh_rows <= 0:
            raise ValueError("lsh_rows must be > 0")
        if self.lsh_bands * self.lsh_rows != self.permutations:
            raise ValueError("lsh_bands * lsh_rows must equal permutations")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0, 1]")
        if self.max_shingles <= 0:
            raise ValueError("max_shingles must be > 0")
        if self.cache_size <= 0:
            raise ValueError("cache_size must be > 0")
        if self.ttl_s <= 0.0:
            raise ValueError("ttl_s must be > 0")
        if self.purge_every <= 0:
            raise ValueError("purge_every must be > 0")
        if self.max_clusters_per_agent <= 0:
            raise ValueError("max_clusters_per_agent must be > 0")
        if self.max_clusters_total <= 0:
            raise ValueError("max_clusters_total must be > 0")
        if self.staleness_k <= 0.0:
            raise ValueError("staleness_k must be > 0")
        if not (0.0 < self.iat_alpha <= 1.0):
            raise ValueError("iat_alpha must be in (0, 1]")
        if self.representation_max_utf8_bytes <= 0:
            raise ValueError("representation_max_utf8_bytes must be > 0")
        if not self.representation_policy:
            raise ValueError("representation_policy must not be empty")
        if not self.representation_version:
            raise ValueError("representation_version must not be empty")

    @property
    def profile_id(self) -> str:
        return (
            "minhash-v1"
            f"|seed={self.seed}"
            f"|n={self.ngram_size}"
            f"|perms={self.permutations}"
            f"|bands={self.lsh_bands}"
            f"|rows={self.lsh_rows}"
            f"|repr_policy={self.representation_policy}"
            f"|repr_version={self.representation_version}"
            f"|repr_max_bytes={self.representation_max_utf8_bytes}"
            f"|max_shingles={self.max_shingles}"
        )
