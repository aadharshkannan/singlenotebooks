from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


class SynonymMap:
    """Maps surface tool tokens to a canonical token per synonym group."""

    def __init__(self, groups: List[List[str]]):
        self.groups = groups
        self._to_canonical: Dict[str, str] = {}
        for group in groups:
            canonical = group[0]
            for tok in group:
                self._to_canonical[tok] = canonical

    def canonical(self, token: str) -> str:
        return self._to_canonical.get(token, token)

    def synonyms(self, canonical: str) -> List[str]:
        for group in self.groups:
            if group[0] == canonical:
                return list(group)
        return [canonical]


@dataclass
class ConceptSpec:
    """A latent behavior concept: a canonical (synonym-normalized) tool subsequence."""
    concept_id: int
    canonical: Tuple[str, ...]

    def __post_init__(self):
        if not self.canonical:
            raise ValueError("ConceptSpec.canonical must be non-empty")


def realize_concept(spec: ConceptSpec, sm: SynonymMap, rng: np.random.Generator,
                    vocab_bias: Optional[Dict[str, str]] = None,
                    edit_prob: float = 0.15) -> Tuple[str, ...]:
    """Turn a concept's canonical sequence into a surface tool sequence.

    * substitute each canonical token with one of its synonyms (biased per-agent
      via ``vocab_bias`` so different agents express the concept differently);
    * apply light edits (drop/duplicate a step) with probability ``edit_prob`` so
      surface sequences vary within a concept.
    """
    vocab_bias = vocab_bias or {}
    out: List[str] = []
    for canon_tok in spec.canonical:
        if canon_tok in vocab_bias:
            surface = vocab_bias[canon_tok]
        else:
            choices = sm.synonyms(canon_tok)
            surface = choices[int(rng.integers(0, len(choices)))]
        if rng.random() < edit_prob:
            continue  # drop this step
        out.append(surface)
        if rng.random() < edit_prob:
            out.append(surface)  # duplicate this step
    if not out:  # never emit an empty sequence
        out.append(vocab_bias.get(spec.canonical[0], spec.canonical[0]))
    return tuple(out)
