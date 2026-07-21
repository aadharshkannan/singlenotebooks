import numpy as np
from trace_sampling.concepts import ConceptSpec, SynonymMap, realize_concept
import pytest

def test_synonym_map_groups_tokens():
    sm = SynonymMap([["search", "query", "find"], ["edit", "modify"]])
    assert sm.canonical("query") == "search"
    assert sm.canonical("modify") == "edit"
    assert sm.canonical("run") == "run"  # ungrouped -> itself

def test_realize_concept_preserves_canonical_sequence():
    sm = SynonymMap([["search", "query", "find"], ["edit", "modify"]])
    spec = ConceptSpec(concept_id=1, canonical=("search", "edit"))
    rng = np.random.default_rng(0)
    surface = realize_concept(spec, sm, rng, vocab_bias={"search": "query", "edit": "modify"},
                              edit_prob=0.0)
    canon = tuple(sm.canonical(t) for t in surface if sm.canonical(t) in ("search", "edit"))
    assert canon[:2] == ("search", "edit")

def test_concept_spec_rejects_empty_canonical():
    with pytest.raises(ValueError):
        ConceptSpec(concept_id=0, canonical=())

