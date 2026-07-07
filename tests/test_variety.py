from trace_sampling.model import Trace
from trace_sampling.variety import VarietyKey, ExactSignatureIndex


def _t(sig, agent="a", ts=0.0):
    return Trace(0, agent, ts, sig, len(sig), 1.0, "ok")


def test_exact_index_first_seen_rarity_is_half():
    idx = ExactSignatureIndex()
    obs = idx.observe(_t(("search",)))
    assert obs.rarity == 0.5
    assert obs.novelty == 1.0
    assert obs.key == VarietyKey("signature", ("search",))


def test_exact_index_rarity_decreases_with_repeats():
    idx = ExactSignatureIndex()
    idx.observe(_t(("search",)))
    obs = idx.observe(_t(("search",)))
    assert obs.rarity == 1.0 / 3.0
    assert obs.novelty == 0.0


def test_exact_index_key_is_tagged():
    idx = ExactSignatureIndex()
    obs = idx.observe(_t(("a", "b")))
    assert obs.key.kind == "signature"
    assert obs.key.value == ("a", "b")
