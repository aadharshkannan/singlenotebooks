import numpy as np
import pytest

from trace_sampling.capi_embedding import CapiEmbedder, CapiEmbedding, CapiTokenizer


class ReversingTransport:
    def embed(self, model, texts):
        assert model == "capi-v1"
        return [
            CapiEmbedding(index=index, vector=[float(index), float(len(text))])
            for index, text in reversed(list(enumerate(texts)))
        ]


def test_capi_embedder_restores_input_order():
    embedder = CapiEmbedder(ReversingTransport(), model="capi-v1", dimensions=2)

    vectors = embedder.embed(["one", "longer"])

    assert vectors.dtype == np.float32
    assert np.array_equal(vectors, np.array([[0.0, 3.0], [1.0, 6.0]], dtype=np.float32))


def test_capi_embedder_rejects_missing_response_indices():
    class MissingTransport:
        def embed(self, model, texts):
            return [CapiEmbedding(index=0, vector=[1.0, 2.0])]

    with pytest.raises(ValueError, match="indices"):
        CapiEmbedder(MissingTransport(), "capi-v1", dimensions=2).embed(["one", "two"])


def test_capi_tokenizer_delegates_to_exact_model_counter():
    calls = []

    def count(model, text):
        calls.append((model, text))
        return len(text.encode("utf-8"))

    tokenizer = CapiTokenizer(count, model="capi-v1", version="2026-07")

    assert tokenizer.count("hé") == 3
    assert calls == [("capi-v1", "hé")]
