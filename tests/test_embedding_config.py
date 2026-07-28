import pytest

from trace_sampling.embedding import EmbeddingCache, FakeEmbedder
from trace_sampling.embedding_config import EmbeddingConfig
from trace_sampling.eval_harness import _make_embedding_cache
from trace_sampling.session_embedding import SessionEmbeddingCache


def _clear_embedding_env(monkeypatch):
    for name in (
        "ENABLE_FULL_SESSION_EMBEDDINGS",
        "SESSION_EMBEDDING_MODEL_ID",
        "SESSION_EMBEDDING_MODEL_VERSION",
        "SESSION_EMBEDDING_TOKENIZER_ID",
        "SESSION_EMBEDDING_TOKENIZER_ENCODING",
        "SESSION_EMBEDDING_MAX_INPUT_TOKENS",
        "SESSION_REPRESENTATION_MAX_UTF8_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_full_session_embeddings_default_to_disabled(monkeypatch):
    _clear_embedding_env(monkeypatch)

    config = EmbeddingConfig.from_env("text-embedding-3-small")
    cache = _make_embedding_cache(FakeEmbedder(), "text-embedding-3-small")

    assert config.full_session_enabled is False
    assert isinstance(cache, EmbeddingCache)


def test_false_uses_original_signature_embedding_cache(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("ENABLE_FULL_SESSION_EMBEDDINGS", "FALSE")

    cache = _make_embedding_cache(FakeEmbedder(), "text-embedding-3-small")

    assert isinstance(cache, EmbeddingCache)


def test_true_uses_full_session_embedding_cache(monkeypatch):
    class CharacterTokenizer:
        name = "characters"
        version = "1"

        def __init__(self, model_name, encoding_name=None):
            pass

        def count(self, text):
            return len(text)

    _clear_embedding_env(monkeypatch)
    monkeypatch.setattr(
        "trace_sampling.session_embedding.TiktokenTokenizer",
        CharacterTokenizer,
    )
    monkeypatch.setenv("ENABLE_FULL_SESSION_EMBEDDINGS", "TRUE")
    monkeypatch.setenv("SESSION_EMBEDDING_MODEL_ID", "capi-embedding")
    monkeypatch.setenv("SESSION_EMBEDDING_MODEL_VERSION", "2026-07")
    monkeypatch.setenv("SESSION_EMBEDDING_TOKENIZER_ID", "text-embedding-3-small")
    monkeypatch.setenv("SESSION_EMBEDDING_MAX_INPUT_TOKENS", "1024")
    monkeypatch.setenv("SESSION_REPRESENTATION_MAX_UTF8_BYTES", "4096")

    cache = _make_embedding_cache(FakeEmbedder(), "text-embedding-3-small")

    assert isinstance(cache, SessionEmbeddingCache)
    assert cache.profile.model_id == "capi-embedding"
    assert cache.profile.model_version == "2026-07"
    assert cache.profile.max_input_tokens == 1024
    assert cache.profile.representation_max_utf8_bytes == 4096


def test_invalid_boolean_is_rejected(monkeypatch):
    _clear_embedding_env(monkeypatch)
    monkeypatch.setenv("ENABLE_FULL_SESSION_EMBEDDINGS", "sometimes")

    with pytest.raises(ValueError, match="ENABLE_FULL_SESSION_EMBEDDINGS"):
        EmbeddingConfig.from_env("text-embedding-3-small")
