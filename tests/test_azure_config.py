import os
from trace_sampling.azure_config import AzureConfig
from trace_sampling.embedding import build_openai_embedding_client


def test_azure_config_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "openai-key-123")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://y.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "search-key-456")
    monkeypatch.setenv("AZURE_SEARCH_INDEX", "trace-clusters")
    cfg = AzureConfig.from_env()
    assert cfg.openai_endpoint.endswith("openai.azure.com/")
    assert cfg.openai_api_key == "openai-key-123"
    assert cfg.embedding_deployment == "text-embedding-3-small"
    assert cfg.search_endpoint == "https://y.search.windows.net"
    assert cfg.search_api_key == "search-key-456"
    assert cfg.search_index == "trace-clusters"
    assert "openai-key-123" not in repr(cfg)
    assert "search-key-456" not in repr(cfg)


def test_azure_config_keeps_legacy_positional_construction(monkeypatch):
    cfg = AzureConfig(
        "https://legacy.example.com",
        "2024-02-01",
        "embedding-model",
        "https://search.example.com",
        "trace-clusters",
    )
    assert cfg.openai_api_key is None
    assert cfg.search_api_key is None


def test_build_openai_embedding_client_uses_modern_foundry_route():
    seen = {}

    class FakeOpenAI:
        def __init__(self, *, base_url, api_key):
            seen["base_url"] = base_url
            seen["api_key"] = api_key

    class FakeAzureOpenAI:
        def __init__(self, **kwargs):
            raise AssertionError("legacy client should not be used for Foundry endpoints")

    cfg = AzureConfig(
        openai_endpoint="https://example.services.ai.azure.com",
        openai_api_version="2024-02-01",
        embedding_deployment="text-embedding-3-small",
        search_endpoint="https://search.example.com",
        search_index="trace-clusters",
        openai_api_key="secret-key",
        search_api_key="search-secret",
    )

    client = build_openai_embedding_client(cfg, openai_cls=FakeOpenAI, azure_openai_cls=FakeAzureOpenAI)
    assert client is not None
    assert seen["base_url"] == "https://example.services.ai.azure.com/openai/v1/"
    assert seen["api_key"] == "secret-key"
