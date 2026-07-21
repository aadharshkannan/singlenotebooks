import os
from trace_sampling.azure_config import AzureConfig

def test_azure_config_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://y.search.windows.net")
    monkeypatch.setenv("AZURE_SEARCH_INDEX", "trace-clusters")
    cfg = AzureConfig.from_env()
    assert cfg.openai_endpoint.endswith("openai.azure.com/")
    assert cfg.embedding_deployment == "text-embedding-3-small"
    assert cfg.search_endpoint == "https://y.search.windows.net"
    assert cfg.search_index == "trace-clusters"
