import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AzureConfig:
    openai_endpoint: str
    openai_api_version: str
    embedding_deployment: str
    search_endpoint: str
    search_index: str

    @classmethod
    def from_env(cls) -> "AzureConfig":
        # Load a local .env if present (no-op if python-dotenv isn't installed
        # or the file is absent). Real env vars always win over .env values.
        try:
            from dotenv import load_dotenv
            load_dotenv(override=False)
        except Exception:
            pass
        return cls(
            openai_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            embedding_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            search_index=os.environ.get("AZURE_SEARCH_INDEX", "trace-clusters"),
        )


def get_credential():
    """Entra-only credential (API keys are policy-disabled)."""
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential()


def openai_token_provider():
    from azure.identity import get_bearer_token_provider
    return get_bearer_token_provider(
        get_credential(), "https://cognitiveservices.azure.com/.default")
