import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AzureConfig:
    openai_endpoint: str
    openai_api_version: str
    embedding_deployment: str
    search_endpoint: str
    search_index: str
    openai_api_key: str | None = field(default=None, repr=False)
    search_api_key: str | None = field(default=None, repr=False)

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
            openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            search_api_key=os.environ.get("AZURE_SEARCH_API_KEY"),
        )


def get_credential():
    """Construct the default Entra credential used for Azure AD token flows."""
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential()


def openai_token_provider():
    from azure.identity import get_bearer_token_provider
    return get_bearer_token_provider(
        get_credential(), "https://cognitiveservices.azure.com/.default")


def get_openai_token() -> str:
    """Acquire a concrete OpenAI bearer token string for SDKs that require string API keys."""
    provider = openai_token_provider()
    token = provider()
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("Failed to acquire Azure OpenAI bearer token")
    return token
