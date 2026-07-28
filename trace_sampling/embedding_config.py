import os
from dataclasses import dataclass
from typing import Optional


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:
        pass


def _parse_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off; got {value!r}"
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    full_session_enabled: bool
    model_id: str
    model_version: str
    tokenizer_id: str
    tokenizer_encoding: Optional[str]
    max_input_tokens: int

    @classmethod
    def from_env(cls, default_model_id: str) -> "EmbeddingConfig":
        _load_dotenv()
        model_id = os.environ.get("SESSION_EMBEDDING_MODEL_ID", default_model_id)
        max_input_tokens = int(
            os.environ.get("SESSION_EMBEDDING_MAX_INPUT_TOKENS", "8191")
        )
        if max_input_tokens < 1:
            raise ValueError("SESSION_EMBEDDING_MAX_INPUT_TOKENS must be >= 1")
        return cls(
            full_session_enabled=_parse_bool("ENABLE_FULL_SESSION_EMBEDDINGS"),
            model_id=model_id,
            model_version=os.environ.get("SESSION_EMBEDDING_MODEL_VERSION", model_id),
            tokenizer_id=os.environ.get("SESSION_EMBEDDING_TOKENIZER_ID", model_id),
            tokenizer_encoding=os.environ.get("SESSION_EMBEDDING_TOKENIZER_ENCODING") or None,
            max_input_tokens=max_input_tokens,
        )
