from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-5.6-sol", alias="OPENAI_CHAT_MODEL")
    zenrows_api_key: str | None = Field(default=None, alias="ZENROWS_API_KEY")

    mongodb_uri: str | None = Field(default=None, alias="MONGODB_URI")
    mongodb_db: str = Field(default="RAG_assessment", alias="MONGODB_DB")
    mongodb_sessions_collection: str = Field(
        default="sessions", alias="MONGODB_SESSIONS_COLLECTION"
    )
    mongodb_chat_collection: str = Field(default="chat", alias="MONGODB_CHAT_COLLECTION")

    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        alias="LANGFUSE_BASE_URL",
    )

    litellm_proxy_api_base: str = Field(
        default="http://localhost:4000",
        alias="LITELLM_PROXY_API_BASE",
    )
    litellm_proxy_api_key: str | None = Field(
        default=None,
        alias="LITELLM_PROXY_API_KEY",
    )

    def require_openai(self) -> str:
        if not self.openai_api_key:
            raise ConfigurationError("OPENAI_API_KEY is not configured")
        return self.openai_api_key

    def require_zenrows(self) -> str:
        if not self.zenrows_api_key:
            raise ConfigurationError("ZENROWS_API_KEY is not configured")
        return self.zenrows_api_key

    def require_mongodb(self) -> str:
        if not self.mongodb_uri:
            raise ConfigurationError("MONGODB_URI is not configured")
        return self.mongodb_uri

    def require_langfuse(self) -> tuple[str, str]:
        if not self.langfuse_public_key or not self.langfuse_secret_key:
            raise ConfigurationError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured"
            )
        return self.langfuse_public_key, self.langfuse_secret_key


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    return settings
