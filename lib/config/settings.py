from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-5.6-sol", alias="OPENAI_CHAT_MODEL")
    openai_vector_store_id = str = Field(
        default="openai_vector_store_id", alias="OPENAI_VECTOR_STORE_ID"
    )

    mongodb_uri: str | None = Field(default=None, alias="MONGODB_URI")
    mongodb_db: str = Field(default="RAG_assessment", alias="MONGODB_DB")
    mongodb_sessions_collection: str = Field(
        default="sessions", alias="MONGODB_SESSIONS_COLLECTION"
    )
    mongodb_pdf_collection: str = Field(default="pdf", alias="MONGODB_PDF_COLLECTION")

    langfuse_secret_key: str = Field(
        default="langfuse_secret_key", alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_public_key: str = Field(
        default="langfuse_public_key", alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_base_url: str = Field(
        default="langfuse_base_url", alias="LANGFUSE_BASE_URL"
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
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return self.openai_api_key

    def require_mongodb(self) -> str:
        if not self.mongodb_uri:
            raise RuntimeError("MONGODB_URI is not configured")
        return self.mongodb_uri

    def require_langfuse(self) -> None:
        if not self.langfuse_public_key or not self.langfuse_secret_key:
            raise RuntimeError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured"
            )


def get_settings():
    settings = Settings()
    return settings
