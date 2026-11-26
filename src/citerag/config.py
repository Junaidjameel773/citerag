from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CITERAG_")

    llm_provider: str = "openai"
    llm_model: str = ""  # empty: provider default (see llm.DEFAULT_MODELS)
    chroma_path: str = "./data/chroma"
    max_upload_mb: int = 20
    embeddings: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_retries: int = 5
    openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")
    gemini_api_key: str = Field("", validation_alias="GEMINI_API_KEY")


settings = Settings()
