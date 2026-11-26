from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CITERAG_")

    llm_provider: str = "openai"
    chroma_path: str = "./data/chroma"
    max_upload_mb: int = 20
    embeddings: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_retries: int = 5
    openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")


settings = Settings()
