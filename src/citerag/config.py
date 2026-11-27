from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CITERAG_")

    llm_provider: str = "openai"
    llm_model: str = ""  # empty: provider default (see llm.DEFAULT_MODELS)
    ollama_url: str = "http://localhost:11434"
    chroma_path: str = "./data/chroma"
    max_upload_mb: int = 20
    embeddings: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_retries: int = 5
    hybrid_vector_weight: float = Field(1.0, ge=0)
    hybrid_keyword_weight: float = Field(1.0, ge=0)
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")
    gemini_api_key: str = Field("", validation_alias="GEMINI_API_KEY")
    groq_api_key: str = Field("", validation_alias="GROQ_API_KEY")


settings = Settings()
