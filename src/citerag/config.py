from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CITERAG_")

    llm_provider: str = "openai"
    chroma_path: str = "./data/chroma"


settings = Settings()
