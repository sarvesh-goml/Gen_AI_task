import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/shop",
        validation_alias="DATABASE_URL"
    )
    GROQ_API_KEY: str = Field(
        default="",
        validation_alias="GROQ_API_KEY"
    )
    GROQ_MODEL_NAME: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="GROQ_MODEL_NAME"
    )
    GROQ_API_BASE: str = Field(
        default="https://api.groq.com/openai/v1",
        validation_alias="GROQ_API_BASE"
    )
    EMBEDDING_MODEL_NAME: str = Field(
        default="BAAI/bge-small-en-v1.5",
        validation_alias="EMBEDDING_MODEL_NAME"
    )
    ENVIRONMENT: str = Field(
        default="development",
        validation_alias="ENVIRONMENT"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
