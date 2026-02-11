from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Keys for translation engines
    GOOGLE_TRANSLATE_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    HF_TOKEN: Optional[str] = None
    
    # API Model names
    NLLB_MODEL_NAME: str = "facebook/nllb-200-distilled-600M"
    GEMINI_MODEL_NAME: str = "gemini-3-flash-preview"
    
    # App configuration
    APP_NAME: str = "BackTrans-Metrics-Hub"
    DEBUG: bool = False
    ALLOWED_ORIGINS: list[str] = ["*"]
    MAX_ROWS_PER_FILE: int = 1000
    
    # Concurrency configuration (Safely lowered to prevent OOM)
    CONCURRENT_ROWS: int = 3
    CONCURRENT_EVAL_TASKS: int = 1
    GEMINI_MAX_CONCURRENT: int = 2
    GOOGLE_MAX_CONCURRENT: int = 2

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
