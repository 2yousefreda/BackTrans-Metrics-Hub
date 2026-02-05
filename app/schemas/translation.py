from pydantic import BaseModel, Field
from enum import Enum

class engineEnum(str, Enum):
    GOOGLE = "google"
    GEMINI = "gemini"
    NLLB = "nllb"

class TranslationRequest(BaseModel):
    text: str = Field(..., example="Hello world", description="The source text to translate")
    source_lang: str = Field(..., example="en", description="Source language code (ISO 639-1)")
    target_lang: str = Field(..., example="ar", description="Target language code (ISO 639-1)")
    engine: engineEnum = Field(..., description="Translation engine to use")

class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    back_translated_text: str
    engine: str
    source_lang: str
    target_lang: str
