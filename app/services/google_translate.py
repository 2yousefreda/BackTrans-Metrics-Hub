import httpx
import asyncio
from app.core.config import settings

class GoogleTranslateService:
    def __init__(self):
        self.api_key = settings.GOOGLE_TRANSLATE_API_KEY
        self.base_url = "https://translation.googleapis.com/language/translate/v2"
        self._client = None
        self._semaphore = asyncio.Semaphore(settings.GOOGLE_MAX_CONCURRENT)

    async def _get_client(self):
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def translate(self, text: str, source_lang: str, target_lang: str, retries: int = 3) -> str:
        """
        Translates text using Google Translate API with pooling, rate limiting, and retries.
        """
        if not self.api_key:
            raise RuntimeError("GOOGLE_TRANSLATE_API_KEY is missing via .env or config")

        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key
        }

        client = await self._get_client()

        for attempt in range(retries + 1):
            async with self._semaphore:
                try:
                    response = await client.post(self.base_url, json=payload, headers=headers)
                    
                    if response.status_code == 429:
                        wait_time = 2 + (attempt * 2)
                        print(f"Google Translate Rate Limited (429). Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                        
                    response.raise_for_status()
                    result = response.json()
                    
                    if "data" in result and "translations" in result["data"]:
                        return result["data"]["translations"][0]["translatedText"]
                    else:
                        raise RuntimeError(f"Unexpected Google Translate response: {result}")
                        
                except httpx.HTTPStatusError as e:
                    if attempt < retries:
                        print(f"Google API Error {e.response.status_code}. Retrying...")
                        await asyncio.sleep(1)
                        continue
                    raise RuntimeError(f"Google Translate API Error {e.response.status_code}: {e.response.text}")
                except (httpx.RequestError, Exception) as e:
                    if attempt < retries:
                        print(f"Google Translate network error: {e}. Retrying...")
                        await asyncio.sleep(1)
                        continue
                    raise RuntimeError(f"Google Translate failed: {e}")

        raise RuntimeError("Google Translate failed after multiple retries")

# Instantiate for use in the app
google_service = GoogleTranslateService()
