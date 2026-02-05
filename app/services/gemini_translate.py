import google.generativeai as genai
import asyncio
from app.core.config import settings

class GeminiTranslateService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing")
        
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL_NAME)
        self._semaphore = asyncio.Semaphore(settings.GEMINI_MAX_CONCURRENT)

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        retries: int = 2
    ) -> str:
        
        # Strict Prompting to ensure ONLY translation is returned
        prompt = (
            f"You are a strict translation engine. Your task is to translate the following text from {source_lang} to {target_lang}.\n"
            "CRITICAL RULES:\n"
            "1. Output ONLY the translated text.\n"
            "2. DO NOT include any introductory or concluding remarks (e.g., 'Here is the translation', 'Sure').\n"
            "3. DO NOT add any notes or explanations.\n"
            "4. Maintain the original tone and formatting exactly.\n"
            "5. If the text is already in the target language, return it exactly as is.\n"
            "6. DO NOT change or fix any grammatical errors or incoherence - translate exactly as written.\n\n"
            f"Input Text:\n{text}"
        )

        generation_config = genai.types.GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            max_output_tokens=1024,
        )

        for attempt in range(retries + 1):
            async with self._semaphore:
                try:
                    # Run the blocking SDK call in a thread to remain async-friendly
                    response = await asyncio.to_thread(
                        self.model.generate_content,
                        prompt,
                        generation_config=generation_config
                    )
                    
                    if response.text:
                        return response.text.strip()
                    else:
                        raise RuntimeError("Gemini returned empty response")

                except Exception as e:
                    # Check for rate limit or quota issues if possible, or just retry generic errors
                    # The SDK raises variations of GoogleAPIError
                    if "429" in str(e) or "Resource exhausted" in str(e):
                        wait_time = 2 + (attempt * 3)
                        print(f"Gemini Rate Limited (429). Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if attempt < retries:
                        print(f"Gemini error: {e}. Retrying...")
                        await asyncio.sleep(1)
                        continue
                    
                    raise RuntimeError(f"Gemini translation failed: {e}")

        raise RuntimeError("Gemini Flash failed after multiple retries")


gemini_service = GeminiTranslateService()
